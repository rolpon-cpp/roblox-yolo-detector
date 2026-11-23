# This script was written by Claude Opus 5.5.
# Test-time views. A view runs one model on the whole frame or on regions of it (tiles, zoom crops), optionally
# mirrored, and maps every box back to full-frame pixels. Each result records the regions the model looked at and
# flags boxes that touch an inner region edge (characters cut by a tile or crop border), so fusion can drop them and
# can tell which views were able to see a given box.
# View names: full@S, flip@S (mirrored), tiles<g>@S (g x g overlapping tiles), tflip<g>@S (tiles of the mirrored frame).

import re

import numpy as np

EDGE_TOL = 3


def grid_regions(w, h, g, overlap=0.1):
    """g x g tiles with the frame's aspect ratio, neighbours overlapping by `overlap` of the frame width."""
    if g == 1:
        return [(0, 0, w, h)]
    ov = int(round(overlap * w))
    tw = int(np.ceil((w + (g - 1) * ov) / g))
    th = int(np.ceil((h + (g - 1) * ov * h / w) / g))
    xs = np.linspace(0, w - tw, g).astype(int)
    ys = np.linspace(0, h - th, g).astype(int)
    return [(int(x), int(y), int(x + tw), int(y + th)) for y in ys for x in xs]


def crop_region(box, w, h, ctx=4.0, min_frac=0.2):
    """Crop with the frame's aspect ratio centred on `box`; its height is ctx x the box's (scaled to cover the box's
    width too), at least min_frac of the frame height, at most the whole frame."""
    bw, bh = box[2] - box[0], box[3] - box[1]
    ar = w / h
    ch = max(bh * ctx, bw * ctx / ar, min_frac * h)
    ch = min(ch, h)
    cw = min(ch * ar, w)
    ch = cw / ar
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    x1 = max(int(round(min(max(cx - cw / 2, 0), w - cw))), 0)
    y1 = max(int(round(min(max(cy - ch / 2, 0), h - ch))), 0)
    # rounding can put the far edge 1 px past the frame, which gives an empty crop on the mirrored frame
    return (x1, y1, min(int(round(x1 + cw)), w), min(int(round(y1 + ch)), h))


def unflip(b, w):
    if len(b) == 0:
        return b
    o = b.copy()
    o[:, 0] = w - b[:, 2]
    o[:, 2] = w - b[:, 0]
    return o


def flip_region(r, w):
    return (w - r[2], r[1], w - r[0], r[3])


def predict_regions(model, img, regions, size, flip=False):
    """Runs `model` on each region of `img` (mirrored first if flip) and returns one view result in frame pixels."""
    h, w = img.shape[:2]
    src = np.ascontiguousarray(img[:, ::-1]) if flip else img
    regs = [flip_region(r, w) for r in regions] if flip else list(regions)
    outs = model.predict([src[y1:y2, x1:x2] for (x1, y1, x2, y2) in regs], size)
    B, C, E, R = [], [], [], []
    for j, ((x1, y1, x2, y2), (b, c)) in enumerate(zip(regs, outs)):
        inner = np.zeros(len(b), bool)
        if x1 > 0:
            inner |= b[:, 0] <= EDGE_TOL
        if y1 > 0:
            inner |= b[:, 1] <= EDGE_TOL
        if x2 < w:
            inner |= b[:, 2] >= (x2 - x1) - EDGE_TOL
        if y2 < h:
            inner |= b[:, 3] >= (y2 - y1) - EDGE_TOL
        B.append(b + np.array([x1, y1, x1, y1], np.float32))
        C.append(c)
        E.append(inner)
        R.append(np.full(len(b), j, np.int32))
    b = np.concatenate(B) if B else np.zeros((0, 4), np.float32)
    if flip:
        b = unflip(b, w)
    return dict(boxes=b.astype(np.float32), conf=np.concatenate(C).astype(np.float32) if C else np.zeros(0, np.float32),
                edge=np.concatenate(E) if E else np.zeros(0, bool),
                region_idx=np.concatenate(R) if R else np.zeros(0, np.int32), regions=np.array(regions, np.float32).reshape(-1, 4))


def parse_view(name):
    m = re.fullmatch(r"(full|flip|tiles|tflip)(\d*)@(\d+)", name)
    if not m:
        raise ValueError(f"bad view name {name}")
    kind, g, size = m.group(1), int(m.group(2) or 2), int(m.group(3))
    return kind, g, size


def run_view(model, img, name):
    kind, g, size = parse_view(name)
    h, w = img.shape[:2]
    regions = [(0, 0, w, h)] if kind in ("full", "flip") else grid_regions(w, h, g)
    return predict_regions(model, img, regions, size, flip=kind in ("flip", "tflip"))
