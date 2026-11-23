# This script was written by Claude Opus 5.5.
# Merging many views of many models into one set of boxes (numpy only).
# prepare(): per view, drop boxes cut by an inner tile or crop edge, drop boxes under `skip`, merge tile overlaps with
#   NMS, keep the top `topk`.
# wbf(): weighted box fusion with a visibility-aware score: the fused confidence is the summed best confidence of each
#   member that found the box, divided by the total weight of the members that could have seen it (a tile or zoom view
#   cannot see a box outside its regions or cut by their inner edges, so it does not count against that box). Box
#   coordinates are a weighted mean of the cluster, with an extra per-member coordinate weight (lets sharper views,
#   such as zoomed ones, decide the box edges more than they decide the score).

import numpy as np


def iou_one(f, b):
    ix1 = np.maximum(f[:, 0], b[0]); iy1 = np.maximum(f[:, 1], b[1])
    ix2 = np.minimum(f[:, 2], b[2]); iy2 = np.minimum(f[:, 3], b[3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ua = (f[:, 2] - f[:, 0]) * (f[:, 3] - f[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / np.maximum(ua, 1e-9)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def nms(boxes, conf, thr=0.6, max_det=1000):
    if len(boxes) == 0:
        return np.zeros(0, int)
    order = np.argsort(-conf, kind="stable")
    b = boxes[order]
    alive = np.ones(len(b), bool)
    keep = []
    for i in range(len(b)):
        if not alive[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        rest = np.nonzero(alive[i + 1:])[0] + i + 1
        if not len(rest):
            break
        alive[rest[iou_one(b[rest], b[i]) > thr]] = False
    return order[np.array(keep, int)]


def prepare(view, skip=0.0, topk=300, nms_thr=0.6):
    """Boxes and confidences of one view result, cleaned for fusion."""
    b, c, e = view["boxes"], view["conf"], view["edge"]
    k = (~e) & (c >= skip)
    b, c = b[k], c[k]
    if len(view["regions"]) > 1 and len(b):
        k = nms(b, c, nms_thr)
        b, c = b[k], c[k]
    o = np.argsort(-c)[:topk]
    return b[o], c[o]


def visible(regions, F, w, h, tol=4.0):
    """(n_fused,) bool: some region contains the box without cutting it at an inner edge."""
    if len(F) == 0:
        return np.zeros(0, bool)
    R = np.asarray(regions, np.float32)
    if len(R) == 1 and R[0, 0] <= 0 and R[0, 1] <= 0 and R[0, 2] >= w and R[0, 3] >= h:
        return np.ones(len(F), bool)
    x1, y1, x2, y2 = (R[None, :, i] for i in range(4))
    f0, f1, f2, f3 = (F[:, None, i] for i in range(4))
    ok = (((x1 <= 0) | (f0 > x1 + tol)) & ((y1 <= 0) | (f1 > y1 + tol))
          & ((x2 >= w) | (f2 < x2 - tol)) & ((y2 >= h) | (f3 < y2 - tol))
          & (f0 >= x1 - tol) & (f1 >= y1 - tol) & (f2 <= x2 + tol) & (f3 <= y2 + tol))
    return ok.any(1)


def wbf(members, w, h, iou_thr=0.7, conf_mode="vis", max_det=300):
    """members: list of dicts with boxes, conf (already transformed), weight, coord_weight (default 1), regions.
    Returns fused boxes (n,4) float32 and confidences (n,) sorted high to low, plus the member count per box."""
    M = len(members)
    if M == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    wts = np.array([m["weight"] for m in members], np.float64)
    B, C, CW, I = [], [], [], []
    for i, m in enumerate(members):
        if len(m["boxes"]):
            B.append(m["boxes"])
            c = m["conf"].astype(np.float64) * m["weight"]
            C.append(c)
            CW.append(c * m.get("coord_weight", 1.0))
            I.append(np.full(len(c), i, np.int32))
    if not B:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    B = np.concatenate(B).astype(np.float64)
    C, CW, I = np.concatenate(C), np.concatenate(CW), np.concatenate(I)
    o = np.argsort(-C, kind="stable")
    B, C, CW, I = B[o], C[o], CW[o], I[o]
    n = len(B)
    fused = np.zeros((n, 4)); acc = np.zeros((n, 4)); accw = np.zeros(n); mem = np.zeros((n, M))
    nf = 0
    for j in range(n):
        b = B[j]
        if nf:
            iou = iou_one(fused[:nf], b)
            k = int(iou.argmax())
            if iou[k] > iou_thr:
                acc[k] += b * CW[j]
                accw[k] += CW[j]
                if C[j] > mem[k, I[j]]:
                    mem[k, I[j]] = C[j]
                if accw[k] > 0:
                    fused[k] = acc[k] / accw[k]
                continue
        acc[nf] = b * CW[j]; accw[nf] = CW[j]; mem[nf, I[j]] = C[j]; fused[nf] = b
        nf += 1
    fused, mem = fused[:nf], mem[:nf]
    if conf_mode == "vis":
        el = np.stack([visible(m["regions"], fused, w, h) for m in members], 1) | (mem > 0)
        conf = mem.sum(1) / np.maximum((el * wts[None]).sum(1), 1e-9)
    elif conf_mode == "avg":
        conf = mem.sum(1) / wts.sum()
    elif conf_mode == "max":
        conf = (mem / wts[None]).max(1)
    else:
        raise ValueError(conf_mode)
    o = np.argsort(-conf, kind="stable")[:max_det]
    return fused[o].astype(np.float32), conf[o].astype(np.float32), (mem[o] > 0).sum(1)
