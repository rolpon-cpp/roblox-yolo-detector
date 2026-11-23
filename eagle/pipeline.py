# This script was written by Claude Opus 5.5.
# MaxDetector: the most accurate Roblox character predictor we can build from existing models, regardless of speed.
# Stages (each one kept only if it raised mAP50-95 on clean validation images, see README.md):
#   1. Several models x several views (whole frame, mirror, tiles), merged by visibility-aware weighted box fusion.
#   2. Zoom: crops around uncertain boxes are re-predicted closer up and fused in as extra views.
#   3. Box tightening: each confident box is re-predicted in a close crop and its edges are pulled toward the
#      close-up box.
# Config: a dict or a JSON file (see config.json); model paths are relative to the repository folder.

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .fusion import iou_matrix, prepare, wbf
from .models import load_model
from .views import crop_region, predict_regions, run_view

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
DEFAULT_CONFIG = PKG / "config.json"


@dataclass
class Detections:
    boxes: np.ndarray  # (n, 4) x1, y1, x2, y2 in pixels
    conf: np.ndarray   # (n,)
    width: int
    height: int

    def __len__(self):
        return len(self.conf)

    def above(self, thr):
        k = self.conf >= thr
        return Detections(self.boxes[k], self.conf[k], self.width, self.height)

    def to_yolo_lines(self, with_conf=True):
        w, h = self.width, self.height
        out = []
        for (x1, y1, x2, y2), c in zip(self.boxes, self.conf):
            s = f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            out.append(s + (f" {c:.5f}" if with_conf else ""))
        return out

    def draw(self, img, thr=0.5, color=(0, 220, 255)):
        """Copy of img (BGR) with every box at confidence thr or more and its confidence; line width follows image size."""
        im = img.copy()
        t = max(2, round(max(im.shape[:2]) / 700))
        for (x1, y1, x2, y2), c in zip(self.boxes, self.conf):
            if c >= thr:
                cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), color, t)
                cv2.putText(im, f"{c:.2f}", (int(x1), max(int(y1) - 2 * t, 12 * t)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35 * t, color, max(1, t // 2))
        return im


def _resolve(p, root):
    p = Path(p)
    return p if p.is_absolute() else Path(root) / p


class MaxDetector:
    def __init__(self, config=None, device="cuda:0", root=ROOT, half=False):
        if config is None or isinstance(config, (str, Path)):
            config = json.loads(Path(config or DEFAULT_CONFIG).read_text(encoding="utf-8"))
        self.cfg = config
        self.device = device
        self.root = root
        self.half = half  # RF-DETR members in float16 on a GPU: about 2x faster, same boxes within noise
        self._models = {}

    # ---- models -------------------------------------------------------------------------------------------------
    def model(self, name):
        if name not in self._models:
            spec = self.cfg["models"][name]
            self._models[name] = load_model(_resolve(spec["path"], self.root), spec.get("imgsz"), self.device, self.half)
        return self._models[name]

    # ---- stage 1: models x views ----------------------------------------------------------------------------------
    def stage1_views(self, img, members=None):
        """Raw view results {(model, view): result} for the stage-1 members (or the given (model, view) pairs)."""
        h, w = img.shape[:2]
        s1 = self.cfg["stage1"]
        pairs = members if members is not None else [(m["model"], m["view"]) for m in s1["members"]]
        out = {}
        for name, view in pairs:
            if view.startswith(("tiles", "tflip")) and max(h, w) < s1.get("min_tile_side", 0):
                continue
            out[(name, view)] = run_view(self.model(name), img, view)
        return out

    def stage1_members(self, views):
        s1 = self.cfg["stage1"]
        mems = []
        for m in s1["members"]:
            v = views.get((m["model"], m["view"]))
            if v is None:
                continue
            b, c = prepare(v, s1.get("skip", 0.0), s1.get("topk", 300))
            p = s1.get("conf_power", {}).get(m["model"], 1.0)
            mems.append(dict(boxes=b, conf=c ** p if p != 1.0 else c, weight=m.get("weight", 1.0),
                             coord_weight=m.get("coord_weight", 1.0), regions=v["regions"]))
        return mems

    # ---- stage 2: zoom on uncertain boxes ------------------------------------------------------------------------
    def zoom_regions(self, boxes, conf, w, h):
        z = self.cfg["zoom"]
        k = np.nonzero((conf >= z["lo"]) & (conf <= z["hi"]))[0][:z["max_regions"]]
        return [crop_region(boxes[i], w, h, z["ctx"], z["min_frac"]) for i in k]

    def zoom_views(self, img, regions):
        z = self.cfg["zoom"]
        out = {}
        if not regions:
            return out
        for name in z["models"]:
            out[(name, "zoom")] = predict_regions(self.model(name), img, regions, z["size"])
            if z.get("flip"):
                out[(name, "zoomflip")] = predict_regions(self.model(name), img, regions, z["size"], flip=True)
        return out

    def zoom_members(self, zviews):
        z = self.cfg["zoom"]
        mems = []
        for v in zviews.values():
            b, c = prepare(v, z.get("skip", 0.0), z.get("topk", 300))
            mems.append(dict(boxes=b, conf=c, weight=z.get("weight", 1.0), coord_weight=z.get("coord_weight", 1.0),
                             regions=v["regions"]))
        return mems

    # ---- stage 3: box tightening ----------------------------------------------------------------------------------
    def refine_targets(self, boxes, conf):
        r = self.cfg["refine"]
        return np.nonzero(conf >= r["min_conf"])[0][:r["max_boxes"]]

    def refine_views(self, img, boxes, idx):
        """Close-up predictions per target box: {(model, flip): [(boxes, conf) per target]} (cut boxes removed)."""
        r = self.cfg["refine"]
        h, w = img.shape[:2]
        regions = [crop_region(boxes[i], w, h, r["ctx"], r["min_frac"]) for i in idx]
        out = {}
        if not regions:
            return out
        for name in r["models"]:
            for fl in ([False, True] if r.get("flip") else [False]):
                v = predict_regions(self.model(name), img, regions, r["size"], flip=fl)
                keep = ~v["edge"]
                out[(name, fl)] = [(v["boxes"][keep & (v["region_idx"] == j)], v["conf"][keep & (v["region_idx"] == j)])
                                   for j in range(len(regions))]
        return out

    def apply_refine(self, boxes, idx, rviews):
        r = self.cfg["refine"]
        boxes = boxes.copy()
        for j, i in enumerate(idx):
            cand, wts = [], []
            for preds in rviews.values():
                b, c = preds[j]
                if not len(b):
                    continue
                iou = iou_matrix(boxes[i:i + 1], b)[0]
                k = int(iou.argmax())
                if iou[k] >= r["min_iou"]:
                    cand.append(b[k])
                    wts.append(c[k])
            if cand:
                tgt = np.average(np.array(cand), axis=0, weights=np.array(wts) + 1e-6)
                boxes[i] = (1 - r["alpha"]) * boxes[i] + r["alpha"] * tgt
        return boxes

    # ---- full pipeline -------------------------------------------------------------------------------------------
    def predict(self, image, return_debug=False):
        """image: path or BGR array. Returns Detections (all boxes down to low confidence, sorted high to low)."""
        img = cv2.imread(str(image), cv2.IMREAD_COLOR) if isinstance(image, (str, Path)) else image
        h, w = img.shape[:2]
        s1 = self.cfg["stage1"]
        mems = self.stage1_members(self.stage1_views(img))
        boxes, conf, _ = wbf(mems, w, h, s1["iou_thr"], max_det=self.cfg.get("max_det", 300))
        dbg = dict(stage1=(boxes, conf))
        z = self.cfg.get("zoom", {})
        if z.get("enabled"):
            regions = self.zoom_regions(boxes, conf, w, h)
            zmems = self.zoom_members(self.zoom_views(img, regions))
            if zmems:
                boxes, conf, _ = wbf(mems + zmems, w, h, z.get("iou_thr", s1["iou_thr"]),
                                     max_det=self.cfg.get("max_det", 300))
            dbg["zoom"] = (boxes, conf, regions)
        r = self.cfg.get("refine", {})
        if r.get("enabled"):
            idx = self.refine_targets(boxes, conf)
            boxes = self.apply_refine(boxes, idx, self.refine_views(img, boxes, idx))
        det = Detections(boxes, conf, w, h)
        return (det, dbg) if return_debug else det
