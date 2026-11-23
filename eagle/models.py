# This script was written by Claude Opus 5.5.
# Model wrappers with one interface: predict(list of BGR images, size) -> list of (boxes xyxy float32, conf float32).
# YOLO (Ultralytics .pt) letterboxes to `size`; RF-DETR (.pth) resizes to a size x size square, like in training.
# Both return every box down to confidence 0.001 (YOLO: NMS IoU 0.7, up to 300 boxes; RF-DETR: its top 300 queries).

import contextlib

import cv2
import numpy as np

MIN_CONF = 0.001


class YoloModel:
    kind = "yolo"

    def __init__(self, path, imgsz=1536, device="cuda:0", batch=8):
        from ultralytics import YOLO
        self.model = YOLO(str(path))
        self.imgsz = imgsz
        self.device = device
        self.batch = batch

    def predict(self, imgs, size=None):
        out = []
        for s in range(0, len(imgs), self.batch):
            rs = self.model.predict(imgs[s:s + self.batch], imgsz=size or self.imgsz, conf=MIN_CONF, iou=0.7,
                                    max_det=300, verbose=False, device=self.device)
            out += [(r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32))
                    for r in rs]
        return out


class RfdetrModel:
    kind = "rfdetr"

    def __init__(self, path, device="cuda", batch=4, half=False):
        from rfdetr import RFDETR
        self.model = RFDETR.from_checkpoint(str(path), device=device)
        self.imgsz = self.model.model_config.resolution
        self.batch = batch
        self.half = half

    def predict(self, imgs, size=None):
        shape = None if size in (None, self.imgsz) else (int(size), int(size))
        out = []
        for s in range(0, len(imgs), self.batch):
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs[s:s + self.batch]]
            with self._autocast():
                dets = self.model.predict(rgb, threshold=MIN_CONF, shape=shape, include_source_image=False)
            dets = dets if isinstance(dets, list) else [dets]
            out += [(np.asarray(d.xyxy, np.float32).reshape(-1, 4), np.asarray(d.confidence, np.float32).reshape(-1))
                    for d in dets]
        return out


    def _autocast(self):
        if not self.half:
            return contextlib.nullcontext()
        import torch
        return torch.autocast("cuda", dtype=torch.float16)


def load_model(path, imgsz=None, device="cuda:0", half=False):
    """Picks the wrapper by file extension (.pt = YOLO, .pth = RF-DETR)."""
    if str(path).lower().endswith(".pth"):
        on_gpu = str(device).startswith("cuda")
        return RfdetrModel(path, device="cuda" if on_gpu else device, half=half and on_gpu)
    return YoloModel(path, imgsz=imgsz or 1536, device=device)
