# This script was written by Claude Opus 5.5.
"""eagle: the most accurate Roblox character detector in this repository (a multi-model ensemble, slow by design).

    from eagle import Eagle
    det = Eagle()                       # config.json: models, views, zoom and box tightening
    d = det.predict("frame.png")        # Detections: boxes (x1, y1, x2, y2 pixels), conf, sorted high to low
    d.above(0.4).boxes
"""
from .pipeline import Detections, MaxDetector

Eagle = MaxDetector

__all__ = ["Eagle", "MaxDetector", "Detections"]
