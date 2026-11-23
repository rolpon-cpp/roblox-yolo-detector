# This script was written by Claude Sonnet 5.5.
# Runs Eagle (the max-accuracy ensemble in ../eagle) on images and writes YOLO label files with confidences
# (class cx cy w h conf) and, with --draw, copies of the images with the boxes drawn on them.
# Eagle is slow by design (several models and many views per image): use it for labeling and offline work.
#
# Usage, from the repository root:
#   python examples/eagle_predict.py --source screenshot.png --out eagle_out --draw
#   python examples/eagle_predict.py --source folder_of_images --out eagle_out --thr 0.4 --half

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eagle import Eagle  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Run Eagle, the max-accuracy ensemble, on images (slow by design).")
    ap.add_argument("--source", required=True, help="an image or a folder of images")
    ap.add_argument("--out", required=True, help="output folder (labels/ and drawn/ are created inside)")
    ap.add_argument("--thr", type=float, default=0.4, help="confidence for printed and drawn boxes (default 0.4)")
    ap.add_argument("--min-conf", type=float, default=0.001, help="lowest confidence written to the label files")
    ap.add_argument("--half", action="store_true", help="run the RF-DETR members in half precision (about 2x faster)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--draw", action="store_true", help="also save images with the boxes drawn")
    a = ap.parse_args()

    src = Path(a.source)
    paths = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXT)
    out = Path(a.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if a.draw:
        (out / "drawn").mkdir(exist_ok=True)

    det = Eagle(device=a.device, half=a.half)
    t0 = time.time()
    for k, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        d = det.predict(img)
        keep = d.above(a.min_conf)
        (out / "labels" / (p.stem + ".txt")).write_text("\n".join(keep.to_yolo_lines()) + ("\n" if len(keep) else ""))
        if a.draw:
            cv2.imwrite(str(out / "drawn" / (p.stem + ".jpg")), d.draw(img, a.thr))
        sure = d.above(a.thr)
        print(f"{k + 1}/{len(paths)} {p.name}: {len(sure)} boxes at {a.thr}+ "
              f"({(time.time() - t0) / (k + 1):.1f} s per image)", flush=True)
        for box, c in zip(sure.boxes, sure.conf):
            print("   ", [round(float(v), 1) for v in box], round(float(c), 3))


if __name__ == "__main__":
    main()
