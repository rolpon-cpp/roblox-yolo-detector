![Roblox YOLO](assets/roblox_yolo_bg.png)

<h1 align="center">Roblox YOLO</h1>

<p align="center">
  Object detection models for Roblox characters, built on <a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLO26</a>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-26-blue" alt="YOLO26"/>
  <img src="https://img.shields.io/badge/task-object%20detection-green" alt="Object detection"/>
  <img src="https://img.shields.io/badge/mAP50--95-70.2%25-brightgreen" alt="mAP50-95 70.2%"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-lightgrey" alt="License AGPL-3.0"/>
</p>

The models detect player and NPC characters in screenshots and video frames using a single class, `character`. Training data covers gameplay from several hundred Roblox games.

![Detection examples](assets/detections.png)

## Contents

- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Evaluation](#evaluation)
- [Training](#training)
- [Labeling Specification](#labeling-specification)
- [Limitations](#limitations)
- [License](#license)
- [Disclaimers](#disclaimers)

## Installation

```bash
git clone https://github.com/rolpon-cpp/roblox-yolo-detector
cd roblox-yolo-detector
pip install -r requirements.txt
```

## Usage

```python
from ultralytics import YOLO

model = YOLO("models/rblx-yolo-orca.pt")
results = model("screenshot.png", conf=0.5)

for box in results[0].boxes:
    print(box.xyxy[0].tolist(), box.conf.item())
```

Each model file stores its input size, so `imgsz` does not need to be specified.

A real-time screen capture example is provided in [`examples/real_time_inference.py`](examples/real_time_inference.py).

### Export

```python
model.export(format="onnx")
model.export(format="engine", dynamic=False, quantize="fp16", batch=1, imgsz=[864, 1536])
```

TensorRT engines must be built on the GPU that will run them.

<h2 align="center" id="models">Models</h2>

<table align="center">
<tr>
<td align="center" width="33%"><img src="assets/orca.png" alt="Orca" height="160"/><br/><b>Orca</b></td>
<td align="center" width="33%"><img src="assets/rabbit.png" alt="Rabbit" height="160"/><br/><b>Rabbit</b></td>
<td align="center" width="33%"><img src="assets/cheetah.png" alt="Cheetah" height="160"/><br/><b>Cheetah</b></td>
</tr>
</table>

| Model | Architecture | Input size | mAP50-95 | mAP50 | PyTorch FP32 (ms) | TensorRT FP16 (ms) | Parameters (M) |
|---|---|---|---|---|---|---|---|
| Orca | YOLO26s | 1536 | 70.2 | 86.9 | 10 | 2.2 | 9.9 |
| Rabbit | YOLO26n | 1280 | 61.9 | 79.9 | 6 | 1.5 | 2.5 |
| Cheetah | YOLO26n | 1024 | 59.3 | 78.4 | 5 | 1.5 | 2.5 |

Orca is recommended where an NVIDIA GPU is available. Rabbit and Cheetah are intended for lower-end hardware.

Speeds were measured on an RTX 5080 at batch 1 with 2560x1440 input frames. PyTorch timings are end to end. TensorRT timings cover the network only. FP16 engines are within 0.2 mAP of the PyTorch models.

## Evaluation

All results are measured on a held-out test set of 200 screenshots at 2560x1440, containing 506 labeled characters and 32 images without characters. No test images, and no training images of the same scenes, were used in training. Scores are from `yolo val` at each model's input size.

### Orca by character size

Size is the longest side of the bounding box at 1536 input.

| Size | < 16 px | 16 to 32 px | 32 to 64 px | 64 to 128 px | ≥ 128 px |
|---|---|---|---|---|---|
| AP50-95 | 24.0 | 44.5 | 69.1 | 75.2 | 83.7 |

### Orca at reduced input size

| Input width | 1536 | 1280 | 1024 | 768 | 640 |
|---|---|---|---|---|---|
| mAP50-95 | 70.1 | 67.1 | 64.0 | 57.9 | 54.7 |

Measured with `model.predict` at 16:9 input without retraining. This method reads approximately 0.1 mAP below `yolo val`.

## Training

Training data consists of approximately 5,000 hand-labeled images and a larger set of automatically labeled gameplay frames. Automatic labels were limited to high-confidence detections. Orca was additionally trained on synthetic images of difficult cases. Released weights are averages of checkpoints from a single training run.

The hand-labeled dataset and the test set are available on Roboflow:

- Dataset: https://universe.roboflow.com/rolpon/roblox-yolo26-dataset
- Test set: https://universe.roboflow.com/rolpon/roblox-public-200img-set

Some images were removed from the published datasets because they contained private information, so the final sizes are slightly lower than expected: the dataset has about 5,600 images (about 6,000 before), and the test set has 199 images (200 before). The results in [Evaluation](#evaluation) were measured on the original 200-image test set.

Example fine-tuning command:

```python
from ultralytics import YOLO

model = YOLO("models/rblx-yolo-orca.pt")
model.train(data="data.yaml", imgsz=1536, epochs=25, batch=5, optimizer="MuSGD", lr0=0.002, lrf=0.1)
```

Set `optimizer` explicitly. The `auto` setting selects different optimizers depending on dataset size.

## Labeling Specification

- Each character receives one bounding box enclosing all visible pixels. Name tags are excluded.
- Boxes include attached items such as wings, weapons and accessories. Particle effects are excluded.
- Overlapping characters receive separate boxes where possible.
- Player characters, NPCs, custom rigs and dead characters are labeled as characters.
- Menu renders, leaderboard statues, paintings, drawings and real people are not labeled.
- Images that are not Roblox gameplay are treated as background.

## Limitations

- Characters smaller than 16 px at the model input size are frequently missed.
- Recall is reduced in dark scenes and for overlapping or partially visible characters.
- Leaderboard statues and character figures in game menus account for most false positives.
- Training data consists of desktop gameplay. Accuracy on mobile layouts has not been measured.

## License

This project is licensed under [AGPL-3.0](LICENSE), consistent with the Ultralytics YOLO framework. The models are provided without warranty.

## Disclaimers

This project is not affiliated with or endorsed by Roblox Corporation. Roblox is a trademark of Roblox Corporation. Dataset images are screenshots of Roblox gameplay and remain the property of their respective owners.

The models may be used for aim assistance. This use is not prohibited but is discouraged, and related issues or pull requests will not be accepted. Automated input programs violate the Roblox Terms of Use and may result in account termination.

Claude (Anthropic) assisted with model development and documentation. All manual labels were created or reviewed by the author.
