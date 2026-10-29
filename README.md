![Roblox YOLO cover image](assets/roblox_yolo_bg.png)

<div align="center">

<img src="https://img.shields.io/badge/YOLO-26-blue" alt="YOLO26"/>
<img src="https://img.shields.io/badge/task-detect-green" alt="Detect"/>
<img src="https://img.shields.io/badge/mAP50--95-70.2-brightgreen" alt="mAP50-95 70.2"/>
<img src="https://img.shields.io/badge/license-AGPL--3.0-lightgrey" alt="AGPL-3.0"/>

</div>

Roblox YOLO is a set of [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) models for detecting Roblox characters in screenshots and video frames. The models are trained on gameplay from hundreds of Roblox games and detect players and NPCs, alive or dead, as a single `character` class. They are suitable for bots, analytics, accessibility tools and research.

![Detection examples](assets/detections.png)

## 📄 Documentation

See below for installation, usage, model performance, training and the labeling spec.

<details open>
<summary>Install</summary>

Clone the repo and install the requirements in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/rolpon-cpp/roblox-yolo-detector
cd roblox-yolo-detector
pip install -r requirements.txt
```

</details>

<details open>
<summary>Usage</summary>

### CLI

```bash
yolo predict model=models/rblx-yolo-orca.pt source=path/to/screenshot.png
```

### Python

```python
from ultralytics import YOLO

# Load a model
model = YOLO("models/rblx-yolo-orca.pt")

# Run inference (each model stores its own input size)
results = model("path/to/screenshot.png", conf=0.5)

# Process results
for box in results[0].boxes:
    print(box.xyxy[0].tolist(), box.conf.item())

results[0].show()
```

A confidence threshold of `0.5` is a reasonable default. Lower it to detect more distant characters.

### Real-time

[`examples/real_time_inference.py`](examples/real_time_inference.py) captures the screen with `dxcam`, runs inference in a worker thread and draws detections in an overlay window. Press `t` to exit.

### Export

```python
from ultralytics import YOLO

model = YOLO("models/rblx-yolo-orca.pt")
model.export(format="onnx")  # ONNX
model.export(format="engine", dynamic=False, quantize="fp16", batch=1, imgsz=[864, 1536])  # TensorRT FP16, 16:9 input
```

TensorRT engines are tied to the GPU and TensorRT version they were built on. Build the engine on the target machine.

</details>

## ✨ Models

<table>
<tr>
<td align="center" width="33%"><img src="assets/orca.png" alt="Orca" height="160"/><br/><b>Orca</b></td>
<td align="center" width="33%"><img src="assets/rabbit.png" alt="Rabbit" height="160"/><br/><b>Rabbit</b></td>
<td align="center" width="33%"><img src="assets/cheetah.png" alt="Cheetah" height="160"/><br/><b>Cheetah</b></td>
</tr>
</table>

| Model | Base | size<br><sup>(pixels)</sup> | mAP<sup>test<br>50-95</sup> | mAP<sup>test<br>50</sup> | Speed<br><sup>RTX 5080 PyTorch<br>(ms)</sup> | Speed<br><sup>RTX 5080 TensorRT FP16<br>(ms)</sup> | params<br><sup>(M)</sup> | FLOPs<br><sup>(B)</sup> |
|---|---|---|---|---|---|---|---|---|
| [Orca](models/rblx-yolo-orca.pt) | YOLO26s | 1536 | **70.2** | **86.9** | 10 | 2.2 | 9.9 | 137.7 |
| [Rabbit](models/rblx-yolo-rabbit.pt) | YOLO26n | 1280 | 61.9 | 79.9 | 6 | 1.5 | 2.5 | 25.0 |
| [Cheetah](models/rblx-yolo-cheetah.pt) | YOLO26n | 1024 | 59.3 | 78.4 | 5 | 1.5 | 2.5 | 15.6 |

- **Orca** is the most accurate model and the recommended choice on NVIDIA GPUs. **Rabbit** trades some accuracy for lower compute. **Cheetah** is the lightest, for CPUs and older GPUs.
- **mAP<sup>test</sup>** values are single-model, single-scale on the Roblox YOLO test set (200 images at 2560x1440, 506 characters, 32 background images; no overlap with the training data). Reproduce with `yolo val model=models/rblx-yolo-orca.pt data=path/to/test/data.yaml` using the [test set](#train-your-own) export.
- **Speed** is measured at batch 1 on 2560x1440 frames. PyTorch speed is end to end in FP32; TensorRT speed is the network only. FP16 engines score within 0.2 mAP of the PyTorch models.
- **FLOPs** are given for a square input at the model's size.

<details>
<summary>Accuracy by character size (Orca)</summary>

Character size is the longest side of the box at 1536 input.

| Size | <16 px | 16-32 px | 32-64 px | 64-128 px | ≥128 px |
|---|---|---|---|---|---|
| AP<sup>test</sup><sub>50-95</sub> | 24.0 | 44.5 | 69.1 | 75.2 | 83.7 |

</details>

<details>
<summary>Orca at lower input sizes</summary>

Orca at reduced 16:9 input widths, without retraining. Measured with `model.predict`, which reads about 0.1 mAP below `yolo val`.

| Width | 1536 | 1280 | 1024 | 768 | 640 |
|---|---|---|---|---|---|
| mAP<sup>test</sup><sub>50-95</sub> | 70.1 | 67.1 | 64.0 | 57.9 | 54.7 |

Most of the loss below 1024 comes from small characters. At 1024, Orca is still more accurate than Rabbit and Cheetah at their native sizes.

</details>

## 🧠 Training

The models are fine-tuned from YOLO26 on two kinds of data:

- About 5,000 hand-labeled screenshots and video frames, split into train and val by capture session so near-duplicate frames never cross the split.
- Tens of thousands of gameplay frames from hundreds of games, labeled by a larger detector ensemble. Only confident detections are kept as labels, and uncertain regions are masked.

Orca is also trained on a small set of synthetic images covering dark scenes, overlapping characters, characters cut off by the frame edge and very small characters. Each released model is a uniform average of the checkpoints from its training run. Rabbit and Cheetah are trained on an earlier version of the data.

### Train your own

The hand-labeled dataset and the test set are available on Roboflow:

| Resource | Link |
|---|---|
| Dataset | https://universe.roboflow.com/rolpon/roblox-yolo26-dataset |
| Test set | https://universe.roboflow.com/rolpon/roblox-public-200img-set |

```python
from ultralytics import YOLO

model = YOLO("models/rblx-yolo-orca.pt")  # or "yolo26s.pt" to train from COCO
model.train(data="data.yaml", imgsz=1536, epochs=25, batch=5, optimizer="MuSGD", lr0=0.002, lrf=0.1, save_period=5)
```

- Fine-tuning from a Roblox YOLO model takes about 25 epochs. Training from COCO takes 150 to 250 epochs with `lr0=0.01`.
- Set `optimizer` explicitly. With `optimizer="auto"`, Ultralytics picks AdamW or MuSGD based on dataset size and epochs.
- Input sizes above 1024 improve small-character recall. Batch 5 at 1536 fits in 16 GB of VRAM.

## 🏷️ Labeling Spec

Use the same rules when adding data:

- One tight box per character covering all visible pixels, without name tags.
- Boxes include wings, large arms, held items and accessories. Highlights are included, particles are not.
- Overlapping characters get separate boxes where possible.
- Players and NPCs are characters, including custom rigs, monsters, mascots and anything with the Roblox body style. Dead characters are characters.
- Menu character renders, leaderboard statues, paintings, drawings, fan art and real people are not characters.
- Anything that is not Roblox gameplay (real photos, browsers, desktops, the Roblox website) is background.

A recording of the labeling process is available [here](https://youtu.be/3P5Tr6R4R_I).

## ⚠️ Known Limitations

- Characters under 16 px at the model's input size are frequently missed.
- Recall is lower in dark scenes, on heavily overlapping characters, on characters cut off by the frame edge and on characters behind UI or effects.
- Most false positives are leaderboard and donation-board statues, character figures in menus and tutorials, and small blocky objects.
- The training data is desktop gameplay. Mobile layouts and heavily compressed video may reduce accuracy.

## 📜 License

Roblox YOLO is released under the [AGPL-3.0 License](LICENSE), in line with the Ultralytics YOLO framework it is built on. The models are provided as is, without warranty.

This project is not affiliated with or endorsed by Roblox Corporation. Roblox is a trademark of Roblox Corporation. Dataset images are screenshots of Roblox gameplay, and the games and assets shown belong to their creators.

## 📌 Disclaimers

- **Intended use.** Hobby projects, research and utility tools such as bots, analytics and accessibility aids. The models are not intended for cheating.
- **Aimbots.** Aim-assist use is not prohibited but is strongly discouraged. Issues and pull requests related to aimbots will not be accepted. Automated aiming or input programs violate the Roblox Terms of Use and can result in an account ban.
- **AI assistance.** Claude (Anthropic) assisted with planning, training, evaluation and tooling, including parts of this README. All hand labels were made or reviewed by a human.
