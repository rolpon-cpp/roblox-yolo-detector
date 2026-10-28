![RobloxYolo Main Cover Image](assets/roblox_yolo_bg.png)

<h1 align="center">Roblox YOLO</h1>

<p align="center">
  <b>High-accuracy Roblox character detection, built on YOLO26.</b><br/>
  Three models, one class, from real-time lightweight to maximum accuracy.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-26-blue" alt="YOLO26"/>
  <img src="https://img.shields.io/badge/task-object%20detection-green" alt="Object detection"/>
  <img src="https://img.shields.io/badge/best%20mAP50-85.7%25-brightgreen" alt="Best mAP50 85.7%"/>
  <img src="https://img.shields.io/badge/license-see%20LICENSE-lightgrey" alt="License"/>
</p>

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Available Models](#available-models)
- [Speed and Input Size](#speed-and-input-size)
- [How the Models Were Trained](#how-the-models-were-trained)
- [Training Your Own](#training-your-own)
- [Documentation](#documentation)
- [Disclaimers & Disclosures](#disclaimers--disclosures)

---

## Overview

Roblox-YOLO is a family of high-accuracy Roblox character detection models developed by Rolpon. Built on YOLO26, they are general purpose detectors suited to autonomous bots and utility tools, and have been trained on a variety of Roblox games to provide detection that holds up across the platform.

**Highlights**

- **Three models** covering the full speed and accuracy range.
- **Up to 85.7% mAP50** and **68.8% mAP50-95** on a 200 image test set that no model has trained on (Orca).
- **Single class**, `character`, which detects both alive and dead Roblox characters.
- **Real-time ready**: Orca runs in about 2 ms per frame as a TensorRT FP16 engine on an RTX 5080, with no measurable accuracy loss.
- **Public dataset and test set** hosted on Roboflow.

![RobloxYolo Detection Sample Image](assets/detections.png)

---

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run inference on an image:

```python
from ultralytics import YOLO

# Load a rblx YOLO model
model = YOLO("models/rblx-yolo-orca.pt")

# Perform object detection on an image (use the model's own image size)
results = model("sample_images/image1.png", imgsz=1536)
results[0].show()  # Display results
```

### Real-Time Inference

See [`examples/real_time_inference.py`](examples/real_time_inference.py) for a complete real-time example.
Export to ONNX or TensorRT if you plan to use these models in real time.

### Export a Model

```python
from ultralytics import YOLO

# Load a rblx YOLO model
model = YOLO("models/rblx-yolo-orca.pt")

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model

# Export to TensorRT FP16 on CUDA. A fixed 16:9 input (height, width) fits 1080p/1440p/4K frames with no padding.
path = model.export(format="engine", dynamic=False, quantize="fp16", batch=1, imgsz=[864, 1536])
```

TensorRT engines are specific to the GPU and TensorRT version they were built on, so build the engine on the machine that will run it.

---

## Available Models

> mAP numbers come from Ultralytics `yolo val` on the 200 image test set, each model at its own image size.

| Model | Architecture | Image Size | mAP50 | mAP50-95 | Best For |
|---|---|---|---|---|---|
| **Orca** | YOLO26s | 1536 | **85.7%** | **68.8%** | Maximum accuracy |
| **Rabbit** | YOLO26n | 1280 | 79.9% | 61.9% | Balance of speed and accuracy |
| **Cheetah** | YOLO26n | 1024 | 78.4% | 59.3% | Real-time inference, fastest |

<table>
<tr>
<td align="center" width="33%">
<img src="assets/orca.png" alt="Orca" width="180"/><br/>
<b>Orca</b>
</td>
<td align="center" width="33%">
<img src="assets/rabbit.png" alt="Rabbit" width="180"/><br/>
<b>Rabbit</b>
</td>
<td align="center" width="33%">
<img src="assets/cheetah.png" alt="Cheetah" width="180"/><br/>
<b>Cheetah</b>
</td>
</tr>
</table>

### The Test Set

- 200 desktop screenshots at 2560x1440 from many different games, with 506 labeled characters and 32 background images (no characters).
- No model was trained on it, and training images that show the same scene as a test image were removed before training.
- About 1 in 5 test characters is smaller than 32 pixels at 1536 input, so the set is hard on far-away characters on purpose.
### Orca

The most powerful and accurate model in the family. YOLO26s at imgsz 1536, reaching **85.7% mAP50** and **68.8% mAP50-95**.

- Trained on about **4,800 hand-labeled images** combined with about **9,400 distilled images** (see [How the Models Were Trained](#how-the-models-were-trained)).

### Rabbit

YOLO26n at imgsz 1280. A balance of speed and accuracy, reaching 79.9% mAP50 and 61.9% mAP50-95. Trained with the same data and recipe as Orca.

### Cheetah

YOLO26n at imgsz 1024. The fastest model, reaching 78.4% mAP50 and 59.3% mAP50-95. Also trained with distillation. Recommended for real-time inference.

---

## Speed and Input Size

Measured on an RTX 5080 at batch 1 with 2560x1440 frames already in memory.

| Model | PyTorch FP32 (whole loop) | TensorRT FP16 (network only) | TensorRT FP16 (whole loop) |
|---|---|---|---|
| Orca (YOLO26s, 1536) | about 10 ms | 2.2 ms | 6.7 ms |
| Rabbit (YOLO26n, 1280) | about 6 ms | about 1.5 ms | about 4.3 ms |
| Cheetah (YOLO26n, 1024) | about 5 ms | about 1.6 ms | about 4.9 ms |

- The FP16 engine of Orca scores within 0.2 points of the PyTorch model, so FP16 is safe to use.
- At these speeds most of the time per frame is resizing the frame on the CPU, not the network itself. Resizing on the GPU, or capturing a smaller region, helps more than a smaller model.
- A slower GPU will be slower: as a rough guide, an RTX 3060 has about a quarter of the FP16 compute of an RTX 5080.

**Running a model below its image size** (Orca, mAP50-95 on the test set, 16:9 input):

| Input width | 1536 | 1280 | 1024 | 768 | 640 |
|---|---|---|---|---|---|
| mAP50-95 | 68.4% | 65.4% | 62.1% | 56.6% | 53.5% |

Accuracy falls quickly below 1024, almost all of it on small, far-away characters. If you need a smaller input, Orca at 1024 is still more accurate than the nano models at the same size. (This table uses a slightly different measurement than `yolo val`, so its 1536 value is 0.4 points lower than the table above.)

---

## How the Models Were Trained

1. **Hand-labeled data.** About 5,000 images labeled by hand with the spec below, split 80/20 into train and validation. Images from the same capture session or the same scene were kept on the same side of the split, so validation scores are not inflated by near-copies.
2. **Distillation.** A much slower, more accurate detector (an ensemble of two RF-DETR models and a YOLO model that looks at each frame many times, whole, tiled, mirrored and zoomed; 80.8% mAP50-95 on the test set) labeled thousands of extra gameplay frames from YouTube. Only its confident boxes became labels; image regions with doubtful boxes were painted gray so the student is never taught that a real character is background.
3. **Fine-tuning.** The YOLO models were fine-tuned for 25 epochs on the hand-labeled plus distilled images, starting from an earlier Roblox-trained model.
4. **Checkpoint averaging.** The final weights are the average of the checkpoints saved during training. This is a single normal model file, with no extra cost at inference, and it scored 1 to 4 points of mAP50-95 higher than the single best checkpoint in every run tested.

**What made a difference, and what did not**

| Change | Effect on the test set |
|---|---|
| Adding distilled frames from games the models had not seen | Clear gain (the largest single improvement) |
| More frames from the same videos and games | No gain: variety matters, not volume |
| Checkpoint averaging | +1 to +4 points mAP50-95 |
| Training longer (150 instead of 25 fine-tune epochs) | No gain |
| A bigger model (YOLO26m instead of YOLO26s) | No gain on this data |
| Higher training resolution (2048 instead of 1536) for a fine-tune | No gain |
| Tiled inference (whole frame plus 2x2 tiles) | About +2 points, at about 3.5x the time per frame |

---

## Training Your Own

The dataset is public on Roboflow. You will need a capable GPU machine or Google Colab to train a model.

| Resource | Link |
|---|---|
| Dataset | https://universe.roboflow.com/rolpon/roblox-yolo26-dataset |
| Test Set | https://universe.roboflow.com/rolpon/roblox-public-200img-set |

```python
from ultralytics import YOLO

# Load a YOLO model from COCO
model = YOLO("yolo26s.pt")

# Train a YOLO model. Set the optimizer explicitly: with optimizer="auto", Ultralytics switches between
# AdamW and MuSGD depending on dataset size and epochs, which makes runs hard to compare.
model.train(data="data.yaml", imgsz=1536, epochs=250, patience=250, batch=5, workers=7,
            optimizer="MuSGD", lr0=0.01, save_period=5, name="roblox-yolo-custom")
```

**Training trends**

- Start from `yolo26s.pt` for a new model, or from one of these models to fine-tune (far fewer epochs are needed).
- 150 to 250 epochs from COCO; 25 epochs is enough for a fine-tune.
- imgsz 1536 works well; above 1024 clearly helps with small characters.
- Batch 5 at imgsz 1536 fits in 16 GB of VRAM (workers 7).
- Save checkpoints during training (`save_period=5`) so you can average them afterwards.
- Do not reuse the test set for choosing checkpoints; pick on the validation split and check the test set once at the end.

**Dataset creation**

1. Copy in hand-labeled data.
2. Split train/val 80%/20%, keeping frames from the same session or video on the same side.

---

## Documentation

### Labeling Spec

- Include all visible pixels of a character (1 box per character), with the exceptions below.
- If multiple characters intersect, attempt to give each character a unique box.
- Use the tightest possible bounding boxes, with no name tags included.
- If individual bounding boxes are truly not possible, use a single-blob box.
- Boxes include all extents of a character, including wings, large arms, and accessories. Particles are not counted.
- Highlights on characters are counted.
- All non-Roblox images are background (real-life photos, browsers, desktop environments, the Roblox website, etc.).
- Fake Roblox characters, UI character renders, and paintings do not count as characters. Only NPCs and player characters count.
- Non-standard 3D characters (custom rigs, monsters and mascot characters such as game entities) count as characters, as does anything with the Roblox body style. Drawings, fan art and real-life people do not.
- Dead characters are accounted for, either in their own class or merged with the character class.

### Known Pain Points

Measured on the test set with Orca (share of characters found at its best confidence threshold):

| Pain Point | How Orca does |
|---|---|
| Tiny characters (under 16 px at 1536 input) | Finds about 1 in 5 |
| Small characters (16 to 32 px) | Finds a bit over half |
| Dark scenes and dark characters | Finds about 2 in 3 |
| Heavily overlapping characters | Finds a bit under half |
| Characters cut off by the frame edge | Finds about 2 in 3 |
| Large characters (64 px and up) | Finds over 80% |

Most remaining false detections are leaderboard and donation-board statues, cartoon or tutorial figures in the game UI, and small blocky objects such as rocks and trees. Other known weak spots: UI and overlays on top of characters, effects and particles on characters, close-up shots, and MM2 paintings.

---

## Disclaimers & Disclosures

**AI assistance.** Claude (Anthropic) assisted with planning, training, evaluation, and the tooling behind these models, including parts of this README. All labels in the hand-labeled dataset were made or reviewed by a human.

**Intended use.** These models were built for hobby projects, research, and utility tools such as bots, analytics, and accessibility aids. They are not meant for cheating.

**Aimbots.** The models can technically be used for aiming assistance. I do not prohibit this, but I strongly discourage it, and I will not accept issues or pull requests related to aimbots. Using automated aiming or input programs in Roblox violates the Roblox Terms of Use and can get your account banned. You are responsible for how you use these models.

**No affiliation.** This project is not affiliated with or endorsed by Roblox Corporation. Roblox is a trademark of Roblox Corporation. Images in the dataset are screenshots of Roblox gameplay.

**No warranty.** The models are provided as is, without any guarantee of accuracy. See [LICENSE](LICENSE) for the full terms.

Watch the labeling process: https://youtu.be/3P5Tr6R4R_I
