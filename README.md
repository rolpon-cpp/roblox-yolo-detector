![Roblox YOLO cover image](assets/roblox_yolo_bg.png)

<h1 align="center">Roblox YOLO</h1>

<p align="center">
  <b>Roblox character detection models, built on YOLO26.</b><br/>
  Three sizes: one for accuracy, one for balance, one for speed.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/YOLO-26-blue" alt="YOLO26"/>
  <img src="https://img.shields.io/badge/task-object%20detection-green" alt="Object detection"/>
  <img src="https://img.shields.io/badge/mAP50--95-70.2%25-brightgreen" alt="mAP50-95 70.2%"/>
  <img src="https://img.shields.io/badge/license-AGPL--3.0-lightgrey" alt="License AGPL-3.0"/>
</p>

---

## Table of Contents

- [What this is](#what-this-is)
- [Models](#models)
- [Quick Start](#quick-start)
- [Performance](#performance)
- [How they were trained](#how-they-were-trained)
- [Training your own](#training-your-own)
- [Labeling spec](#labeling-spec)
- [Limitations](#limitations)
- [Disclaimers & Disclosures](#disclaimers--disclosures)

---

## What this is

These are object detection models that find Roblox characters in screenshots and video frames. Players, NPCs, alive or dead: there is one class, `character`, and that's it.

A detector trained on one game usually falls apart in the next one, so these models are trained on gameplay from hundreds of games and should work wherever you point them. They're a good fit for bots, small tools, analytics, accessibility projects and research.

The best model, Orca, scores **70.2% mAP50-95** (86.9% mAP50) on a test set it has never seen. As a TensorRT FP16 engine on an RTX 5080 it runs in about 2 ms per frame. All three are normal Ultralytics weights, so you can load, export and fine-tune them the usual way.

![Detection examples](assets/detections.png)

---

## Models

<table>
<tr>
<td align="center" width="33%">
<img src="assets/orca.png" alt="Orca" height="160"/><br/>
<b>Orca</b><br/>
<sub>Most accurate</sub>
</td>
<td align="center" width="33%">
<img src="assets/rabbit.png" alt="Rabbit" height="160"/><br/>
<b>Rabbit</b><br/>
<sub>Balanced</sub>
</td>
<td align="center" width="33%">
<img src="assets/cheetah.png" alt="Cheetah" height="160"/><br/>
<b>Cheetah</b><br/>
<sub>Fastest</sub>
</td>
</tr>
</table>

| Model | File | Architecture | Input size | Parameters | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **Orca** | `models/rblx-yolo-orca.pt` | YOLO26s | 1536 | 9.9M | **86.9%** | **70.2%** |
| **Rabbit** | `models/rblx-yolo-rabbit.pt` | YOLO26n | 1280 | 2.5M | 79.9% | 61.9% |
| **Cheetah** | `models/rblx-yolo-cheetah.pt` | YOLO26n | 1024 | 2.5M | 78.4% | 59.3% |

Which one to pick: if you have a decent NVIDIA GPU, use **Orca**. It's the most accurate by a wide margin and it's still fast once exported to TensorRT. **Rabbit** is for weaker GPUs where Orca doesn't keep up. **Cheetah** is for the CPU, older hardware, or when every millisecond counts.

Scores are from Ultralytics `yolo val` on the test set described under [Performance](#performance), each model at its own input size.

---

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from ultralytics import YOLO

model = YOLO("models/rblx-yolo-orca.pt")

# Each model remembers its own input size, so you don't need to pass imgsz.
results = model("screenshot.png", conf=0.5)
results[0].show()

for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"character at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}), confidence {box.conf.item():.2f}")
```

`conf=0.5` is a good place to start. Go lower if you're missing far-away characters, higher if you're getting false detections.

### Real-time

There's a full example in [`examples/real_time_inference.py`](examples/real_time_inference.py). It grabs the screen with `dxcam`, runs the model in a separate thread and draws the boxes in an overlay window (press `t` to quit). If you want real-time speed, export to TensorRT or ONNX first.

### Exporting

```python
from ultralytics import YOLO

model = YOLO("models/rblx-yolo-orca.pt")

# ONNX, for CPU or other runtimes
model.export(format="onnx")

# TensorRT FP16 for NVIDIA GPUs. A fixed 16:9 input (height, width) fits 1080p, 1440p and 4K frames without padding.
model.export(format="engine", dynamic=False, quantize="fp16", batch=1, imgsz=[864, 1536])
```

TensorRT engines only work on the GPU and TensorRT version they were built with, so build yours on the machine that will run it.

---

## Performance

### The test set

The test set is 200 desktop screenshots at 2560x1440 from a lot of different games. They contain 506 labeled characters, and 32 of the images have no characters at all. None of the models were trained on these images, and I also removed training images that showed the same scene as a test image. About one in five test characters is under 32 pixels tall at 1536 input, so the set is deliberately hard on far-away characters.

### Accuracy by character size (Orca)

| Character size (longest side at 1536 input) | under 16 px | 16 to 32 px | 32 to 64 px | 64 to 128 px | 128 px and up |
|---|---|---|---|---|---|
| AP50-95 | 24% | 45% | 69% | 75% | 84% |

Large and mid-size characters are reliable. Tiny ones are still the hardest part.

### Speed

Measured on an RTX 5080 at batch 1, with 2560x1440 frames already in memory.

| Model | PyTorch FP32, full frame | TensorRT FP16, network only |
|---|---|---|
| Orca | about 10 ms | about 2.2 ms |
| Rabbit | about 6 ms | about 1.5 ms |
| Cheetah | about 5 ms | about 1.5 ms |

A few things I found while measuring:

- FP16 TensorRT engines score within 0.2 points of the PyTorch models, so there's no reason not to use them.
- On a fast GPU, Rabbit and Cheetah end up about the same speed, because most of the time goes to resizing the frame and copying results, not to the network. On slower GPUs and on the CPU the difference shows up, and that's where Cheetah makes sense.
- Resizing frames on the GPU, or capturing a smaller part of the screen, saves more time than switching to a smaller model.

### Running Orca at a smaller size

You can run Orca below 1536 without retraining. This is what it costs (mAP50-95 on the test set, 16:9 input, measured with `model.predict`, which reads about 0.1 points below `yolo val`):

| Input width | 1536 | 1280 | 1024 | 768 | 640 |
|---|---|---|---|---|---|
| mAP50-95 | 70.1% | 67.1% | 64.0% | 57.9% | 54.7% |

It falls off quickly below 1024, almost all of it on small characters. Even at 1024, Orca beats both nano models.

---

## How they were trained

It started with about 5,000 screenshots and video frames that I labeled by hand. When splitting them into train and validation, I kept frames from the same session or scene on the same side, so the validation scores aren't inflated by near-duplicates.

That isn't enough data to cover Roblox, and labeling everything by hand doesn't scale. So I built a much slower and more accurate detector ensemble and used it to label tens of thousands of extra gameplay frames from hundreds of games. Only its confident detections became labels. Anywhere it was unsure got masked out, so the models are never taught that a real character is background. This made the biggest difference of anything I tried. One thing I learned: frames from *new* games help a lot, while more frames from games the models had already seen barely help.

Orca also saw a small set of synthetic images with the cases models struggle with most: dark scenes, overlapping characters, characters cut off by the screen edge, and tiny characters.

Finally, each released model is an average of the checkpoints saved during its training run. It's still a normal model file and costs nothing extra at inference, and it beat the single best checkpoint every time I tried it.

Rabbit and Cheetah were trained the same way on an earlier version of the data.

---

## Training your own

The hand-labeled dataset and the test set are both public on Roboflow. You'll need a decent GPU or Google Colab.

| Resource | Link |
|---|---|
| Dataset | https://universe.roboflow.com/rolpon/roblox-yolo26-dataset |
| Test set | https://universe.roboflow.com/rolpon/roblox-public-200img-set |

```python
from ultralytics import YOLO

# Fine-tune one of these models, or start from "yolo26s.pt" to train from COCO
model = YOLO("models/rblx-yolo-orca.pt")

# Set the optimizer yourself. With optimizer="auto", Ultralytics switches between AdamW and MuSGD
# depending on dataset size and epochs, which makes runs hard to compare.
model.train(data="data.yaml", imgsz=1536, epochs=25, batch=5, workers=7,
            optimizer="MuSGD", lr0=0.002, lrf=0.1, warmup_epochs=1, save_period=5, name="roblox-yolo-custom")
```

Some tips from my own runs:

- Fine-tuning one of these models takes about 25 epochs. Training from COCO takes 150 to 250 (use `lr0=0.01`).
- Input sizes above 1024 help a lot with small characters. 1536 works well.
- Batch 5 at 1536 fits in 16 GB of VRAM.
- Keep checkpoints during training (`save_period`) so you can average them afterwards.
- Pick checkpoints using the validation split only, and look at the test set once at the end.

---

## Labeling spec

If you add your own data, label it the same way, or the models will learn two different ideas of what a box is.

- One box per character, as tight as possible, covering every visible pixel. No name tags.
- Include the full character: wings, large arms, held items and accessories. Highlights count; particles don't.
- Give overlapping characters their own boxes when you can. If it really can't be done, one box for the group is fine.
- Players and NPCs count, including custom rigs, monsters, mascots and anything with the Roblox body style.
- These are not characters: character renders in menus, leaderboard statues, paintings, drawings, fan art and real people.
- Anything that isn't Roblox gameplay is background (real photos, browsers, desktops, the Roblox website).
- Dead characters are labeled as characters.

---

## Limitations

No detector is perfect, and these are the places where these ones struggle:

- **Tiny characters** (under 16 px at the model's input) get missed a lot. A bigger input size or tiled inference helps.
- **Hard scenes:** dark lighting, characters stacked on top of each other, characters cut off by the screen edge, and characters behind UI or effects.
- **False detections** mostly come from leaderboard and donation-board statues, character figures in menus and tutorials, and small blocky things like rocks and trees.
- **Mobile and video:** the models were trained on desktop gameplay, so mobile layouts, unusual camera angles and heavily compressed video can lower accuracy.

---

## Disclaimers & Disclosures

**AI assistance.** I used Claude (Anthropic) to help with planning, training, evaluation and tooling, including parts of this README. Every label in the hand-labeled dataset was made or checked by me.

**Intended use.** I made these for hobby projects, research and utility tools like bots, analytics and accessibility aids. They aren't meant for cheating.

**Aimbots.** The models can technically be used for aim assistance. I don't prohibit it, but I strongly discourage it, and I won't accept issues or pull requests about aimbots. Automated aiming or input programs break the Roblox Terms of Use and can get your account banned. What you do with the models is on you.

**No affiliation.** This project isn't affiliated with or endorsed by Roblox Corporation. Roblox is a trademark of Roblox Corporation. The dataset images are screenshots of Roblox gameplay, and the games and assets in them belong to their creators.

**License.** The code and model weights are released under [AGPL-3.0](LICENSE), the same license as the Ultralytics YOLO framework they're built on. They come as is, with no guarantee of accuracy.

If you want to see how the labeling was done, I recorded it: https://youtu.be/3P5Tr6R4R_I
