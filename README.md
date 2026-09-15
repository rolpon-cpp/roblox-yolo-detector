## Roblox YOLO26 Overview

Rolpon introduces high-accuracy, open-source, roblox character detectors based off the State of the Art YOLO26 SOTA Models.

## Usage

Check out examples/real_time_inference.py for using the models in real time!

```python
from ultralytics import YOLO

# Load a rblx YOLO model
model = YOLO("models/rblx-yolo-cheetah.pt")

# Perform object detection on an image
results = model("sample_images/image0.png")  # Predict on an image
results[0].show()  # Display results

# Export the model to ONNX format for deployment
path = model.export(format="onnx")  # Returns the path to the exported model

```

## Available Models

**Cheetah** - YOLO26n model at imgsz 1280.
This model was trained on hand-labeled images only and had high mAP scores on it's val set (83 mAP50, 60 mAP50-95).
It is recommended for real time inference since it is nano. Was trained via hand-labeled data only.

**Horse** - YOLO26s model at imgsz 1024.
This is a bit of an older model, however it is still good. It also got high mAP (82 mAP50, 67 mAP50-95)
This can also be used for real time, and is a reliable model. Was trained using hand + syn data.

**Orca** - YOLO26s model at imgsz 1024.
This is also an older model but it's very good with 73 mAP50-95 and 90 mAP50.
This numbers are likely inflated, expect Horse or Cheetah to be better in practice.
Was trained using hand + syn data.

## Documentation

**Labeling Spec**
- Try to include all visible pixels of a character (1 box per char) (exceptions, see below)
- If multiple characters intersect, attempt to give each character a unique box
- Tightest possible bounding boxes, no name tags included
- If a avatar is holding a large item or cosmetic item (ex: big wings on back, a huge sword, etc.) the bounding box should only include the visible extends of the avatar's body. If the item is small, it may be included in the bounding box.
- large items: items that take up a similar amount of space to the actual character or increase width\height by 40% or more (easily eyeball-able)
- ALL non-roblox images are background (real life photos, browsers, desktop environment, roblox website, etc)
- If a roblox screenshot contains a painting or object with roblox characters inside it, but the characters are not actual characters, the characters should be ignored as background. Otherwise, all characters should be included.

**Dataset Creation**
- Copy in hand labeled data
- Train/Val Split 80%/20%
- Generate icon syn data
- Generate BG syn data
- Generate regular syn data

**Model and Training Trends**
- More syn data = better model
- External models have a 50/50 chance of being better than COCO
- yolo26s.pt works best (for now)
- 250 - 400 epochs works best
- imgsz 1024 seems to be good
- batch 16, workers 8 ideal setup

**Pain Points**
- P: UI/Overlays on top of characters
- S: Syn data OR manually collect

- P: Overlapping characters
- S: Studio syn/Manual collect

- P: UI elements being confused for characters
- S: Syn data

- P: View models
- S: Manual collect FPS data OR syn data

- P: Real human faces
- S: Grab human faces from online

- P: Close up shots of characters
- S: Manual collect

- P: Far shots of characters
- S: Studio syn and isle manual collect

- P: Effects/particles on characters
- S: Syn data/Studio syn

- P: MM2 Paintings
- S: Go into MM2, Doors, Online pictures and manual collect

- P: Extremely large or odd avatars (wings, arms, etc)
- S: Join a hangout game and manual collect

## Disclaimers & Disclosures

Claude AI was used to assist in the planning, use, and production of these AI models.
Rolpon does not condone cheating, hacking, or exploiting on Roblox. These models were developed for educational and hobbyist purposes.
Rolpon is not responsible for any potential malicious use of these tools.
