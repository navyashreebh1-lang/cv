# AGRIVISION training

Two models, trained separately, with separate jobs:

| | model | question it answers | where |
|---|---|---|---|
| **Model 1** | COCO-SSD (in use) or a custom YOLOv8n | *Is there a plant, and where?* | this file |
| **Model 2** | MobileNetV3-Small classifier | *What crop, is it healthy, what condition?* | [`plant_health/`](plant_health/README.md) |

The app currently runs **COCO-SSD** as Model 1 (`DETECTION_CONFIG.engine` in
`script.js`) because the custom detector below reports plants on person-only
frames. Model 2 is built but **not trained yet** — see
[`plant_health/README.md`](plant_health/README.md).

---

# Model 1 — training a custom plant detector

The live demo ships with a generic **COCO-SSD** model (80 everyday object
classes, few plant-related). This trains a dedicated single-class **plant
detector** so the demo can count *whole plants* reliably.

## What changed and why (2026-09-06)

The first trained model classified people / faces / background as `plant` at
~90%+ confidence. Two causes:

1. **Diluted class.** The dataset merged Open Images `Flower`, `Flowerpot`,
   `Fruit`, `Vegetable` into `plant`. Open Images boxes those as *separate
   sub-objects* (one bloom, an empty pot, a single tomato, a produce pile), so
   the model learned "plant = any round / potted / produce-like blob" - which a
   face or torso satisfies. The notebook now trains only on **whole-organism**
   classes: `Plant`, `Houseplant`, `Tree`, `Palm tree`.
2. **No negatives.** Every training image contained a plant, so the model never
   saw a "nothing here" frame. The notebook now also downloads a **background /
   negative set** (people, faces, hands, vehicles, furniture, buildings, empty
   pots) with plant images scrubbed out, and exports them with empty label
   files. YOLOv8 trains on these as hard negatives.

A separate inference bug was fixed in `script.js`: the exported model emits box
coords **normalized to [0,1]**, but the browser decoded them as 640-pixel
values, so every box collapsed into the top-left corner. `CustomModel.detect()`
now auto-detects the coordinate space.

## Steps

1. Open `train_plant_detector.ipynb` in [Google Colab](https://colab.research.google.com/) (Upload notebook).
2. `Runtime > Change runtime type` -> select **GPU** (T4 is fine).
3. `Runtime > Run all`. This will:
   - Download the **positive** set (`Plant`, `Houseplant`, `Tree`, `Palm tree`,
     merged into one `plant` class). Default ~2000 images/class; raise
     `MAX_SAMPLES_PER_CLASS` for more.
   - Download the **negative** set (`MAX_NEGATIVES_TRAIN` background images,
     default 4000) and scrub any that contain a plant.
   - Fine-tune YOLOv8n for 80 epochs on positives + negatives.
   - Export to ONNX, then TFLite (float32) via `onnx2tf`.
   - Download `plant_detector_tflite.zip`.
4. Unzip and copy `plant_detector.tflite` + `metadata.json` into
   `cv-prototype/model/`.
5. **Switch the app back to the custom engine.** Because the *current* shipped
   model reports plants on person-only frames, `script.js` defaults to
   COCO-SSD:

   ```js
   const DETECTION_CONFIG = {
       engine: 'coco-ssd',   // <- change to 'custom' after retraining
   ```

   Set it to `'custom'`, reload the AGRIVISION page (`python -m http.server 8080`
   from `cv-prototype/`), and confirm the acceptance table below still passes.
   If it doesn't, put it back to `'coco-ssd'` and train with more negatives.

## Acceptance check (do this before trusting the model)

Open the browser console (`[AGRIVISION]` filter) and confirm:

| Scene                        | `FINAL PLANTS` |
|------------------------------|----------------|
| person, no plant in frame    | 0              |
| one whole plant              | 1              |
| two separated plants         | 2              |
| three separated plants       | 3              |
| person + one plant           | 1              |

If a person still reads as > 0, raise `MAX_NEGATIVES_TRAIN` and retrain - do
**not** paper over it by raising the confidence threshold or filtering the
count in the UI.

## Notes

- ~12k images (8k positive + 4k negative) for 80 epochs on a Colab T4 is
  roughly 1-1.5 hours.
- To iterate on accuracy: raise `MAX_SAMPLES_PER_CLASS` and
  `MAX_NEGATIVES_TRAIN` together (keep negatives at roughly half the
  positives), increase `epochs`, or try `yolov8s.pt` (larger, slower - fine on
  desktop, may be too slow on a rover's onboard compute).
- Watch `metrics/precision(B)` in the training log, not just `mAP50`. Low
  precision with decent recall means more negatives are needed.
