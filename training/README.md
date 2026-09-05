# Training a custom plant detector

The live demo ships with a generic **COCO-SSD** model (trained on 80 everyday
object classes, of which only a few relate to plants). This trains a
dedicated single-class **plant detector** on a much larger, plant-specific
slice of the Open Images V7 dataset, which should give better recall and
tighter boxes on real plant/crop footage than the generic model.

## Steps

1. Open `train_plant_detector.ipynb` in [Google Colab](https://colab.research.google.com/) (Upload notebook).
2. `Runtime > Change runtime type` → select **GPU**.
3. `Runtime > Run all`. This will:
   - Download a plant-related subset of Open Images V7 (Plant, Houseplant,
     Flower, Tree, Flowerpot, Fruit, Vegetable, Palm tree — merged into one
     `plant` class). Default is ~1500 images/class (~10k+ images); raise
     `MAX_SAMPLES_PER_CLASS` in the notebook for more data.
   - Fine-tune YOLOv8n for 60 epochs.
   - Export the trained model to ONNX, then convert it to a channels-last
     (NHWC) TensorFlow Lite model via `onnx2tf` — this is the format
     `script.js`'s `CustomModel` runtime (`tfjs-tflite`) expects.
   - Download `plant_detector_tflite.zip`.
4. Unzip it and copy `plant_detector.tflite` and `metadata.json` into
   `cv-prototype/model/` in this repo.
5. Reload the AGRIVISION page (with `python -m http.server 8080` still
   running from `cv-prototype/`). `script.js` looks for
   `model/plant_detector.tflite` on startup and uses it automatically —
   falling back to COCO-SSD if it isn't found.

## Notes

- Training ~10k images for 60 epochs on a Colab T4 GPU takes roughly
  30-60 minutes.
- To iterate on accuracy: raise `MAX_SAMPLES_PER_CLASS`, increase `epochs`,
  or try `yolov8s.pt` instead of `yolov8n.pt` (larger model, slower
  inference — fine on desktop, may be too slow on constrained hardware like
  a rover's onboard compute).
