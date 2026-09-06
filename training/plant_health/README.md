# AGRIVISION Model 2 — Plant Health / Crop Analysis

The **second** model in the AGRIVISION pipeline. Model 1 (COCO-SSD, in the
browser) answers *"is there a plant, and where?"*. Model 2 answers *"what crop,
is it healthy, and what condition does it show?"* for each plant Model 1 found.

```
CAMERA
  ↓
MODEL 1  COCO-SSD  ──── no plant ───→  ⚠️ NO PLANT DETECTED
  ↓ plant boxes
crop each box out of the frame
  ↓
MODEL 2  MobileNetV3-Small
  ↓
crop + health + condition + confidence   (per plant, independently)
  ↓
AGRIVISION UI
```

Model 2 never sees the full frame — only the pixels inside a box Model 1 already
class-validated as a plant. A face or background cannot influence the health
verdict.

---

## Status

**The pipeline is complete and runnable. The model is not trained yet.**

Training needs a GPU. This machine has Python 3.14.5 (no TensorFlow wheel
exists for it), no TensorFlow, and no GPU — so training must run in Google
Colab. Until you run it, the browser app detects and counts plants exactly as
it does now and the analysis card reads *"Analysis model not installed"*.

---

## Layout

```
plant_health/
  config.py                       every hyperparameter + the label space
  DATASETS.md                     provenance, licences, limitations  ← read this
  dataset/                        populated by prepare_dataset.py (git-ignored)
  models/                         checkpoints, reports, TFLite export
  notebooks/
    train_plant_health.ipynb      Colab runner — start here
  scripts/
    prepare_dataset.py            download → clean → de-duplicate → split
    train.py                      two-phase transfer learning
    evaluate.py                   full metrics, in-domain AND out-of-domain
    export_tflite.py              convert + verify against Keras
```

Scripts are the source of truth; the notebook just orchestrates them, so a Colab
run and a local run cannot diverge.

---

## Run it

1. Open `notebooks/train_plant_health.ipynb` in Colab.
2. **Runtime → Change runtime type → GPU (T4)**.
3. Set `REPO_URL` in cell 1 to your repo, or upload a zip of this folder.
4. **Runtime → Run all.** ~30–60 min including the ~2 GB download.
5. Download `plant_health_model.zip`, unzip, and copy **both** files into
   `cv-prototype/model/`:
   - `plant_health_classifier.tflite`
   - `class_names.json`
6. Reload the page. `script.js` picks them up automatically.

Smoke-test the pipeline first with `python scripts/train.py --quick` (2+2 epochs).

---

## Design decisions

### One head, not three

The model predicts a single distribution over PlantVillage's native
`Crop___Condition` labels. Crop, health and condition are **derived** by exact
marginalisation:

```
P(crop)      = Σ probabilities of that crop's classes
P(healthy)   = Σ probabilities of the healthy classes
condition    = the single most likely class, with its own probability
```

Three independent heads could return `crop=Tomato` alongside
`condition=Potato Early blight`. Marginalising one distribution makes that
structurally impossible.

### MobileNetV3-Small

~2.5 M parameters, ~0.06 GFLOPs. Chosen because the rover's onboard compute is
the binding constraint, Keras ships it with ImageNet weights, and it converts to
TFLite cleanly including INT8. EfficientNet-Lite is comparable but needs
TF-Hub plumbing; MobileNetV2 is larger and less accurate at the same latency.

### Preprocessing is baked into the graph

Exported with `include_preprocessing=True`, so the model takes **raw 0–255
floats** and does its own rescaling. The browser cannot mis-normalise. The
contract is written into `class_names.json` and *read* by `script.js` — not
hard-coded. This is deliberate: Model 1 shipped broken because the browser
assumed a normalisation the model did not use.

### Abstention instead of an "unknown" class

There is no trained `Other` class. If the distribution is not confident enough
(`config.CONFIDENCE`: crop ≥ 0.60, condition ≥ 0.55, health ≥ 0.60, top-1/top-2
margin ≥ 0.10) the UI shows **ANALYSIS UNCERTAIN**.

A trained negative class only recognises the specific negatives it was shown; a
reject option on the distribution covers everything the model was never trained
for. `DATASETS.md §6` explains how to add a real `Other` class if you later want
one as well.

Verified behaviour: an even spread across crops → UNCERTAIN; a confident
`Tomato` with early-blight-vs-late-blight split 45/44 → UNCERTAIN (it refuses to
pick between two similar diseases rather than guessing).

### Leakage prevention

PlantVillage contains many near-identical shots of the same physical leaf.
Splitting per-image puts one shot in train and another in test and inflates the
score. `prepare_dataset.py` computes a 64-bit difference hash, keeps one copy of
each visual duplicate, and assigns **whole duplicate clusters** to a split.

---

## The accuracy warning

Read `DATASETS.md` before quoting any number.

PlantVillage is lab imagery — one detached leaf on a uniform background — and its
backgrounds leak the label. Noyan (2022,
[arXiv:2206.04374](https://arxiv.org/abs/2206.04374)) trained a classifier on
**8 background pixels alone** and got **49.0%** against a **2.6%** random
baseline. Models scoring ~99% in-domain have been measured at **~31%** on other
datasets.

So `evaluate.py` reports **two** numbers and treats the second as the headline:

| set | meaning |
|---|---|
| PlantVillage test split | in-domain; tells you almost nothing about a camera |
| **PlantDoc (field photos)** | out-of-domain; **this is the real number** |

If the gap exceeds 25 points the script prints a warning and refuses to let the
in-domain number stand as "the accuracy". Expect a large gap on the first run —
that is the honest starting point, not a failure of the pipeline.

There is a second domain shift on top of it: PlantVillage is *single detached
leaves*, while Model 1 hands Model 2 a crop of a *whole plant*. Closing both gaps
means training on field imagery (PlantDoc's train split, or your own rover
captures), not lowering the thresholds.

### Metrics reported

Accuracy, macro/weighted precision, recall, F1, per-class table, confusion
matrix (CSV + PNG), and a health-level binary view that isolates:

- **MISSED DISEASE** — unhealthy called healthy
- **FALSE ALARM** — healthy called unhealthy

These are reported separately because they are not equally bad. A false alarm
costs an inspection; a missed disease costs a crop.
