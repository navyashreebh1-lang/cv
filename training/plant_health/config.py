"""
AGRIVISION Model 2 - plant health/crop analysis classifier.
Single source of truth for dataset layout, label space, model and training
hyperparameters. Every script in scripts/ imports from here so the notebook,
the training run and the TFLite export can never drift apart.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"          # untouched downloads
WORK_DIR = DATASET_DIR / "work"        # PlantVillage (lab), cleaned + split
FIELD_DIR = DATASET_DIR / "field"      # PlantDoc (field), cleaned + split
# The held-out field test set. This is PlantDoc's OWN published test split and
# nothing in it is ever trained on. evaluate.py reads this path, so pointing it
# here is what makes the second reported set the held-out FIELD test.
#
# Naming note: evaluate.py still labels this section "out_of_domain_plantdoc".
# Since PlantDoc's train split now contributes to training, this set is no
# longer out-of-DOMAIN in the strict sense - it is a held-out set from a domain
# the model has seen. It remains a genuine generalisation test (those exact
# images, and every near-duplicate of them, are excluded from training), but
# the key name overstates it. Renaming it means editing evaluate.py, which is
# deliberately untouched in this change.
OOD_DIR = FIELD_DIR / "test"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = MODELS_DIR / "reports"

SPLITS = ("train", "val", "test")

# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
# We train ONE head over PlantVillage's native "Crop___Condition" labels rather
# than three separate heads. Crop, health and condition are then *derived* from
# the single predicted distribution:
#
#   P(crop)      = sum of probabilities of every class with that crop prefix
#   P(condition) = probability of the individual class
#   P(healthy)   = sum of probabilities of the crop's "healthy" classes
#
# This is exact marginalisation, and it makes contradictory outputs structurally
# impossible (three independent heads could return crop="Tomato" alongside
# condition="Potato Early blight"; this cannot).
LABEL_SEPARATOR = "___"
HEALTHY_TOKEN = "healthy"

# Crops we are willing to *report*. A class is kept only if the prepared dataset
# actually holds >= MIN_IMAGES_PER_CLASS usable images for it - see
# prepare_dataset.py, which prunes the label space and rewrites class_names.json
# accordingly. Nothing here is assumed to exist.
#
# NOTE ON INDIAN AGRICULTURE: PlantVillage covers apple, blueberry, cherry,
# corn(maize), grape, orange, peach, bell pepper, potato, raspberry, soybean,
# squash, strawberry and tomato. Of the crops requested, that gives us Tomato,
# Potato, Maize/Corn, Grape, Chilli/Pepper (bell pepper) and Soybean. It does
# NOT contain Rice, Wheat, Cotton, Sugarcane, Groundnut, Onion, Banana or Mango.
# Those are deliberately absent from the label space instead of being faked -
# adding them needs a separate sourced dataset (see DATASETS.md).
CANDIDATE_CROPS = [
    "Tomato",
    "Potato",
    "Corn_(maize)",
    "Grape",
    "Pepper,_bell",
    "Apple",
    "Peach",
    "Strawberry",
    "Cherry_(including_sour)",
    "Soybean",
]

# Human-readable crop names for the UI.
CROP_DISPLAY_NAMES = {
    "Tomato": "Tomato",
    "Potato": "Potato",
    "Corn_(maize)": "Maize / Corn",
    "Grape": "Grape",
    "Pepper,_bell": "Bell Pepper / Chilli",
    "Apple": "Apple",
    "Peach": "Peach",
    "Strawberry": "Strawberry",
    "Cherry_(including_sour)": "Cherry",
    "Soybean": "Soybean",
    "Blueberry": "Blueberry",
    "Orange": "Orange",
    "Raspberry": "Raspberry",
    "Squash": "Squash",
}

MIN_IMAGES_PER_CLASS = 150   # below this a class is dropped, not trained badly

# ---------------------------------------------------------------------------
# Image / model
# ---------------------------------------------------------------------------
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# MobileNetV3-Small: ~2.5M params, ~0.06 GFLOPs, designed for mobile/edge NPUs,
# first-class Keras support, and converts to TFLite cleanly (including INT8).
# Chosen over EfficientNet-Lite because Keras ships MobileNetV3 directly with
# ImageNet weights, and over MobileNetV2 because V3 gets better accuracy at
# lower latency. The rover's onboard compute is the binding constraint, so the
# smallest architecture that clears our accuracy bar wins.
BACKBONE = "MobileNetV3Small"

# include_preprocessing=True bakes the [0,255] -> normalised rescaling INTO the
# graph. The browser then feeds raw 0-255 floats and cannot get normalisation
# wrong - which is exactly the class of bug that broke Model 1's first version.
# This contract is written into class_names.json and read by script.js.
INPUT_RANGE = "0-255"

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SEED = 1337
BATCH_SIZE = 32
EPOCHS_HEAD = 12          # frozen backbone, train the new classifier head
EPOCHS_FINETUNE = 18      # unfreeze the top of the backbone
FINETUNE_AT = 0.70        # unfreeze the last 30% of backbone layers
LR_HEAD = 1e-3
LR_FINETUNE = 1e-5        # much lower - fine-tuning with a high LR destroys
                          # the pretrained features
LABEL_SMOOTHING = 0.05
EARLY_STOPPING_PATIENCE = 6
REDUCE_LR_PATIENCE = 3

# Split fractions. Applied to *duplicate clusters*, never to individual images -
# see prepare_dataset.py. PlantVillage contains many near-identical shots of the
# same physical leaf; splitting per-image puts the same leaf in train and test
# and inflates the score. Clusters are assigned atomically to one split.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

# ---------------------------------------------------------------------------
# Field domain (PlantDoc)
# ---------------------------------------------------------------------------
# The first training run scored 96.12% on the PlantVillage test split and
# 18.18% on PlantDoc - a 78-point domain gap. Its binary health accuracy on
# PlantDoc (82.85%) was only ~4.7 points better than always answering
# "unhealthy", because 78.2% of that set IS unhealthy. The model had learned
# PlantVillage's capture conditions, not the disease.
#
# The only reliable fix for a domain gap is data from that domain, so PlantDoc's
# own train split now contributes to training. Its published TEST split stays
# completely held out and is the honest field number.
#
# Test is NOT re-split: keeping PlantDoc's published split means the result
# stays comparable with the PlantDoc literature and cannot be accused of split
# shopping. Only its train split is divided, and by duplicate CLUSTER, never
# per image.
FIELD_SPLIT_FRACTIONS = {"train": 0.85, "val": 0.15}

# Fraction of each training batch drawn from the field domain.
# There are ~1.5k field images against ~44k lab ones. Concatenating them would
# make field data ~3% of every batch and the model would carry on ignoring it,
# so the two streams are sampled at a fixed ratio instead.
DOMAIN_MIX = 0.5

# Validation is blended in the same proportion. Left as pure PlantVillage,
# early stopping and checkpoint selection would be decided by ~6.7k lab images
# and would keep selecting the lab-specialised model this change exists to
# avoid.
VAL_FIELD_FRACTION = 0.5

# Augmentation - realistic field variation only. Nothing here should be able to
# turn a healthy leaf into something that looks diseased (no hue shifts, no
# heavy colour jitter), because colour IS the signal for several conditions.
AUG = {
    "horizontal_flip": True,
    "vertical_flip": False,     # plants have a gravity direction
    "rotation": 0.08,           # +/- ~29 degrees
    "zoom": 0.15,
    "translation": 0.10,
    "contrast": 0.20,
    "brightness": 0.20,
    "gaussian_noise_stddev": 4.0,   # on the 0-255 scale; mild sensor noise
}

# ---------------------------------------------------------------------------
# Inference-time confidence policy (shared with the browser)
# ---------------------------------------------------------------------------
# Abstention thresholds. Below these the app must say "ANALYSIS UNCERTAIN"
# rather than guess. There is no trained "unknown" class: a reject option based
# on the predicted distribution is the honest way to handle out-of-distribution
# input, because a trained "other" class only recognises the specific negatives
# it was shown. See DATASETS.md for how to add a real negative class later.
CONFIDENCE = {
    "crop_min": 0.60,        # below -> crop is not reported
    "condition_min": 0.55,   # below -> condition is not reported
    "health_min": 0.60,      # below -> health is not reported
    # If the top-1 and top-2 classes are this close the prediction is unstable.
    "margin_min": 0.10,
}


def crop_of(label: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato'"""
    return label.split(LABEL_SEPARATOR)[0]


def condition_of(label: str) -> str:
    """'Tomato___Early_blight' -> 'Early_blight'"""
    parts = label.split(LABEL_SEPARATOR)
    return parts[1] if len(parts) > 1 else "unknown"


def is_healthy(label: str) -> bool:
    return condition_of(label).lower() == HEALTHY_TOKEN


def pretty_crop(crop: str) -> str:
    return CROP_DISPLAY_NAMES.get(crop, crop.replace("_", " "))


def pretty_condition(condition: str) -> str:
    return condition.replace("_", " ").replace("  ", " ").strip().title()
