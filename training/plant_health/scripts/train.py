"""
AGRIVISION Model 2 - training.

    python scripts/train.py [--epochs-head N] [--epochs-finetune N] [--quick]

Two-phase transfer learning on MobileNetV3-Small:
  Phase 1  frozen backbone, train the classifier head at LR 1e-3
  Phase 2  unfreeze the top 30% of the backbone, fine-tune at LR 1e-5

Writes to models/:
  best_model.keras          lowest val_loss checkpoint  (this is what we export)
  final_model.keras         last epoch, for inspection
  class_names.json          label space + derived maps + preprocessing contract
  training_config.json      every hyperparameter actually used
  history.csv               per-epoch metrics
"""

import argparse
import math
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


def _load_split(directory: Path, class_names: list[str] | None, shuffle: bool):
    """One directory -> a batched dataset. `class_names` pins the label order."""
    present = None
    if class_names is not None:
        present = [c for c in class_names if (directory / c).is_dir()
                   and any((directory / c).iterdir())]
        if not present:
            return None, []
    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        class_names=present,
        image_size=(C.IMG_SIZE, C.IMG_SIZE),
        batch_size=C.BATCH_SIZE,
        shuffle=shuffle,
        seed=C.SEED,
        interpolation="bilinear",
    )
    return ds, list(ds.class_names)


def _to_global_labels(ds, local_names: list[str], global_names: list[str]):
    """Re-express one-hot labels in the GLOBAL class space.

    The field split only covers 22 of the 34 trained classes, so a dataset
    loaded from it produces 22-wide one-hot vectors whose indices mean
    something different from the lab dataset's. Feeding those to the model
    would train every field image against the wrong class. This maps each local
    index onto its global index with a constant gather matrix.
    """
    if local_names == global_names:
        return ds
    m = np.zeros((len(local_names), len(global_names)), dtype="float32")
    for i, name in enumerate(local_names):
        m[i, global_names.index(name)] = 1.0
    matrix = tf.constant(m)
    return ds.map(lambda x, y: (x, tf.matmul(y, matrix)),
                  num_parallel_calls=tf.data.AUTOTUNE)


def _count_images(directory: Path, class_names: list[str]) -> int:
    if not directory.exists():
        return 0
    return sum(len(list((directory / c).glob("*")))
               for c in class_names if (directory / c).is_dir())


def build_datasets():
    """Lab (PlantVillage) + field (PlantDoc) training and validation streams.

    The lab set is ~44k images and the field set ~1.5k. Concatenating them
    would make field data roughly 3% of every batch, and the model would go on
    ignoring exactly the domain it is failing on - so the two streams are
    sampled at a fixed ratio (config.DOMAIN_MIX) instead. Validation is blended
    in the same proportion, because a validation set dominated by 6.7k lab
    images would keep selecting the lab-specialised checkpoint this change
    exists to avoid.

    Returns (train_ds, val_ds, class_names, steps_per_epoch, sizes).
    """
    if not (C.WORK_DIR / "train").exists():
        sys.exit("[train] No prepared dataset. Run scripts/prepare_dataset.py first.")

    lab_train, class_names = _load_split(C.WORK_DIR / "train", None, True)
    lab_val, val_names = _load_split(C.WORK_DIR / "val", None, False)

    # Sanity: val must expose exactly the same label space, in the same order.
    if val_names != class_names:
        sys.exit(f"[train] FATAL: train/val class mismatch.\n"
                 f"  train={class_names}\n  val={val_names}")

    n_lab_train = _count_images(C.WORK_DIR / "train", class_names)
    n_lab_val = _count_images(C.WORK_DIR / "val", class_names)

    field_train_dir = C.FIELD_DIR / "train"
    field_val_dir = C.FIELD_DIR / "val"
    has_field = field_train_dir.exists() and _count_images(field_train_dir, class_names) > 0

    if not has_field:
        print("[train] WARNING: no field/train split found - training LAB-ONLY. "
              "Run scripts/prepare_dataset.py to build it. The lab-only model is "
              "the one that scored 18% on field images.")
        steps = math.ceil(n_lab_train / C.BATCH_SIZE)
        return (lab_train.prefetch(tf.data.AUTOTUNE),
                lab_val.prefetch(tf.data.AUTOTUNE),
                class_names, steps,
                {"lab_train": n_lab_train, "field_train": 0,
                 "lab_val": n_lab_val, "field_val": 0})

    field_train, f_train_names = _load_split(field_train_dir, class_names, True)
    field_val, f_val_names = _load_split(field_val_dir, class_names, False)
    field_train = _to_global_labels(field_train, f_train_names, class_names)

    n_field_train = _count_images(field_train_dir, class_names)
    n_field_val = _count_images(field_val_dir, class_names)
    print(f"[train] lab train {n_lab_train:,} | field train {n_field_train:,} "
          f"({len(f_train_names)} of {len(class_names)} classes present)")

    # ---- training stream -------------------------------------------------
    # Unbatch before sampling so a single batch contains BOTH domains. Sampling
    # batched datasets would give homogeneous batches, which is worse for
    # BatchNorm. Both streams repeat, so the epoch length is set explicitly and
    # stays what it was before this change - the LR schedule and epoch counts
    # keep their meaning.
    mix = float(C.DOMAIN_MIX)
    train_ds = tf.data.Dataset.sample_from_datasets(
        [lab_train.unbatch().repeat(), field_train.unbatch().repeat()],
        weights=[1.0 - mix, mix],
        seed=C.SEED,
        stop_on_empty_dataset=False,
    ).batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    steps = math.ceil(n_lab_train / C.BATCH_SIZE)
    print(f"[train] domain mix {1 - mix:.0%} lab / {mix:.0%} field, "
          f"{steps} steps per epoch")

    # ---- validation blend ------------------------------------------------
    # All of field/val, plus a deterministic sample of lab/val of comparable
    # size. Shuffled with a fixed seed and reshuffle_each_iteration=False, so
    # the validation set is identical on every epoch and every run - a moving
    # validation set would make early stopping and val_loss meaningless.
    if field_val is not None and n_field_val > 0:
        field_val = _to_global_labels(field_val, f_val_names, class_names)
        share = float(C.VAL_FIELD_FRACTION)
        n_lab_keep = int(round(n_field_val * (1 - share) / max(share, 1e-6)))
        n_lab_keep = max(1, min(n_lab_keep, n_lab_val))
        lab_val_part = (lab_val.unbatch()
                        .shuffle(min(n_lab_val, 10000), seed=C.SEED,
                                 reshuffle_each_iteration=False)
                        .take(n_lab_keep))
        val_ds = (lab_val_part.concatenate(field_val.unbatch())
                  .batch(C.BATCH_SIZE).prefetch(tf.data.AUTOTUNE))
        print(f"[train] validation blend: {n_lab_keep:,} lab + {n_field_val:,} field")
    else:
        print("[train] WARNING: no field/val split - validating on lab images only, "
              "which will select a lab-specialised checkpoint.")
        val_ds = lab_val.prefetch(tf.data.AUTOTUNE)
        n_field_val = 0

    return (train_ds, val_ds, class_names, steps,
            {"lab_train": n_lab_train, "field_train": n_field_train,
             "lab_val": n_lab_val, "field_val": n_field_val})


def build_augmenter():
    """Realistic field variation only. No hue/saturation shifts - colour is the
    actual signal for chlorosis, rust and blight, so distorting it would teach
    the model to ignore the very thing it must read."""
    a = C.AUG
    layers = [tf.keras.layers.RandomFlip("horizontal")] if a["horizontal_flip"] else []
    layers += [
        tf.keras.layers.RandomRotation(a["rotation"], fill_mode="reflect"),
        tf.keras.layers.RandomZoom(a["zoom"], fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(a["translation"], a["translation"],
                                          fill_mode="reflect"),
        tf.keras.layers.RandomContrast(a["contrast"]),
        tf.keras.layers.RandomBrightness(a["brightness"], value_range=(0, 255)),
    ]

    # Sensor noise, applied in [0,1] and scaled straight back to 0-255.
    #
    # This augmenter runs on RAW 0-255 pixels: image_dataset_from_directory()
    # does no rescaling, the RandomBrightness above declares value_range=(0,255),
    # and normalisation happens later inside MobileNetV3 (include_preprocessing).
    # config.AUG["gaussian_noise_stddev"] = 4.0 is therefore 4 grey levels out
    # of 255 - mild sensor noise, which is the intended strength.
    #
    # Keras 3 rejects GaussianNoise(stddev > 1) because it assumes normalised
    # inputs. Passing 4/255 directly to a 0-255 tensor would apply ~250x too
    # little noise and quietly disable the augmentation, so instead the noise is
    # applied where Keras expects it and immediately scaled back:
    #
    #     (x/255 + N(0, 4/255)) * 255  ==  x + N(0, 4)
    #
    # identical to the original in both mean and standard deviation. The two
    # Rescaling layers are exact inverses, GaussianNoise is a no-op at inference,
    # so the model still consumes and forwards raw 0-255 exactly as
    # class_names.json promises the browser. Rescaling is a plain built-in layer
    # (a MUL in TFLite) - no custom objects, so load_model() in evaluate.py and
    # export_tflite.py keeps working untouched.
    stddev_01 = a["gaussian_noise_stddev"] / 255.0
    layers += [
        tf.keras.layers.Rescaling(1.0 / 255.0),
        tf.keras.layers.GaussianNoise(stddev_01),
        tf.keras.layers.Rescaling(255.0),
    ]
    return tf.keras.Sequential(layers, name="augment")


def build_model(num_classes: int):
    # include_preprocessing=True bakes [0,255] -> normalised INTO the graph, so
    # the browser feeds raw 0-255 floats and cannot mis-normalise. This contract
    # is recorded in class_names.json.
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=C.IMG_SHAPE,
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=C.IMG_SHAPE, name="image")
    x = build_augmenter()(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax",
                                    name="predictions")(x)
    model = tf.keras.Model(inputs, outputs, name="agrivision_plant_health")
    return model, base


def compute_class_weights(train_dirs, class_names: list[str]) -> dict:
    """Counts are summed over EVERY training directory - lab and field.

    Weighting on the lab counts alone would misstate the balance of the pool
    the model actually sees now that field data is mixed in at
    config.DOMAIN_MIX.
    """
    if isinstance(train_dirs, Path):
        train_dirs = [train_dirs]
    counts = np.array(
        [sum(len(list((d / c).glob("*"))) for d in train_dirs if (d / c).is_dir())
         for c in class_names],
        dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return {i: float(w) for i, w in enumerate(weights)}


def write_class_metadata(class_names: list[str], path: Path) -> None:
    """Everything the browser needs to interpret the model, so script.js never
    hard-codes a label, an index or a preprocessing assumption."""
    crops = sorted({C.crop_of(l) for l in class_names})
    meta = {
        "model": "agrivision_plant_health",
        "backbone": C.BACKBONE,
        "input": {
            "width": C.IMG_SIZE, "height": C.IMG_SIZE, "channels": 3,
            "layout": "NHWC",
            "dtype": "float32",
            "range": C.INPUT_RANGE,
            "note": "Preprocessing is baked into the graph "
                    "(MobileNetV3 include_preprocessing=True). Feed raw 0-255 "
                    "floats. Do NOT divide by 255.",
        },
        "output": {"type": "softmax", "num_classes": len(class_names)},
        "class_names": class_names,
        "crops": crops,
        "crop_display_names": {c: C.pretty_crop(c) for c in crops},
        # index -> derived attributes, so the browser marginalises correctly
        "class_meta": [
            {
                "index": i,
                "label": l,
                "crop": C.crop_of(l),
                "condition": C.condition_of(l),
                "condition_display": ("Healthy" if C.is_healthy(l)
                                      else C.pretty_condition(C.condition_of(l))),
                "healthy": C.is_healthy(l),
            }
            for i, l in enumerate(class_names)
        ],
        "confidence": C.CONFIDENCE,
    }
    path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs-head", type=int, default=C.EPOCHS_HEAD)
    ap.add_argument("--epochs-finetune", type=int, default=C.EPOCHS_FINETUNE)
    ap.add_argument("--quick", action="store_true",
                    help="2+2 epochs, for verifying the pipeline runs")
    args = ap.parse_args()
    if args.quick:
        args.epochs_head, args.epochs_finetune = 2, 2

    tf.keras.utils.set_random_seed(C.SEED)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[train] TensorFlow {tf.__version__}, GPUs: {tf.config.list_physical_devices('GPU')}")

    train_ds, val_ds, class_names, steps_per_epoch, sizes = build_datasets()
    num_classes = len(class_names)
    print(f"[train] {num_classes} classes")

    train_dirs = [C.WORK_DIR / "train"]
    if sizes["field_train"]:
        train_dirs.append(C.FIELD_DIR / "train")
    class_weight = compute_class_weights(train_dirs, class_names)
    imbalance = max(class_weight.values()) / min(class_weight.values())
    print(f"[train] class imbalance ratio: {imbalance:.1f}x (weights applied)")

    # build_datasets() already batched, mixed and prefetched both streams; the
    # training stream repeats indefinitely, so it must not be cached here.

    model, base = build_model(num_classes)

    # Metrics follow the label format chosen in build_datasets() - one-hot,
    # so the non-sparse variants. The reported NAMES are unchanged
    # ("acc" / "top3"), so history.csv and every callback monitor still work.
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="acc"),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
    ]
    # LABEL_SMOOTHING (0.05) is preserved exactly; only the class carrying it
    # changed. Both of these are BUILT-IN Keras losses, which matters more
    # than it looks: the compiled loss is serialised into best_model.keras,
    # and evaluate.py / export_tflite.py both call load_model() with no
    # custom_objects. A hand-written smoothing loss would train fine and then
    # fail to load afterwards - a worse bug than the one being fixed.
    loss = tf.keras.losses.CategoricalCrossentropy(
        label_smoothing=C.LABEL_SMOOTHING) if C.LABEL_SMOOTHING else \
        tf.keras.losses.CategoricalCrossentropy()

    best_path = C.MODELS_DIR / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path, monitor="val_loss",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=C.EARLY_STOPPING_PATIENCE,
                                         restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                                             patience=C.REDUCE_LR_PATIENCE, verbose=1),
        tf.keras.callbacks.CSVLogger(C.MODELS_DIR / "history.csv", append=True),
    ]

    # ---- Phase 1: head only -------------------------------------------------
    print(f"\n[train] PHASE 1 - frozen backbone, {args.epochs_head} epochs @ LR {C.LR_HEAD}")
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LR_HEAD),
                  loss=loss, metrics=metrics)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
              steps_per_epoch=steps_per_epoch,
              class_weight=class_weight, callbacks=callbacks, verbose=2)

    # ---- Phase 2: fine-tune -------------------------------------------------
    base.trainable = True
    freeze_upto = int(len(base.layers) * C.FINETUNE_AT)
    for layer in base.layers[:freeze_upto]:
        layer.trainable = False
    # BatchNorm must stay in inference mode while fine-tuning a small batch, or
    # the running statistics get wrecked and val accuracy collapses.
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    trainable = sum(1 for l in base.layers if l.trainable)
    print(f"\n[train] PHASE 2 - fine-tune {trainable}/{len(base.layers)} backbone "
          f"layers, {args.epochs_finetune} epochs @ LR {C.LR_FINETUNE}")
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LR_FINETUNE),
                  loss=loss, metrics=metrics)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_finetune,
              steps_per_epoch=steps_per_epoch,
              class_weight=class_weight, callbacks=callbacks, verbose=2)

    model.save(C.MODELS_DIR / "final_model.keras")
    write_class_metadata(class_names, C.MODELS_DIR / "class_names.json")

    (C.MODELS_DIR / "training_config.json").write_text(json.dumps({
        "backbone": C.BACKBONE, "img_size": C.IMG_SIZE,
        "batch_size": C.BATCH_SIZE, "seed": C.SEED,
        "epochs_head": args.epochs_head, "epochs_finetune": args.epochs_finetune,
        "lr_head": C.LR_HEAD, "lr_finetune": C.LR_FINETUNE,
        "finetune_at_fraction": C.FINETUNE_AT,
        "optimizer": "Adam", "label_smoothing": C.LABEL_SMOOTHING,
        "augmentation": C.AUG, "split_fractions": C.SPLIT_FRACTIONS,
        "class_weighting": "balanced (n/(k*n_c)) over lab+field train",
        "class_imbalance_ratio": round(imbalance, 2),
        "num_classes": num_classes, "tensorflow": tf.__version__,
        # Which domains this run actually saw, so a result can never be
        # attributed to the wrong training set after the fact.
        "domain_mix_field_fraction": C.DOMAIN_MIX,
        "val_field_fraction": C.VAL_FIELD_FRACTION,
        "steps_per_epoch": steps_per_epoch,
        "dataset_sizes": sizes,
    }, indent=2))

    print(f"\n[train] done."
          f"\n  best  -> {best_path}"
          f"\n  final -> {C.MODELS_DIR / 'final_model.keras'}"
          f"\n  meta  -> {C.MODELS_DIR / 'class_names.json'}"
          f"\n\n  NEXT: python scripts/evaluate.py   (do not trust training accuracy)")


if __name__ == "__main__":
    main()
