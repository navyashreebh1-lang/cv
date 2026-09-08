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
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


def build_datasets():
    if not (C.WORK_DIR / "train").exists():
        sys.exit("[train] No prepared dataset. Run scripts/prepare_dataset.py first.")

    def load(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            C.WORK_DIR / split,
            labels="inferred",
            # One-hot, not integer, labels. Keras 3 removed `label_smoothing`
            # from SparseCategoricalCrossentropy - and could not sensibly have
            # kept it, because smoothing spreads epsilon mass across the whole
            # label vector and a bare class index has no vector to spread it
            # over. CategoricalCrossentropy does support it and needs one-hot
            # targets, so the conversion happens here, at load time.
            # `class_weight` is unaffected: Keras maps one-hot targets back to
            # class ids with argmax before applying the weights.
            label_mode="categorical",
            image_size=(C.IMG_SIZE, C.IMG_SIZE),
            batch_size=C.BATCH_SIZE,
            shuffle=shuffle,
            seed=C.SEED,
            interpolation="bilinear",
        )

    train_ds = load("train", True)
    val_ds = load("val", False)
    class_names = list(train_ds.class_names)

    # Sanity: val must expose exactly the same label space, in the same order.
    val_names = list(val_ds.class_names)
    if val_names != class_names:
        sys.exit(f"[train] FATAL: train/val class mismatch.\n  train={class_names}\n  val={val_names}")

    return train_ds, val_ds, class_names


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


def compute_class_weights(train_dir: Path, class_names: list[str]) -> dict:
    counts = np.array([len(list((train_dir / c).glob("*"))) for c in class_names],
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

    train_ds, val_ds, class_names = build_datasets()
    num_classes = len(class_names)
    print(f"[train] {num_classes} classes")

    class_weight = compute_class_weights(C.WORK_DIR / "train", class_names)
    imbalance = max(class_weight.values()) / min(class_weight.values())
    print(f"[train] class imbalance ratio: {imbalance:.1f}x (weights applied)")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

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
        "class_weighting": "balanced (n/(k*n_c))",
        "class_imbalance_ratio": round(imbalance, 2),
        "num_classes": num_classes, "tensorflow": tf.__version__,
    }, indent=2))

    print(f"\n[train] done."
          f"\n  best  -> {best_path}"
          f"\n  final -> {C.MODELS_DIR / 'final_model.keras'}"
          f"\n  meta  -> {C.MODELS_DIR / 'class_names.json'}"
          f"\n\n  NEXT: python scripts/evaluate.py   (do not trust training accuracy)")


if __name__ == "__main__":
    main()
