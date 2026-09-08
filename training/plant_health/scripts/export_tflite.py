"""
AGRIVISION Model 2 - TensorFlow Lite export.

    python scripts/export_tflite.py [--model models/best_model.keras] [--int8]

Produces, in models/export/:
  plant_health_classifier.tflite    float32 (default) or dynamic-range INT8
  class_names.json                  copied verbatim from training

Then VERIFIES the converted model by running the TFLite interpreter against the
Keras model on real test images and comparing top-1 agreement and max absolute
probability drift. An export that silently changes behaviour is worse than no
export - Model 1 shipped with exactly that class of bug (a normalisation the
browser did not know about), so this check is not optional.

Copy both files into cv-prototype/model/ to activate the analysis stage.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

EXPORT_DIR = C.MODELS_DIR / "export"
TFLITE_NAME = "plant_health_classifier.tflite"


def representative_dataset(class_names, n=200):
    """Calibration data for INT8. Must be real images from the training
    distribution, fed exactly as the model expects (raw 0-255 floats)."""
    files = []
    train_dir = C.WORK_DIR / "train"
    for c in class_names:
        files.extend(sorted((train_dir / c).glob("*"))[:max(1, n // len(class_names))])

    def gen():
        for f in files[:n]:
            img = tf.keras.utils.load_img(f, target_size=(C.IMG_SIZE, C.IMG_SIZE))
            arr = tf.keras.utils.img_to_array(img)          # 0-255 float32
            yield [np.expand_dims(arr, 0).astype(np.float32)]
    return gen


def _make_converter(model):
    """Build a TFLiteConverter in a way that works on BOTH Keras 2 and Keras 3.

    Colab now ships TensorFlow 2.16+ with Keras 3, where
    `TFLiteConverter.from_keras_model()` on a functional Keras 3 model is no
    longer the supported route and can fail outright. The supported Keras 3
    path is to export a SavedModel first and convert that. Try the SavedModel
    route when Keras 3 is detected, and keep the direct route as the fallback
    (and the primary on Keras 2), so this script does not break on either.
    """
    keras_major = 0
    try:
        import keras
        keras_major = int(str(keras.__version__).split(".")[0])
    except Exception:
        pass

    attempts = []
    if keras_major >= 3:
        attempts = ["saved_model", "keras_model"]
    else:
        attempts = ["keras_model", "saved_model"]

    last_error = None
    for how in attempts:
        try:
            if how == "keras_model":
                conv = tf.lite.TFLiteConverter.from_keras_model(model)
            else:
                export_dir = C.MODELS_DIR / "_saved_model"
                if export_dir.exists():
                    shutil.rmtree(export_dir)
                # Keras 3 exposes .export(); Keras 2 needs tf.saved_model.save.
                if hasattr(model, "export"):
                    model.export(str(export_dir))
                else:
                    tf.saved_model.save(model, str(export_dir))
                conv = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
            print(f"[export] converter route: {how}")
            return conv
        except Exception as exc:      # noqa: BLE001 - report and try the other route
            last_error = exc
            print(f"[export] converter route '{how}' failed: {type(exc).__name__}: {exc}")

    raise RuntimeError(f"Could not build a TFLiteConverter: {last_error}")


def _make_interpreter(tflite_bytes: bytes):
    """tf.lite.Interpreter is being moved out of TF into ai-edge-litert.
    Use whichever this runtime actually provides."""
    try:
        return tf.lite.Interpreter(model_content=tflite_bytes)
    except Exception:
        from ai_edge_litert.interpreter import Interpreter    # TF >= 2.20
        return Interpreter(model_content=tflite_bytes)


def convert(model, class_names, int8: bool) -> bytes:
    converter = _make_converter(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if int8:
        converter.representative_dataset = representative_dataset(class_names)
        # Keep float I/O: tfjs-tflite in the browser feeds/reads float tensors,
        # and full-integer I/O would require the browser to know the quant
        # scale/zero-point. Weights are still quantised, which is where the
        # size saving comes from.
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.TFLITE_BUILTINS]
    return converter.convert()


def verify(tflite_bytes: bytes, model, class_names, n=60) -> dict:
    interp = _make_interpreter(tflite_bytes)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"[export] TFLite input : shape={inp['shape']} dtype={inp['dtype'].__name__}")
    print(f"[export] TFLite output: shape={out['shape']} dtype={out['dtype'].__name__}")

    test_dir = C.WORK_DIR / "test"
    files = []
    for c in class_names:
        files.extend(sorted((test_dir / c).glob("*"))[:max(1, n // len(class_names))])
    files = files[:n]
    if not files:
        return {"verified": False, "reason": "no test images available"}

    agree, drifts = 0, []
    for f in files:
        img = tf.keras.utils.load_img(f, target_size=(C.IMG_SIZE, C.IMG_SIZE))
        x = np.expand_dims(tf.keras.utils.img_to_array(img), 0).astype(np.float32)
        keras_p = model.predict(x, verbose=0)[0]
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        tfl_p = interp.get_tensor(out["index"])[0]
        agree += int(np.argmax(keras_p) == np.argmax(tfl_p))
        drifts.append(float(np.max(np.abs(keras_p - tfl_p))))

    return {
        "verified": True,
        "images": len(files),
        "top1_agreement": agree / len(files),
        "max_prob_drift": max(drifts),
        "mean_prob_drift": float(np.mean(drifts)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.MODELS_DIR / "best_model.keras"))
    ap.add_argument("--int8", action="store_true",
                    help="dynamic-range INT8 weights (smaller, slightly less accurate)")
    args = ap.parse_args()

    meta_path = C.MODELS_DIR / "class_names.json"
    if not meta_path.exists():
        sys.exit("[export] class_names.json missing - run scripts/train.py first.")
    meta = json.loads(meta_path.read_text())
    class_names = meta["class_names"]

    model = tf.keras.models.load_model(args.model)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[export] converting {args.model} (int8={args.int8})...")
    blob = convert(model, class_names, args.int8)
    tflite_path = EXPORT_DIR / TFLITE_NAME
    tflite_path.write_bytes(blob)

    result = verify(blob, model, class_names)
    meta["export"] = {
        "quantization": "dynamic-range-int8" if args.int8 else "float32",
        "size_bytes": len(blob),
        "size_mb": round(len(blob) / 1e6, 2),
        "verification": result,
    }
    (EXPORT_DIR / "class_names.json").write_text(json.dumps(meta, indent=2))

    print(f"\n[export] {tflite_path}  ({len(blob)/1e6:.2f} MB)")
    if result.get("verified"):
        print(f"[export] top-1 agreement with Keras: {result['top1_agreement']:.1%}")
        print(f"[export] max probability drift     : {result['max_prob_drift']:.4f}")
        if result["top1_agreement"] < 0.98:
            print("[export] WARNING: conversion changed predictions on >2% of images.\n"
                  "         Re-run without --int8, or investigate before shipping.")
    else:
        print(f"[export] VERIFICATION SKIPPED: {result.get('reason')}")

    print(f"\n[export] To activate in the browser app, copy BOTH files:\n"
          f"    {tflite_path}\n"
          f"    {EXPORT_DIR / 'class_names.json'}\n"
          f"  ->  cv-prototype/model/\n"
          f"  then reload the page. script.js picks them up automatically and\n"
          f"  leaves the detection stage untouched if they are absent.")


if __name__ == "__main__":
    main()
