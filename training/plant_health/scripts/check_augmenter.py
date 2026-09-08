"""AGRIVISION Model 2 - minimal post-fix smoke test.

    python scripts/check_augmenter.py

Checks four things and nothing else. It builds no dataset, trains nothing and
writes nothing, so it is safe to run at any time:

  1. build_model() succeeds                    (the Keras 3 GaussianNoise crash)
  2. training-mode noise really is ~4 grey levels on the 0-255 scale
     (a fix that quietly weakened the augmentation would pass step 1 but fail
      here - which is the whole point of measuring it)
  3. inference mode is the identity, so the model still consumes and forwards
     RAW 0-255 exactly as class_names.json promises the browser
  4. the model produces a sane softmax

Exits non-zero on failure so it can gate a training run.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from train import build_augmenter, build_model  # noqa: E402

TOLERANCE = 0.25          # grey levels; sampling noise on 64 images is well under this
failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


print(f"TensorFlow {tf.__version__} / Keras {tf.keras.__version__}")
print(f"config.AUG['gaussian_noise_stddev'] = {C.AUG['gaussian_noise_stddev']} (0-255 scale)")
print()

# ---- 1. the model builds -------------------------------------------------
print("1. build_model()")
try:
    model, base = build_model(num_classes=34)
    built = True
except Exception as exc:                       # noqa: BLE001 - report, do not raise
    built = False
    check("build_model() returns a model", False, f"{type(exc).__name__}: {exc}")
if built:
    check("build_model() returns a model", True,
          f"{model.count_params():,} params, output {model.output_shape}")
    check("input is 224x224x3", model.input_shape[1:] == C.IMG_SHAPE,
          str(model.input_shape))

# ---- 2. augmentation strength is unchanged -------------------------------
print("\n2. augmentation strength (the part a lazy fix would break)")
aug = build_augmenter()
x = tf.random.stateless_uniform((64, C.IMG_SIZE, C.IMG_SIZE, 3), seed=(C.SEED, 1),
                                minval=0.0, maxval=255.0)

# Flip/rotate/zoom/translate/contrast/brightness all move pixels around, so
# isolate the noise: run the noise sub-layers only.
noise_only = tf.keras.Sequential([l for l in aug.layers
                                  if isinstance(l, (tf.keras.layers.Rescaling,
                                                    tf.keras.layers.GaussianNoise))])
delta = (noise_only(x, training=True) - x).numpy()
measured = float(delta.std())
check("training-mode noise stddev ~= config value",
      abs(measured - C.AUG["gaussian_noise_stddev"]) < TOLERANCE,
      f"measured {measured:.4f}, expected {C.AUG['gaussian_noise_stddev']:.4f}")
check("noise is zero-mean", abs(float(delta.mean())) < 0.1, f"mean {float(delta.mean()):+.5f}")

# ---- 3. inference is the identity (0-255 contract to the browser) --------
print("\n3. inference-mode passthrough (0-255 contract)")
out = noise_only(x, training=False).numpy()
max_err = float(np.abs(out - x.numpy()).max())
check("inference adds no noise and no scale drift", max_err < 1e-3,
      f"max abs error {max_err:.3e} on a 0-255 scale")

# ---- 4. forward pass ------------------------------------------------------
if built:
    print("\n4. forward pass")
    probs = model(x[:4], training=False).numpy()
    check("output is a probability distribution",
          probs.shape == (4, 34) and np.allclose(probs.sum(axis=1), 1.0, atol=1e-4),
          f"shape {probs.shape}, row sums {np.round(probs.sum(axis=1), 5).tolist()}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) -> {failures}")
    sys.exit(1)
print("All checks passed. Safe to run scripts/train.py.")
