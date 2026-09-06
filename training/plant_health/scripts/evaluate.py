"""
AGRIVISION Model 2 - evaluation.

    python scripts/evaluate.py [--model models/best_model.keras]

Reports, for BOTH the in-domain test split and the out-of-domain PlantDoc set:
  accuracy, macro/weighted precision, recall, F1, per-class table,
  confusion matrix (CSV + PNG), and health-level binary metrics.

The health-level section is the one that matters operationally. Telling a farmer
a diseased plant is healthy (a MISSED DISEASE) is far worse than getting the
crop name wrong, so we report that error rate on its own rather than burying it
in a macro average.

THE HEADLINE NUMBER IS THE OUT-OF-DOMAIN ONE. PlantVillage is lab imagery whose
backgrounds leak the label (Noyan 2022: 49.0% accuracy from 8 background pixels
alone vs a 2.6% random baseline), so a high in-domain score is not evidence the
model works on a camera. If OOD accuracy is far below in-domain, that gap IS the
result - report it, do not average it away.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402


def load_split(directory: Path, class_names: list[str]):
    """Load a directory as a dataset with a label order pinned to class_names."""
    if not directory.exists() or not any(directory.iterdir()):
        return None, None
    ds = tf.keras.utils.image_dataset_from_directory(
        directory, labels="inferred", label_mode="int",
        image_size=(C.IMG_SIZE, C.IMG_SIZE), batch_size=C.BATCH_SIZE,
        shuffle=False, class_names=[c for c in class_names
                                    if (directory / c).is_dir()],
        interpolation="bilinear",
    )
    present = list(ds.class_names)
    # Remap local indices -> global class_names indices.
    remap = np.array([class_names.index(c) for c in present])
    return ds, remap


def predict(model, ds, remap) -> tuple[np.ndarray, np.ndarray]:
    probs = model.predict(ds, verbose=0)
    y_true_local = np.concatenate([y.numpy() for _, y in ds])
    return remap[y_true_local], probs


def save_confusion(cm: np.ndarray, labels: list[str], stem: str) -> None:
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = C.REPORTS_DIR / f"{stem}_confusion.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("true\\pred," + ",".join(labels) + "\n")
        for lbl, row in zip(labels, cm):
            f.write(lbl + "," + ",".join(str(v) for v in row) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(labels)
        fig, ax = plt.subplots(figsize=(max(8, n * 0.45), max(6, n * 0.45)))
        norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{stem} (row-normalised)")
        fig.tight_layout()
        fig.savefig(C.REPORTS_DIR / f"{stem}_confusion.png", dpi=160)
        plt.close(fig)
    except Exception as exc:
        print(f"  (confusion PNG skipped: {exc})")


def health_metrics(y_true, y_pred, class_names) -> dict:
    """Binary healthy(1)/unhealthy(0) view, derived from the class predictions."""
    healthy_mask = np.array([C.is_healthy(c) for c in class_names])
    t = healthy_mask[y_true].astype(int)
    p = healthy_mask[y_pred].astype(int)

    # unhealthy = positive class for "disease present"
    true_unhealthy = (t == 0)
    pred_unhealthy = (p == 0)
    tp = int((true_unhealthy & pred_unhealthy).sum())
    fn = int((true_unhealthy & ~pred_unhealthy).sum())   # MISSED DISEASE
    fp = int((~true_unhealthy & pred_unhealthy).sum())   # false alarm
    tn = int((~true_unhealthy & ~pred_unhealthy).sum())

    n_unhealthy = tp + fn
    n_healthy = tn + fp
    return {
        "health_accuracy": float((t == p).mean()) if len(t) else 0.0,
        "unhealthy_recall_disease_caught": tp / n_unhealthy if n_unhealthy else None,
        "missed_disease_rate_unhealthy_called_healthy": fn / n_unhealthy if n_unhealthy else None,
        "false_alarm_rate_healthy_called_unhealthy": fp / n_healthy if n_healthy else None,
        "counts": {"true_unhealthy": n_unhealthy, "true_healthy": n_healthy,
                   "missed_disease": fn, "false_alarm": fp},
    }


def evaluate_one(model, directory: Path, class_names: list[str], stem: str) -> dict | None:
    ds, remap = load_split(directory, class_names)
    if ds is None:
        print(f"\n--- {stem}: NOT AVAILABLE ({directory}) ---")
        return None

    y_true, probs = predict(model, ds, remap)
    # The model's output axis is already the global class_names order, so argmax
    # gives a global index directly. y_true was remapped from the directory's
    # local order in predict().
    y_pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)

    present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_labels = [class_names[i] for i in present]

    acc = accuracy_score(y_true, y_pred)
    res = {
        "images": int(len(y_true)),
        "accuracy": float(acc),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mean_confidence": float(conf.mean()),
        "health": health_metrics(y_true, y_pred, class_names),
        "per_class": classification_report(y_true, y_pred, labels=present,
                                           target_names=present_labels,
                                           output_dict=True, zero_division=0),
    }

    # Abstention: how much does the confidence policy actually save us?
    keep = conf >= C.CONFIDENCE["condition_min"]
    res["abstention"] = {
        "threshold": C.CONFIDENCE["condition_min"],
        "kept_fraction": float(keep.mean()),
        "accuracy_on_kept": float(accuracy_score(y_true[keep], y_pred[keep])) if keep.any() else None,
    }

    cm = confusion_matrix(y_true, y_pred, labels=present)
    save_confusion(cm, present_labels, stem)

    print(f"\n--- {stem} ---")
    print(f"  images            : {res['images']}")
    print(f"  accuracy          : {acc:.4f}")
    print(f"  precision (macro) : {res['precision_macro']:.4f}")
    print(f"  recall    (macro) : {res['recall_macro']:.4f}")
    print(f"  F1        (macro) : {res['f1_macro']:.4f}")
    print(f"  F1     (weighted) : {res['f1_weighted']:.4f}")
    h = res["health"]
    print(f"  health accuracy   : {h['health_accuracy']:.4f}")
    if h["missed_disease_rate_unhealthy_called_healthy"] is not None:
        print(f"  MISSED DISEASE    : {h['missed_disease_rate_unhealthy_called_healthy']:.4f} "
              f"({h['counts']['missed_disease']}/{h['counts']['true_unhealthy']}) "
              f"<-- the dangerous error")
        print(f"  false alarm       : {h['false_alarm_rate_healthy_called_unhealthy']:.4f} "
              f"({h['counts']['false_alarm']}/{h['counts']['true_healthy']})")
    print(f"  abstain>= {res['abstention']['threshold']:.2f}    : keeps "
          f"{res['abstention']['kept_fraction']:.1%}, acc on kept "
          f"{res['abstention']['accuracy_on_kept']}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(C.MODELS_DIR / "best_model.keras"))
    args = ap.parse_args()

    meta_path = C.MODELS_DIR / "class_names.json"
    if not meta_path.exists():
        sys.exit("[eval] class_names.json missing - run scripts/train.py first.")
    class_names = json.loads(meta_path.read_text())["class_names"]

    model = tf.keras.models.load_model(args.model)
    print(f"[eval] loaded {args.model} ({model.count_params():,} params)")

    results = {
        "model": args.model,
        "in_domain_test_plantvillage": evaluate_one(
            model, C.WORK_DIR / "test", class_names, "in_domain_test"),
        "out_of_domain_plantdoc": evaluate_one(
            model, C.OOD_DIR, class_names, "ood_plantdoc"),
    }

    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORTS_DIR / "evaluation.json").write_text(json.dumps(results, indent=2))

    ind = results["in_domain_test_plantvillage"]
    ood = results["out_of_domain_plantdoc"]
    print("\n================ VERDICT ================")
    if ind:
        print(f"  in-domain  (PlantVillage test): {ind['accuracy']:.1%}")
    if ood:
        print(f"  OUT-OF-DOMAIN (PlantDoc field): {ood['accuracy']:.1%}   <-- HEADLINE")
        if ind:
            gap = ind["accuracy"] - ood["accuracy"]
            print(f"  domain gap                   : {gap:.1%}")
            if gap > 0.25:
                print("\n  WARNING: large domain gap. The model has substantially learned\n"
                      "  PlantVillage's lab conditions rather than the disease itself.\n"
                      "  Do NOT report the in-domain number as the model's accuracy.\n"
                      "  Fixes, in order of expected effect:\n"
                      "    1. train on field imagery (PlantDoc train split, or your own\n"
                      "       rover captures) instead of only PlantVillage\n"
                      "    2. stronger background randomisation / random cropping\n"
                      "    3. raise the abstention thresholds in config.CONFIDENCE")
    else:
        print("  OUT-OF-DOMAIN: unavailable. In-domain accuracy alone does NOT\n"
              "  demonstrate real-world skill - obtain the PlantDoc set before\n"
              "  reporting this model as working.")
    print(f"\n  reports -> {C.REPORTS_DIR}")


if __name__ == "__main__":
    main()
