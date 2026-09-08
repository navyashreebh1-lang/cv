"""
AGRIVISION Model 2 - dataset preparation.

    python scripts/prepare_dataset.py [--skip-download] [--max-per-class N]
                                      [--pv-source auto|github|tfds]

Pipeline:
  1. Materialise PlantVillage to class folders, from the authors' GitHub mirror
     (no Kaggle account, API token or login needed). TFDS is kept as a fallback.
  2. Fetch PlantDoc (field photos) as an OUT-OF-DOMAIN test set. Never trained on.
  3. Drop corrupt / tiny / greyscale-degenerate images.
  4. Cluster near-duplicates with a difference hash and assign whole clusters to
     one split -> no leakage of the same physical leaf across train/val/test.
  5. Prune classes below MIN_IMAGES_PER_CLASS.
  6. Write work/{train,val,test}/<label>/ and ood/<label>/, plus a manifest.

WHY THE OOD SET MATTERS (read this before trusting any accuracy number):
PlantVillage is lab imagery - one detached leaf on a uniform background. Noyan
(2022, arXiv:2206.04374) trained a classifier on *8 background pixels alone* and
got 49.0% on the PlantVillage test set against a 2.6% random baseline, i.e. the
backgrounds leak the label. Models scoring ~99% in-domain have been measured at
~31% on other datasets. So the PlantVillage test split is NOT evidence the model
works on a rover camera. PlantDoc is. evaluate.py reports both and treats the
PlantDoc number as the headline.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

PLANTDOC_REPO = "https://github.com/pratikkayal/PlantDoc-Dataset.git"

MIN_DIM = 48            # anything smaller cannot show a lesion
MIN_STDDEV = 4.0        # near-blank / single-colour images
DHASH_SIZE = 8          # 64-bit difference hash


# ---------------------------------------------------------------------------
# 1. PlantVillage
# ---------------------------------------------------------------------------
# ACQUISITION ONLY. Everything downstream of this function - cleaning,
# de-duplication, class pruning, cluster-based splitting, the OOD set - is
# unchanged and still reads `raw/plantvillage/<Crop___Condition>/*.jpg`.
#
# Why this is not just `tfds.load("plant_village")` any more: TFDS fetches the
# archive from data.mendeley.com, which now answers programmatic requests with
# **HTTP 403**, so the TFDS route fails before a single image is written. The
# GitHub mirror below is the dataset as published by the PlantVillage authors
# (Mohanty et al., the spMohanty/PlantVillage-Dataset repo), needs no account,
# no API token and no login, and its `raw/color/` folders are named with the
# same `Crop___Condition` convention this pipeline already expects - so nothing
# downstream has to change.
#
# TFDS is kept as a fallback rather than deleted: if Mendeley starts answering
# again, or the machine already has a TFDS cache, that path still works.
PLANTVILLAGE_REPO = "https://github.com/spMohanty/PlantVillage-Dataset.git"
PLANTVILLAGE_BRANCH = "master"
PLANTVILLAGE_SUBDIR = "raw/color"     # color images; grayscale/ and segmented/ are not fetched

# A complete copy is 38 class folders / ~54k images. Anything far below that is
# a half-finished download, not a dataset.
MIN_CLASSES_EXPECTED = 30
MIN_IMAGES_EXPECTED = 20000


def _survey(out_dir: Path) -> tuple[int, int]:
    """(class folders, image files) currently in out_dir."""
    if not out_dir.exists():
        return 0, 0
    class_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
    images = sum(1 for d in class_dirs for f in d.iterdir()
                 if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    return len(class_dirs), images


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _download_plantvillage_github(out_dir: Path) -> None:
    """Sparse-clone just raw/color, then move it into place.

    The clone lands in a temporary directory and is only moved to its final
    name once git has finished. That is what makes re-running safe: an
    interrupted download can never leave something that looks like a complete
    dataset, and the next run simply starts again from a clean slate.
    """
    tmp_repo = out_dir.parent / "_plantvillage_clone"
    if tmp_repo.exists():
        print(f"[prepare] Removing leftover partial clone at {tmp_repo}")
        shutil.rmtree(tmp_repo, ignore_errors=True)

    out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Partial clone + sparse checkout: fetch blobs only for raw/color, so the
    # grayscale and segmented copies of the same 54k images are never
    # transferred. Falls back to a plain shallow clone on an older git.
    try:
        print(f"[prepare] Cloning PlantVillage from {PLANTVILLAGE_REPO}")
        print(f"[prepare]   (sparse: {PLANTVILLAGE_SUBDIR} only - this takes a few minutes)")
        _run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
              "--branch", PLANTVILLAGE_BRANCH, PLANTVILLAGE_REPO, str(tmp_repo)])
        _run(["git", "-C", str(tmp_repo), "sparse-checkout", "init", "--cone"])
        _run(["git", "-C", str(tmp_repo), "sparse-checkout", "set", PLANTVILLAGE_SUBDIR])
        _run(["git", "-C", str(tmp_repo), "checkout", PLANTVILLAGE_BRANCH])
    except subprocess.CalledProcessError as exc:
        print(f"[prepare] Sparse clone unavailable ({exc}); retrying as a full shallow clone.")
        shutil.rmtree(tmp_repo, ignore_errors=True)
        _run(["git", "clone", "--depth", "1", "--branch", PLANTVILLAGE_BRANCH,
              PLANTVILLAGE_REPO, str(tmp_repo)])

    src = tmp_repo / PLANTVILLAGE_SUBDIR
    if not src.is_dir():
        raise RuntimeError(f"{PLANTVILLAGE_SUBDIR} not found in the clone at {tmp_repo}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    # Same filesystem, so this is a rename, not a 54k-file copy.
    shutil.move(str(src), str(out_dir))
    shutil.rmtree(tmp_repo, ignore_errors=True)


def _download_plantvillage_tfds(out_dir: Path) -> None:
    """Original route. Broken while data.mendeley.com returns 403, kept because
    it works from a warm TFDS cache and may start working again."""
    import tensorflow_datasets as tfds

    print("[prepare] Trying TFDS 'plant_village'...")
    ds, info = tfds.load("plant_village", split="train", with_info=True,
                         as_supervised=False, shuffle_files=False)
    names = info.features["label"].names
    print(f"[prepare] {info.splits['train'].num_examples} images, {len(names)} classes")

    out_dir.mkdir(parents=True, exist_ok=True)
    counters = defaultdict(int)
    for rec in tfds.as_numpy(ds):
        label = names[int(rec["label"])]
        cls_dir = out_dir / label
        cls_dir.mkdir(exist_ok=True)
        idx = counters[label]
        counters[label] += 1
        Image.fromarray(rec["image"]).save(cls_dir / f"{label}_{idx:05d}.jpg",
                                           quality=95)
    print(f"[prepare] Wrote {sum(counters.values())} images to {out_dir}")


def download_plantvillage(out_dir: Path, source: str = "auto") -> None:
    """Materialise PlantVillage as raw/plantvillage/<Crop___Condition>/*.

    Idempotent: an existing complete copy is detected and left alone, so
    re-running prepare_dataset.py after a crash does not re-download 54k images.
    """
    n_classes, n_images = _survey(out_dir)
    if n_classes >= MIN_CLASSES_EXPECTED and n_images >= MIN_IMAGES_EXPECTED:
        print(f"[prepare] PlantVillage already present at {out_dir} "
              f"({n_classes} classes, {n_images:,} images) - skipping download.")
        return
    if n_classes or n_images:
        print(f"[prepare] Found an INCOMPLETE PlantVillage at {out_dir} "
              f"({n_classes} classes, {n_images:,} images); re-fetching.")

    routes = {"auto": ["github", "tfds"], "github": ["github"], "tfds": ["tfds"]}[source]
    errors = []
    for route in routes:
        try:
            if route == "github":
                _download_plantvillage_github(out_dir)
            else:
                _download_plantvillage_tfds(out_dir)
        except Exception as exc:                      # noqa: BLE001 - try the next route
            errors.append(f"{route}: {type(exc).__name__}: {exc}")
            print(f"[prepare] Source '{route}' failed -> {type(exc).__name__}: {exc}")
            continue

        n_classes, n_images = _survey(out_dir)
        print(f"[prepare] PlantVillage ready via {route}: "
              f"{n_classes} class folders, {n_images:,} images at {out_dir}")
        if n_classes < MIN_CLASSES_EXPECTED or n_images < MIN_IMAGES_EXPECTED:
            errors.append(f"{route}: only {n_classes} classes / {n_images} images")
            print("[prepare] ...that is short of a complete copy; trying the next source.")
            continue
        return

    raise RuntimeError(
        "Could not obtain PlantVillage from any source.\n  " + "\n  ".join(errors) +
        f"\n\nManual fallback: place the class folders yourself as\n"
        f"  {out_dir}/<Crop___Condition>/*.jpg\n"
        f"then re-run with --skip-download.")


# ---------------------------------------------------------------------------
# 2. PlantDoc (out-of-domain field images)
# ---------------------------------------------------------------------------
# PlantDoc's folder names differ from PlantVillage's. Only classes that exist in
# BOTH label spaces can be used for OOD evaluation; everything else is reported
# as unusable rather than silently force-mapped onto a wrong label.
PLANTDOC_TO_PV = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}


def download_plantdoc(raw_dir: Path) -> Path:
    repo_dir = raw_dir / "PlantDoc-Dataset"
    if repo_dir.exists():
        print(f"[prepare] PlantDoc already present at {repo_dir}, skipping clone.")
        return repo_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("[prepare] Cloning PlantDoc (CC-BY-4.0)...")
    subprocess.run(["git", "clone", "--depth", "1", PLANTDOC_REPO, str(repo_dir)],
                   check=True)
    return repo_dir


def build_ood_set(repo_dir: Path, out_dir: Path, keep_labels: set) -> dict:
    """Copy the mappable PlantDoc images into ood/<plantvillage_label>/."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"copied": 0, "unmapped_dirs": [], "dropped_not_in_label_space": 0}
    for split in ("train", "test"):          # OOD set: we use all of PlantDoc
        split_dir = repo_dir / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            pv_label = PLANTDOC_TO_PV.get(cls_dir.name)
            if pv_label is None:
                if cls_dir.name not in stats["unmapped_dirs"]:
                    stats["unmapped_dirs"].append(cls_dir.name)
                continue
            if pv_label not in keep_labels:
                stats["dropped_not_in_label_space"] += 1
                continue
            dest = out_dir / pv_label
            dest.mkdir(exist_ok=True)
            for img in cls_dir.iterdir():
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if not is_usable_image(img):
                    continue
                shutil.copy2(img, dest / f"{split}_{img.name}")
                stats["copied"] += 1
    return stats


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------
def is_usable_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im = im.convert("RGB")
            if min(im.size) < MIN_DIM:
                return False
            arr = np.asarray(im.resize((32, 32)), dtype=np.float32)
            if float(arr.std()) < MIN_STDDEV:     # blank / flat image
                return False
        return True
    except Exception:
        return False


def dhash(path: Path, size: int = DHASH_SIZE) -> str:
    """Difference hash - stable under resize/recompression, so it catches the
    'same leaf photographed twice' duplicates that per-image splitting leaks."""
    with Image.open(path) as im:
        im = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        arr = np.asarray(im, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    return hashlib.md5(np.packbits(bits).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# 4. Split with cluster-level assignment
# ---------------------------------------------------------------------------
def assign_split(cluster_key: str) -> str:
    """Deterministic hash-bucket assignment. Every image sharing a dhash lands in
    the same split, so a duplicate can never straddle train and test."""
    h = int(hashlib.sha256(cluster_key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for split in C.SPLITS:
        acc += C.SPLIT_FRACTIONS[split]
        if h < acc:
            return split
    return "test"


def prepare(src_dir: Path, work_dir: Path, max_per_class: int | None) -> dict:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    for s in C.SPLITS:
        (work_dir / s).mkdir(parents=True, exist_ok=True)

    report = {
        "classes_seen": 0, "classes_kept": 0, "classes_dropped": [],
        "corrupt_or_blank": 0, "duplicates_collapsed": 0,
        "per_split": {s: 0 for s in C.SPLITS}, "per_class": {},
    }

    class_dirs = sorted(p for p in src_dir.iterdir() if p.is_dir())
    report["classes_seen"] = len(class_dirs)

    for cls_dir in class_dirs:
        label = cls_dir.name
        crop = C.crop_of(label)
        if crop not in C.CANDIDATE_CROPS:
            report["classes_dropped"].append({"label": label, "reason": "crop not in CANDIDATE_CROPS"})
            continue

        seen_hashes: dict[str, str] = {}      # dhash -> split
        kept: list[tuple[Path, str]] = []
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if not is_usable_image(img):
                report["corrupt_or_blank"] += 1
                continue
            try:
                h = dhash(img)
            except Exception:
                report["corrupt_or_blank"] += 1
                continue
            if h in seen_hashes:
                # Exact-visual duplicate: keep ONE copy only. Duplicates inflate
                # both the class count and the test score.
                report["duplicates_collapsed"] += 1
                continue
            split = assign_split(h)
            seen_hashes[h] = split
            kept.append((img, split))
            if max_per_class and len(kept) >= max_per_class:
                break

        if len(kept) < C.MIN_IMAGES_PER_CLASS:
            report["classes_dropped"].append(
                {"label": label, "reason": f"only {len(kept)} usable images "
                                           f"(< {C.MIN_IMAGES_PER_CLASS})"})
            continue

        counts = {s: 0 for s in C.SPLITS}
        for img, split in kept:
            dest = work_dir / split / label
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dest / img.name)
            counts[split] += 1
            report["per_split"][split] += 1
        report["per_class"][label] = counts
        report["classes_kept"] += 1

    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="cap images per class (useful for a fast smoke run)")
    ap.add_argument("--pv-source", choices=("auto", "github", "tfds"), default="auto",
                    help="where PlantVillage comes from. 'auto' tries the GitHub "
                         "mirror then TFDS; 'tfds' forces the old Mendeley route, "
                         "which currently returns HTTP 403.")
    args = ap.parse_args()

    pv_dir = C.RAW_DIR / "plantvillage"
    if not args.skip_download:
        download_plantvillage(pv_dir, source=args.pv_source)

    print("[prepare] Cleaning, de-duplicating and splitting...")
    report = prepare(pv_dir, C.WORK_DIR, args.max_per_class)

    labels = sorted(report["per_class"].keys())
    if not labels:
        sys.exit("[prepare] FATAL: no class survived pruning. Check the download.")

    # OOD set, restricted to the surviving label space.
    ood_stats = {"copied": 0, "unmapped_dirs": [], "dropped_not_in_label_space": 0}
    if not args.skip_download:
        try:
            repo = download_plantdoc(C.RAW_DIR)
            ood_stats = build_ood_set(repo, C.OOD_DIR, set(labels))
        except Exception as exc:                     # non-fatal
            print(f"[prepare] WARNING: PlantDoc OOD set unavailable ({exc}). "
                  f"Evaluation will be in-domain ONLY, which overstates real-world skill.")

    healthy = sum(v for k, vs in report["per_class"].items()
                  if C.is_healthy(k) for v in vs.values())
    unhealthy = sum(v for k, vs in report["per_class"].items()
                    if not C.is_healthy(k) for v in vs.values())

    manifest = {
        "labels": labels,
        "num_classes": len(labels),
        "crops": sorted({C.crop_of(l) for l in labels}),
        "healthy_images": healthy,
        "unhealthy_images": unhealthy,
        "prepare_report": report,
        "ood": {"images": ood_stats["copied"],
                "unmapped_plantdoc_dirs": ood_stats["unmapped_dirs"],
                "source": "PlantDoc (CC-BY-4.0)"},
    }
    C.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (C.DATASET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n=== DATASET SUMMARY ===")
    print(f"  classes seen / kept : {report['classes_seen']} / {report['classes_kept']}")
    print(f"  dropped             : {len(report['classes_dropped'])}")
    for d in report["classes_dropped"]:
        print(f"      - {d['label']}: {d['reason']}")
    print(f"  corrupt or blank    : {report['corrupt_or_blank']}")
    print(f"  duplicates collapsed: {report['duplicates_collapsed']}")
    print(f"  train/val/test      : {report['per_split']['train']} / "
          f"{report['per_split']['val']} / {report['per_split']['test']}")
    print(f"  healthy / unhealthy : {healthy} / {unhealthy}")
    print(f"  OOD (PlantDoc)      : {ood_stats['copied']} images")
    print(f"\n  manifest -> {C.DATASET_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
