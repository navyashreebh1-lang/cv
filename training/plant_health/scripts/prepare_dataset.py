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


def assign_field_split(cluster_key: str) -> str:
    """Deterministic train/val bucket for a PlantDoc duplicate cluster.

    Salted differently from assign_split() so a cluster's field assignment is
    not correlated with the PlantVillage bucketing, and driven only by the
    image content hash - so it does not depend on file order, on how many
    images exist, or on the run.
    """
    h = int(hashlib.sha256(("field:" + cluster_key).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for split in ("train", "val"):
        acc += C.FIELD_SPLIT_FRACTIONS[split]
        if h < acc:
            return split
    return "val"


def build_field_splits(repo_dir: Path, field_dir: Path, keep_labels: set,
                       pv_hashes: set) -> dict:
    """Turn PlantDoc into field/{train,val,test}/<plantvillage_label>/.

    PlantDoc's OWN published test split becomes field/test and is never trained
    on; its train split is divided into field/train and field/val.

    Leakage rules, in order of precedence:

      1. Clusters are built from the 64-bit dHash over the WHOLE PlantDoc
         corpus, so a cluster can span the published train and test splits.
      2. TEST WINS. If a cluster contains any test-split image, every
         train-split image in that cluster is DROPPED rather than moved. That
         removes the leak while leaving the published test set exactly as
         published - moving images into it would quietly redefine the
         benchmark.
      3. field/test keeps every published test image, duplicates included. It
         is the benchmark; collapsing duplicates inside it would make results
         non-comparable with published PlantDoc numbers. Any duplicates found
         are reported, not silently removed.
      4. Remaining (train-only) clusters are assigned whole to train or val, so
         a near-duplicate pair can never straddle them, and only ONE image per
         cluster is kept for training.
      5. Any PlantDoc image whose hash also appears in PlantVillage is dropped.
         Overlap is unlikely between lab and web imagery, but it would be a
         leak straight into the PlantVillage test split, and the check is free
         once both hash sets exist.
    """
    if field_dir.exists():
        shutil.rmtree(field_dir)

    stats = {
        "per_split": {"train": 0, "val": 0, "test": 0},
        "per_class": {},
        "unmapped_dirs": [],
        "dropped_not_in_label_space": 0,
        "dropped_unusable": 0,
        "dropped_train_leaking_into_test": 0,
        "dropped_duplicate_within_train": 0,
        "dropped_overlapping_plantvillage": 0,
        "duplicate_pairs_inside_published_test": 0,
    }

    # ---- 1. collect and hash everything --------------------------------
    records = []                      # (origin_split, pv_label, path, dhash)
    for origin in ("train", "test"):
        split_dir = repo_dir / origin
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
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                    continue
                if not is_usable_image(img):
                    stats["dropped_unusable"] += 1
                    continue
                try:
                    h = dhash(img)
                except Exception:
                    stats["dropped_unusable"] += 1
                    continue
                records.append((origin, pv_label, img, h))

    # ---- 2. cluster by hash, across BOTH published splits ---------------
    clusters = defaultdict(list)
    for rec in records:
        clusters[rec[3]].append(rec)

    test_hashes = {h for origin, _, _, h in records if origin == "test"}

    to_copy = []                      # (split, pv_label, path)
    for h, group in clusters.items():
        in_test = [r for r in group if r[0] == "test"]
        in_train = [r for r in group if r[0] == "train"]

        if in_test:
            # Rule 3: publish the test images as they are.
            if len(in_test) > 1:
                stats["duplicate_pairs_inside_published_test"] += len(in_test) - 1
            for _, pv_label, img, _ in in_test:
                to_copy.append(("test", pv_label, img))
            # Rule 2: everything on the train side of this cluster is dropped.
            stats["dropped_train_leaking_into_test"] += len(in_train)
            continue

        # Rule 5: overlap with the lab dataset.
        if h in pv_hashes:
            stats["dropped_overlapping_plantvillage"] += len(in_train)
            continue

        # Rule 4: one image per cluster, whole cluster to one split.
        split = assign_field_split(h)
        keep = sorted(in_train, key=lambda r: str(r[2]))[0]
        stats["dropped_duplicate_within_train"] += len(in_train) - 1
        to_copy.append((split, keep[1], keep[2]))

    # ---- 3. materialise --------------------------------------------------
    for split, pv_label, img in to_copy:
        dest = field_dir / split / pv_label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dest / img.name)
        stats["per_split"][split] += 1
        stats["per_class"].setdefault(pv_label, {"train": 0, "val": 0, "test": 0})
        stats["per_class"][pv_label][split] += 1

    # ---- 4. assert the guarantee rather than assume it -------------------
    # Cheap, and it is the whole point of the exercise: no hash may appear in
    # more than one field split.
    seen = {}
    for split, _, img in to_copy:
        h = dhash(img)
        if h in seen and seen[h] != split:
            raise RuntimeError(
                f"LEAK: hash {h} appears in both '{seen[h]}' and '{split}' "
                f"({img}). Refusing to write a contaminated split.")
        seen[h] = split
    stats["verified_no_cross_split_hash"] = True
    stats["published_test_images_kept"] = len(test_hashes)

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


def prepare(src_dir: Path, work_dir: Path,
            max_per_class: int | None) -> tuple[dict, set]:
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

    # Every PlantVillage hash seen, so build_field_splits() can drop any
    # PlantDoc image that duplicates a lab image. Collected here because the
    # hashes are computed anyway; re-deriving them would re-hash ~44k files.
    pv_hashes: set[str] = set()

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
            pv_hashes.add(h)
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

    return report, pv_hashes


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
    report, pv_hashes = prepare(pv_dir, C.WORK_DIR, args.max_per_class)

    labels = sorted(report["per_class"].keys())
    if not labels:
        sys.exit("[prepare] FATAL: no class survived pruning. Check the download.")

    # Field domain (PlantDoc), restricted to the surviving label space.
    # Its published TRAIN split feeds field/train + field/val; its published
    # TEST split becomes field/test and is never trained on.
    field_stats = None
    if not args.skip_download:
        try:
            repo = download_plantdoc(C.RAW_DIR)
            field_stats = build_field_splits(repo, C.FIELD_DIR, set(labels), pv_hashes)
        except Exception as exc:                     # non-fatal
            print(f"[prepare] WARNING: PlantDoc field set unavailable ({exc}). "
                  f"Training will be lab-only and evaluation in-domain ONLY, "
                  f"which overstates real-world skill.")

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
        # Regression guard: the PlantVillage split is assigned by a pure
        # function of each image's dhash, so this count must stay identical
        # across runs. If it moves, the lab test set moved and no comparison
        # with a previous run is valid.
        "plantvillage_test_images": report["per_split"]["test"],
        "field": ({"source": "PlantDoc (CC-BY-4.0)",
                   "published_train_split": "-> field/train + field/val",
                   "published_test_split": "-> field/test (never trained on)",
                   **field_stats} if field_stats else
                  {"source": "PlantDoc (CC-BY-4.0)", "available": False}),
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

    print("\n=== FIELD DOMAIN (PlantDoc) ===")
    if not field_stats:
        print("  NOT AVAILABLE - training will be lab-only.")
    else:
        f = field_stats
        print(f"  train / val / test  : {f['per_split']['train']} / "
              f"{f['per_split']['val']} / {f['per_split']['test']}")
        print(f"  classes covered     : {len(f['per_class'])} of {len(labels)}")
        print(f"  verified no hash spans two splits: "
              f"{f['verified_no_cross_split_hash']}")
        print("  dropped:")
        print(f"      leaking into published test : "
              f"{f['dropped_train_leaking_into_test']}")
        print(f"      duplicate within field/train: "
              f"{f['dropped_duplicate_within_train']}")
        print(f"      also in PlantVillage        : "
              f"{f['dropped_overlapping_plantvillage']}")
        print(f"      unusable / unmapped label   : "
              f"{f['dropped_unusable']} / {f['dropped_not_in_label_space']}")
        if f["duplicate_pairs_inside_published_test"]:
            print(f"  NOTE: {f['duplicate_pairs_inside_published_test']} duplicate(s) "
                  f"exist INSIDE the published test set; left in place so the "
                  f"benchmark stays as published.")
        fh = sum(v["test"] for k, v in f["per_class"].items() if C.is_healthy(k))
        fu = sum(v["test"] for k, v in f["per_class"].items() if not C.is_healthy(k))
        print(f"  field/test balance  : {fh} healthy / {fu} unhealthy "
              f"(always-unhealthy baseline = {fu / max(fh + fu, 1):.1%})")

    print(f"\n  manifest -> {C.DATASET_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
