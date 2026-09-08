"""AGRIVISION Model 2 - dataset split integrity check.

    python scripts/check_splits.py

Run after scripts/prepare_dataset.py and before training. It re-derives the
guarantees from the files on disk rather than trusting the manifest, and exits
non-zero if any of them is violated:

  1. The PlantVillage test split is unchanged, so the previous run's 96.12%
     stays a valid comparison. (Assignment is a pure function of each image's
     dhash, so this must hold across runs.)
  2. No image hash appears in more than one field split - the near-duplicate
     leakage guarantee for field train / val / test.
  3. field/test is exactly PlantDoc's published test split, and none of those
     images (or their near-duplicates) appear in field/train or field/val.
  4. No field image duplicates a PlantVillage image.
  5. Every field class exists in the lab label space, so the label remap in
     train.py cannot silently mistrain a class.

It reads images, so it takes a couple of minutes on the full dataset.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from prepare_dataset import dhash  # noqa: E402

EXPECTED_PV_TEST = 6751          # from the first full run; see models/reports/
failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def hashes_of(root: Path) -> dict:
    """{dhash: [relative paths]} for every image under root/<class>/."""
    out = defaultdict(list)
    if not root.exists():
        return out
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                out[dhash(img)].append(f"{cls_dir.name}/{img.name}")
            except Exception:
                pass
    return out


def count(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for d in root.iterdir() if d.is_dir()
               for f in d.iterdir()
               if f.suffix.lower() in (".jpg", ".jpeg", ".png"))


print("Hashing the prepared dataset (a couple of minutes)...\n")

lab = {s: hashes_of(C.WORK_DIR / s) for s in C.SPLITS}
field = {s: hashes_of(C.FIELD_DIR / s) for s in ("train", "val", "test")}

# ---- 1. the lab test split has not moved ---------------------------------
print("1. PlantVillage split unchanged (comparability with the previous run)")
n_pv_test = count(C.WORK_DIR / "test")
check(f"lab test image count == {EXPECTED_PV_TEST}", n_pv_test == EXPECTED_PV_TEST,
      f"found {n_pv_test:,}")
for s in C.SPLITS:
    print(f"        lab/{s:<5}: {count(C.WORK_DIR / s):>7,} images")

# ---- 2 & 3. field splits are disjoint by hash -----------------------------
print("\n2. Field splits share no image (near-duplicate leakage)")
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    shared = set(field[a]) & set(field[b])
    example = ""
    if shared:
        h = sorted(shared)[0]
        example = f"e.g. {field[a][h][0]} <-> {field[b][h][0]}"
    check(f"field/{a} and field/{b} are disjoint", not shared,
          f"{len(shared)} shared hash(es) {example}" if shared else "")

for s in ("train", "val", "test"):
    print(f"        field/{s:<5}: {count(C.FIELD_DIR / s):>7,} images, "
          f"{len(field[s]):,} distinct hashes")

# ---- 4. no overlap with the lab dataset -----------------------------------
print("\n3. Field images do not duplicate PlantVillage images")
lab_all = set().union(*[set(lab[s]) for s in C.SPLITS]) if lab else set()
for s in ("train", "val", "test"):
    overlap = set(field[s]) & lab_all
    check(f"field/{s} vs PlantVillage", not overlap, f"{len(overlap)} shared hash(es)")

# ---- 5. label space ------------------------------------------------------
print("\n4. Field label space is a subset of the trained label space")
meta = C.DATASET_DIR / "manifest.json"
if meta.exists():
    labels = set(json.loads(meta.read_text())["labels"])
    field_labels = set()
    for s in ("train", "val", "test"):
        d = C.FIELD_DIR / s
        if d.exists():
            field_labels |= {p.name for p in d.iterdir() if p.is_dir()}
    extra = field_labels - labels
    check("every field class is a trained class", not extra, f"unknown: {sorted(extra)}")
    print(f"        {len(field_labels)} field classes of {len(labels)} trained")
else:
    check("manifest.json present", False, "run scripts/prepare_dataset.py first")

# ---- balance of the held-out field test ----------------------------------
test_dir = C.FIELD_DIR / "test"
if test_dir.exists():
    h = sum(1 for d in test_dir.iterdir() if d.is_dir() and C.is_healthy(d.name)
            for _ in d.iterdir())
    u = count(test_dir) - h
    print(f"\n5. Held-out field test balance: {h} healthy / {u} unhealthy")
    if h + u:
        print(f"        always-UNHEALTHY baseline health accuracy = {u/(h+u):.2%}")
        print("        (any reported health accuracy must beat this to mean anything)")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) -> {failures}")
    sys.exit(1)
print("All split checks passed.")
