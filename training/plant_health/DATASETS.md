# Model 2 — dataset provenance, licensing and limitations

Every dataset the training pipeline touches, what it is used for, and what it
cannot support. Nothing is downloaded that is not listed here.

---

## 1. PlantVillage — primary training set

| field | value |
|---|---|
| **Name** | PlantVillage (Hughes & Salathé, 2015) |
| **Source** | **Used:** [github.com/spMohanty/PlantVillage-Dataset](https://github.com/spMohanty/PlantVillage-Dataset) `raw/color/` (the authors' own release) · *fallback:* [TFDS `plant_village`](https://www.tensorflow.org/datasets/catalog/plant_village) · [Kaggle mirror](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage) |
| **Images** | 54,303 (256×256 RGB, single detached leaf) |
| **Classes** | 38, named `Crop___Condition` |
| **Crops** | apple, blueberry, cherry, corn (maize), grape, orange, peach, bell pepper, potato, raspberry, soybean, squash, strawberry, tomato — 14 total |
| **Healthy classes** | 12 (one `___healthy` per crop that has one) |
| **Disease classes** | 26 |
| **License** | Public domain / CC0 as distributed; redistributed through TFDS |
| **Permitted use** | Research, education, commercial — no restriction asserted by the distributors |
| **How we use it** | Train / validation / in-domain test. Fetched by a sparse `git clone` of the authors' GitHub repo - no Kaggle account, API token or login. |
| **Why not TFDS** | TFDS downloads the archive from `data.mendeley.com`, which now answers programmatic requests with **HTTP 403**, so `tfds.load("plant_village")` fails outright. `prepare_dataset.py --pv-source tfds` still forces that route if it ever starts working again. |

### The critical limitation

**PlantVillage is laboratory imagery: one detached leaf, uniform background,
consistent lighting and angle.** It is not field data, and its backgrounds carry
label information.

Noyan (2022), *Uncovering bias in the PlantVillage dataset*
([arXiv:2206.04374](https://arxiv.org/abs/2206.04374)), trained a classifier on
**8 background pixels alone** and reached **49.0%** accuracy against a **2.6%**
random baseline. Removing the background does not remove the bias — the capture
conditions themselves (camera, lighting, session) correlate with the label.

Reported consequences: models scoring ~99% on PlantVillage have been measured at
**~31%** on other datasets, and ~95% in-domain drops to **~62%** on custom field
images.

**Therefore a high PlantVillage test accuracy is not evidence this model works on
a camera.** `evaluate.py` treats the out-of-domain number as the headline and
prints a warning when the gap exceeds 25 points.

### Crops it does NOT contain

Of the crops requested for Indian agriculture, PlantVillage covers **Tomato,
Potato, Maize/Corn, Grape, Bell Pepper (as a proxy for Chilli) and Soybean**.

It contains **no Rice, Wheat, Cotton, Sugarcane, Groundnut, Onion, Banana or
Mango.** Those are deliberately absent from `config.CANDIDATE_CROPS` rather than
being faked. Adding them requires a separately sourced and licence-checked
dataset — candidates worth evaluating are listed at the bottom of this file.

---

## 2. PlantDoc — out-of-domain test set (never trained on)

| field | value |
|---|---|
| **Name** | PlantDoc (Singh et al., CoDS-COMAD 2020) |
| **Source** | [github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) · [paper](https://arxiv.org/abs/1911.10317) |
| **Images** | 2,598 |
| **Classes** | 13 plant species, up to 17 disease classes (~28 folders) |
| **License** | **CC-BY-4.0** — attribution required |
| **Permitted use** | Research, education and commercial, with attribution and citation |
| **How we use it** | **Split by its own published split.** PlantDoc's `train/` feeds our `field/train` + `field/val`; its `test/` becomes `field/test` and is **never trained on**. |

Field photographs scraped from the web and hand-annotated (~300 human hours), so
they carry the messiness the rover camera will actually see: dirt, hands,
overlapping canopy, shadows, variable sunlight.

### Why it is now a TRAINING domain (changed 2026-09-08)

The first full run scored **96.12%** on the PlantVillage test split and **18.18%**
on PlantDoc - a 78-point domain gap. Its binary health accuracy on PlantDoc
(82.85%) was only ~4.7 points above always answering "unhealthy", because 78.2%
of that set is unhealthy. The model had learned PlantVillage's capture
conditions rather than the disease.

A domain gap is only closed with data from that domain, so PlantDoc's published
**train** split now contributes to training. What this costs, stated plainly:
`field/test` is a **held-out field test**, not an unseen *domain* - the model has
now seen other images from the same corpus. It stays a genuine generalisation
test (those images and every near-duplicate of them are excluded from training),
but it no longer measures transfer to a completely unseen distribution. The only
thing that would is a set captured on the actual rover camera.

**Split integrity** (enforced in `prepare_dataset.py`, verified independently by
`scripts/check_splits.py`):

| rule | handling |
|---|---|
| Published test split | Used as `field/test` verbatim - never re-split, never trained on |
| Train images that duplicate a test image | **Dropped from training**, not moved, so the published benchmark is left exactly as published |
| Duplicates inside the published test set | Left in place and reported - collapsing them would make results non-comparable with published PlantDoc numbers |
| field train vs val | Assigned per duplicate CLUSTER via a deterministic content hash, so a near-duplicate pair cannot straddle them |
| Duplicates within field/train | Collapsed to one copy |
| Overlap with PlantVillage | Dropped - it would be a leak straight into the lab test split |

**Training mix**: ~1.5k field images against ~44k lab ones. Concatenating them
would make field data ~3% of each batch, so the two streams are sampled at
`config.DOMAIN_MIX` (0.5) and validation is blended in the same proportion -
otherwise early stopping keeps selecting the lab-specialised checkpoint.

`prepare_dataset.py` maps PlantDoc's folder names onto PlantVillage's label space
via an explicit table (`PLANTDOC_TO_PV`). Only classes present in **both** label
spaces are used; unmapped folders are reported, not force-fitted onto a
plausible-looking wrong label.

### Required attribution

> Singh, D., Jain, N., Jain, P., Kayal, P., Kumawat, S., & Batra, N. (2020).
> *PlantDoc: A Dataset for Visual Plant Disease Detection.* Proceedings of the
> 7th ACM IKDD CoDS and 25th COMAD. https://doi.org/10.1145/3371158.3371196

---

## 3. Data quality handling

Both sets pass through the same cleaning in `prepare_dataset.py`:

| problem | handling |
|---|---|
| Corrupt / unreadable files | `PIL.verify()` then a re-open; failures dropped and counted |
| Images below 48 px | dropped — too small to show a lesion |
| Blank / flat images | dropped when 32×32 pixel std-dev < 4.0 |
| Duplicate images | difference-hash (dHash, 64-bit); only one copy of each visual duplicate is kept |
| **Train/test leakage** | splits are assigned **per duplicate cluster**, not per image, via a deterministic hash bucket — so the same physical leaf photographed twice can never land in both train and test |
| Under-populated classes | any class with fewer than `MIN_IMAGES_PER_CLASS` (150) usable images is **dropped**, not trained badly |
| Inconsistent label spaces | explicit `PLANTDOC_TO_PV` mapping; unmapped classes excluded and reported |

Leakage prevention matters here specifically because PlantVillage contains many
near-identical shots of the same leaf. Splitting per-image puts one shot in train
and another in test, which is a large part of why reported PlantVillage accuracy
is so high.

---

## 4. Not used, and why

| dataset | why not |
|---|---|
| Roboflow Universe agricultural sets | Licences vary per project — many are unspecified or non-commercial. Would need per-dataset review; none is currently needed. |
| Kaggle "New Plant Diseases Dataset" | An augmented re-release of PlantVillage. Adding it would inflate counts with synthetic variants of images we already have and defeat de-duplication. |
| Scraped web images | No licence, no verified labels. Excluded. |

---

## 5. If you need the missing Indian crops

To extend beyond PlantVillage's 14 crops, source and licence-check each addition,
then add its label folders under `dataset/raw/` in the same `Crop___Condition`
naming and add the crop to `config.CANDIDATE_CROPS`. Places worth checking:

- **Rice** — IRRI / Kaggle rice leaf disease sets (bacterial blight, blast, brown spot)
- **Wheat** — CGIAR / Global Wheat Head datasets (mostly detection, not disease)
- **Cotton** — Kaggle cotton disease sets
- **Sugarcane / Groundnut / Onion / Banana / Mango** — mostly small university
  collections; verify class balance against `MIN_IMAGES_PER_CLASS` before adding

Do not add a crop until it clears the 150-image-per-class floor. A crop the model
recognises unreliably is worse than one it abstains on, because the confidence
policy can only reject what it is uncertain about — it cannot rescue a class that
was trained on 30 images and is confidently wrong.

---

## 6. Adding a real "unknown / other" class

The shipped design handles unfamiliar input by **abstention**: if the predicted
distribution is not confident enough (`config.CONFIDENCE`), the UI says
*ANALYSIS UNCERTAIN* instead of guessing. This is preferred over a trained
`Other` class because a trained negative class only recognises the specific
negatives it was shown, and gives no protection against everything else.

If you later want an explicit class as well, it needs a genuine out-of-
distribution image set — non-plant objects, soil, sky, hands, machinery — added
as `Other___unknown`. Do **not** build it from crops you already train on, and
never label a person as a plant class.
