# Curation log: MVTec AD bottle

Produced by `python -m src.data` (seed 42). Splits are in `splits/`. `splits/manifest.json` pins the SHA-256 of each split CSV.

## Source and license
- Dataset: MVTec AD, category `bottle` (Bergmann et al., "MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019).
- Obtained from a third-party mirror (URL: TODO Tom). Stored read-only at `data/mvtec/bottle`, not committed to git.
- License: CC BY-NC-SA 4.0, so non-commercial use only. This project is coursework. A commercial MedVision deployment would need the manufacturer's own images under its own terms.

## Why bottle
The bottle is a rigid part photographed top-down from a fixed viewpoint against a fixed background. Of the MVTec categories, it is the closest stand-in for a molded medical-device component on an inspection line. Its defect types (large breaks, small chips on the rim, contamination inside) are the kinds of failures a molded part shows.

## Integrity checks (every run of data.py)
- Folder names and counts must match MVTec AD bottle exactly: train/good 209; test good 20, broken_large 20, broken_small 22, contamination 21.
- Every image must be 900x900, the size of the original release. This check is there because the data came from a mirror, and matching counts alone do not prove these are the original files. It cannot prove the files are byte-identical to MVTec's release, because we have no official per-file checksums. The SHA-256 column in each CSV pins exactly which files we used.
- Every image is hashed with SHA-256. A re-run re-hashes every file and checks it against the CSVs, checks each CSV against the manifest, and stops on any mismatch. A tampered `val.csv` was caught this way during testing.
- `--force` rebuilds byte-identical CSVs (verified: all four manifest hashes unchanged).

## Exact duplicates
Duplicates were checked by SHA-256 within each split and across splits.

| check | count |
|---|---|
| within fit / val / calib / test | 0 / 0 / 0 / 0 |
| fit x val, fit x test, val x test | 0, 0, 0 |
| calib not in fit, calib x val, calib x test | 0, 0, 0 |

## Splits
| split | source | images | good | defect | used for |
|---|---|---|---|---|---|
| fit | train/good minus val | 188 | 188 | 0 | memory bank (coreset) |
| val | 10% of train/good, seed 42 | 21 | 21 | 0 | thresholds (with synthetic defects made from val only), attack smoke tests |
| calib | 50 drawn from fit, seed 42 | 50 | 50 | 0 | INT8 activation-range calibration only |
| test | official test folder | 83 | 20 | 63 | scored once by `src/eval.py` |

Rationale:
- PatchCore learns from normal images only, so every training-side split is good images.
- val is held out from the bank. If threshold images were also in the bank, their nearest-neighbor distances would be near zero, and the threshold would come out too low.
- calib is a subset of fit on purpose. INT8 calibration only sets activation ranges from images the model was already built from. Thresholds come from val, which shares no image with fit or calib.
- test is used exactly as MVTec released it, so results are comparable to the published benchmark. Only `src/eval.py` may load it: `load_split("test")` raises PermissionError otherwise. data.py reads test only to check image sizes from the file headers and to hash the bytes.

## Class balance
- Training side: 209 good, 0 defect. This is by design for one-class anomaly detection.
- Test: 20 good vs 63 defect, so 76% of the test set is defective, the opposite of a real production line. Read precision and F1 against that base rate. A detector that flags every image scores precision 63/83 = 0.759, recall 1.000 and F1 = 0.863 while catching nothing useful.
- Counts are small, so every rate is reported with its counts. With 20 good images, one false alarm moves FPR by 5 points. Per defect type, one miss moves recall by 4.5 to 5 points.

## Preprocessing
- Images are resized from 900x900 to 224x224 with PIL bilinear filtering, which smooths while shrinking so thin cracks fade instead of breaking up into noise.
- No center crop. The paper's resize-then-crop would cut the image border, where rim defects sit.
- Images are kept in RGB, because contamination is partly a color and darkness cue.
- Pixel values are in [0, 1]. ImageNet mean/std normalization happens inside the model's forward with the fixed ImageNet constants, never computed from any split. This keeps attack epsilon on the 0 to 1 pixel scale.
- Every script loads pixels through the one `load_images` function in `src/data.py`, so the PyTorch, FP32 ONNX and INT8 ONNX paths see identical inputs.

## What was dropped
Nothing. All 292 images are used: 209 train/good and 83 test. Pixel-level ground-truth masks are not used to fit, calibrate or score, and all metrics are image-level.

## Architecture
**Why PatchCore (Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022) over a reconstruction autoencoder:**
- No defect labels are needed. Both approaches share this advantage over a supervised classifier, which would need many labeled defects we do not have.
- It works from 188 normal images. An autoencoder trained on 188 images risks overfitting, and it often reconstructs defects well enough to hide them.
- There is no training loop. The backbone stays frozen and ImageNet-pretrained, and the memory bank is a coreset of fit-set patch features. Nothing is tuned, so no hyperparameter search can drift toward the test set.
- Each score comes with nearest-neighbor evidence per patch. Every one of the 28x28 patches has a distance to its closest normal patch, so a flagged image comes with a map of where it differs and which normal patch it was compared against.

**Why resnet18:**
- CPU edge budget. This laptop's CPU (Intel Core Ultra 7 155U) is the declared edge target. resnet18 has 11.7M parameters, wide_resnet50_2 has 68.9M. Its layer2 + layer3 features are 384 dims, against 1,536 for wide_resnet50_2, which shrinks the memory bank and the nearest-neighbor cost by about 4x.
- Attack runtime. Every PGD step is a forward and backward pass through the backbone plus a nearest-neighbor search, repeated over 83 test images, several eps values and restarts. A smaller backbone keeps the full attack grid within minutes on CPU.
- wide_resnet50_2 is the paper's default and a one-line swap (`backbone` in `eval_config.yaml`).
