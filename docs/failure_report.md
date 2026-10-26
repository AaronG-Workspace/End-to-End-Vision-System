# Failure report

## Threshold calibration: synthetic defects carry uncertainty into test
Thresholds come from val only: 21 normals plus 42 synthetic defects made from them (21 CutPaste, 21 dark blobs; `src/calibrate.py`, `results/synth_grid.png`). No real defect was used to set them.

- Val is perfectly separated for both variants (AUROC 1.000). For FP32, val normals score at most 1.279 and synthetic defects at least 1.979, so every threshold in that gap gives val F1 1.000. The F1-optimal threshold is the midpoint of the gap: 1.629 for FP32 and 1.665 for INT8 (gap 1.334 to 1.995). The alternative max-normal thresholds are 1.279 for FP32 and 1.334 for INT8.
- Synthetic defects may be easier or harder than real ones, and here they look easier. Most CutPastes leave sharp seams, some paste white background onto the bottle, and one blob sits on the background outside the part. Real `broken_small` and `contamination` defects are subtler. A real defect scoring between 1.279 and 1.629 would be missed at the primary threshold and caught at the alternative.
- The max-normal threshold has the opposite risk. It is set by the single highest of 21 val normals. If normal scores are exchangeable, each test normal has a 1/22 chance (4.5%) of scoring above it. With 20 good test images, that means about 0.9 expected false alarms, and each false alarm adds 5 points of FPR.
- Both thresholds are frozen in `eval_config.yaml` before test is read. The primary is F1-optimal. Where the two thresholds disagree on test, the result tells us how much the synthetic defects misled the calibration.
