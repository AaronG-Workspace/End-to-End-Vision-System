# Failure report

## Threshold calibration: synthetic defects carry uncertainty into test
Thresholds come from val only: 21 normals plus 42 synthetic defects made from them (21 CutPaste, 21 dark blobs; `src/calibrate.py`, `results/synth_grid.png`). No real defect was used to set them.

- Val is perfectly separated for both variants (AUROC 1.000). For FP32, val normals score at most 1.279 and synthetic defects at least 1.979, so every threshold in that gap gives val F1 1.000. The F1-optimal threshold is the midpoint of the gap: 1.629 for FP32 and 1.665 for INT8 (gap 1.334 to 1.995). The alternative max-normal thresholds are 1.279 for FP32 and 1.334 for INT8.
- Synthetic defects may be easier or harder than real ones, and here they look easier. Most CutPastes leave sharp seams, some paste white background onto the bottle, and one blob sits on the background outside the part. Real `broken_small` and `contamination` defects are subtler. A real defect scoring between 1.279 and 1.629 would be missed at the primary threshold and caught at the alternative.
- The max-normal threshold has the opposite risk. It is set by the single highest of 21 val normals. If normal scores are exchangeable, each test normal has a 1/22 chance (4.5%) of scoring above it. With 20 good test images, that means about 0.9 expected false alarms, and each false alarm adds 5 points of FPR.
- Decision (Tom, before freeze): the primary threshold is `max_val_normal` for both variants (FP32 1.279, INT8 1.334). In medical-device QC a missed defect costs more than a false reject. The F1-optimal value is set by synthetic defects we now know are unrealistic, while `max_val_normal` comes from real good parts. F1-optimal is still reported as the alternative.
- The synthetic defects were left as generated. Their unrealistic placement (seams, background patches, a blob off the part) is a known caveat, not something fixed.
- The max-normal threshold has almost no margin. In the `src/attack.py` smoke test, val normal 1 scored 1.2774, 0.0018 below the FP32 primary threshold of 1.2792. Random-sign noise at eps 0.01 pushed it to 1.3416, a false alarm. That noise is ±2.5 gray levels per pixel, about the size of camera noise, and it is not an attack. Expect good parts that score near the threshold to flip under ordinary sensor noise.
- Both thresholds are frozen in `eval_config.yaml` before test is read. Where the two thresholds disagree on test, the result tells us how much the synthetic defects misled the calibration.

## Sanity check 4 revised after the val dry run, before freeze
- As first written, check 4 compared mean score shift toward the attacker's goal and required random-sign shift < 0.25 x FGSM shift. The `src/eval.py --dry-run` on val (21 normals + 42 synthetic defects) failed it narrowly at eps 0.02: random-sign 0.1734 against a limit of 0.1715 (0.25 x FGSM 0.6861).
- There were two causes. FGSM stops getting stronger with eps (0.653 at eps 0.01, 0.686 at eps 0.02). Random noise raises every score, so all 21 val normals became false alarms at eps 0.02, while two-sided AUROC stayed at 1.000.
- Revised rule (Tom, before freeze.py ran): at each eps > 0, the random-sign drop in two-sided AUROC must be at most 0.25 x the FGSM drop, or at most 0.01. Score shifts are still printed for information. Checks 3 and 5 still use the score shift, because PGD's AUROC bottoms out at 0.
- The false alarms random noise caused are a real finding and stay in the results through FPR in `results/attack.csv`.
