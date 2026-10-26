# Failure report: PatchCore bottle QC

Results come from the single test run on 2026-10-26 (`results/`). Thresholds were frozen on val before the run, and all attacks are on the FP32 PyTorch model. Every rate is shown with its counts. Perturbation size eps is on the 0 to 1 pixel scale and also in gray levels out of 255.

## 1. Executive summary
On clean test images the system ranks every defect above every good part (AUROC 1.000 for FP32 and INT8). At the primary threshold it catches all 63 defects but rejects 4 of 20 good parts (FP32; INT8 rejects 3 of 20).

The main failure mode is false rejects. A white-box perturbation of 1.3 gray levels (eps 0.005, FGSM or PGD) turns every good part into a reject. Ordinary random noise of ±3.8 gray levels (eps 0.015), which needs no attacker at all, does the same.

No successful escape (a defect pushed below the threshold) was found: up to eps 0.02 with PGD (10 steps, 1 restart), and up to eps 0.05 with FGSM. However, the closest attacked defect ended only 0.031 above the threshold, and the escape search gets weaker as eps grows (section 4.4).

Severity: false rejects cost scrap and throughput, not patient safety. Still, a threshold that flips under sensor-sized noise is not fit for an unattended line.

Recommended action: do not deploy unattended. Run a supervised pilot with human review of every reject, recalibrate the threshold on real good parts from the target line, and pass a natural-shift suite and a stronger escape search before release (section 5).

## 2. Quantitative findings

### 2.1 Clean test (83 images: 20 good, 63 defect)
| variant | threshold | value | AUROC | AP | recall | FPR | precision | F1 |
|---|---|---|---|---|---|---|---|---|
| fp32 | max_val_normal (primary) | 1.2792 | 1.0000 | 1.0000 | 1.000 (63/63) | 0.200 (4/20) | 0.940 (63/67) | 0.969 |
| fp32 | f1_optimal (alternative) | 1.6292 | 1.0000 | 1.0000 | 1.000 (63/63) | 0.000 (0/20) | 1.000 (63/63) | 1.000 |
| int8 | max_val_normal (primary) | 1.3339 | 1.0000 | 1.0000 | 1.000 (63/63) | 0.150 (3/20) | 0.955 (63/66) | 0.977 |
| int8 | f1_optimal (alternative) | 1.6647 | 1.0000 | 1.0000 | 1.000 (63/63) | 0.000 (0/20) | 1.000 (63/63) | 1.000 |

- Recall by defect type was 1.000 for every variant and threshold: broken_large 20/20, broken_small 22/22, contamination 21/21.
- FP32 test score ranges:
  - Good parts: 1.020 to 1.365 (mean 1.166). Val goods for comparison: 0.974 to 1.279 (mean 1.151).
  - Lowest defect by type: contamination 1.765, broken_small 2.148, broken_large 2.710.
- The false alarms at the FP32 primary threshold are `test/good/001` (1.345), `002` (1.365), `009` (1.287) and `015` (1.310). INT8 flags the same parts except `009`.
- Exact 95% intervals (Clopper-Pearson, from the counts):

  | rate | count | interval |
  |---|---|---|
  | recall, every variant | 63/63 | [0.943, 1.000] |
  | FP32 FPR | 4/20 | [0.057, 0.437] |
  | INT8 FPR | 3/20 | [0.032, 0.379] |

- The stratified bootstrap intervals in `results/clean.csv` collapse to [1.000, 1.000] for AUROC and recall. Perfect separation makes every resample perfect, so those intervals carry no information. Use the exact intervals above.

### 2.2 Attacks on FP32 at the primary threshold 1.2792
The escape half compares attacked defects with clean goods. The false-alarm half compares clean defects with attacked goods. At eps 0, 4 good parts are already false alarms, so "new false alarms" counts the attack-caused flips among the other 16. PGD is 10 steps, alpha 0.25 x eps, 1 restart, random start.

| attack | eps | gray levels | two-sided AUROC | escape AUROC | false-alarm AUROC | recall (escapes) | FPR (false alarms) | new escapes | new false alarms |
|---|---|---|---|---|---|---|---|---|---|
| FGSM | 0 | 0.0 | 1.0000 | 1.0000 | 1.0000 | 1.000 (63/63, FN 0) | 0.200 (4/20) | 0 | 0 |
| FGSM | 0.005 | 1.3 | 0.9500 | 1.0000 | 0.9857 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| FGSM | 0.01 | 2.6 | 0.8905 | 1.0000 | 0.9389 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| FGSM | 0.015 | 3.8 | 0.8690 | 1.0000 | 0.9135 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| FGSM | 0.02 | 5.1 | 0.8659 | 1.0000 | 0.8857 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| FGSM | 0.03 | 7.6 | 0.8206 | 1.0000 | 0.8206 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| FGSM | 0.05 | 12.8 | 0.7095 | 1.0000 | 0.6024 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| PGD | 0.005 | 1.3 | 0.1905 | 0.9960 | 0.4786 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| PGD | 0.01 | 2.6 | 0.0000 | 1.0000 | 0.0111 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| PGD | 0.02 | 5.1 | 0.0000 | 1.0000 | 0.0000 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| random-sign noise | 0 | 0.0 | 1.0000 | 1.0000 | 1.0000 | 1.000 (63/63, FN 0) | 0.200 (4/20) | 0 | 0 |
| random-sign noise | 0.005 | 1.3 | 1.0000 | 1.0000 | 1.0000 | 1.000 (63/63, FN 0) | 0.200 (4/20) | 0 | 0 |
| random-sign noise | 0.01 | 2.6 | 1.0000 | 1.0000 | 1.0000 | 1.000 (63/63, FN 0) | 0.300 (6/20) | 0 | 2 |
| random-sign noise | 0.015 | 3.8 | 0.9992 | 1.0000 | 1.0000 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| random-sign noise | 0.02 | 5.1 | 0.9968 | 1.0000 | 0.9984 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| random-sign noise | 0.03 | 7.6 | 0.9960 | 1.0000 | 0.9786 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |
| random-sign noise | 0.05 | 12.8 | 0.9730 | 1.0000 | 0.9190 | 1.000 (63/63, FN 0) | 1.000 (20/20) | 0 | 16 |

- Two-sided AUROC first falls below the frozen minimum of 0.85 at eps 0.03 with FGSM and at eps 0.005 with PGD. Random-sign noise stays at or above 0.85 up to eps 0.05.
- No successful escape was found up to eps 0.05 with FGSM, up to eps 0.02 with PGD (10 steps, 1 restart), or up to eps 0.05 with random-sign noise.
- The INT8 variant was not attacked.

### 2.3 Sanity checks (`results/sanity.txt`)
| # | check | result | values |
|---|---|---|---|
| 1 | eps 0 reproduces clean scores exactly | PASS | FGSM and random-sign scores are bit-identical to clean |
| 2 | max \|x_adv - x\| <= eps, pixels in [0, 1] | PASS | largest excess +2.51e-08 (float32 rounding); pixels in [0.0598, 1.0000] |
| 3 | damage grows with eps | **FAIL** | FGSM 0.433, 0.506, 0.507, 0.488, 0.452, 0.381 at eps 0.005 to 0.05; PGD 1.029, 1.604, 2.122 (grows) |
| 4 | random-sign does much less damage than FGSM | PASS | AUROC drop 0.000 to 0.027 for random-sign, against 0.050 to 0.291 for FGSM |
| 5 | PGD damage >= FGSM damage | PASS | 1.029 vs 0.433, 1.604 vs 0.506, 2.122 vs 0.488 |

The cutoff for check 4 was revised after the val dry run and before freeze:
- As first written, check 4 compared mean score shift toward the attacker's goal and required random-sign shift < 0.25 x FGSM shift. The `src/eval.py --dry-run` on val (21 normals + 42 synthetic defects) failed it narrowly at eps 0.02: random-sign 0.1734 against a limit of 0.1715 (0.25 x FGSM 0.6861).
- There were two causes. FGSM stops getting stronger with eps (0.653 at eps 0.01, 0.686 at eps 0.02). Random noise raises every score, so all 21 val normals became false alarms at eps 0.02, while two-sided AUROC stayed at 1.000.
- Revised rule (Tom, before freeze.py ran): at each eps > 0, the random-sign drop in two-sided AUROC must be at most 0.25 x the FGSM drop, or at most 0.01. Score shifts are still printed for information. Checks 3 and 5 still use the score shift, because PGD's AUROC bottoms out at 0.
- The false alarms random noise caused are a real finding and stay in the results through FPR in `results/attack.csv`.

## 3. Failure mode class
- **Primary: adversarial fragility on the false-alarm side.** A white-box L-infinity perturbation of 1.3 gray levels makes every good part a reject, with both FGSM and PGD.
- **Secondary: a calibration failure under sensor-sized noise and a small distribution shift.** No attacker is needed. On clean test, 4 of 20 good parts already score above a threshold set from 21 val goods, and random noise of ±3.8 gray levels rejects all 20.
- **Not observed: escapes.** No defect was pushed below the threshold by any attack run. The search was limited, though (1 restart, 10 steps), and it gets weaker at larger eps (section 4.4), so this is an absence of evidence, not a guarantee.

## 4. Root cause with evidence

### 4.1 Max-over-patches scoring makes false alarms easy and escapes hard
The image score is the largest of 784 patch distances to the memory bank. To raise the score, an attacker has to push only one patch away from the bank. To lower it, every patch has to go under the threshold, and each time the top patch drops, the next one takes over. The evidence:
- FGSM and PGD at eps 0.005 flip all 16 good parts that were not already false alarms.
- Escape AUROC stays at or above 0.996 for every attack.
- The anomaly map of `test/contamination/003` shows how PGD tries to escape: at eps 0.01 it spreads the hot spot across the whole part instead of removing it (1.765 down to 1.391), so the maximum stays above the threshold (`results/misses/pgd001_low1_contamination_003.png`).

### 4.2 The threshold has almost no margin, and it came from synthetic defects
Thresholds come from val only: 21 normals plus 42 synthetic defects made from them (21 CutPaste, 21 dark blobs; `src/calibrate.py`, `results/synth_grid.png`). No real defect was used to set them.
- Val is perfectly separated for both variants (AUROC 1.000). For FP32, val normals score at most 1.279 and synthetic defects at least 1.979, so every threshold in that gap gives val F1 1.000. The F1-optimal threshold is the midpoint of the gap: 1.629 for FP32 and 1.665 for INT8 (gap 1.334 to 1.995). The alternative max-normal thresholds are 1.279 for FP32 and 1.334 for INT8.
- Synthetic defects may be easier or harder than real ones, and here they look easier. Most CutPastes leave sharp seams, some paste white background onto the bottle, and one blob sits on the background outside the part. Real `broken_small` and `contamination` defects are subtler. A real defect scoring between 1.279 and 1.629 would be missed at the alternative threshold and caught at the primary.
- The max-normal threshold has the opposite risk. It is set by the single highest of 21 val normals. If normal scores are exchangeable, each test normal has a 1/22 chance (4.5%) of scoring above it. With 20 good test images, that means about 0.9 expected false alarms, and each false alarm adds 5 points of FPR.
- Decision (Tom, before freeze): the primary threshold is `max_val_normal` for both variants (FP32 1.279, INT8 1.334). In medical-device QC a missed defect costs more than a false reject. The F1-optimal value is set by synthetic defects we now know are unrealistic, while `max_val_normal` comes from real good parts. F1-optimal is still reported as the alternative.
- The synthetic defects were left as generated. Their unrealistic placement (seams, background patches, a blob off the part) is a known caveat, not something fixed.
- The max-normal threshold has almost no margin. In the `src/attack.py` smoke test, val normal 1 scored 1.2774, 0.0018 below the FP32 primary threshold of 1.2792. Random-sign noise at eps 0.01 pushed it to 1.3416, a false alarm. That noise is ±2.5 gray levels per pixel, about the size of camera noise, and it is not an attack. On test, the same noise flipped 2 good parts, and noise at eps 0.015 (±3.8 gray levels) flipped all of them.
- Both thresholds were frozen in `eval_config.yaml` before test was read.

### 4.3 Test good parts score higher than val good parts
4 of 20 test goods score above the highest of 21 val goods. If val and test goods came from one distribution, the chance of 4 or more would be C(20,4)/C(41,4) = 0.048. That suggests MVTec's test good images differ slightly from train/good. With these counts it is a suggestion, not proof. The four false-alarm anomaly maps (`results/misses/fp_good_*.png`) show no compact hot spot: their warmest patches are mild and sit on the inner rim. So these parts are flagged for overall appearance, not for anything that looks like a defect.

### 4.4 The escape search weakens as eps grows (post hoc)
This analysis was computed from `results/attack_scores.csv` after the run. It changes no frozen setting.

| attack | eps | mean defect score drop | lowest attacked defect (margin to 1.2792) |
|---|---|---|---|
| FGSM | 0.005 | 0.393 | 1.478 (+0.199) |
| FGSM | 0.01 | 0.382 | 1.841 (+0.562) |
| FGSM | 0.02 | 0.253 | 2.151 (+0.872) |
| FGSM | 0.05 | -0.084 | 2.785 (+1.506) |
| PGD | 0.005 | 0.698 | 1.310 (+0.031) |
| PGD | 0.01 | 0.836 | 1.391 (+0.112) |
| PGD | 0.02 | 0.786 | 1.702 (+0.423) |

Larger sign-gradient perturbations add their own anomaly, and at eps 0.05 FGSM raises defect scores on average. So "no successful escape found up to eps X" shows the limits of this attack, not that defects are safe at that eps.

### 4.5 Why sanity check 3 failed
Check 3 failed because FGSM damage peaks at eps 0.015 (0.507) and falls to 0.381 at eps 0.05. A single full-size gradient step overshoots once eps is large enough that the linear approximation no longer holds. PGD, which takes small projected steps, grows as expected (1.029, then 1.604, then 2.122).

*Post hoc note:* two-sided AUROC falls monotonically under FGSM (1.000, 0.950, 0.891, 0.869, 0.866, 0.821, 0.710). It never rises under PGD (0.191, 0.000, 0.000) and reaches 0.0. That gives no sign of gradient masking on the two-sided metric. The escape side alone does weaken with eps (4.4), so the masking question stays open for escapes specifically.

### 4.6 Threshold comparison under attack (post hoc)
On clean test, the alternative threshold would have done better: 0/20 false alarms against 4/20. Under attack the trade reverses. At f1_optimal (FP32 1.6292), these would have escaped out of 63 defects:
- PGD: 4 at eps 0.005 and 4 at eps 0.01
- FGSM: 2 at eps 0.005
- random-sign noise: 1 at eps 0.015 and 1 at eps 0.02

At the primary threshold, none escaped. The primary threshold traded 4 clean false alarms for zero escapes under every attack tried. This is reported for the record, and no frozen setting was changed because of it.

## 5. Mitigations
Owners are named by role on the MedVision team.

1. **Recalibrate the threshold on real good parts from the target line.**
   - Owner: Tom (CV lead), with line QA.
   - Due: 2026-12-04.
   - Success: at least 500 good parts from at least 3 production lots, kept out of the bank fit, set the threshold. On a separate set of at least 300 good parts, the FPR 95% upper bound is at most 2%, and the recall lower bound on all labeled defects is at least 0.95.
2. **Build a natural-shift suite. This closes the EU AI Act Art 15(4) gap: resilience to errors and inconsistencies in the operating environment.**
   - Owner: Tom (CV lead).
   - Due: 2026-11-20.
   - Covers: sensor noise up to 4 gray levels, brightness ±10%, defocus blur, JPEG compression, 1-2 px shifts, small rotations.
   - Success: at every level, FPR rises by at most 2 points and recall drops by at most 0.02 from clean. Today random noise of ±3.8 gray levels raises FPR from 4/20 to 20/20, so the current system fails.
3. **Control access to the camera feed and image path.** White-box adversarial examples need write access to the pixels (Art 15(5), adversarial examples).
   - Owner: Manufacturing IT / OT security.
   - Due: 2026-11-30.
   - Success: every frame is authenticated (MAC or signature) from camera to inference and verified before scoring, with 100% of verifications logged. A penetration test finds 0 ways to inject or alter frames.
4. **Human review of every reject during the pilot.**
   - Owner: QA lead.
   - Due: from pilot start, 2026-11-16, until the pilot exit criteria are met.
   - Success: 100% of rejects reviewed within one shift, each decision logged. The pilot exits only when the reviewed false-reject rate is at most 2% over at least 2,000 parts and an audit of passed parts finds 0 escapes.
5. **Run a stronger escape search.**
   - Owner: Tom (CV lead).
   - Due: 2026-11-13.
   - Scope: PGD with at least 50 steps, at least 5 restarts and a smaller step (alpha eps/10), plus a top-k patch objective that lowers the whole top of the map rather than only its maximum. Eps grid up to 0.05 (12.8 gray levels).
   - Success: the minimum escape margin is reported for every eps, and there are 0 escapes at the deployed threshold up to eps 0.02 (5.1 gray levels). If an escape is found, the smallest escaping eps is documented and fed into mitigation 3.
6. **Cut nearest-neighbor latency.**
   - Owner: Tom (CV lead).
   - Due: 2026-12-11.
   - Approach: a 1% coreset (1,474 patches) or an approximate nearest-neighbor index.
   - Success: NN search p95 at most 20 ms on the Core Ultra 7 155U benchmark (now 91 to 98 ms). Image AUROC drops by at most 0.005, and recall and FPR at the threshold are unchanged on the recalibration set.
7. **Replace MVTec with the manufacturer's own licensed images.**
   - Owner: Regulatory affairs and procurement, with the manufacturer.
   - Due: 2027-01-15.
   - Success: a signed data agreement permits commercial use, at least 50 labeled images per defect type come from the target line, and every result in this report is re-run on that data. MVTec AD (CC BY-NC-SA 4.0, non-commercial) is removed from the training and evaluation pipeline.
