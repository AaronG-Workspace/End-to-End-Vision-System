# CST 620 Week 7 Capstone: PatchCore QC system with adversarial evaluation, INT8 edge deployment, governance memo

Read ASSIGNMENT.md first. Where it conflicts with this file, ASSIGNMENT.md wins.
Due Mon Oct 26 2026 11:59 PM Central. Build the smallest thing that covers every rubric line. No extras.

## Environment
- Windows 11, PowerShell, VS Code, CPU only (Intel Core Ultra 7 155U, 16 GB RAM). This laptop is the declared edge-hardware stand-in.
- venv at .venv made with uv. Install with `uv pip install`, never plain pip.
- Console output ASCII only. No Greek letters or special symbols in print() (Week 2's eval.py crashed printing sigma on this console). Write eps, sigma, alpha.
- Every script finishes in minutes. Seed 42 everywhere (torch.manual_seed, numpy default_rng(42), random.seed).
- reuse/week3 and reuse/week6 hold code from earlier weeks. Read them for patterns (split guard, manifest hashing, ONNX Runtime INT8, latency harness, freeze hashes, scored-once guard). Never import from reuse/. Copy and adapt into src/.
- git init in the project root. Commit after data.py, after calibrate.py, and after freeze.py (before eval.py runs).

## Data
- MVTec AD bottle at data/mvtec/bottle (read-only). Expected counts: train/good 209; test good 20, broken_large 20, broken_small 22, contamination 21; ground_truth masks for defects.
- License is CC BY-NC-SA 4.0 (non-commercial). Record it in docs/curation_log.md.
- Preprocessing: resize to 224x224 (no center crop, so rim defects near the edge are kept), RGB, pixel values in [0, 1].
- ImageNet mean/std normalization happens INSIDE the model's forward, so attacks and epsilon live on the 0 to 1 pixel scale. Normalization constants are the fixed ImageNet values, never computed from any split.

## Splits (src/data.py, run once)
- val = 10% of train/good (21 images, seed 42). fit = remaining 188. calib = 50 images sampled from fit (seed 42) for INT8 calibration. test = the official test folder.
- Write splits/fit.csv, val.csv, calib.csv, test.csv. test.csv holds path, label, defect_type, sha256 only.
- SHA-256 every image. Report exact duplicates within and across splits (expected 0, report the number either way).
- Write splits/manifest.json with the SHA-256 of every CSV.
- Re-running data.py must not re-split. Load existing CSVs unless --force is passed.
- Print counts per split and per class.
- Write docs/curation_log.md: source and license, class balance (test is 20 good vs 63 defect, so precision and F1 read against that base rate), split rationale, preprocessing choices and why, what was dropped (expected nothing), and why bottle (a rigid part with a fixed top-down view, closest MVTec stand-in for a molded medical device component).

## Test set lock
- ONLY src/eval.py opens test images. data.py may read test paths and bytes to hash them, nothing else.
- load_split("test") raises PermissionError unless allow_test=True. Only eval.py passes it.
- No statistic, threshold, attack setting, coreset choice, or quantization setting comes from test images.
- eval.py runs once. On success it writes results/.scored and refuses to run again without --force. A forced run appends a dated line to docs/leakage_audit.md.
- src/freeze.py runs before eval.py. It writes results/freeze.json with the SHA-256 of eval_config.yaml, src/eval.py, src/patchcore.py, src/attack.py, both memory banks, and both ONNX files. eval.py checks freeze.json first and aborts on any mismatch.

## Model (src/patchcore.py, hand-written, no anomalib)
- PatchCore (Roth et al., 2022). torchvision resnet18 ImageNet weights, frozen, eval mode.
- Features from layer2 and layer3, 3x3 average pooling (stride 1, padding 1), layer3 upsampled bilinearly to layer2 size, concatenated: 384 dims on a 28x28 patch grid.
- Memory bank = greedy coreset, 10% of fit-set patches (random projection to 128 dims for selection only, seed 42). Save results/bank_fp32.npy.
- Score: each patch's distance to its nearest bank vector (torch.cdist). Image score = max over patches. Keep the 28x28 map for error-analysis figures.
- score(x) must be differentiable with respect to the input image: no torch.no_grad, no .detach(), no numpy inside it. torch.inference_mode is allowed only in fit and benchmark code.
- Backbone name lives in eval_config.yaml so wide_resnet50_2 (the paper default) can be swapped in with one line.

## Threshold (src/calibrate.py, val only)
- Build synthetic defects from val normals only: CutPaste-style patch from another val normal, plus small dark contamination blobs. Never from test.
- Score val normals and synthetic defects. Threshold = F1-optimal on val. Also record the max val normal score as the alternative estimator.
- Write into eval_config.yaml: backbone, coreset ratio, both thresholds per variant (FP32 and INT8), minimum acceptable AUROC 0.85, epsilon grid, PGD settings, variants. Then freeze.

## Metrics (eval.py)
- Image-level AUROC, average precision, precision-recall curve saved as PNG.
- At the frozen threshold: precision, recall, F1, FPR, confusion counts, recall by defect type.
- Counts next to every rate. 20 good test images means one false alarm moves FPR by 5 points.

## Adversarial (src/attack.py, settings smoke-tested on val only)
- Inputs in [0, 1]. Two-sided: defect images step to LOWER the score (escape), good images step to RAISE it (false alarm). Report both halves separately.
- FGSM, white box, L-infinity, epsilon grid 0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05 on the 0 to 1 pixel scale.
- PGD: 10 steps, alpha = eps/4, random start, 1 restart (3 if time), at eps 0.005, 0.01, 0.02.
- Five sanity checks printed by eval.py to results/sanity.txt:
  1. eps 0 reproduces clean scores exactly
  2. max |x_adv - x| <= eps and pixels stay in [0, 1]
  3. damage grows with eps
  4. random-sign noise at the same eps does much less damage
  5. PGD does at least as much damage as FGSM at the same eps
- Name the attack next to every number. Never write "robust" or "defended". Write "no successful attack found up to eps X with PGD, k steps, r restarts".

## Quantization and edge benchmark (src/quantize.py, src/bench.py)
- Do NOT use torch.quantization or torch.ao.quantization (deprecated).
- Export the backbone feature extractor (input to concatenated 384-dim patch features) to ONNX. Quantize with ONNX Runtime quantize_static: QDQ format, per-channel weights, calibration reader over calib.csv. Week 6's working version is in reuse/week6.
- Rebuild the memory bank from INT8 backbone features (results/bank_int8.npy). Derive the INT8 threshold on val the same way as FP32, before freezing.
- Verify INT8 really runs: count quantized nodes, compare output shapes and mean feature difference to FP32 on calib images.
- Latency on this CPU through ONNX Runtime, batch 1, 4 intra-op threads, 50 warmup, 1000 timed runs. Report p50, p95, p99 for backbone plus nearest-neighbor scoring. Size in MB = ONNX file plus bank file.
- Tradeoff table (results/tradeoff.csv and .md): variant, AUROC, AP, F1 and recall at its frozen threshold, p50 ms, p95 ms, size MB.

## Outputs
- results/: clean.csv, attack.csv, tradeoff.csv, tradeoff.md, sanity.txt, pr_curve.png, degradation_curve.png, misses/ (thumbnails and anomaly maps of missed defects and false alarms).
- report.ipynb reads only results/ (never test images) and is saved run top to bottom with outputs.
- docs/: curation_log.md, leakage_audit.md (Tom will paste the 11-item checklist), failure_report.md, challenges.md.
- README.md: run order, commands, expected outputs, hardware, seed.

## Working with Tom
- Propose before writing. One script at a time. Show the plan and wait for approval.
- After each script runs, print the counts or numbers Tom must check.
- When Tom corrects you, append one line to docs/challenges.md: what you proposed, what he caught, what changed.
- Commands are for PowerShell. Code blocks flush left, no leading spaces.
