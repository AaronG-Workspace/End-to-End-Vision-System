# CST 620 Week 7: PatchCore QC system

Work in progress. Run order, commands and expected outputs are added as each script lands.

## Run order (PowerShell, project root, seed 42)
```powershell
.\.venv\Scripts\Activate.ps1
python -m src.data          # splits/*.csv, splits/manifest.json (re-runs verify, --force re-splits identically)
python -m src.patchcore     # results/bank_fp32.npy, smoke test on val
python -m src.quantize      # results/fp32.onnx, int8.onnx, bank_int8.npy, quant.json (INT8 checks, val only)
python -m src.calibrate     # thresholds into eval_config.yaml, results/calib.json, synth_grid.png (val only)
python -m src.bench --notes "plugged in, Zoom closed"   # results/bench.json, bench.md (~4 min, one val image)
python -m src.attack        # attack smoke test and timing on val; writes pgd.restarts to eval_config.yaml
python -m src.eval --dry-run   # eval's exact code path on val + synthetic defects -> results/dryrun/ (~5 min)
python -m src.freeze        # results/freeze.json: sha256 of the 11 files eval.py depends on
python -m src.eval          # test, once (~12 min); refuses on any freeze mismatch or if results/.scored exists
jupyter nbconvert --to notebook --execute --inplace report.ipynb   # report from results/ and docs/ only
```

## Not in git
- `results/*.npy` (memory banks) and `results/*.onnx` (exported models) are not committed. `results/freeze.json` records their SHA-256 values. The bank rebuild is deterministic: two runs of `python -m src.patchcore` produced the same `bank_fp32.npy` (sha256 `b75ff119426420dc68a30df66f5af6d629d591905de3eb202f98c15f6a47503c`). Two runs of `python -m src.quantize` also produced identical `fp32.onnx`, `int8.onnx` and `bank_int8.npy`. Anyone can regenerate these files with the commands above and check them against `results/freeze.json`. Use the same library versions: torch 2.14.1, torchvision 0.29.1, onnxruntime 1.31.0.
- `data/` is not committed. MVTec AD is licensed CC BY-NC-SA 4.0. Place the bottle category at `data/mvtec/bottle`.
