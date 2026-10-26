"""Freeze everything src/eval.py depends on before it opens test: write results/freeze.json with each file's sha256.

src/eval.py recomputes every hash and refuses to run on any mismatch. Refuses to re-freeze once results/.scored exists.

Run from the project root: python -m src.freeze
"""
import json
from datetime import datetime, timezone

import yaml

from src.data import MANIFEST, ROOT, SPLITS, sha256_of

FILES = ["eval_config.yaml", "src/data.py", "src/patchcore.py", "src/attack.py", "src/eval.py", "splits/manifest.json",
         "results/bank_fp32.npy", "results/bank_int8.npy", "results/fp32.onnx", "results/int8.onnx",
         "results/bench.json"]
FREEZE = ROOT / "results" / "freeze.json"


def main():
    if (ROOT / "results" / ".scored").exists():
        raise SystemExit("REFUSING TO FREEZE: results/.scored exists; re-freezing after test was scored would hide changes")
    missing = [f for f in FILES if not (ROOT / f).exists()]
    assert not missing, f"missing: {missing}"
    cfg = yaml.safe_load((ROOT / "eval_config.yaml").read_text(encoding="utf-8"))
    nulls = [f"{v}.{k}" for v, t in cfg["thresholds"].items() for k, val in t.items() if val is None]
    assert not nulls and cfg["primary_threshold"] in cfg["thresholds"]["fp32"], f"thresholds not calibrated: {nulls}"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    stale = [n for n, h in manifest.items() if sha256_of(SPLITS / n) != h]
    assert not stale, f"split CSVs no longer match splits/manifest.json: {stale}"

    frozen = {f: sha256_of(ROOT / f) for f in FILES}
    FREEZE.write_text(json.dumps({
        "frozen_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "rule": "src/eval.py recomputes every sha256 below (and the split CSVs against splits/manifest.json) "
                "and refuses to open test on any mismatch",
        "sha256": frozen,
    }, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"froze {len(frozen)} files (split CSVs verified against the manifest; thresholds non-null, primary "
          f"{cfg['primary_threshold']}, pgd restarts {cfg['pgd']['restarts']})")
    for f, h in frozen.items():
        print(f"  {f:<24}{h}")
    print(f"wrote {FREEZE.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
