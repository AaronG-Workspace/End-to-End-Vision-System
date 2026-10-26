"""Latency and size of the FP32 and INT8 PatchCore variants on this CPU (the declared edge stand-in).

ONNX Runtime CPU session (4 intra-op, 1 inter-op threads), batch 1, one val image. Each run times the extractor
(sess.run) and the nearest-neighbor search (nn_score, the same torch code for both variants, 4 threads) separately.
50 warmup runs per variant, then 1000 iterations that each time both variants, alternating which goes first, so
throttling hits both equally. Writes results/bench.json and results/bench.md. Never reads test.

Run from the project root: python -m src.bench --notes "plugged in, Zoom closed"
"""
import argparse
import json
import os
import platform
import time
import winreg
from datetime import date

import numpy as np
import onnxruntime as ort
import torch

from src.data import ROOT, load_images, load_split
from src.patchcore import BANK_FP32, MB, nn_score, ort_session
from src.quantize import BANK_INT8, FP32_ONNX, INT8_ONNX

WARMUP, RUNS, THREADS = 50, 1000, 4
VARIANTS = {"fp32": (FP32_ONNX, BANK_FP32), "int8": (INT8_ONNX, BANK_INT8)}
JSON_OUT, MD_OUT = ROOT / "results" / "bench.json", ROOT / "results" / "bench.md"


def cpu_name():
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()


def run_once(sess, bank, feed):
    """One batch-1 inference -> (extractor ns, nearest-neighbor ns)."""
    t0 = time.perf_counter_ns()
    feats = sess.run(None, feed)[0]
    t1 = time.perf_counter_ns()
    nn_score(torch.from_numpy(feats), bank)
    t2 = time.perf_counter_ns()
    return t1 - t0, t2 - t1


def main(notes):
    torch.set_num_threads(THREADS)
    row = load_split("val").iloc[:1]
    x = load_images(row).numpy()
    models = {}
    for v, (onnx_path, bank_path) in VARIANTS.items():
        sess = ort_session(onnx_path)
        models[v] = (sess, torch.from_numpy(np.load(bank_path)), {sess.get_inputs()[0].name: x})
    times = {v: [] for v in VARIANTS}
    with torch.inference_mode():
        for v in VARIANTS:
            for _ in range(WARMUP):
                run_once(*models[v])
        for i in range(RUNS):
            for v in (("fp32", "int8") if i % 2 == 0 else ("int8", "fp32")):
                times[v].append(run_once(*models[v]))

    rows = {}
    for v, (onnx_path, bank_path) in VARIANTS.items():
        ms = np.array(times[v]) / 1e6
        parts = {"extractor": ms[:, 0], "nn": ms[:, 1], "total": ms.sum(axis=1)}
        rows[v] = {k: dict(zip(("p50", "p95", "p99"), np.percentile(a, [50, 95, 99]).round(3).tolist()))
                   for k, a in parts.items()}
        onnx_mb, bank_mb = onnx_path.stat().st_size / MB, bank_path.stat().st_size / MB
        rows[v]["size_mb"] = {"onnx": round(onnx_mb, 2), "bank": round(bank_mb, 2), "total": round(onnx_mb + bank_mb, 2)}

    info = {
        "date": date.today().isoformat(),
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "runtime": f"ONNX Runtime {ort.__version__} CPUExecutionProvider, graph optimization "
                   f"{ort.SessionOptions().graph_optimization_level.name}",
        "intra_op_threads": THREADS,
        "inter_op_threads": 1,
        "torch_threads_nn": torch.get_num_threads(),
        "batch": 1,
        "image": row["path"].iloc[0],
        "warmup_per_variant": WARMUP,
        "timed_runs_per_variant": RUNS,
        "schedule": "interleaved, order alternated",
        "timer": "perf_counter_ns; extractor = sess.run, nn = nn_score (torch.cdist to bank, min, max), total = sum",
        "size": "MB = 2^20 bytes; ONNX file + bank file",
        "versions": {"onnxruntime": ort.__version__, "torch": torch.__version__, "numpy": np.__version__,
                     "python": platform.python_version()},
        "notes": notes,
        "variants": rows,
    }
    JSON_OUT.write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")

    fmt = lambda d: f"{d['p50']:.2f} / {d['p95']:.2f} / {d['p99']:.2f}"
    md = [
        "# Edge latency benchmark",
        "",
        f"{info['cpu']} ({info['logical_cores']} logical cores), {info['os']}. {info['runtime']}, "
        f"{THREADS} intra-op / 1 inter-op threads; NN search in torch {torch.__version__} on {THREADS} threads. "
        f"Batch 1, one val image, {WARMUP} warmup runs per variant, {RUNS} timed runs per variant, {info['schedule']}. "
        f"{info['date']}. Notes: {notes}.",
        "",
        "| variant | extractor p50 / p95 / p99 ms | NN search p50 / p95 / p99 ms | total p50 / p95 / p99 ms | size MB (ONNX + bank) |",
        "|---|---|---|---|---|",
    ]
    md += [f"| {v} | {fmt(r['extractor'])} | {fmt(r['nn'])} | {fmt(r['total'])} | "
           f"{r['size_mb']['onnx']:.2f} + {r['size_mb']['bank']:.2f} = {r['size_mb']['total']:.2f} |" for v, r in rows.items()]
    MD_OUT.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nwrote {JSON_OUT.relative_to(ROOT).as_posix()}, {MD_OUT.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notes", required=True, help='run conditions, e.g. "plugged in, Zoom closed"')
    main(parser.parse_args().notes)
