"""INT8 post-training static quantization of the PatchCore feature extractor with ONNX Runtime (no torch.quantization).

Exports the extractor (normalization inside, input in [0, 1]) to results/fp32.onnx, quantizes it with quantize_static
(QDQ, per-channel, U8S8) calibrated on calib.csv only, keeps MinMax or Percentile by feature difference to FP32 on
val, verifies the INT8 graph, and rebuilds the memory bank from INT8 fit features. Never reads test.

Run from the project root: python -m src.quantize
"""
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.data import ROOT, SEED, load_images, load_split, sha256_of
from src.patchcore import BANK_FP32, MB, FeatureExtractor, fit_bank, nn_score, ort_features, ort_session

RESULTS = ROOT / "results"
FP32_ONNX, INT8_ONNX = RESULTS / "fp32.onnx", RESULTS / "int8.onnx"
BANK_INT8 = RESULTS / "bank_int8.npy"
LOG = RESULTS / "quant.json"
OPSET = 17
METHODS = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}
# Week 6 settings that worked on this CPU: U8S8 QDQ (uint8 activations, int8 weights), symmetric weights per channel.
QUANT_ARGS = {"quant_format": QuantFormat.QDQ, "per_channel": True, "activation_type": QuantType.QUInt8,
              "weight_type": QuantType.QInt8, "reduce_range": False}
EXTRA_OPTIONS = {"ActivationSymmetric": False, "WeightSymmetric": True}


class CalibReader(CalibrationDataReader):
    """The 50 calib.csv images, one per get_next(); reads nothing else."""

    def __init__(self, images):
        self.items = iter([{"image": images[i:i + 1].numpy()} for i in range(len(images))])

    def get_next(self):
        return next(self.items, None)


def compare(feats, ref):
    """INT8 vs FP32 features: mean |diff| over all values, and cosine similarity per 384-dim patch vector."""
    cos = F.cosine_similarity(feats, ref, dim=1)
    return {"mean_abs_diff": (feats - ref).abs().mean().item(), "cos_mean": cos.mean().item(), "cos_min": cos.min().item()}


def graph_ops(path):
    """Op counts in the file, Conv nodes fed a dequantized int8 weight, the output's producer, and the runtime graph."""
    model = onnx.load(path)
    file_ops = Counter(n.op_type for n in model.graph.node)
    dq_out = {o for n in model.graph.node if n.op_type == "DequantizeLinear" for o in n.output}
    q_convs = sum(n.op_type == "Conv" and n.input[1] in dq_out for n in model.graph.node)
    producer = next(n.op_type for n in model.graph.node if model.graph.output[0].name in n.output)
    with tempfile.TemporaryDirectory() as tmp:
        opts = ort.SessionOptions()
        opts.optimized_model_filepath = str(Path(tmp) / "optimized.onnx")
        opts.log_severity_level = 3   # the saved graph is only counted, so ORT's "hardware specific" warning is noise
        ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        runtime = Counter(n.op_type for n in onnx.load(Path(tmp) / "optimized.onnx").graph.node)
    return file_ops, q_convs, producer, runtime


def main():
    torch.manual_seed(SEED)
    extractor = FeatureExtractor()
    val = load_images(load_split("val"))
    RESULTS.mkdir(exist_ok=True)

    torch.onnx.export(extractor, (val[:1],), FP32_ONNX, input_names=["image"], output_names=["features"],
                      opset_version=OPSET, dynamo=False)
    ref = ort_features(ort_session(FP32_ONNX), val)
    parity = (ref[:5] - extractor(val[:5])).abs().max().item()
    assert parity < 1e-4, f"fp32.onnx parity failed: max abs diff {parity:.2e}"
    print(f"fp32.onnx: opset {OPSET}, input (1, 3, 224, 224) in [0, 1], output (1, {', '.join(map(str, ref.shape[1:]))})")
    print(f"parity fp32.onnx vs PyTorch on 5 val images: max abs diff {parity:.2e} (must be < 1e-4)")
    print(f"FP32 feature scale on val: mean |x| {ref.abs().mean():.4f}, max {ref.max():.4f}")

    calib = load_images(load_split("calib"))
    rows = {}
    print(f"\nquantize_static on calib.csv ({len(calib)} images), compared to fp32.onnx on val ({len(val)} images)")
    with tempfile.TemporaryDirectory() as tmp:
        prep = Path(tmp) / "prep.onnx"
        quant_pre_process(FP32_ONNX, prep)
        for method in METHODS:
            out = Path(tmp) / f"int8_{method}.onnx"
            quantize_static(prep, out, CalibReader(calib), calibrate_method=METHODS[method],
                            extra_options=dict(EXTRA_OPTIONS), **QUANT_ARGS)
            rows[method] = compare(ort_features(ort_session(out), val), ref)
            r = rows[method]
            print(f"  {method:<11} mean |int8 - fp32| {r['mean_abs_diff']:.4f}   cosine mean {r['cos_mean']:.4f}  min {r['cos_min']:.4f}")
        picked = min(rows, key=lambda m: rows[m]["mean_abs_diff"])
        shutil.copyfile(Path(tmp) / f"int8_{picked}.onnx", INT8_ONNX)
    print(f"  picked {picked} (lower mean |int8 - fp32| on val) -> results/int8.onnx")

    file_ops, q_convs, producer, runtime = graph_ops(INT8_ONNX)
    int8 = ort_session(INT8_ONNX)
    feats = ort_features(int8, val)
    assert feats.shape == ref.shape, f"int8 output {tuple(feats.shape)} != fp32 {tuple(ref.shape)}"
    float_convs = runtime["Conv"] + runtime["FusedConv"]
    print("\nint8.onnx verification")
    print(f"  QuantizeLinear {file_ops['QuantizeLinear']}, DequantizeLinear {file_ops['DequantizeLinear']}")
    print(f"  Conv nodes with dequantized int8 weights: {q_convs} of {file_ops['Conv']}")
    print(f"  ORT runtime graph: QLinearConv {runtime['QLinearConv']}, float Conv {float_convs}")
    print(f"  graph output produced by {producer}")
    print(f"  output shape on val {tuple(feats.shape)} matches fp32")
    print(f"  per-patch cosine to fp32 on val: mean {rows[picked]['cos_mean']:.4f}, min {rows[picked]['cos_min']:.4f}")

    fit_feats = ort_features(int8, load_images(load_split("fit")))
    np.save(BANK_INT8, fit_bank(fit_feats).numpy())
    bank_int8, bank_fp32 = torch.from_numpy(np.load(BANK_INT8)), torch.from_numpy(np.load(BANK_FP32))
    print(f"\nbank_int8: {len(fit_feats)} fit images -> {tuple(bank_int8.shape)} (same coreset code, seed {SEED})")

    val_scores = {name: torch.cat([nn_score(f[i:i + 4], bank)[0] for i in range(0, len(f), 4)])
                  for name, f, bank in (("fp32 (onnx)", ref, bank_fp32), ("int8", feats, bank_int8))}
    print(f"\nval normals ({len(val)}), image score")
    for name, s in val_scores.items():
        print(f"  {name:<12} min {s.min():.4f}  mean {s.mean():.4f}  max {s.max():.4f}")

    files = (FP32_ONNX, INT8_ONNX, BANK_FP32, BANK_INT8)
    mb = {p.name: p.stat().st_size / MB for p in files}
    print("\nsize (MB = 2^20 bytes)")
    for p in files:
        print(f"  {p.name:<15}{mb[p.name]:7.2f}")
    print(f"  variant fp32 = fp32.onnx + bank_fp32.npy: {mb['fp32.onnx'] + mb['bank_fp32.npy']:.2f}")
    print(f"  variant int8 = int8.onnx + bank_int8.npy: {mb['int8.onnx'] + mb['bank_int8.npy']:.2f}")
    sha = {p.name: sha256_of(p) for p in (FP32_ONNX, INT8_ONNX, BANK_INT8)}
    print("\nsha256")
    for name, h in sha.items():
        print(f"  {name:<15}{h}")

    LOG.write_text(json.dumps({
        "calibration": "splits/calib.csv, 50 images",
        "selection": "lower mean |int8 - fp32.onnx| feature difference on val (21 images)",
        "settings": {**{k: str(v) for k, v in QUANT_ARGS.items()}, **EXTRA_OPTIONS, "opset": OPSET},
        "methods": rows,
        "picked": picked,
        "parity_fp32_onnx_vs_pytorch_max_abs_diff": parity,
        "verification": {"QuantizeLinear": file_ops["QuantizeLinear"], "DequantizeLinear": file_ops["DequantizeLinear"],
                         "conv_int8_weights": q_convs, "conv_total": file_ops["Conv"],
                         "runtime_QLinearConv": runtime["QLinearConv"], "runtime_float_conv": float_convs,
                         "output_producer": producer},
        "val_scores": {k: {"min": v.min().item(), "mean": v.mean().item(), "max": v.max().item()}
                       for k, v in val_scores.items()},
        "size_mb": mb,
        "sha256": sha,
        "versions": {"torch": torch.__version__, "onnxruntime": ort.__version__, "onnx": onnx.__version__},
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {LOG.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
