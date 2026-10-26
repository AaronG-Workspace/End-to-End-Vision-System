"""Thresholds from val only. Synthetic defects (CutPaste + dark blobs) are made from the 21 val normals and scored with
the FP32 (PyTorch) and INT8 (ONNX Runtime) variants, each against its own bank. Writes the four threshold values into
eval_config.yaml (nothing else changes), results/calib.json and results/synth_grid.png. Never reads test.

Run from the project root: python -m src.calibrate
"""
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import yaml
from sklearn.metrics import roc_auc_score

from src.data import IMG, ROOT, SEED, load_images, load_split
from src.patchcore import BANK_FP32, FeatureExtractor, nn_score, ort_features, ort_session, score
from src.quantize import BANK_INT8, INT8_ONNX

CONFIG_PATH = ROOT / "eval_config.yaml"
GRID = ROOT / "results" / "synth_grid.png"
LOG = ROOT / "results" / "calib.json"
KINDS = ("cutpaste", "blob")


def cutpaste(img, donor, rng):
    """Paste a randomly rotated 24-56 px patch from a different val normal at a random spot -> (image, bbox)."""
    s = int(rng.integers(24, 57))
    sy, sx, y, x = (int(v) for v in rng.integers(0, IMG - s + 1, size=4))
    angle = float(rng.uniform(0, 360))
    patch = TF.rotate(donor[:, sy:sy + s, sx:sx + s], angle)
    mask = TF.rotate(torch.ones(1, s, s), angle)   # 0 outside the rotated square, so no black corners are pasted
    out = img.clone()
    out[:, y:y + s, x:x + s] = mask * patch + (1 - mask) * out[:, y:y + s, x:x + s]
    return out, (x, y, s, s)


def dark_blob(img, rng):
    """Darken a soft-edged ellipse (radii 4-10 px) by 60-90% at a random spot -> (image, bbox)."""
    ry, rx = (float(v) for v in rng.uniform(4, 10, size=2))
    cy, cx = (float(v) for v in rng.uniform(16, IMG - 16, size=2))
    yy, xx = torch.meshgrid(torch.arange(IMG), torch.arange(IMG), indexing="ij")
    mask = torch.clamp(1 - ((yy - cy) / ry) ** 2 - ((xx - cx) / rx) ** 2, 0, 1).sqrt()   # 1 at center, 0 at edge
    return img * (1 - float(rng.uniform(0.6, 0.9)) * mask), (cx - rx, cy - ry, 2 * rx, 2 * ry)


def synthesize(val, rng):
    """Two synthetic defects per val normal: CutPaste from a different val normal, and a dark blob."""
    out = {k: [] for k in KINDS}
    boxes = {k: [] for k in KINDS}
    for i in range(len(val)):
        j = (i + int(rng.integers(1, len(val)))) % len(val)   # never the same image
        for kind, (img, box) in (("cutpaste", cutpaste(val[i], val[j], rng)), ("blob", dark_blob(val[i], rng))):
            out[kind].append(img)
            boxes[kind].append(box)
    return {k: torch.stack(v) for k, v in out.items()}, boxes


def metrics_at(scores, labels, kinds, t):
    """Confusion counts, precision, recall, F1 and recall per synthetic type for the rule: defect if score > t."""
    pred = scores > t
    tp, fp = int((pred & labels).sum()), int((pred & ~labels).sum())
    fn, tn = int((~pred & labels).sum()), int((~pred & ~labels).sum())
    return {"threshold": t, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall_by_type": {k: [int(pred[kinds == k].sum()), int((kinds == k).sum())] for k in KINDS}}


def f1_threshold(scores, labels, kinds):
    """Best-F1 cut among midpoints of consecutive distinct val scores; ties go to the lower cut (higher recall)."""
    u = np.unique(scores)
    cuts = (u[:-1] + u[1:]) / 2
    return float(cuts[int(np.argmax([metrics_at(scores, labels, kinds, t)["f1"] for t in cuts]))])


def write_thresholds(thresholds):
    """Replace only the two threshold lines in eval_config.yaml; assert every other setting is unchanged."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    before = yaml.safe_load(text)
    for variant, t in thresholds.items():
        line = f"{{f1_optimal: {t['f1_optimal']!r}, max_val_normal: {t['max_val_normal']!r}}}"
        text, n = re.subn(rf"^(  {variant}: )\{{.*?\}}", lambda m: m.group(1) + line, text, flags=re.M)
        assert n == 1, f"expected one '{variant}:' threshold line in eval_config.yaml, found {n}"
    after = yaml.safe_load(text)
    assert after["thresholds"] == thresholds and {**after, "thresholds": 0} == {**before, "thresholds": 0}
    CONFIG_PATH.write_text(text, encoding="utf-8", newline="\n")   # LF: freeze.py hashes this file


def save_grid(synth, boxes, fp32, primary, t):
    """4x4 grid for the video: two rows of CutPaste, two rows of dark blobs, red box on the synthetic region."""
    fig, axes = plt.subplots(4, 4, figsize=(10, 10.6))
    for k, ax in enumerate(axes.flat):
        kind, i = KINDS[k // 8], k % 8
        ax.imshow(synth[kind][i].permute(1, 2, 0).numpy())
        x, y, w, h = boxes[kind][i]
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1.5))
        ax.set_title(f"{kind} (val {i})  score {fp32[kind][i]:.3f}", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Synthetic defects made from val normals only. Red box = synthetic region.\n"
                 f"FP32 image score; primary FP32 threshold ({primary}) {t:.3f}, defect if score > threshold",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(GRID, dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    val = load_images(load_split("val"))
    synth, boxes = synthesize(val, rng)
    images = torch.cat([val, synth["cutpaste"], synth["blob"]])
    kinds = np.array(["normal"] * len(val) + [k for k in KINDS for _ in range(len(val))])
    labels = kinds != "normal"

    extractor, sess = FeatureExtractor(), ort_session(INT8_ONNX)
    bank_fp32, bank_int8 = (torch.from_numpy(np.load(p)) for p in (BANK_FP32, BANK_INT8))
    scorers = {"fp32": lambda x: score(x, extractor, bank_fp32)[0],
               "int8": lambda x: nn_score(ort_features(sess, x), bank_int8)[0]}
    results, thresholds, all_scores = {}, {}, {}
    for variant, fn in scorers.items():
        s = torch.cat([fn(images[i:i + 4]) for i in range(0, len(images), 4)]).numpy().astype(np.float64)
        t_f1, t_max = f1_threshold(s, labels, kinds), float(s[~labels].max())
        separated = bool(t_max < s[labels].min())
        if separated:   # the whole gap is F1-optimal; the only midpoint inside it is (max normal + min synthetic) / 2
            assert t_f1 == (t_max + float(s[labels].min())) / 2
        thresholds[variant] = {"f1_optimal": t_f1, "max_val_normal": t_max}
        all_scores[variant] = s
        results[variant] = {
            "auroc": float(roc_auc_score(labels, s)),
            "separated": separated,
            "scores": {k: {"min": float(s[kinds == k].min()), "mean": float(s[kinds == k].mean()),
                           "max": float(s[kinds == k].max())} for k in ("normal", *KINDS)},
            "f1_optimal": metrics_at(s, labels, kinds, t_f1),
            "max_val_normal": metrics_at(s, labels, kinds, t_max),
        }

    print(f"val: {len(val)} normals + {len(images) - len(val)} synthetic ({len(val)} cutpaste, {len(val)} blob), "
          "rule: defect if score > threshold")
    print("\nscores        normal (min/mean/max)     cutpaste (min/mean/max)   blob (min/mean/max)")
    for variant, r in results.items():
        print(f"  {variant:<10}" + "".join(f"  {r['scores'][k]['min']:.3f} {r['scores'][k]['mean']:.3f} {r['scores'][k]['max']:.3f}  "
                                         for k in ("normal", *KINDS)))
    print(f"\n{'variant':<8}{'AUROC':>7}  {'separated':<10}{'threshold':<16}{'value':>8}{'P':>7}{'R':>7}{'F1':>7}"
          f"{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}  {'cutpaste':>8}{'blob':>7}")
    for variant, r in results.items():
        for name in ("f1_optimal", "max_val_normal"):
            m = r[name]
            hits = [f"{m['recall_by_type'][k][0]}/{m['recall_by_type'][k][1]}" for k in KINDS]
            print(f"{variant:<8}{r['auroc']:>7.4f}  {str(r['separated']):<10}{name:<16}{m['threshold']:>8.4f}"
                  f"{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}{m['tn']:>4}"
                  f"  {hits[0]:>8}{hits[1]:>7}")
    for variant, r in results.items():
        if r["separated"]:
            print(f"note: {variant} val is perfectly separated; F1-optimal = midpoint of max normal and min synthetic")

    write_thresholds(thresholds)
    primary = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["primary_threshold"]
    print(f"\neval_config.yaml: wrote 4 threshold values (primary = {primary}); every other setting unchanged (asserted)")
    for variant, t in thresholds.items():
        print(f"  {variant}: f1_optimal {t['f1_optimal']!r}, max_val_normal {t['max_val_normal']!r}")

    s = all_scores["fp32"]
    save_grid(synth, boxes, {k: s[kinds == k] for k in KINDS}, primary, thresholds["fp32"][primary])
    LOG.write_text(json.dumps({
        "split": "val only (21 normals); test never read",
        "synthetic": {"cutpaste": "24-56 px square from a different val normal, rotated uniform 0-360 deg, pasted at a "
                                  "random spot through a rotated mask", "blob": "soft-edged ellipse, radii 4-10 px, "
                                  "darkened 60-90%, center 16 px or more from the border", "seed": SEED},
        "rule": "defect if score > threshold",
        "threshold_selection": "f1_optimal = best F1 among midpoints of consecutive distinct val scores, ties to the "
                               "lower cut; max_val_normal = highest val normal score",
        "primary": primary,
        "thresholds": thresholds,
        "variants": results,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {GRID.relative_to(ROOT).as_posix()}, {LOG.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
