"""Score the frozen PatchCore variants once on the locked test split: clean metrics for FP32 and INT8 (ONNX Runtime),
two-sided white-box attacks on FP32 (PyTorch), five sanity checks, figures, the tradeoff table and error-analysis
thumbnails. The only file that opens test images.

Run from the project root:
  python -m src.eval --dry-run   same code path on val (21 normals + 42 synthetic) into results/dryrun/; never reads test
  python -m src.eval             test, once: checks results/freeze.json, then results/.scored
  python -m src.eval --force     re-score test after .scored exists; the forced run is logged in docs/leakage_audit.md
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.attack import fgsm, pgd, random_sign
from src.data import ROOT, SEED, load_images, load_split, sha256_of
from src.patchcore import BANK_FP32, FeatureExtractor, nn_score, ort_features, ort_session, score

RESULTS = ROOT / "results"
FREEZE, SCORED = RESULTS / "freeze.json", RESULTS / ".scored"
AUDIT = ROOT / "docs" / "leakage_audit.md"
CONFIG_PATH = ROOT / "eval_config.yaml"
VARIANTS = ("fp32", "int8")
ONNX = {"fp32": RESULTS / "fp32.onnx", "int8": RESULTS / "int8.onnx"}
BANKS = {"fp32": BANK_FP32, "int8": RESULTS / "bank_int8.npy"}
FROZEN_REQUIRED = {"eval_config.yaml", "src/data.py", "src/patchcore.py", "src/attack.py", "src/eval.py",
                   "splits/manifest.json", "results/bank_fp32.npy", "results/bank_int8.npy", "results/fp32.onnx",
                   "results/int8.onnx", "results/bench.json"}
B = 4                        # fixed batches: clean and attacked FP32 scores come from identical batches
DRY_EPS = [0.0, 0.01, 0.02]
N_BOOT = 1000
RANDOM_VS_FGSM, NOISE_FLOOR = 0.25, 0.01   # sanity 4: random-sign AUROC drop <= 0.25 x FGSM drop, or <= 0.01
TEST_OPENED = False


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def append_log(path, line):
    """Append one line; the leakage audit gets a run-log heading the first time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.exists() else ("# Leakage audit\n" if path == AUDIT else "")
    if path == AUDIT and "## eval.py run log" not in text:
        text += "\n## eval.py run log\n\n"
    path.write_text(text + line + "\n", encoding="utf-8", newline="\n")


def check_freeze():
    """Abort unless results/freeze.json covers the required files and every listed sha256 still matches."""
    if not FREEZE.exists():
        raise SystemExit("REFUSING TO RUN: results/freeze.json is missing; run python -m src.freeze first")
    frozen = json.loads(FREEZE.read_text(encoding="utf-8"))["sha256"]
    missing = sorted(FROZEN_REQUIRED - set(frozen))
    changed = [p for p, h in frozen.items() if not (ROOT / p).exists() or sha256_of(ROOT / p) != h]
    manifest = json.loads((ROOT / "splits" / "manifest.json").read_text(encoding="utf-8"))
    changed += [f"splits/{n}" for n, h in manifest.items() if sha256_of(ROOT / "splits" / n) != h]
    if missing or changed:
        raise SystemExit(f"REFUSING TO RUN: not frozen {missing}, changed since freeze {changed}")
    print(f"freeze OK: {len(frozen)} files match results/freeze.json; {len(manifest)} split CSVs match the manifest")


def check_scored(force):
    if SCORED.exists() and not force:
        raise SystemExit("REFUSING TO RUN: results/.scored exists, test was already scored once "
                         "(--force re-scores and is logged in docs/leakage_audit.md)")


def load_test():
    """The only allow_test=True call in the repo."""
    global TEST_OPENED
    TEST_OPENED = True
    df = load_split("test", allow_test=True)
    changed = [p for p, h in zip(df["path"], df["sha256"]) if sha256_of(ROOT / p) != h]
    assert not changed, f"{len(changed)} test images changed since splitting, first: {changed[0]}"
    return load_images(df), df["label"].to_numpy(), df["defect_type"].to_numpy(), df["path"].tolist()


def load_dry():
    """Val normals plus the 42 synthetic defects calibrate.py makes from them (same seed)."""
    from src.calibrate import synthesize
    df = load_split("val")
    val = load_images(df)
    synth, _ = synthesize(val, np.random.default_rng(SEED))
    n, paths = len(val), df["path"].tolist()
    return (torch.cat([val, synth["cutpaste"], synth["blob"]]), np.array([0] * n + [1] * 2 * n),
            np.array(["good"] * n + ["cutpaste"] * n + ["blob"] * n),
            paths + [f"{p}#cutpaste" for p in paths] + [f"{p}#blob" for p in paths])


def onnx_scores(variant, bank, x):
    feats = ort_features(ort_session(ONNX[variant]), x)
    return torch.cat([nn_score(feats[i:i + B], bank)[0] for i in range(0, len(feats), B)]).numpy()


def metrics(s, labels, types, t):
    """Confusion counts and rates for the rule: defect if score > t; recall per defect type as 'hits/n'."""
    pos, pred = labels == 1, s > t
    tp, fp = int((pred & pos).sum()), int((pred & ~pos).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & ~pos).sum())
    return {"threshold": t, "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fp + fn), "fpr": fp / (fp + tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            **{f"recall_{d}": f"{int(pred[types == d].sum())}/{int((types == d).sum())}"
               for d in sorted(set(types[pos]))}}


def bootstrap(s, labels, t):
    """Stratified bootstrap 95% CIs (good and defect images resampled separately) for AUROC and recall at t."""
    rng = np.random.default_rng(SEED)
    good, bad = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    aucs, recalls = [], []
    for _ in range(N_BOOT):
        g, b = rng.choice(good, len(good)), rng.choice(bad, len(bad))
        idx = np.concatenate([g, b])
        aucs.append(roc_auc_score(labels[idx], s[idx]))
        recalls.append(float((s[b] > t).mean()))
    return np.percentile(aucs, [2.5, 97.5]).tolist(), np.percentile(recalls, [2.5, 97.5]).tolist()


def rate(num, den):
    return f"{num / den:.3f} ({num}/{den})" if den else "n/a (0/0)"


def attack_stats(s, clean_t, labels, t):
    """Two-sided AUROC/AP, each half's AUROC (escape: attacked defects vs clean goods; false alarm: clean defects vs
    attacked goods), recall and FPR at t with counts, attack-caused flips, and mean score shift toward the goal."""
    pos, pred, pred0 = labels == 1, s > t, clean_t > t
    tp, fn = int((pred & pos).sum()), int((~pred & pos).sum())
    fp, tn = int((pred & ~pos).sum()), int((~pred & ~pos).sum())
    return {"auroc": roc_auc_score(labels, s), "auroc_escape": roc_auc_score(labels, np.where(pos, s, clean_t)),
            "auroc_false_alarm": roc_auc_score(labels, np.where(pos, clean_t, s)),
            "ap": average_precision_score(labels, s), "recall": tp / (tp + fn),
            "tp": tp, "fn_escapes": fn, "fpr": fp / (fp + tn), "fp_false_alarms": fp, "tn": tn,
            "new_escapes": int((pred0 & ~pred & pos).sum()), "new_false_alarms": int((~pred0 & pred & ~pos).sum()),
            "damage": float(np.mean(np.where(pos, -1.0, 1.0) * (s - clean_t)))}


def sanity(att, stats, clean_t, clean_auc, eps_grid, pgd_eps):
    """The five sanity checks as PASS/FAIL lines with the values they were judged on."""
    dmg = lambda name, eps: 0.0 if eps == 0 else stats[(name, eps)]["damage"]
    drop = lambda name, eps: clean_auc - stats[(name, eps)]["auroc"]
    lines = ["damage (checks 3 and 5) = mean over images of the score shift toward the attacker's goal "
             "(+ for good images pushed up, + for defects pushed down)",
             f"check 4 = drop in two-sided AUROC from clean ({clean_auc:.4f}); its cutoff was revised after the val "
             "dry run, before freeze (docs/failure_report.md)", ""]

    def check(n, ok, text):
        lines.append(f"{n} {'PASS' if ok else 'FAIL'}  {text}")

    exact = {a: bool(np.array_equal(att[(a, 0.0)]["scores"], clean_t)) for a in ("fgsm", "random_sign")}
    check(1, all(exact.values()), f"eps 0 reproduces clean scores exactly: {exact}")
    worst = max(att, key=lambda k: att[k]["linf"].max() - k[1])
    ok = all(r["linf"].max() <= e + 1e-6 and r["pixel_min"] >= 0 and r["pixel_max"] <= 1 for (_, e), r in att.items())
    lo, hi = min(r["pixel_min"] for r in att.values()), max(r["pixel_max"] for r in att.values())
    check(2, ok, f"max |x_adv - x| <= eps + 1e-6 for every attack x eps (largest excess: {worst[0]} eps {worst[1]}, "
                 f"{att[worst]['linf'].max() - worst[1]:+.2e}); pixels in [{lo:.4f}, {hi:.4f}]")
    grows = {}
    for name, es in (("fgsm", eps_grid), ("pgd", [0.0] + pgd_eps)):
        d = [dmg(name, e) for e in es]
        grows[name] = (all(b > a for a, b in zip(d, d[1:])), ", ".join(f"{e:g}: {v:.4f}" for e, v in zip(es, d)))
    check(3, all(g for g, _ in grows.values()), "damage grows with eps: " +
          "; ".join(f"FGSM [{grows['fgsm'][1]}]" if n == "fgsm" else f"PGD [{grows['pgd'][1]}]" for n in grows))
    pairs = [(e, drop("random_sign", e), drop("fgsm", e), dmg("random_sign", e), dmg("fgsm", e))
             for e in eps_grid if e > 0]
    check(4, all(rd <= RANDOM_VS_FGSM * fd or rd <= NOISE_FLOOR for _, rd, fd, _, _ in pairs),
          f"random-sign AUROC drop <= {RANDOM_VS_FGSM} x FGSM AUROC drop, or <= {NOISE_FLOOR}: "
          + ", ".join(f"eps {e:g} drop {rd:.4f} vs {fd:.4f} (score shift, info only: {rs:.4f} vs {fs:.4f})"
                      for e, rd, fd, rs, fs in pairs))
    pairs = [(e, dmg("pgd", e), dmg("fgsm", e)) for e in pgd_eps]
    check(5, all(p >= g for _, p, g in pairs), "PGD damage >= FGSM damage: "
          + ", ".join(f"eps {e:g} PGD {p:.4f} vs FGSM {g:.4f}" for e, p, g in pairs))
    return lines


def thumb(path, panels, title, vmax):
    """Side-by-side panels of (image, 28x28 anomaly map or None, subtitle); maps share one color scale."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 5), layout="constrained")
    for ax, (img, amap, sub) in zip(axes, panels):
        ax.imshow(img.permute(1, 2, 0).numpy())
        if amap is not None:
            up = F.interpolate(amap[None, None], size=img.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            ax.imshow(up.numpy(), cmap="jet", alpha=0.45, vmin=0, vmax=vmax)
        ax.set_title(sub, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run(out, dry_run, cfg):
    t_run = time.perf_counter()
    misses = out / "misses"
    misses.mkdir(parents=True, exist_ok=True)
    for p in misses.glob("*.png"):
        p.unlink()
    x, labels, types, names = load_dry() if dry_run else load_test()
    pos = labels == 1
    n_def, n_good = int(pos.sum()), int((~pos).sum())
    primary, thr = cfg["primary_threshold"], cfg["thresholds"]
    alternative = next(k for k in thr["fp32"] if k != primary)
    p_cfg = cfg["pgd"]
    assert p_cfg["random_start"], "src/attack.py pgd always starts from a uniform random point"
    eps_grid = DRY_EPS if dry_run else [float(e) for e in cfg["eps_grid"]]
    pgd_eps = [float(e) for e in p_cfg["eps"] if not dry_run or float(e) in DRY_EPS]
    r = p_cfg["restarts"]
    label = {"fgsm": "FGSM", "random_sign": "random-sign noise",
             "pgd": f"PGD {p_cfg['steps']} steps, alpha {p_cfg['alpha_over_eps']} x eps, {r} restart{'s' * (r != 1)}, "
                    "random start"}
    where = "DRY RUN on val (normals + synthetic defects)" if dry_run else "test"
    print(f"{where}: {len(x)} images ({n_good} good, {n_def} defect); primary threshold {primary}")

    banks = {v: torch.from_numpy(np.load(BANKS[v])) for v in VARIANTS}
    clean = {v: onnx_scores(v, banks[v], x) for v in VARIANTS}
    extractor = FeatureExtractor()
    f = lambda z: score(z, extractor, banks["fp32"])[0]
    batches = [(x[i:i + B], torch.from_numpy(labels[i:i + B])) for i in range(0, len(x), B)]
    clean_t = torch.cat([f(xb) for xb, _ in batches]).numpy()
    gap = float(np.abs(clean_t - clean["fp32"]).max())
    assert gap < 1e-4, f"FP32 torch and FP32 ONNX clean scores differ by {gap:.2e}"
    print(f"FP32 torch vs FP32 ONNX clean scores: max |diff| {gap:.2e} (< 1e-4)")
    pd.DataFrame({"path": names, "label": labels, "defect_type": types, "fp32_onnx": clean["fp32"],
                  "int8_onnx": clean["int8"], "fp32_torch": clean_t}).to_csv(
        out / "clean_scores.csv", index=False, lineterminator="\n", float_format="%.6f")

    rows, prim = [], {}
    print(f"\nclean, rule: defect if score > threshold; 95% CI = stratified bootstrap, {N_BOOT} resamples, seed {SEED}")
    for v in VARIANTS:
        s = clean[v]
        auc, ap = roc_auc_score(labels, s), average_precision_score(labels, s)
        auc_ci, rec_ci = bootstrap(s, labels, thr[v][primary])
        for name in (primary, alternative):
            m = metrics(s, labels, types, thr[v][name])
            is_primary = name == primary
            rows.append({"variant": v, "threshold_name": name, "role": "primary" if is_primary else "alternative",
                         "auroc": auc, "auroc_ci_lo": auc_ci[0], "auroc_ci_hi": auc_ci[1], "ap": ap, **m,
                         "recall_ci_lo": rec_ci[0] if is_primary else None,
                         "recall_ci_hi": rec_ci[1] if is_primary else None})
            if is_primary:
                prim[v] = rows[-1]
            ci = f" [{rec_ci[0]:.3f}, {rec_ci[1]:.3f}]" if is_primary else ""
            print(f"  {v} {name:<15}({rows[-1]['role']}) t {m['threshold']:.4f}  AUROC {auc:.4f} "
                  f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]  AP {ap:.4f}  P {rate(m['tp'], m['tp'] + m['fp'])}  "
                  f"R {rate(m['tp'], n_def)}{ci}  F1 {m['f1']:.3f}  FPR {rate(m['fp'], n_good)}  "
                  f"TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}")
            print("      recall by type: " + ", ".join(f"{k[7:]} {val}" for k, val in m.items() if k.startswith("recall_")))
    pd.DataFrame(rows).to_csv(out / "clean.csv", index=False, lineterminator="\n", float_format="%.6f")

    fig, ax = plt.subplots(figsize=(6.4, 5))
    for v, marker in zip(VARIANTS, ("o", "s")):
        p, rc, _ = precision_recall_curve(labels, clean[v])
        (line,) = ax.plot(rc, p, drawstyle="steps-post", label=f"{v} (AP {prim[v]['ap']:.3f})")
        ax.plot(prim[v]["recall"], prim[v]["precision"], marker, color=line.get_color(), mfc="none", ms=9,
                label=f"{v} at {primary} {prim[v]['threshold']:.3f}")
    ax.axhline(n_def / len(labels), ls=":", color="gray", label=f"base rate {n_def}/{len(labels)}")
    ax.set(xlabel="recall", ylabel="precision", ylim=(0, 1.02), title=f"Precision-recall, {where}")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=120)
    plt.close(fig)

    fns = {"fgsm": fgsm, "random_sign": random_sign,
           "pgd": lambda g, z, y, e: pgd(g, z, y, e, steps=p_cfg["steps"], alpha=e * p_cfg["alpha_over_eps"],
                                         restarts=r)}
    plan = [("fgsm", e) for e in eps_grid] + [("pgd", e) for e in pgd_eps] + [("random_sign", e) for e in eps_grid]
    t_fp32 = thr["fp32"][primary]
    att, stats, pgd_adv = {}, {}, None
    print(f"\nattacks on FP32 torch, two-sided, L-infinity on 0 to 1 pixels, batches of {B}; at threshold {t_fp32:.4f}")
    for name, eps in plan:
        t0 = time.perf_counter()
        adv = torch.cat([fns[name](f, xb, yb, eps) for xb, yb in batches])
        s = torch.cat([f(adv[i:i + B]) for i in range(0, len(adv), B)]).numpy()
        att[(name, eps)] = {"scores": s, "linf": (adv - x).abs().amax(dim=(1, 2, 3)).numpy(),
                            "pixel_min": adv.min().item(), "pixel_max": adv.max().item()}
        stats[(name, eps)] = st = attack_stats(s, clean_t, labels, t_fp32)
        if name == "pgd" and eps == 0.01:
            pgd_adv = adv
        print(f"  {label[name]:<55} eps {eps:<6g} AUROC {st['auroc']:.4f} (escape {st['auroc_escape']:.4f}, "
              f"false alarm {st['auroc_false_alarm']:.4f})  recall {rate(st['tp'], n_def)}  "
              f"FPR {rate(st['fp_false_alarms'], n_good)}  new escapes {st['new_escapes']}  "
              f"new false alarms {st['new_false_alarms']}  ({time.perf_counter() - t0:.0f} s)")
    pd.DataFrame([{"attack": n, "label": label[n], "eps": e, "steps": p_cfg["steps"] if n == "pgd" else None,
                   "restarts": r if n == "pgd" else None, "threshold": t_fp32, **stats[(n, e)], "n_defect": n_def,
                   "n_good": n_good, "max_linf": att[(n, e)]["linf"].max(), "pixel_min": att[(n, e)]["pixel_min"],
                   "pixel_max": att[(n, e)]["pixel_max"]} for n, e in plan]).to_csv(
        out / "attack.csv", index=False, lineterminator="\n", float_format="%.6f")
    pd.DataFrame([{"path": names[i], "label": labels[i], "defect_type": types[i], "attack": n, "eps": e,
                   "clean_score": clean_t[i], "adv_score": att[(n, e)]["scores"][i], "linf": att[(n, e)]["linf"][i]}
                  for n, e in plan for i in range(len(x))]).to_csv(
        out / "attack_scores.csv", index=False, lineterminator="\n", float_format="%.6f")

    clean_st = attack_stats(clean_t, clean_t, labels, t_fp32)
    lines = sanity(att, stats, clean_t, clean_st["auroc"], eps_grid, pgd_eps)
    (out / "sanity.txt").write_text(f"Sanity checks, {where}, attacks on FP32 torch\n" + "\n".join(lines) + "\n",
                                    encoding="utf-8")
    print("\nsanity checks (results/" + ("dryrun/" if dry_run else "") + "sanity.txt)")
    print("\n".join("  " + ln for ln in lines if ln))

    min_auroc = cfg["min_auroc"]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    print(f"\nfirst eps where each AUROC < {min_auroc} (frozen minimum), and attack-caused flips at {primary} "
          f"{t_fp32:.4f}")
    for k, name in enumerate(("fgsm", "pgd", "random_sign")):
        es = ([0.0] if name == "pgd" else []) + [e for n, e in plan if n == name]
        st = [clean_st if (name == "pgd" and e == 0) else stats[(name, e)] for e in es]
        c, short = f"C{k}", label[name].split(",")[0]
        a1.plot(es, [x_["auroc"] for x_ in st], "o-", color=c, label=label[name])
        a2.plot(es, [x_["auroc_escape"] for x_ in st], "o-", color=c, label=f"{short} escape half")
        a2.plot(es, [x_["auroc_false_alarm"] for x_ in st], "x--", color=c, label=f"{short} false-alarm half")
        a3.plot(es, [x_["recall"] for x_ in st], "o-", color=c, label=f"{short} recall")
        a3.plot(es, [x_["fpr"] for x_ in st], "x--", color=c, label=f"{short} FPR")
        print(f"  {label[name]}:")
        for key, what in (("auroc", "two-sided AUROC"), ("auroc_escape", "escape AUROC"),
                          ("auroc_false_alarm", "false-alarm AUROC")):
            below = [e for e, x_ in zip(es, st) if x_[key] < min_auroc]
            print(f"    {what}: " + (f"first below {min_auroc} at eps {below[0]:g}" if below
                                     else f"stayed >= {min_auroc} at every eps tested, up to {es[-1]:g}"))
        for kind, col in (("escape", "new_escapes"), ("false alarm", "new_false_alarms")):
            hit = [e for e in es[1:] if stats[(name, e)][col] > 0]
            print(f"    {kind}s: " + (f"first successful {kind} at eps {hit[0]:g} ({stats[(name, hit[0])][col]} images)"
                                       if hit else f"no successful {kind} found up to eps {es[-1]:g} with {label[name]}"))
    eps_label = "eps (L-infinity, 0 to 1 pixels)"
    for ax in (a1, a2):
        ax.axhline(min_auroc, color="red", ls=":", label=f"frozen minimum AUROC {min_auroc}")
    a1.set(xlabel=eps_label, ylabel="AUROC", ylim=(0, 1.02), title=f"two-sided AUROC, FP32, {where}")
    a2.set(xlabel=eps_label, ylabel="AUROC", ylim=(0, 1.02),
           title="halves: escape (attacked defects vs clean goods),\nfalse alarm (clean defects vs attacked goods)")
    a3.set(xlabel=eps_label, ylabel="rate", ylim=(-0.02, 1.02), title=f"recall and FPR at {primary} {t_fp32:.3f}")
    a1.legend(fontsize=8)
    a2.legend(fontsize=7, ncol=2)
    a3.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "degradation_curve.png", dpi=120)
    plt.close(fig)

    bench = json.loads((RESULTS / "bench.json").read_text(encoding="utf-8"))
    trade = [{"variant": v, "auroc": prim[v]["auroc"], "ap": prim[v]["ap"], "f1": prim[v]["f1"],
              "recall": prim[v]["recall"], "recall_counts": f"{prim[v]['tp']}/{n_def}",
              "threshold": prim[v]["threshold"], "p50_ms": bench["variants"][v]["total"]["p50"],
              "p95_ms": bench["variants"][v]["total"]["p95"], "size_mb": bench["variants"][v]["size_mb"]["total"]}
             for v in VARIANTS]
    pd.DataFrame(trade).to_csv(out / "tradeoff.csv", index=False, lineterminator="\n", float_format="%.6f")
    md = [f"# Accuracy-latency-size tradeoff ({where})", "",
          f"Accuracy at the primary threshold ({primary}, frozen on val). Latency = ONNX Runtime extractor + "
          f"nearest-neighbor search, batch 1, {bench['intra_op_threads']} threads, on {bench['cpu']} "
          f"(results/bench.json). Size = ONNX file + bank file, MB = 2^20 bytes.", "",
          "| variant | AUROC | AP | F1 | recall | p50 ms | p95 ms | size MB |", "|---|---|---|---|---|---|---|---|"]
    md += [f"| {t['variant']} | {t['auroc']:.4f} | {t['ap']:.4f} | {t['f1']:.3f} | {t['recall']:.3f} "
           f"({t['recall_counts']}) | {t['p50_ms']:.2f} | {t['p95_ms']:.2f} | {t['size_mb']:.2f} |" for t in trade]
    (out / "tradeoff.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md[4:]))

    vmax = 2 * t_fp32
    stem = lambda i: f"{types[i]}_{Path(names[i].split('#')[0]).stem}"
    short = lambda i: names[i].replace("data/mvtec/bottle/", "")
    amap = lambda z: score(z[None], extractor, banks["fp32"])[1][0]
    pred = clean["fp32"] > t_fp32
    for kind, idx in (("fn", np.flatnonzero(pos & ~pred)), ("fp", np.flatnonzero(~pos & pred))):
        for i in idx:
            thumb(misses / f"{kind}_{stem(i)}.png", [(x[i], None, "image"), (x[i], amap(x[i]), "anomaly map")],
                  f"clean {'missed defect' if kind == 'fn' else 'false alarm'}: {short(i)}\n{types[i]}, FP32 score "
                  f"{clean['fp32'][i]:.3f} vs {primary} {t_fp32:.3f}", vmax)
    if pgd_adv is not None:
        s = att[("pgd", 0.01)]["scores"]
        for rank, i in enumerate(np.flatnonzero(pos)[np.argsort(s[pos])][:3], 1):
            thumb(misses / f"pgd001_low{rank}_{stem(i)}.png",
                  [(x[i], amap(x[i]), f"clean, score {clean_t[i]:.3f}"),
                   (pgd_adv[i], amap(pgd_adv[i]), f"PGD eps 0.01, score {s[i]:.3f}")],
                  f"lowest-scoring defect #{rank} under PGD eps 0.01\n{label['pgd']}\n"
                  f"{short(i)} ({types[i]}), threshold {primary} {t_fp32:.3f}", vmax)
    written = sorted(p.relative_to(ROOT).as_posix() for p in out.rglob("*") if p.is_file() and p.name != "run_log.txt")
    print(f"\nwrote {len(written)} files in {time.perf_counter() - t_run:.0f} s:\n  " + "\n  ".join(written))


def main(dry_run=False, force=False):
    out = RESULTS / "dryrun" if dry_run else RESULTS
    log_path = out / "run_log.txt" if dry_run else AUDIT
    if not dry_run:
        check_freeze()
        check_scored(force)
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    mode = "dry run on val" if dry_run else ("test, FORCED re-run" if SCORED.exists() else "test")
    append_log(log_path, f"- {now()} eval.py start ({mode}), sha256 {sha256_of(Path(__file__))}")
    try:
        run(out, dry_run, cfg)
    except BaseException as e:
        append_log(log_path, f"- {now()} eval.py FAILED ({mode}): {type(e).__name__}: {e}")
        raise
    assert not (dry_run and TEST_OPENED), "dry run opened test.csv"
    append_log(log_path, f"- {now()} eval.py end ({mode}), all outputs written")
    if not dry_run:
        SCORED.write_text(f"test scored {now()}\n", encoding="utf-8")
    print(f"\n{log_path.relative_to(ROOT).as_posix()}: start and end logged"
          + ("" if dry_run else "; results/.scored written") + ("; test.csv never opened" if dry_run else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="same code path on val + synthetic, results/dryrun/")
    parser.add_argument("--force", action="store_true", help="re-score test after .scored exists (logged)")
    args = parser.parse_args()
    main(args.dry_run, args.force)
