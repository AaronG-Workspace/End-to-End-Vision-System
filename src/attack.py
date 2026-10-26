"""Two-sided white-box L-infinity attacks on the FP32 PatchCore score: FGSM, PGD (random start, restarts) and a
random-sign noise baseline. Inputs and outputs are on the 0 to 1 pixel scale, clamped after every step.

Label 1 (defect) steps to LOWER the score (escape); label 0 (good) steps to RAISE it (false alarm). Each attack takes
f(x) -> differentiable image scores (n,) first; eval.py passes the frozen FP32 PyTorch model.

Run from the project root: python -m src.attack   (smoke test and timing on val only; writes pgd.restarts)
"""
import re
import time

import numpy as np
import torch
import yaml

from src.data import ROOT, SEED, load_images, load_split
from src.patchcore import BANK_FP32, FeatureExtractor, score

CONFIG_PATH = ROOT / "eval_config.yaml"
N_TEST, PGD_EPS, FGSM_EPS = 83, 3, 7   # test images; PGD eps values; FGSM eps values incl. the eps 0 check
BUDGET_S = 20 * 60


def _direction(y):
    """+1 raises the score (label 0, false alarm), -1 lowers it (label 1, escape)."""
    return 1 - 2 * y.to(torch.float32)


def _grad_sign(f, x, d):
    """Scores at x and the sign of d(sum(d * f(x)))/dx; images in a batch never interact (eval mode, per-image max)."""
    x = x.clone().requires_grad_(True)
    s = f(x)
    (g,) = torch.autograd.grad((d * s).sum(), x)
    return s.detach(), g.sign()


def fgsm(f, x, y, eps):
    """One signed-gradient step of size eps toward the attacker's goal, clamped to [0, 1]."""
    return (x + eps * _grad_sign(f, x, _direction(y))[1]).clamp(0, 1)


def random_sign(f, x, y, eps):
    """Noise baseline: +-eps per pixel with seeded random signs, clamped to [0, 1]. f and y unused (same call shape)."""
    signs = torch.randint(0, 2, x.shape, generator=torch.Generator().manual_seed(SEED), dtype=torch.float32) * 2 - 1
    return (x + eps * signs).clamp(0, 1)


def pgd(f, x, y, eps, steps=10, alpha=None, restarts=1):
    """L-infinity PGD from a uniform random start in [-eps, eps]; returns the worst iterate per image over every step
    of every restart (start and final point included): lowest score for defects, highest for normals."""
    alpha = eps / 4 if alpha is None else alpha
    d = _direction(y)
    lo, hi = x - eps, x + eps
    gen = torch.Generator().manual_seed(SEED)
    best, best_obj = x.clone(), torch.full((len(x),), -float("inf"))

    def keep(x_k, s):
        better = d * s > best_obj
        best[better], best_obj[better] = x_k[better], (d * s)[better]

    for _ in range(restarts):
        x_k = (x + (torch.rand(x.shape, generator=gen) * 2 - 1) * eps).clamp(0, 1)
        for _ in range(steps):
            s, sign = _grad_sign(f, x_k, d)
            keep(x_k, s)
            x_k = (x_k + alpha * sign).clamp(lo, hi).clamp(0, 1)
        keep(x_k, f(x_k))
    return best


def write_restarts(r):
    """Replace only pgd.restarts in eval_config.yaml; assert every other setting is unchanged."""
    text = CONFIG_PATH.read_text(encoding="utf-8")
    want = yaml.safe_load(text)
    want["pgd"]["restarts"] = r
    text, n = re.subn(r"^(  restarts: )\d+", lambda m: f"{m.group(1)}{r}", text, flags=re.M)
    assert n == 1 and yaml.safe_load(text) == want, "eval_config.yaml: expected exactly one pgd restarts line"
    CONFIG_PATH.write_text(text, encoding="utf-8", newline="\n")   # LF: freeze.py hashes this file


def main():
    from src.calibrate import synthesize   # smoke test only; eval.py never needs calibrate.py
    torch.manual_seed(SEED)
    extractor = FeatureExtractor()
    bank = torch.from_numpy(np.load(BANK_FP32))
    f = lambda x: score(x, extractor, bank)[0]
    val = load_images(load_split("val"))
    synth, _ = synthesize(val, np.random.default_rng(SEED))
    x = torch.cat([val[:4], synth["cutpaste"][:2], synth["blob"][:2]])
    y = torch.tensor([0] * 4 + [1] * 4)
    names = [f"normal {i}" for i in range(4)] + ["cutpaste 0", "cutpaste 1", "blob 0", "blob 1"]
    attacks = {"fgsm": fgsm, "pgd": pgd, "random_sign": random_sign}

    clean = f(x)
    for name, attack in attacks.items():
        assert torch.equal(f(attack(f, x, y, 0.0)), clean), f"{name} at eps 0 changed the clean scores"
    print("eps 0: fgsm, pgd (10 steps, 1 restart) and random_sign reproduce the clean scores exactly (torch.equal)")

    eps = 0.01
    adv = {name: attack(f, x, y, eps) for name, attack in attacks.items()}
    adv_scores = {name: f(a) for name, a in adv.items()}
    print(f"\nval only, eps {eps} (L-infinity, 0 to 1 pixels); PGD 10 steps, alpha eps/4, 1 restart")
    print(f"{'image':<12}{'goal':<14}{'clean':>8}{'fgsm':>8}{'pgd':>8}{'random_sign':>13}")
    for i, name in enumerate(names):
        goal = "lower (escape)" if y[i] else "raise (alarm)"
        print(f"{name:<12}{goal:<14}{clean[i]:>8.4f}" + "".join(f"{adv_scores[a][i]:>{w}.4f}"
                                                            for a, w in (("fgsm", 8), ("pgd", 8), ("random_sign", 13))))
    print()
    for name, a in adv.items():
        linf, lo, hi = (a - x).abs().max().item(), a.min().item(), a.max().item()
        assert linf <= eps + 1e-6 and lo >= 0 and hi <= 1, f"{name}: |x_adv - x| {linf}, pixels [{lo}, {hi}]"
        print(f"{name:<12} max |x_adv - x| {linf:.6f} (<= eps {eps} + 1e-6 float slack)   pixels [{lo:.4f}, {hi:.4f}]")

    x1, y1 = x[4:5], y[4:5]
    t0 = time.perf_counter()
    f(fgsm(f, x1, y1, eps))
    t1 = time.perf_counter()
    f(pgd(f, x1, y1, eps))
    t2 = time.perf_counter()
    t_fgsm, t_pgd = t1 - t0, t2 - t1
    est = {"fgsm": N_TEST * FGSM_EPS * t_fgsm, **{f"pgd r={r}": N_TEST * PGD_EPS * r * t_pgd for r in (1, 3)}}
    print(f"\ntiming, 1 image (attack + scoring the result): fgsm {t_fgsm:.2f} s, pgd 10 steps 1 restart {t_pgd:.2f} s")
    print(f"test estimate: fgsm {N_TEST} x {FGSM_EPS} eps {est['fgsm'] / 60:.1f} min; pgd {N_TEST} x {PGD_EPS} eps: "
          f"r=1 {est['pgd r=1'] / 60:.1f} min, r=3 {est['pgd r=3'] / 60:.1f} min")
    restarts = 3 if est["pgd r=3"] <= BUDGET_S else 1
    write_restarts(restarts)
    print(f"rule: 3 restarts if PGD at r=3 fits in {BUDGET_S // 60} min, else 1 -> pgd.restarts = {restarts} "
          "written to eval_config.yaml (nothing else changed, asserted)")


if __name__ == "__main__":
    main()
