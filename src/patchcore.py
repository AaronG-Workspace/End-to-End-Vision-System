"""PatchCore (Roth et al., 2022), hand-written: frozen ImageNet backbone, layer2 + layer3 patch features,
greedy coreset memory bank, nearest-neighbor patch scores.

Run from the project root: python -m src.patchcore
Fits results/bank_fp32.npy on the 188 fit images, then smoke-tests on val only (score range, gradient check).
"""
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml

from src.data import ROOT, SEED, load_images, load_split, sha256_of

CONFIG = yaml.safe_load((ROOT / "eval_config.yaml").read_text(encoding="utf-8"))
BANK_FP32 = ROOT / "results" / "bank_fp32.npy"
PROJ_DIM = 128   # random projection used for coreset selection only
MB = 1_048_576
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FeatureExtractor(nn.Module):
    """[0, 1] RGB (n, 3, 224, 224) -> patch features (n, C2 + C3, 28, 28): 384 dims for resnet18."""

    def __init__(self, backbone=CONFIG["backbone"]):
        super().__init__()
        net = getattr(torchvision.models, backbone)(weights="DEFAULT")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        # fixed ImageNet constants inside forward, so attack eps lives on the 0 to 1 pixel scale
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        f2 = self.layer2(self.stem((x - self.mean) / self.std))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, stride=1, padding=1), size=f2.shape[-2:], mode="bilinear",
                           align_corners=False)
        return torch.cat([f2, f3], dim=1)


def greedy_coreset(patches, ratio=CONFIG["coreset_ratio"]):
    """Greedy k-center: keep adding the patch farthest from those picked so far, measured in a seeded projection."""
    g = torch.Generator().manual_seed(SEED)
    proj = patches @ torch.randn(patches.shape[1], PROJ_DIM, generator=g)
    sq = (proj * proj).sum(dim=1)
    idx = torch.empty(int(len(patches) * ratio), dtype=torch.long)
    idx[0] = torch.randint(len(patches), (1,), generator=g)
    min_d = torch.full((len(patches),), float("inf"))
    for i in range(1, len(idx)):
        last = idx[i - 1]
        min_d = torch.minimum(min_d, sq - 2 * (proj @ proj[last]) + sq[last])   # squared distance, one matvec
        idx[i] = torch.argmax(min_d)
    return patches[idx]


def fit_bank(feats):
    """(N, C, 28, 28) features -> coreset bank (ratio * N * 784, C). Takes features so INT8 ONNX features reuse it."""
    return greedy_coreset(feats.permute(0, 2, 3, 1).reshape(-1, feats.shape[1]))


def nn_score(feats, bank):
    """Each patch's distance to its nearest bank row -> (image scores (n,), maps (n, 28, 28)); image = max patch."""
    n, c, h, w = feats.shape
    maps = torch.cdist(feats.permute(0, 2, 3, 1).reshape(-1, c), bank).min(dim=1).values.view(n, h, w)
    return maps.amax(dim=(1, 2)), maps


def score(x, extractor, bank):
    """Differentiable in x (no no_grad, no detach, no numpy): (image scores, 28x28 maps)."""
    return nn_score(extractor(x), bank)


def ort_session(path):
    """ONNX Runtime CPU session with the benchmark threading: 4 intra-op threads, 1 inter-op thread."""
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_features(sess, images):
    """Batch-1 ONNX features for (n, 3, 224, 224) images in [0, 1] -> torch tensor (n, C, 28, 28). Not differentiable."""
    name = sess.get_inputs()[0].name
    return torch.from_numpy(np.concatenate([sess.run(None, {name: images[i:i + 1].numpy()})[0]
                                            for i in range(len(images))]))


def main():
    torch.manual_seed(SEED)
    extractor = FeatureExtractor()
    fit = load_images(load_split("fit"))
    t0 = time.perf_counter()
    with torch.inference_mode():
        feats = torch.cat([extractor(fit[i:i + 16]) for i in range(0, len(fit), 16)])
        t1 = time.perf_counter()
        bank = fit_bank(feats)
    t2 = time.perf_counter()
    BANK_FP32.parent.mkdir(exist_ok=True)
    np.save(BANK_FP32, bank.numpy())
    n_patches = feats.shape[0] * feats.shape[2] * feats.shape[3]
    print(f"fit: {len(fit)} images -> {n_patches} patches of {feats.shape[1]} dims ({CONFIG['backbone']})")
    print(f"bank: {tuple(bank.shape)} float32, {BANK_FP32.stat().st_size / MB:.1f} MB, sha256 {sha256_of(BANK_FP32)}")
    print(f"fit time: {t2 - t0:.1f} s (features {t1 - t0:.1f} s, coreset {t2 - t1:.1f} s)")

    bank = torch.from_numpy(np.load(BANK_FP32))   # reload: smoke-test the saved file, as a normal (non-inference) tensor
    val = load_images(load_split("val"))
    scores = torch.cat([score(val[i:i + 4], extractor, bank)[0] for i in range(0, len(val), 4)])
    print(f"val normals ({len(scores)}): score min {scores.min():.4f}  mean {scores.mean():.4f}  max {scores.max():.4f}")

    x = val[:1].clone().requires_grad_(True)
    score(x, extractor, bank)[0].sum().backward()
    g = x.grad.abs().mean().item()
    assert g > 0 and np.isfinite(g), f"gradient check failed: mean |grad| = {g}"
    print(f"gradient check OK: val image 0, mean |d score / d x| = {g:.3e} (nonzero, finite)")


if __name__ == "__main__":
    main()
