"""Widen a trained net (Net2Net style) without changing what it plays.

The new hidden units get small random incoming weights and zero outgoing weights, so the wider net computes exactly
the same logits and win estimate as the original; training then decides what the new units are for.

    python experiments/widen.py runs/z1/net_800.pt runs/z1w_init.pt --nh 512
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.net import Net, load, save  # noqa: E402


def widen(src: Net, nh: int, seed: int = 0) -> Net:
    assert nh > src.nh
    torch.manual_seed(seed)
    dst = Net(nh)
    o = src.nh
    with torch.no_grad():
        dst.l1.weight[o:] *= 0.5                 # new first-layer units: default init, a bit smaller
        dst.l1.weight[:o] = src.l1.weight
        dst.l1.bias[:o] = src.l1.bias
        dst.l2.weight.mul_(0.5)
        dst.l2.weight[:o, :o] = src.l2.weight
        dst.l2.weight[:o, o:] = 0                # old second-layer units ignore the new first-layer units
        dst.l2.bias[:o] = src.l2.bias
        dst.pi.weight.zero_()
        dst.pi.weight[:, :o] = src.pi.weight     # new second-layer units start with no say in the output
        dst.pi.bias.copy_(src.pi.bias)
        dst.v.weight.zero_()
        dst.v.weight[:, :o] = src.v.weight
        dst.v.bias.copy_(src.v.bias)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--nh", type=int, default=512)
    a = ap.parse_args()
    src = load(a.src)
    dst = widen(src, a.nh)
    x = torch.randn(64, src.l1.in_features).abs()
    lg = torch.ones(64, src.pi.out_features, dtype=torch.bool)
    (l1, v1), (l2, v2) = src(x, lg), dst(x, lg)
    print("max logit difference", float((l1 - l2).abs().max()), "value", float((v1 - v2).abs().max()))
    save(dst, a.out)
    print("wrote", a.out, "and", a.out.replace(".pt", ".npy"))


if __name__ == "__main__":
    main()
