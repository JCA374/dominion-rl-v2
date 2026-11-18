import os
import random
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.core import GA, NET, AZ_SEAT, NF, NA, Seat, base_genome, lib, play, random_kingdom  # noqa: E402
from rl.net import Net  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_engine_pins_agree():
    """engine/ and az-v1's engine/ (and its dominion-v1's) are the same commit."""
    h = [subprocess.check_output(["git", "-C", os.path.join(ROOT, p), "rev-parse", "HEAD"]).strip()
         for p in ("engine", "az-v1/engine", "az-v1/dominion-v1/engine")]
    assert h[0] == h[1] == h[2]


def test_forward_matches_c():
    torch.manual_seed(0)
    net = Net(64)
    for p in net.parameters():
        torch.nn.init.normal_(p, 0, 0.3)
    flat = net.flat()
    x = np.random.default_rng(0).normal(size=NF).astype(np.float32)
    lg = np.zeros(NA, np.float32)
    v = lib.rl_forward(flat.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)), 64,
                       x.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)),
                       lg.ctypes.data_as(__import__("ctypes").POINTER(__import__("ctypes").c_float)))
    tl, tv = net(torch.from_numpy(x)[None])
    assert np.allclose(lg, tl[0].detach().numpy(), atol=1e-4)
    assert abs(v - float(tv[0])) < 1e-4


def test_ga_seats_replay_az_v1_games():
    import az.core as azc
    rng = random.Random(3)
    for _ in range(10):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        s = rng.getrandbits(32)
        a = play([Seat(GA, g), Seat(GA, g)], k, s)
        b = azc.play([azc.Seat(azc.GA, g), azc.Seat(azc.GA, g)], k, s)
        assert a["scores"] == b["scores"] and a["turns"] == b["turns"]


def test_net_seat_plays_legal_games_and_records():
    net = Net(64)
    flat = net.flat()
    rng = random.Random(4)
    for i in range(10):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        r = play([Seat(NET, g, flat, sample=True, record=True), Seat(GA, g)], k, i, nh=64)
        assert not r["error"]
        rec = r["rec"]
        assert len(rec["A"]) > 0 and (rec["SEAT"] == 0).all()
        assert rec["LEGAL"][np.arange(len(rec["A"])), rec["A"]].all()
        assert (rec["LOGP"] <= 0).all()


def test_mixed_sizes():
    """A 512-unit learner against a 256-unit opponent: each seat's size comes from its weights."""
    import random
    from rl.core import NET, Seat, base_genome, nh_of, play, random_kingdom
    from rl.net import Net
    big = Net(512).flat()
    small = Net(256).flat()
    assert nh_of(big) == 512 and nh_of(small) == 256
    rng = random.Random(3)
    for i in range(5):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        r = play([Seat(NET, g, big, sample=True, record=True), Seat(NET, g, small)], k, i)
        assert not r["error"] and len(r["rec"]["A"]) > 0
