"""play/rlplay.py (numpy, for the browser) computes exactly c/rl.c's features and picks exactly the C net's move:
C games are replayed in the Python reference engine and compared at every recorded decision."""
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "play"))

from rl.core import GA, NET, PASS, Seat, base_genome, play, random_kingdom  # noqa: E402


def _replay(k, seed, rec, check):
    import rlplay
    from dominion_engine.game import Game
    g = Game(k, seed)
    gen = g.run()
    j = 0
    try:
        d = next(gen)
        while True:
            nopt = len(d.options) + bool(d.may_pass)
            if nopt == 1:
                ans = d.options[0] if d.options else -1
            else:
                assert rec["KIND"][j] == d.kind and rec["SEAT"][j] == d.player, (j, d)
                check(j, g, d, rlplay)
                a = int(rec["A"][j])
                ans = -1 if a == PASS else a
                j += 1
            d = gen.send(ans)
    except StopIteration:
        pass
    assert j == len(rec["A"])


def test_features_equal_c():
    rng = random.Random(11)
    for i in range(12):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        r = play([Seat(GA, g, record=True), Seat(GA, g, record=True)], k, 1000 + i, rng_seed=7)
        rec = r["rec"]

        def check(j, game, d, rlplay):
            x = rlplay.features(game, d.player, d.kind, d.subject)
            assert np.array_equal(x, rec["X"][j]), (j, d, np.nonzero(x != rec["X"][j]))
        _replay(k, 1000 + i, rec, check)


def test_net_agent_plays_like_c():
    W = np.load(os.path.join(ROOT, "opponents", "rl_r2_2200.npy"))
    rng = random.Random(12)
    for i in range(6):
        k = random_kingdom(rng)
        g = base_genome(k)["ints"]
        r = play([Seat(NET, g, W, record=True), Seat(GA, g, record=True)], k, 2000 + i, rng_seed=9)
        rec = r["rec"]

        def check(j, game, d, rlplay):
            if d.player == 0:
                a = rlplay.NetAgent(W).decide(game, d)
                assert (PASS if a == -1 else a) == rec["A"][j], (j, d)
        _replay(k, 2000 + i, rec, check)
