"""The rl-v2 network as a Python agent for the Python reference engine: c/rl.c's 298 features and forward pass in
numpy, with no C. Runs in the browser (Pyodide) for the Play-against-it page; tests/test_play_parity.py checks
that the features equal the C ones on every decision of real games, so this agent plays exactly like the C net
(argmax).
"""
from __future__ import annotations

import numpy as np

from dominion_engine import rules
from dominion_engine.cards import NUM_CARDS, PROVINCE
from dominion_engine.game import NUM_KINDS

N = NUM_CARDS
AZ_NF = 5 * N + 10
NF = AZ_NF + 2 * N + 2 + NUM_KINDS + N          # 298
NA = N + 1                                       # a card id (or Sentry code), then PASS
PASS = N


# c/rl.c is built with -ffast-math, which turns x / 10.0f into x * (1 / 10.0f): the same here, to the bit.
R2, R3, R8, R10, R20 = (np.float32(1) / np.float32(d) for d in (2, 3, 8, 10, 20))


def _counts(cards):
    c = np.zeros(N, np.float32)
    for x in cards:
        c[x] += 1
    return c


def features(g, me: int, kind: int, subject: int) -> np.ndarray:
    p, o = g.players[me], g.players[1 - me]
    x = np.zeros(NF, np.float32)
    sup = np.array(g.supply, np.float32)
    x[0:N] = _counts(p.all_cards()) * R10
    x[N:2 * N] = _counts(p.deck) * R10
    x[2 * N:3 * N] = _counts(o.all_cards()) * R10
    x[3 * N:4 * N] = np.where(sup > 0, sup * R10, 0)
    x[4 * N:5 * N] = sup >= 0
    e = 5 * N
    vp, ovp = rules.score(p.all_cards()), rules.score(o.all_cards())
    x[e + 0] = g.coins * R8
    x[e + 1] = g.buys * R2
    x[e + 2] = p.turns * R20
    x[e + 3] = o.turns * R20
    x[e + 4] = vp * R20
    x[e + 5] = ovp * R20
    x[e + 6] = (vp - ovp) * R10
    x[e + 7] = max(g.supply[PROVINCE], 0) * R8
    x[e + 8] = rules.empty_piles(g.supply) * R3
    x[e + 9] = me == 0
    f = AZ_NF
    for c in p.hand:
        x[f + c] += np.float32(0.2)
    for c in p.play:
        x[f + N + c] += np.float32(0.2)
    x[f + 2 * N] = g.actions * R3 if g.current == me else 0
    x[f + 2 * N + 1] = g.current == me
    x[f + 2 * N + 2 + kind] = 1
    if 0 <= subject < N:
        x[f + 2 * N + 2 + NUM_KINDS + subject] = 1
    return x


class Net:
    """Flat float32 weights in c/rl.c's layout: W1 b1 W2 b2 Wp bp Wv bv."""

    def __init__(self, flat: np.ndarray):
        b = NF + NA + 4
        nh = int(round((-b + (b * b + 4 * (flat.size - NA - 1)) ** 0.5) / 2))
        i = 0

        def take(n, shape):
            nonlocal i
            a = flat[i:i + n].reshape(shape)
            i += n
            return a
        self.w1, self.b1 = take(nh * NF, (nh, NF)), take(nh, (nh,))
        self.w2, self.b2 = take(nh * nh, (nh, nh)), take(nh, (nh,))
        self.wp, self.bp = take(NA * nh, (NA, nh)), take(NA, (NA,))
        self.wv, self.bv = take(nh, (nh,)), take(1, (1,))
        assert i == flat.size
        self.nh = nh

    def forward(self, x):
        h = np.maximum(self.w1 @ x + self.b1, 0)
        h = np.maximum(self.w2 @ h + self.b2, 0)
        return self.wp @ h + self.bp, float(np.tanh(self.wv @ h + self.bv[0]))


class NetAgent:
    """Argmax over the legal answers, as the C NET seat in every evaluation. `last` keeps the probabilities of the
    last decision (for the page: what the network considered)."""

    def __init__(self, flat):
        self.net = Net(np.asarray(flat, np.float32))
        self.last = None

    def decide(self, game, d) -> int:
        opts = list(d.options) + ([PASS] if d.may_pass else [])
        if len(opts) == 1:
            return d.options[0] if d.options else -1
        lg, v = self.net.forward(features(game, d.player, d.kind, d.subject))
        legal = np.array(opts)
        z = lg[legal] - lg[legal].max()
        pr = np.exp(z) / np.exp(z).sum()
        a = int(legal[int(np.argmax(lg[legal]))])
        self.last = {"options": [int(o) for o in legal], "probs": [float(q) for q in pr], "value": v, "kind": d.kind}
        return -1 if a == PASS else a
