"""Project hooks for publish.py (ml-project-site): the Play tab's game clips, pictures of the net for
The model tab, and the results file for the custom pages.

Clips use az-v1's drawing (one frame per buy of either player: both decks, the score, what was
bought, and for the network the probabilities it gave the top buys). Drawing lives here only; no
policy imports this file.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os

import numpy as np

from rl import core

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = "z1"
EVERY = 25                     # the Play tab's newest net moves every EVERY iterations (clips are committed)
N = core.NUM_CARDS
NAME = core.NAME

_spec = importlib.util.spec_from_file_location("az_site_hooks", os.path.join(ROOT, "az-v1", "site_hooks.py"))
AZH = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AZH)          # az-v1's frame(), encode_mp4(), GAMES and the model-tab figure helpers


def _nets(run=RUN):
    nets = glob.glob(os.path.join(ROOT, "runs", run, "net_*.npy"))
    return sorted((os.path.relpath(p, ROOT) for p in nets), key=lambda p: int(p.split("_")[-1][:-4]))


def models():
    """Iteration 0 (random), early self-play, the newest net, and rl-v1 (the target) for comparison."""
    nets = _nets()
    its = [int(p.split("_")[-1][:-4]) for p in nets]
    out = []
    if 0 in its:
        out.append({"id": "zero-0", "label": "Iteration 0 (random)", "path": nets[its.index(0)],
                    "note": "Random weights: every legal choice about equally likely. Loses every game."})
    for early in (10, 50):
        if early in its and its[-1] > early:
            out.append({"id": f"zero-{early}", "label": f"Self-play, iteration {early}", "path": nets[its.index(early)],
                        "note": f"After {early} iterations ({early * 1500:,} games) against itself only."})
    newest = [p for p in nets if int(p.split("_")[-1][:-4]) % EVERY == 0 and not p.endswith("net_0.npy")]
    if newest:
        it = int(newest[-1].split("_")[-1][:-4])
        if it not in (10, 50):
            out.append({"id": f"zero-{it}", "label": f"rl-v2 (iteration {it})", "path": newest[-1],
                        "note": "The newest network: everything it knows it learned from games, starting from nothing."})
    out.append({"id": "rl-v1", "label": "rl-v1 (r2, iteration 2,200)", "path": "opponents/rl_r2_2200.npy",
                "note": "The target: the same network, started as a copy of the GA genome. 82.8% against the GA base."})
    return out


def games():
    return AZH.games()


def _seat(path, kingdom):
    g = core.base_genome(kingdom)["ints"]
    if path == "ga":
        return core.Seat(core.GA, g), "GA base"
    if path == "az":
        return core.Seat(core.AZ_SEAT, g, np.load(os.path.join(ROOT, "opponents", "az_r2_357.npy")), sims=400), "AZ search"
    return core.Seat(core.NET, g, np.load(os.path.join(ROOT, path))), "RL net"


def render_play(policy_path, game, out_mp4):
    k = game["kingdom"]
    seat, label = _seat(policy_path, k)
    og = core.base_genome(k)["ints"] if game["opponent"] == "GA base" else core.bot(game["opponent"], k)
    seats = [seat, core.Seat(core.GA, og)]
    labels = [label, game["opponent"]]
    r = core.play(seats, k, game["seed"], rng_seed=game["seed"] + 1, trace=True)
    tr = r["trace"]
    frames, last = [], [None, None]
    for t in tr:
        me, coins, choice = int(t[0]), int(t[2]), int(t[3])
        what = "nothing" if choice == core.PASS else NAME[choice]
        line2 = ""
        pi = t[7 + 2 * N:]
        if me == 0 and label != "GA base" and pi.max() < 0.999:
            top = np.argsort(-pi)[:3]
            who = "search" if label == "AZ search" else "net"
            line2 = f"{who}: " + " · ".join(f"{'nothing' if a == core.PASS else NAME[a]} {pi[a] * 100:.0f}%"
                                            for a in top if pi[a] > 0.005)
            if label == "RL net":
                line2 += f"   win estimate {(float(t[6]) + 1) / 2 * 100:.0f}%"
        last[me] = (f"${coins}: buys {what}", line2)
        frames.append((AZH.frame(t, labels, k, last), 11))
    res = core.outcome(r, 0)
    word = "wins" if res > 0 else "loses" if res < 0 else "shares the win"
    sc = r["scores"]
    fin = f"{label} {word} {sc[0]}–{sc[1]} in {max(r['turns'])} turns"
    frames.append((AZH.frame(tr[-1], labels, k, last, final=fin), 75))
    AZH.encode_mp4(frames, out_mp4)
    return {"ok": res > 0, "result": "win" if res > 0 else "loss" if res < 0 else "shared", "detail": fin + "."}


# ── The model tab ───────────────────────────────────────────────────────────────────────────

def cached(out_png, source_path):
    """True when out_png is newer than the model file it is drawn from (so the figure need not be redrawn)."""
    return os.path.exists(out_png) and os.path.exists(source_path) and os.path.getmtime(out_png) >= os.path.getmtime(source_path)


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 13, 'axes.labelsize': 12, 'xtick.labelsize': 11,
                         'ytick.labelsize': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


# Figures are shown one per row, up to ~900 px wide: draw them 9 inches wide at 120 dpi (1080 px), fonts >= 11 pt.
def fig_arch(layers, out_png, title='Architecture'):
    """layers: [("input", 8), ("dense tanh", 64), ("dense tanh", 64), ("action logits", 4)] -> a box diagram with sizes."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.9 * len(layers)))
    ax.axis('off')
    for i, (name, n) in enumerate(layers):
        y = len(layers) - 1 - i
        w = 2.2 + 5 * (n / max(m for _, m in layers)) ** 0.5
        ax.add_patch(plt.Rectangle((5 - w / 2, y - 0.32), w, 0.64, fc='#dbe4ff', ec='#3b5bdb', lw=1.5))
        ax.text(5, y, f'{name}  ·  {n:,}', ha='center', va='center', fontsize=13)
        if i:
            ax.annotate('', (5, y + 0.34), (5, y + 0.66), arrowprops=dict(arrowstyle='->', color='#6b6b70', lw=1.5))
    ax.set_xlim(0, 10); ax.set_ylim(-0.6, len(layers) - 0.4)
    ax.set_title(title, fontsize=14)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def fig_weight_hist(named_arrays, out_png, title='Weights per layer'):
    """named_arrays: [("layer 1", ndarray), ...] -> one histogram per layer, two per row."""
    plt = _plt()
    import numpy as np
    n = len(named_arrays)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, a) in zip(axes, named_arrays):
        a = np.asarray(a).ravel()
        lim = float(np.percentile(np.abs(a), 99.5)) or 1
        ax.hist(np.clip(a, -lim, lim), bins=50, color='#3b5bdb')
        ax.set_title(f'{name}: {a.size:,} weights, std {a.std():.3f}')
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(title, fontsize=14); fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def fig_bars(labels, values, out_png, title='', ylabel='', colors=None, fmt='{:.2f}'):
    """Horizontal bars, first label on top, value written at the end of each bar: e.g. action probabilities in a fixed
    situation, or input sensitivity (how much the output moves when each input is nudged by one standard deviation).
    ylabel names the value axis."""
    plt = _plt()
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.36 * n))
    ys = list(range(n))[::-1]
    ax.barh(ys, values, color=colors or '#3b5bdb', height=0.7)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    top = max(values) if len(values) else 1
    for y, v in zip(ys, values):
        ax.text(v + top * 0.01, y, fmt.format(v), va='center', fontsize=10, color='#444')
    ax.set_xlim(0, top * 1.12 if top > 0 else 1)
    ax.set_xlabel(ylabel); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


KIND_SHOW = ["buy", "play action", "Militia discard", "Workshop gain", "Chapel trash", "Remodel gain"]


def model_viz(out_dir):
    """Architecture, weights, what it chooses per decision kind (vs the GA genome) and input
    sensitivity, for the newest net of RUN. Redrawn only when that net changes."""
    import torch
    from rl.net import load
    nets = _nets()
    if not nets:
        return []
    src = os.path.join(ROOT, nets[-1])
    pt = src.replace(".npy", ".pt")
    it = nets[-1].split("_")[-1][:-4]
    figs = [("arch", "Architecture", f"{RUN} net {it}: layer sizes read from the saved weights. One trunk, two heads: "
                                    "34 logits (a card, or pass) for whatever decision is asked, and a win estimate."),
            ("weights", "Weights per layer", "A wide, centred histogram means the layer is used; a spike at 0 means "
                                             "dead units."),
            ("agreement", "How often it does what the GA genome would do", "On positions from games against the GA "
             "base: the share of decisions, per kind, where the network's top choice equals the genome's. This network "
             "has never seen the genome's choices: any agreement is something it found on its own."),
            ("buys", "What it buys as self-play goes on", "For snapshots along the run: what the network buys "
             "(top choice) in 40 games against itself, as a share of all its buys. Iteration 0 buys anything; watch "
             "Curses and Coppers disappear and Silver, Gold and Provinces take over, then the action cards it decides "
             "are worth it."),
            ("sensitivity", "What it looks at", "How much its choices move (total variation, averaged over all recorded "
                                                "decisions) when one input is nudged by one standard deviation. The 25 "
                                                "most influential of the 298 inputs.")]
    paths = {k: os.path.join(out_dir, f"{k}.png") for k, _, _ in figs}
    entries = [{"src": f"model/{k}.png", "title": t, "caption": c} for k, t, c in figs]
    try:
        fig_buys(paths["buys"])
    except Exception as ex:                       # a figure must never stop the publisher
        print("buys figure failed:", ex)
    if all(cached(paths[k], src) for k in paths if k != "buys"):
        return entries
    net = load(pt)
    sd = net.state_dict()
    fig_arch([("input: what it knows at a decision", core.NF), ("dense ReLU", net.nh), ("dense ReLU", net.nh),
                  ("34 logits + 1 win estimate", core.NA + 1)], paths["arch"], title=f"Net {it}")
    fig_weight_hist([("W1", sd["l1.weight"].numpy()), ("W2", sd["l2.weight"].numpy()),
                         ("policy", sd["pi.weight"].numpy()), ("value", sd["v.weight"].numpy())], paths["weights"])
    # positions: the GA base genome's decisions (recorded) in the Play games and more
    import random
    rng = random.Random("rl-v2 model viz")
    rows = []
    for i in range(60):
        k = core.random_kingdom(rng)
        g = core.base_genome(k)["ints"]
        r = core.play([core.Seat(core.GA, g, record=True), core.Seat(core.GA, g, record=True)], k, rng.getrandbits(32))
        rows.append(r["rec"])
    X = np.concatenate([r["X"] for r in rows])
    L = np.concatenate([r["LEGAL"] for r in rows])
    A = np.concatenate([r["A"] for r in rows])
    K = np.concatenate([r["KIND"] for r in rows])
    with torch.no_grad():
        lg, _ = net(torch.from_numpy(X), torch.from_numpy(L).bool())
        prob = torch.softmax(lg, -1).numpy()
    pick = prob.argmax(1)
    names, vals = [], []
    for kname in core.KINDS:
        m = K == core.KINDS.index(kname)
        if m.sum() >= 20:
            names.append(f"{kname} ({m.sum()})")
            vals.append(float((pick[m] == A[m]).mean()))
    fig_bars(names, vals, paths["agreement"], title="Top choice = the GA genome's choice, per kind of decision", ylabel="share of decisions (number of decisions in brackets)", fmt="{:.0%}")
    std = X.std(0)
    eff = np.zeros(core.NF)
    with torch.no_grad():
        for j in range(core.NF):
            if std[j] == 0:
                continue
            X2 = X.copy()
            X2[:, j] += std[j]
            lg2, _ = net(torch.from_numpy(X2), torch.from_numpy(L).bool())
            eff[j] = 0.5 * np.abs(torch.softmax(lg2, -1).numpy() - prob).sum(1).mean()
    fn = feature_names()
    top = np.argsort(-eff)[:25]
    fig_bars([fn[j] for j in top], [eff[j] for j in top], paths["sensitivity"],
                 title="How much its choices move when one input rises by 1 std", ylabel="total variation (0 = no change, 1 = a different choice)", fmt="{:.3f}")
    return entries


BUY_GROUPS = [("nothing", "#bbbbbb"), ("Curse", "#7b3fa0"), ("Copper", "#b87333"), ("Estate", "#8fbf6f"),
              ("Silver", "#9aa5b1"), ("Gold", "#e0b000"), ("Duchy", "#4f9a3c"), ("Province", "#1f6a2a"),
              ("action cards", "#3b6fd8")]


def _buy_shares(path, games=40):
    """Shares of the net's buys (top choice, self-play) per group, cached per net file in runs/<run>/buys.json."""
    import random
    rng = random.Random("rl-v2 buys")
    cnt = {g: 0 for g, _ in BUY_GROUPS}
    W = np.load(os.path.join(ROOT, path))
    for _ in range(games):
        k = core.random_kingdom(rng)
        g = core.base_genome(k)["ints"]
        r = core.play([core.Seat(core.NET, g, W, record=True), core.Seat(core.NET, g, W, record=True)], k,
                      rng.getrandbits(32))
        rec = r["rec"]
        for a in rec["A"][rec["KIND"] == core.KINDS.index("buy")]:
            a = int(a)
            name = "nothing" if a == core.PASS else NAME[a]
            cnt[name if name in cnt else "action cards"] += 1
    tot = max(1, sum(cnt.values()))
    return {g: cnt[g] / tot for g in cnt}


def fig_buys(out_png):
    nets = [p for p in _nets() if int(p.split("_")[-1][:-4]) % EVERY == 0 or p.endswith("_10.npy")
            or p.endswith("_5.npy")]
    if not nets:
        return
    cache_p = os.path.join(ROOT, "runs", RUN, "buys.json")
    cache = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    changed = False
    for p in nets:
        if p not in cache:
            cache[p] = _buy_shares(p)
            changed = True
    if changed or not os.path.exists(out_png):
        json.dump(cache, open(cache_p, "w"))
        keep = sorted((p for p in cache if os.path.exists(os.path.join(ROOT, p))),
                      key=lambda p: int(p.split("_")[-1][:-4]))
        if len(keep) > 14:                         # at most 14 rows: first, last, evenly spaced between
            idx = np.linspace(0, len(keep) - 1, 14).round().astype(int)
            keep = [keep[i] for i in sorted(set(idx))]
        plt = _plt()
        fig, ax = plt.subplots(figsize=(9, 1.2 + 0.42 * len(keep)))
        labels = [f"iteration {int(p.split('_')[-1][:-4])}" for p in keep]
        left = np.zeros(len(keep))
        for g, col in BUY_GROUPS:
            v = np.array([cache[p][g] for p in keep])
            ax.barh(labels, v, left=left, color=col, label=g, edgecolor="white", linewidth=0.5)
            for i, (x, w) in enumerate(zip(left, v)):
                if w >= 0.07:
                    ax.text(x + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=10,
                            color="white" if g not in ("nothing", "Estate", "Silver", "Gold") else "black")
            left += v
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of all its buys in 40 self-play games")
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        fig.suptitle("What it buys, iteration by iteration", y=0.995)
        ax.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)
        fig.set_size_inches(9, 2.2 + 0.42 * len(keep))
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)


def feature_names():
    names = AZH.feature_names()
    names += [f"hand: {NAME[c]}" for c in range(N)] + [f"in play: {NAME[c]}" for c in range(N)]
    names += ["actions left", "my turn"] + [f"kind: {k}" for k in core.KINDS] + [f"about: {NAME[c]}" for c in range(N)]
    assert len(names) == core.NF
    return names


def extra(data_dir):
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        out[os.path.basename(p)[:-5]] = json.load(open(p))
    json.dump(out, open(os.path.join(data_dir, "results.json"), "w"), separators=(",", ":"))
