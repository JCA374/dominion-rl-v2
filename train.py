"""rl-v2 training: PPO from a random network, no imitation. The opponents come in three stages:

    1 self-play   only itself and its own past snapshots (it never sees a hand-made strategy)
    2 bots        + the three bots (Big Money, Smithy-BM, Witch-BM)
    3 league      + the GA base genome, az-v1's best net and search, rl-v1's nets

A stage ends when its gate in the fixed evaluation is passed, or after its iteration budget.

    python train.py --run z1                 # new run in runs/z1/
    python train.py --run z1 --continue      # resume from runs/z1/ckpt.pt

Per iteration: `games` games with the learner sampling from its policy against the stage's
opponents, a PPO update on those decisions, then the fixed evaluation (argmax play; the same games
as rl-v1's): 400 games against the GA base genome, 300 against the bots. The evaluation is only
measured, never trained on, so stage 1 shows how far self-play alone gets against strategies it has
never met.
Writes runs/<run>/stats.csv, events.jsonl, net_<iter>.pt/.npy, ckpt.pt, config.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import random
import time

import numpy as np

CFG = {
    "nh": 256,
    "nf": 0,                 # inputs the net reads: 0 = all (with the helper inputs), 298 = rl-v1's only
    "bc_games": 0,           # 0: start from a random network (rl-v1 imitated 20,000 GA games here)
    "bc_epochs": 6,
    "games": 3000,           # PPO games per iteration
    "focus_share": 0.0,      # share of training games on kingdoms without the focus_exclude cards (kingdom curriculum)
    "focus_exclude": "Witch,Militia,Bandit,Bureaucrat,Gardens",
    "snapshot_every": 10,
    "snapshots": 8,
    "temp": 1.0,
    "lam": 0.95,             # GAE
    "clip": 0.2,
    "epochs": 3,
    "batch": 4096,
    "lr": 3e-4,
    "vf": 0.5,               # value loss weight
    "ent": 0.01,             # entropy bonus (after the schedule below)
    "ent_start": 0.0,        # >0: the entropy bonus starts here and falls linearly to ent over ent_iters iterations
    "ent_iters": 600,
    "max_kl": 0.03,          # stop the epoch loop early above this approximate KL
    "eval_kingdoms": 100,    # vs GA base: x 2 seeds x 2 seats = 400 games; bots: 25 x 2 x 2 x 3 = 300
    "workers": max(1, os.cpu_count() - 1),
    "iterations": 100000,
    "goal_every": 25,
    "vs_az_every": 100,      # goal checks with the 1,000 head-to-head games against az-v1's best (~3 min)
    "chunk": 50,
}
BOTS = ["Big Money", "Smithy-BM", "Witch-BM"]
AZ_NET = "opponents/az_r2_357.npy"
RL_V1 = ["opponents/rl_r2_200.npy", "opponents/rl_r2_2200.npy"]   # rl-v1's released net and its r2 at 2,200

# Opponent shares per stage (the rest: the current net, both seats learning). gate: (eval key, level)
# that ends the stage, or ("plateau", n): no new best against the GA base for n iterations (self-play
# runs until it stops improving on its own); budget: iterations after which it ends anyway.
STAGES = [
    {"name": "self-play", "shares": {"past": 0.2}, "gate": ("plateau", 60), "budget": 400,
     "expect": "From a random network to money play: buy Silver, Gold and Provinces, learned only from beating itself."},
    {"name": "bots", "shares": {"past": 0.15, "bots": 0.35}, "gate": ("bots", 0.7), "budget": 400,
     "expect": "Learns to punish the bots' fixed habits (Witch-BM's curses above all)."},
    {"name": "league", "shares": {"past": 0.15, "bots": 0.1, "ga": 0.2, "rlv1": 0.15, "aznet": 0.1,
                                  "azsearch": 0.05}, "gate": None, "budget": None,
     "expect": "Meets every earlier model; the target is rl-v1 (82% against the GA base, 72% against az-v1)."},
]


def chunks(games, n):
    return [games[i:i + n] for i in range(0, len(games), n)]


def _ppo(job):
    from rl.jobs import ppo_chunk
    return ppo_chunk(job)


def _bc(job):
    from rl.jobs import bc_chunk
    return bc_chunk(job)


def _eval(job):
    from rl.jobs import eval_chunk
    return eval_chunk(job)


def eval_games(cfg):
    from rl.core import kingdom_for_bot, random_kingdom
    rng = random.Random("rl-v1 eval")
    games = []
    for _ in range(cfg["eval_kingdoms"]):
        k = random_kingdom(rng)
        for _ in range(2):
            s = rng.getrandbits(32)
            games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "ga", "tag": "GA base"}} for seat in (0, 1)]
    for b in BOTS:
        for _ in range(cfg["eval_kingdoms"] // 4):
            k = kingdom_for_bot(rng, b)
            for _ in range(2):
                s = rng.getrandbits(32)
                games += [{"kingdom": k, "seed": s, "seat": seat, "opp": {"kind": "bot", "name": b, "tag": b}}
                          for seat in (0, 1)]
    return games


def run_eval(pool, cfg, agent):
    jobs = [{"games": c, "agent": agent, "nh": cfg["nh"]} for c in chunks(eval_games(cfg), cfg["chunk"])]
    res = [r for part in pool.imap_unordered(_eval, jobs) for r in part]
    out = {}
    for tag in ["GA base"] + BOTS:
        pts = [r["points"] for r in res if r["opp"] == tag]
        out[tag] = float(np.mean(pts))
    out["bots"] = float(np.mean([out[b] for b in BOTS]))
    out["turns"] = float(np.mean([r["turns"] for r in res]))
    out["capped"] = float(np.mean([r["capped"] for r in res]))
    return out


def fmt(e):
    return f"GA {e['GA base']:.3f} BM {e['Big Money']:.3f} Smithy {e['Smithy-BM']:.3f} Witch {e['Witch-BM']:.3f} bots {e['bots']:.3f}"


class Log:
    def __init__(self, d):
        self.d = d

    def event(self, kind, **kw):
        with open(os.path.join(self.d, "events.jsonl"), "a") as f:
            f.write(json.dumps({"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind, **kw}) + "\n")
        print(f"[{kind}]", json.dumps(kw)[:300], flush=True)

    def row(self, row):
        path = os.path.join(self.d, "stats.csv")
        new = not os.path.exists(path)
        fields = list(row)
        if not new:                                  # a resumed run keeps its file's columns (new ones are dropped)
            with open(path) as f:
                fields = next(csv.reader(f))
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def ppo_update(net, opt, data, cfg, torch):
    """Clipped PPO on one iteration's decisions. Returns mean metrics."""
    n = len(data["A"])
    X = torch.from_numpy(data["X"])
    L = torch.from_numpy(data["LEGAL"]).bool()
    A = torch.from_numpy(data["A"]).long()
    OLD = torch.from_numpy(data["LOGP"])
    adv = data["ADV"]
    ADV = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8))
    RET = torch.from_numpy(data["RET"])
    stats = []
    stop = False
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n)
        for i in range(0, n, cfg["batch"]):
            j = perm[i:i + cfg["batch"]]
            logits, v = net(X[j], L[j])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, A[j][:, None]).squeeze(1)
            ratio = torch.exp(logp - OLD[j])
            s1 = ratio * ADV[j]
            s2 = torch.clamp(ratio, 1 - cfg["clip"], 1 + cfg["clip"]) * ADV[j]
            pg = -torch.min(s1, s2).mean()
            p = logp_all.exp()
            ent = -(p * torch.where(L[j], logp_all, torch.zeros_like(logp_all))).sum(-1).mean()
            vl = ((v - RET[j]) ** 2).mean()
            loss = pg + cfg["vf"] * vl - cfg["ent_now"] * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                kl = (OLD[j] - logp).mean().item()
                clipfrac = ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            stats.append({"pg_loss": pg.item(), "value_loss": vl.item(), "entropy": ent.item(), "kl": kl,
                          "clipfrac": clipfrac})
            if kl > cfg["max_kl"]:
                stop = True
                break
        if stop:
            break
    out = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
    out["epochs_done"] = ep + 1
    return out


def bc_train(net, data, cfg, torch, log):
    """Imitation of the GA genome: cross-entropy on its choice + 0.5 x value error on the result.
    Batches index the numpy arrays (no copies: the data is ~1 GB)."""
    n = len(data["A"])
    idx = np.arange(n)
    hold = idx[idx % 10 == 0][:200000]
    train_idx = idx[idx % 10 != 0]
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    def batch(j):
        return (torch.from_numpy(data["X"][j]), torch.from_numpy(data["LEGAL"][j]).bool(),
                torch.from_numpy(data["A"][j]).long(), torch.from_numpy(data["Z"][j]))

    for ep in range(cfg["bc_epochs"]):
        perm = np.random.default_rng(ep).permutation(train_idx)
        for i in range(0, len(perm), cfg["batch"]):
            x, lg, a, z = batch(np.sort(perm[i:i + cfg["batch"]]))
            logits, v = net(x, lg)
            loss = torch.nn.functional.cross_entropy(logits, a) + 0.5 * ((v - z) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            x, lg, a, z = batch(hold)
            logits, v = net(x, lg)
            pick = logits.argmax(1)
            buy = torch.from_numpy(data["KIND"][hold] == 1)
            res = {"acc": (pick == a).float().mean().item(), "acc_buys": (pick == a)[buy].float().mean().item(),
                   "acc_other": (pick == a)[~buy].float().mean().item(),
                   "ce": torch.nn.functional.cross_entropy(logits, a).item(), "value_mse": ((v - z) ** 2).mean().item()}
        log.event("bc_epoch", epoch=ep, samples=n, holdout=res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r1")
    ap.add_argument("--continue", dest="cont", action="store_true")
    ap.add_argument("--init_from", default=None)
    ap.add_argument("--stage", type=int, default=0, help="with --init_from: the stage to start in (0-based)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    from rl.core import kingdom_for_bot, random_kingdom
    from rl.core import NF
    from rl.net import Net, load, save

    d = os.path.join("runs", args.run)
    os.makedirs(d, exist_ok=True)
    log = Log(d)
    cfg = dict(CFG)
    if args.cont and os.path.exists(os.path.join(d, "config.json")):
        cfg.update(json.load(open(os.path.join(d, "config.json"))))
    for kv in args.set:
        k, v = kv.split("=")
        cfg[k] = type(CFG[k])(v)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=1)
    write_curriculum(d)
    pool = mp.Pool(cfg["workers"])
    t0 = time.time()

    def snap(it):
        return os.path.join(d, f"net_{it}.npy")

    if args.cont:
        ck = torch.load(os.path.join(d, "ckpt.pt"), weights_only=False)
        net = Net(cfg["nh"], ck["net"]["l1.weight"].shape[1])
        net.load_state_dict(ck["net"])
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        opt.load_state_dict(ck["opt"])
        for g in opt.param_groups:                 # the config's learning rate wins over the saved one
            g["lr"] = cfg["lr"]
        it, snaps = ck["iter"] + 1, ck["snaps"]
        stage, stage_start = ck.get("stage", 0), ck.get("stage_start", 1)
        log.event("run_resume", iter=it, stage=STAGES[stage]["name"])
    else:
        log.event("run_start", config=cfg, init_from=args.init_from)
        if args.init_from:
            net = load(args.init_from)
        elif not cfg["bc_games"]:
            torch.manual_seed(0)
            net = Net(cfg["nh"], cfg["nf"] or NF)
        else:
            net = Net(cfg["nh"], cfg["nf"] or NF)
            rng = random.Random("rl-v2 bc")
            games = []
            for i in range(cfg["bc_games"]):
                if i % 2 == 0:
                    games.append({"kingdom": random_kingdom(rng), "seed": rng.getrandbits(32)})
                else:
                    b = BOTS[i // 2 % 3]
                    games.append({"kingdom": kingdom_for_bot(rng, b), "seed": rng.getrandbits(32), "bot": b,
                                  "seat": i // 6 % 2})
            tb = time.time()
            parts = list(pool.imap_unordered(_bc, [{"games": c} for c in chunks(games, cfg["chunk"])]))
            data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
            log.event("bc_data", games=len(games), decisions=int(len(data["A"])), secs=round(time.time() - tb))
            bc_train(net, data, cfg, torch, log)
            del data
        save(net, os.path.join(d, "net_0.pt"))
        opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
        e = run_eval(pool, cfg, {"kind": "net", "path": snap(0)})
        log.event("init_eval", eval=fmt(e))
        log.row(make_row(0, t0, {}, {}, e))
        it, snaps = 1, [snap(0)]
        stage, stage_start = args.stage, 1
        log.event("stage_start", stage=STAGES[stage]["name"], index=stage, iter=1, expect=STAGES[stage].get("expect", ""))
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": 0, "snaps": snaps,
                    "stage": stage, "stage_start": stage_start}, os.path.join(d, "ckpt.pt"))

    stage_best = [-1.0, it - 1]              # best eval vs GA within the current stage, and its iteration
    best_eval = [-1.0]
    if os.path.exists(os.path.join(d, "stats.csv")):
        with open(os.path.join(d, "stats.csv")) as f:
            best_eval[0] = max(float(r["eval_GA_base"]) for r in csv.DictReader(f))
    while it <= cfg["iterations"]:
        ti = time.time()
        learner = snap(it - 1)
        rng = random.Random(f"rl-v2 ppo {args.run} {it}")
        shares = STAGES[stage]["shares"]
        games = []
        for i in range(cfg["games"]):
            u, seat, opp = rng.random(), i % 2, None
            for kind, share in shares.items():
                if u < share:
                    opp = pick_opponent(kind, i, rng, snaps)
                    break
                u -= share
            if opp is None or (opp["kind"] == "net" and opp["path"] is None):
                opp = {"kind": "net", "path": learner, "tag": "self"}
            if opp["kind"] == "bot":
                k = kingdom_for_bot(rng, opp["name"])
            elif rng.random() < cfg["focus_share"]:
                k = focus_kingdom(rng, cfg["focus_exclude"].split(","))
            else:
                k = random_kingdom(rng)
            games.append({"kingdom": k, "seed": rng.getrandbits(32), "seat": seat, "opp": opp})
        jobs = [{"games": c, "learner": learner, "nh": cfg["nh"], "temp": cfg["temp"], "lam": cfg["lam"]}
                for c in chunks(games, cfg["chunk"])]
        parts = list(pool.imap_unordered(_ppo, jobs))
        res = [r for p in parts for r in p["results"]]
        data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "results"}
        t_play = time.time() - ti
        sp = {"selfplay_vs_ga": float(np.mean([r["points"] for r in res if r["opp"] == "GA base"] or [float("nan")])),
              "selfplay_vs_bots": float(np.mean([r["points"] for r in res if r["opp"] in BOTS] or [float("nan")])),
              "selfplay_vs_past": float(np.mean([r["points"] for r in res if r["opp"] == "past"] or [float("nan")])),
              "selfplay_vs_aznet": float(np.mean([r["points"] for r in res if r["opp"] == "AZ net"] or [float("nan")])),
              "selfplay_vs_azsearch": float(np.mean([r["points"] for r in res if r["opp"] == "AZ search"] or [float("nan")])),
              "selfplay_vs_rlv1": float(np.mean([r["points"] for r in res if r["opp"] == "rl-v1"] or [float("nan")])),
              "decisions": int(len(data["A"])),
              "value_mse_fresh": float(((data["V"] - data["Z"]) ** 2).mean()),
              "t_play": t_play}
        tt = time.time()
        net.train()
        cfg["ent_now"] = entropy_bonus(cfg, it)
        m = ppo_update(net, opt, data, cfg, torch)
        m["ent_coef"] = cfg["ent_now"]
        sp["t_train"] = time.time() - tt
        save(net, os.path.join(d, f"net_{it}.pt"))
        e = run_eval(pool, cfg, {"kind": "net", "path": snap(it)})
        sp["t_iter"] = time.time() - ti
        sp["stage"] = stage + 1
        log.row(make_row(it, t0, m, sp, e))
        print(f"iter {it}: {fmt(e)} | train vs GA {sp['selfplay_vs_ga']:.3f} kl {m['kl']:.4f} ent {m['entropy']:.3f} "
              f"| {sp['t_iter']:.0f}s", flush=True)
        if e["GA base"] > best_eval[0]:
            best_eval[0] = e["GA base"]
            log.event("goal_best", iter=it, check={"score": round(e["GA base"], 4), "n": 4 * cfg["eval_kingdoms"]},
                      champion=snap(it), eval=fmt(e))
        if it % cfg["snapshot_every"] == 0:
            snaps = (snaps + [snap(it)])[-cfg["snapshots"]:]
        for old in range(max(0, it - 30), it - 2):          # keep snapshots, goal nets and the last two
            p = os.path.join(d, f"net_{old}.npy")
            if old % cfg["snapshot_every"] and old % cfg["goal_every"] and os.path.exists(p):
                os.remove(p)
                os.remove(p.replace(".npy", ".pt"))
        st = STAGES[stage]
        if e["GA base"] > stage_best[0]:
            stage_best[:] = [e["GA base"], it]
        if stage + 1 < len(STAGES):
            key, level = st["gate"]
            if key == "plateau":
                passed = it - stage_best[1] >= level
                why = f"no new best against the GA base for {level} iterations (best {stage_best[0]:.3f} at {stage_best[1]})"
            else:
                passed = e[key] >= level
                why = f"eval {key} {e[key]:.3f} >= {level}"
            capped = not passed and it - stage_start + 1 >= st["budget"]
            if passed or capped:
                log.event("stage_pass" if passed else "stage_cap", stage=st["name"], index=stage, iter=it,
                          why=why if passed else f"budget of {st['budget']} iterations used", champion=snap(it),
                          eval=fmt(e))
                stage, stage_start, stage_best[:] = stage + 1, it + 1, [-1.0, it]
                log.event("stage_start", stage=STAGES[stage]["name"], index=stage, iter=it + 1,
                          expect=STAGES[stage].get("expect", ""))
        torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "iter": it, "snaps": snaps,
                    "stage": stage, "stage_start": stage_start}, os.path.join(d, "ckpt.pt"))
        if it % cfg["goal_every"] == 0:
            import subprocess
            import sys
            subprocess.run([sys.executable, "experiments/goal_check.py", snap(it), "--name", f"{args.run}_iter{it}",
                            "--label", f"{args.run} iteration {it}", "--nh", str(cfg["nh"])]
                           + (["--vs_az"] if it % cfg["vs_az_every"] == 0 else []), check=False)
        it += 1


def write_curriculum(d):
    """runs/<run>/curriculum.json for the site's Curriculum tab, written before the stages run."""
    names = {"past": "own past snapshots", "bots": "the 3 bots", "ga": "GA base genome", "rlv1": "rl-v1's nets",
             "aznet": "az-v1 net", "azsearch": "az-v1 search"}
    stages = []
    for st in STAGES:
        opp = ", ".join(f"{names[k]} {v:.0%}" for k, v in st["shares"].items())
        rest = 1 - sum(st["shares"].values())
        g = st["gate"]
        stages.append({"name": st["name"], "world": f"opponents: itself {rest:.0%}, {opp}",
                       "reward": "+1 win, 0 shared win, -1 loss, at the end of the game only",
                       "pass": "never (the last stage)" if g is None else
                       f"no new best against the GA base (eval) for {g[1]} iterations" if g[0] == "plateau" else
                       f"eval against the {g[0]} at least {g[1]:.0%}",
                       "cap": f"{st['budget']} iterations" if st["budget"] else "none", "expect": st["expect"]})
    json.dump({"note": "PPO from a random network, no imitation. The evaluation games (GA base, bots) are only "
                       "measured, never trained on, until a stage adds those opponents.", "stages": stages},
              open(os.path.join(d, "curriculum.json"), "w"), indent=1)


def entropy_bonus(cfg, it):
    """The entropy coefficient at iteration `it`: ent_start falling linearly to ent over ent_iters (or ent)."""
    if cfg["ent_start"] <= 0:
        return cfg["ent"]
    f = max(0.0, 1 - it / cfg["ent_iters"])
    return cfg["ent"] + (cfg["ent_start"] - cfg["ent"]) * f


def focus_kingdom(rng, exclude):
    """10 kingdom cards drawn from those not in `exclude`: a kingdom where the excluded strategies are impossible."""
    import rl.core  # noqa: F401  (puts the engine on the path)
    from dominion_engine.cards import KINGDOM_CARDS
    return sorted(rng.sample([c for c in KINGDOM_CARDS if c not in exclude], 10))


def pick_opponent(kind, i, rng, snaps):
    """One opponent of the given league kind ("net" with path None: not available yet, play self)."""
    if kind == "past":
        return {"kind": "net", "path": rng.choice(snaps[:-1]) if len(snaps) > 1 else None, "tag": "past"}
    if kind == "bots":
        b = BOTS[i % 3]
        return {"kind": "bot", "name": b, "tag": b}
    if kind == "ga":
        return {"kind": "ga", "tag": "GA base"}
    if kind == "rlv1":
        return {"kind": "net", "path": rng.choice(RL_V1), "tag": "rl-v1"}
    if kind == "aznet":
        return {"kind": "az", "path": AZ_NET, "sims": 0, "tag": "AZ net"}
    if kind == "azsearch":
        return {"kind": "az", "path": AZ_NET, "sims": 400, "tag": "AZ search"}
    raise ValueError(kind)


def make_row(it, t0, m, sp, e):
    row = {"iter": it, "hours": round((time.time() - t0) / 3600, 4)}
    for k in ("pg_loss", "value_loss", "entropy", "kl", "clipfrac", "epochs_done", "ent_coef"):
        row[k] = round(m[k], 5) if k in m else ""
    for tag in ["GA base"] + BOTS + ["bots"]:
        row[f"eval_{tag.replace(' ', '_').replace('-', '_')}"] = round(e[tag], 4)
    row["eval_turns"] = round(e["turns"], 2)
    row["eval_capped"] = round(e["capped"], 4)
    row["stage"] = sp.get("stage", 1)
    for k in ("selfplay_vs_ga", "selfplay_vs_bots", "selfplay_vs_past", "selfplay_vs_aznet", "selfplay_vs_azsearch",
              "selfplay_vs_rlv1", "decisions", "value_mse_fresh",
              "t_play", "t_train", "t_iter"):
        row[k] = (round(sp[k], 4) if isinstance(sp[k], float) and not math.isnan(sp[k]) else sp[k]) if k in sp else ""
    return row


if __name__ == "__main__":
    main()
