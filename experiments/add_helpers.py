"""Fork a run onto the helper inputs without changing its play: copy runs/<src> to runs/<dst> and give the net's
first layer zero weights for the helper inputs (rl_helpers in c/rl.c), padding Adam's state the same way, so that
`train.py --run <dst> --continue` goes on exactly where <src> was, with the new inputs available to learn from.

    python experiments/add_helpers.py z1 z1h
"""
from __future__ import annotations

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.core import NF, NF0  # noqa: E402


def pad(t: torch.Tensor) -> torch.Tensor:
    assert t.shape[1] == NF0, t.shape
    return torch.cat([t, torch.zeros(t.shape[0], NF - NF0, dtype=t.dtype)], 1)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    s, d = os.path.join("runs", src), os.path.join("runs", dst)
    assert not os.path.exists(d), d
    os.makedirs(d)
    for f in ("stats.csv", "events.jsonl", "config.json", "curriculum.json"):
        if os.path.exists(os.path.join(s, f)):
            shutil.copyfile(os.path.join(s, f), os.path.join(d, f))
    ck = torch.load(os.path.join(s, "ckpt.pt"), weights_only=False)
    ck["net"]["l1.weight"] = pad(ck["net"]["l1.weight"])
    keys = list(ck["net"].keys())                 # Adam's state is indexed in parameter order
    i = keys.index("l1.weight")
    st = ck["opt"]["state"][i]
    st["exp_avg"], st["exp_avg_sq"] = pad(st["exp_avg"]), pad(st["exp_avg_sq"])
    torch.save(ck, os.path.join(d, "ckpt.pt"))
    from rl.net import Net, save                  # the learner's current net, which the next iteration plays
    net = Net(ck["net"]["l1.weight"].shape[0], NF)
    net.load_state_dict(ck["net"])
    save(net, os.path.join(d, f"net_{ck['iter']}.pt"))
    print(f"{d}: iteration {ck['iter']}, first layer {tuple(ck['net']['l1.weight'].shape)}")


if __name__ == "__main__":
    main()
