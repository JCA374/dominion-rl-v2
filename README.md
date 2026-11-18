# dominion-rl-v2: Dominion from zero, by self-play

**🌐 Live site: https://jca374.github.io/dominion-rl-v2/**

The fourth method of the Dominion project. One network (298 inputs → 256 → 256 → 34 choices + a win estimate) makes
every decision of the game. It starts from **random weights**: no imitation, no rules. It learns with PPO, first only
against itself (stage 1, self-play), then against the three bots (stage 2), then against every earlier model: the GA
genome, az-v1 and rl-v1 (stage 3).

The question: is a teacher a head start or a ceiling? [rl-v1](https://github.com/JCA374/dominion-rl-v1) is the same
network and PPO, started as a copy of the GA genome; it scores 82.8% against the GA base genome and 71.7% head to head
against az-v1's best agent.

```
git clone --recursive https://github.com/JCA374/dominion-rl-v2 && cd dominion-rl-v2
python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy pytest pillow matplotlib
.venv/bin/python -m pytest -q tests                        # builds rl.so; 4 passed
.venv/bin/python -u train.py --run z1 --set games=1500     # stages in STAGES at the top of train.py
```

| Folder | What |
|---|---|
| `train.py` | stages, PPO, evaluation; writes `runs/<run>/` |
| `c/rl.c`, `rl/` | the agent in C, the ctypes bridge, the PyTorch net, game jobs and GAE |
| `experiments/` | goal checks, the once-only confirmation, the journal |
| `opponents/` | earlier models met in stage 3 (az-v1's best net, rl-v1's nets) |
| `engine/`, `az-v1/` | submodules: the rules, and az-v1 (with dominion-v1's genomes and bots) |
