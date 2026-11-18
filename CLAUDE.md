# CLAUDE.md

rl-v2: PPO from a random network, self-play first (see README). Copied from rl-v1 (same C agent, net, jobs);
the differences are in `train.py`: `bc_games = 0`, `STAGES`, `pick_opponent`, `write_curriculum`.

- Run: `.venv/bin/python -u train.py --run z1 --set games=1500` (detached with setsid nohup; log in `logs/`).
  Resume: `--continue`. 3,000 games per iteration runs out of memory early (random games are long).
- Stop a run by pid: `pgrep -f "^.venv/bin/python -u train.py --run z1"`. Never `pkill -f` with a pattern that
  also matches your own shell command.
- Evaluation seeds are rl-v1's (comparable numbers). `experiments/confirm.py` has a new seed: use it once, at the end.
- Site: ml-project-site (`publish.py`, `site.json`, `site_hooks.py`, `docs/`), loop `publish_loop.sh`.
- Never put personal names or home paths on the site, in commits or podcasts.
