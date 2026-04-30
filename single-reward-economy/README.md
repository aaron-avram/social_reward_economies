# Single Reward Economy — Legacy Subtree

This directory is **not on the runtime path** for the current experiment pipeline (`experiments/*.py → src/code_debugged.py`). It is kept for historical reference only.

---

## What is tracked and why

| Directory | Why kept |
|---|---|
| `norm/` | The closest thing to Alex He-Mo's original implementation. `train.py`, `norm_agent.py`, `norm.py`, and `norm_actor.py` are the reference for the old status/follower/interaction-rate logic. Useful for comparing old behavior against the current `src/` simulator. |
| `reputation_tests/` | Staged debugging tests for the old reputation/gossip/influencer logic. The README is a compact explanation of what each test validated. Good historical reference. |
| `reputation_toy_experiment/` | Minimal sandbox for the old γ-scaling and Eq. 12 reasoning. Good for historical intuition; conclusions should not be treated as authoritative for the current simulator. |
| `reputation_scaling/` | An older simplified Experiment-B-style implementation. `run_experiment.py` documents the assumptions of that earlier branch. Not the current experiment runner. |

## What is gitignored and why

| Directory | Why removed |
|---|---|
| `model/` | Redundant with `norm/`. `model/` is Alex He-Mo's code imported directly from his repo; `norm/` is the same files with audit comments added. Keeping both was redundant, and `norm/` is more annotated. |
| `personal_utility/` | Four standalone train scripts superseded by `experiments/pu_scaling.py`. No unique logic not covered by the current pipeline. |
| `experiments_results/` | Archival outputs from old Experiment 2 runs (PNGs, NPYs, `run_exp2.py`). The runner is superseded by `experiments/reputation_scaling.py`; the output artifacts are covered by the `*.png`/`*.npy` gitignore rules. |
