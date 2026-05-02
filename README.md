# Social Reward Economies

Extends work done in https://github.com/DeathByThermodynamics/orchard-action-market

---

## Repository Structure

```
.
├── src/
│   ├── code_debugged.py          # current simulator; all experiments run against this
│   └── code_old.py               # pre-fix baseline; kept for doc/BUGS.md reference
├── experiments/                  # experiment runners (A–D)
│   ├── pu_scaling.py             # Experiment A
│   ├── reputation_scaling.py     # Experiment B
│   ├── status_scaling.py         # Experiment C
│   └── perturbation_recovery.py  # Experiment D
├── code_by_peter_tests/          # test suite for code_debugged.py (see Tests below)
├── doc/
│   ├── BUGS.md                   # canonical 14-bug report: code_old.py → code_debugged.py
│   └── approach_by_peter.md      # guide to the code structure for new readers
└── single-reward-economy/        # legacy reference subtree; not on the active pipeline
    ├── norm/                     # Alex He-Mo's original implementation with audit comments
    ├── reputation_tests/         # staged debugging tests for old reputation/gossip logic
    ├── reputation_toy_experiment/ # minimal sandbox for old γ-scaling and Eq. 12 reasoning
    └── reputation_scaling/       # earlier simplified Experiment-B-style runner
```

The following directories are gitignored (local only): `single-reward-economy/model/` (Alex He-Mo's original files, superseded by `norm/`), `single-reward-economy/personal_utility/` (superseded by `experiments/pu_scaling.py`), `single-reward-economy/experiments_results/` (archival outputs), `gossip_test/` (early standalone gossip experiments), and `baseline_debug_reputation_interaction_2026-02-19/` (historical bug tests against `code_old.py`).

---

## Source Code

The active simulator is `src/code_debugged.py` — a single file implementing the full multi-agent model. The two main entry points for callers are `SystemConfig` (a dataclass holding all simulation parameters) and `MultiAgentSystem` (which runs the simulation via repeated calls to `step()`). Detailed in-code annotations will be added separately.

---

## Experiment Call Stacks

All four experiment scripts share the same call structure:

```
python3 experiments/<script>.py [args]
  → main()           # parses args, iterates over parameter grid and seeds
    → run_single()   # one (parameters, seed) combination
      → make_config() → SystemConfig
      → MultiAgentSystem(config)
      → for t in range(num_steps): system.step()
      → _finalize_results(system) → metrics dict
    → writes CSV and plots to --output-dir
```

Experiment-specific notes:

**Experiment A** (`pu_scaling.py`): γ=0, κ=0. `main()` sweeps `num_states` and `reward_model`.

**Experiment B** (`reputation_scaling.py`): κ=0 fixed; `main()` sweeps γ across seeds.

**Experiment C** (`status_scaling.py`): γ fixed; `main()` sweeps κ across seeds.

**Experiment D** (`perturbation_recovery.py`): γ and κ fixed. `run_single()` runs three sequential phases rather than a single loop: (1) `system.step()` until leader convergence is detected, (2) `apply_force_bad_action_perturbation()` injected each step for `perturb_duration` steps, (3) `system.step()` for `post_window` steps to measure recovery.

---

## Tests

`code_by_peter_tests/` contains 187 tests covering all 14 bugs documented in `doc/BUGS.md`. The suite is kept as a record of the debugging process and as a regression guard against changes to `code_debugged.py`. It is not expected to be extended as the simulator evolves; future development should modify `src/code_debugged.py` directly.

---

## Paper

This repository implements and replicates experiments from the report:

> **Norm Emergence and Rupture in a Simulated Social Reward Economy**

The paper "Learning Common Norms in Multi-Agent Systems", the ROP course final report, and related outputs (figures and tables) are not included in this repository. They are kept out of version control because the manuscript is not yet published and the outputs are a part of coursework submissions.

---

## Experiment A (Paper §4.1): Personal Utility Baseline

Script: `experiments/pu_scaling.py`

```bash
python3 experiments/pu_scaling.py \
  --mode static \
  --num-agents 100 \
  --num-states-list "10" \
  --num-actions 2 \
  --num-steps 50000 \
  --selected-seeds "0,1,2,3,4,5,6,7,8,9" \
  --reward-models "shared_base_gaussian" \
  --role-update-base-interval 3000 \
  --fixed-role-update-interval \
  --tracking-mode light \
  --initial-actor-rate 0.7 \
  --initial-participant-rate 0.7 \
  --reward-base-mu 0.5 \
  --reward-base-sigma 0.15 \
  --reward-agent-sigma 0.08 \
  --reward-clip-min 0.01 \
  --reward-clip-max 2.5 \
  --numpy-fast-path \
  --trace-seeds "0" \
  --trace-every 100 \
  --output-dir experiments/outputs/exp_a/final_pu_baseline/
```

**Key fixed parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| N | 100 | agents |
| S | 10 | social states |
| A | 2 | actions per state |
| γ | 0 | reputation disabled |
| κ | 0 | status disabled |
| role update interval | 3,000 steps | fixed |
| total steps | 50,000 | |
| seeds | 0–9 | 10 seeds |
| reward model | shared_base_gaussian | σ_base=0.15, σ_agent=0.08 |

---

## Experiment B (Paper §4.2): Reputation Scaling and Norm Emergence

Script: `experiments/reputation_scaling.py`

**γ sweep (seed 0) — produces follower-count-vs-γ table:**
```bash
python3 experiments/reputation_scaling.py \
  --mode static \
  --gammas "0,1,2,2.75,3,3.25,3.5,5.0" \
  --num-agents 100 \
  --num-states 10 \
  --num-actions 2 \
  --num-steps 50000 \
  --selected-seeds 0 \
  --kappa 0 \
  --delta 1e-6 \
  --role-update-base-interval 3000 \
  --fixed-role-update-interval \
  --tracking-mode light \
  --initial-actor-rate 0.7 \
  --initial-participant-rate 0.7 \
  --reward-model shared_base_gaussian \
  --reward-base-mu 0.5 \
  --reward-base-sigma 0.15 \
  --reward-agent-sigma 0.08 \
  --reward-clip-min 0.01 \
  --reward-clip-max 2.5 \
  --numpy-fast-path \
  --output-dir experiments/outputs/exp_b/gamma_sweep_seed0/
```

**Seed robustness check (10 seeds at γ=5):**
```bash
python3 experiments/reputation_scaling.py \
  --mode static \
  --gammas "5" \
  --num-agents 100 \
  --num-states 10 \
  --num-actions 2 \
  --num-steps 50000 \
  --selected-seeds "0,1,2,3,4,5,6,7,8,9" \
  --kappa 0 \
  --delta 1e-6 \
  --role-update-base-interval 3000 \
  --fixed-role-update-interval \
  --tracking-mode light \
  --initial-actor-rate 0.7 \
  --initial-participant-rate 0.7 \
  --reward-model shared_base_gaussian \
  --reward-base-mu 0.5 \
  --reward-base-sigma 0.15 \
  --reward-agent-sigma 0.08 \
  --reward-clip-min 0.01 \
  --reward-clip-max 2.5 \
  --numpy-fast-path \
  --output-dir experiments/outputs/exp_b/seed_robustness_g5/
```

**Key fixed parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| N | 100 | agents |
| S | 10 | social states |
| A | 2 | actions per state |
| κ | 0 | status disabled |
| B_R | 0.3 | hardcoded in script |
| δ | 1e-6 | tie-breaking threshold |
| role update interval | 3,000 steps | fixed |
| total steps | 50,000 | |
| reward model | shared_base_gaussian | σ_base=0.15, σ_agent=0.08 |

---

## Experiment C (Paper §4.3): Status Incentives and Leader Behavior

Script: `experiments/status_scaling.py`

```bash
python3 experiments/status_scaling.py \
  --mode static \
  --gamma 5 \
  --kappas "0,0.005,0.01,0.015,0.02,0.03,0.05" \
  --num-agents 100 \
  --num-states 10 \
  --num-actions 2 \
  --num-steps 50000 \
  --selected-seeds "0,1,2,3,4,5,6,7,8,9" \
  --delta 1e-6 \
  --role-update-base-interval 3000 \
  --fixed-role-update-interval \
  --tracking-mode full \
  --initial-actor-rate 0.7 \
  --initial-participant-rate 0.7 \
  --reward-model shared_base_gaussian \
  --reward-base-mu 0.5 \
  --reward-base-sigma 0.15 \
  --reward-agent-sigma 0.08 \
  --reward-clip-min 0.01 \
  --reward-clip-max 2.5 \
  --eq9-averaging-mode participants_only \
  --leader-update-mode participants_only_post_eq9 \
  --numpy-fast-path \
  --output-dir experiments/outputs/exp_c/final_status_sweep/
```

**Key fixed parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| N | 100 | agents |
| S | 10 | social states |
| A | 2 | actions per state |
| γ | 5 | fixed; strong reputation ensures leader emerges |
| δ | 1e-6 | tie-breaking threshold |
| B_R | 0.8 | follow threshold |
| B_F | 0.6 | unfollow threshold |
| c | 0.1 | status entry threshold; leader needs at least 10% followers |
| role update interval | 3,000 steps | fixed |
| total steps | 50,000 | |
| seeds | 0–9 | 10 seeds |
| reward model | shared_base_gaussian | σ_base=0.15, σ_agent=0.08 |

---

## Experiment D (Paper §4.4): Norm Collapse and Recovery

Script: `experiments/perturbation_recovery.py`

### Stage 1: Observational Baseline (γ=5, 10 seeds)

```bash
python3 experiments/perturbation_recovery.py \
  --mode static \
  --num-agents 100 \
  --num-states 10 \
  --num-actions 2 \
  --gamma 5 \
  --kappa 2 \
  --B-R 0.15 \
  --B-F 0.10 \
  --delta 1e-6 \
  --role-update-base-interval 6000 \
  --fixed-role-update-interval \
  --num-steps-max 44000 \
  --perturb-duration 24000 \
  --perturb-policy-mode force_bad_action \
  --perturb-strength 16 \
  --post-window 12000 \
  --reputation-shock-factor 1.0 \
  --reward-model shared_good_bad_heterogeneous \
  --reward-good-value 1.0 \
  --reward-bad-value 0.1 \
  --reward-agent-sigma 0.1 \
  --reward-order-gap 0.02 \
  --reward-clip-min 0.01 \
  --reward-clip-max 2.5 \
  --initial-actor-rate 0.7 \
  --initial-participant-rate 0.7 \
  --numpy-fast-path \
  --seeds 10 --seed-start 0 \
  --output-dir experiments/outputs/exp_d/observational_baseline/
```

### Stage 2: γ IV Sweep (γ ∈ {3, 4, 5}, seeds {0, 3, 9})

Run once per γ value, changing `--gamma` and `--output-dir`:

```bash
# γ=3
python3 experiments/perturbation_recovery.py \
  --mode static \
  --num-agents 100 \
  --num-states 10 \
  --num-actions 2 \
  --gamma 3 \
  --kappa 2 \
  --B-R 0.15 \
  --B-F 0.10 \
  --delta 1e-6 \
  --role-update-base-interval 6000 \
  --fixed-role-update-interval \
  --num-steps-max 44000 \
  --perturb-duration 24000 \
  --perturb-policy-mode force_bad_action \
  --perturb-strength 16 \
  --post-window 12000 \
  --reputation-shock-factor 1.0 \
  --reward-model shared_good_bad_heterogeneous \
  --reward-good-value 1.0 \
  --reward-bad-value 0.1 \
  --reward-agent-sigma 0.1 \
  --reward-order-gap 0.02 \
  --reward-clip-min 0.01 \
  --reward-clip-max 2.5 \
  --initial-actor-rate 0.7 \
  --initial-participant-rate 0.7 \
  --numpy-fast-path \
  --selected-seeds "0,3,9" \
  --output-dir experiments/outputs/exp_d/gamma_sweep/gamma3/

# γ=4: same command with --gamma 4 --output-dir .../gamma4/
# γ=5: same command with --gamma 5 --output-dir .../gamma5/
```

**Key fixed parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| N | 100 | agents |
| S | 10 | social states |
| A | 2 | actions per state |
| γ | 5 (Stage 1); 3,4,5 (Stage 2) | |
| κ | 2 | status enabled |
| B_R | 0.15 | follow threshold |
| B_F | 0.10 | unfollow threshold (B_F < B_R → hysteresis) |
| δ | 1e-6 | tie-breaking threshold |
| role update interval | 6,000 steps | fixed; doubled vs Exp B to allow collapse |
| total steps | 44,000 | |
| perturb duration | 24,000 steps | 4 role-update intervals |
| reward model | shared_good_bad_heterogeneous | requires designated bad action per state |
| good/bad rewards | N(1.0, 0.1) / N(0.1, 0.1) | order_gap=0.02 ensures separation |
