# Social Reward Economies

Extends work done in https://github.com/DeathByThermodynamics/orchard-action-market

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
