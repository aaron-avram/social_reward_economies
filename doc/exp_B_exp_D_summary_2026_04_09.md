# Experiment B / D Summary (2026-04-09)

## Scope

This note summarizes:

1. Code changes since the last commit `4d6a8c3` (`status scaliing`)
2. The confirmed working configuration for Experiment B
3. The best currently known configurations for Experiment D
4. Why some Experiment D seeds fail under different configurations
5. The observed effect of varying each important parameter

## Code Changes Since Last Commit

Tracked code changes since `4d6a8c3`:

- `src/code_debugged.py`
- `experiments/perturbation_recovery.py`
- `experiments/reputation_scaling.py`
- `code_by_peter_tests/_shared.py`
- `code_by_peter_tests/test_async_role_switching.py`
- `code_by_peter_tests/test_pu_gossip_role_switching.py`
- `code_by_peter_tests/test_reputation_status_rate_allocation.py`

High-level summary of what changed:

### 1. Core simulator (`src/code_debugged.py`)

- Added paper-aligned Step 1 role-update logic using
  - `gamma * selected_rep_raw > max(B_i, estimated_reward_pu)`
  - with hysteresis `B_R` / `B_F`
- Added `get_current_policy()` so followers use the leader's current behavior weights consistently
- Reworked reputation learning to match the paper's Eq. (9)-style structure:
  - update personal benefit estimates first
  - then update reputation as `avg gossip estimate + delta_v`
- Added broader diagnostics / audit infrastructure:
  - decision-audit support
  - small-`N` trace export helpers
  - checkpoint trace builders
  - true-reputation and consensus diagnostics
  - rate-audit helpers
- Added more tracking outputs for:
  - selected reputation
  - weighted selected reputation
  - estimated PU / reputation / status rewards
  - following history and follower counts
- Added and/or expanded support for reward-table based heterogeneous reward models used in the Exp B / Exp D sweeps

### 2. Perturbation harness (`experiments/perturbation_recovery.py`)

- Added CLI support for per-agent decision audit:
  - `--decision-audit-seeds`
  - `--decision-audit-role-update-steps`
- Added richer seed normalization / seed resolution helpers
- Added Step-1 diagnostic summarization and CSV export
- Added per-seed checkpoint leader summaries:
  - follower counts by leader
  - preleader follower count
  - mean selected reputation
  - mean PU
  - share passing Step 1
- Expanded run outputs to save:
  - `classifications.csv`
  - richer per-seed diagnostics
  - checkpoint summary CSVs for audited runs

### 3. Scaling harness (`experiments/reputation_scaling.py`)

- Expanded CLI and config coverage for gamma / sigma / reward sweeps
- Added much richer trace extraction and CSV writing
- Added toy / small-`N` debugging outputs, including:
  - gate-signal traces
  - choice traces
  - alignment summaries
  - follow-graph snapshots
  - consensus diagnostics
- Added richer aggregate reporting for the Exp B scaling sweeps

### 4. Test coverage

Tests were substantially expanded to cover:

- async role switching and scheduler audits
- paper-gossip scope handling
- Eq. (9)-style reputation updates
- active-zero-payoff vs inactive decay behavior
- fast-path vs python-path reputation learning agreement
- tie handling in highest-reputation selection
- actor-rate allocation rules
- trace writers and small-`N` audit outputs

In short: the codebase now has much stronger paper-alignment diagnostics, richer experiment harness outputs, and much better regression coverage for the role-switching / gossip / Step-1 gate logic that mattered in Experiments B and D.

## Experiment B

### Most Relaxed Robust Config

This is the best confirmed "minimal" robust Exp B configuration:

- `mode=static`
- `gamma=5`
- `initial_actor_rate=0.7`
- `initial_participant_rate=0.7`
- `reward_base_sigma=0.15`
- `reward_agent_sigma=0.08`
- `delta=1e-6`

Confirmed output:

- `experiments/outputs/exp_b/scaled/gammas_0_2_5_seeds_0_9_ir0p7_sigma0p08_base0p15_delta1e6`

Key result:

- `gamma=0`: `0/10` converged
- `gamma=2`: `0/10` converged
- `gamma=5`: `10/10` converged to `99` followers

This is the current recommended Exp B working point because it keeps the lower interaction rates and still converges robustly.

### Earlier Stronger-Rate Robust Config

Earlier robust success also existed with more aggressive interaction rates:

- `gamma=5`
- `initial_actor_rate=2.0`
- `initial_participant_rate=2.0`
- `reward_base_sigma=0.15`
- `reward_agent_sigma=0.02`
- `delta=1e-6`

Confirmed output:

- `experiments/outputs/exp_b/scaled/all10_reward_structure_delta1e6_g5_sigma002_base015`

This was useful as a proof of concept, but it is not the most relaxed working Exp B configuration.

## Experiment D

## Fixed Structural Baseline Used in the Best D Runs

Unless stated otherwise, the best D runs used:

- `mode=static`
- `100` agents, `10` states, `2` actions
- `num_steps_max=80000`
- `tracking_mode=light`
- `kappa=2`
- `delta=1e-6`
- `perturb_policy_mode=force_bad_action`
- `collapse_followers_on_perturb=False`
- `reputation_shock_factor=1.0`
- `role_update_base_interval=3000`
- `fixed_role_update_interval=True`
- `initial_actor_rate=0.7`
- `initial_participant_rate=0.7`
- `reward_model=shared_good_bad_heterogeneous`

## Best Current D Configurations

### 1. Best Safe Config

This is the best all-10 Experiment D configuration without front-end regression:

- `gamma=5`
- `reward_good_value=1.5`
- `reward_bad_value=0.1`
- `reward_order_gap=0.02`
- `reward_agent_sigma=0.05`
- `reward_clip_min=0.01`
- `reward_clip_max=2.5`
- `B_R=0.15`
- `B_F=0.10`
- `perturb_duration=18000`
- `perturb_strength=12`

Confirmed output:

- `experiments/outputs/exp_d/tuning/recovery_longdur_goodlift_20260407/bestdur_good1p5_all10`

Outcome:

- `8/10` full new-leader recoveries
- `0/10` same-leader recoveries
- `2/10` no-stable-recovery seeds
- `0/10` front-end regressions

Unresolved seeds:

- seed `2`: `no_stable_recovery`
- seed `3`: `no_stable_recovery`

This is still the best safe baseline because no later all-10 candidate improved beyond `8/10` without introducing a new problem.

### 2. Best Raw / Risky Config

This is the highest raw full-success count we observed, but it is not acceptable as a final config because it breaks collapse for one seed:

- `gamma=7.5`
- `reward_good_value=1.30`
- `reward_bad_value=0.1`
- `reward_order_gap=0.05`
- `reward_agent_sigma=0.05`
- `reward_clip_min=0.01`
- `reward_clip_max=2.5`
- `B_R=0.15`
- `B_F=0.10`
- `perturb_duration=18000`
- `perturb_strength=12`

Confirmed output:

- `experiments/outputs/exp_d/tuning/recovery_rewardshape_gamma_20260408/bestshape_all10_gamma7p5_corrected`

Outcome:

- `9/10` full new-leader recoveries
- `0/10` same-leader recoveries
- `0/10` no-stable-recovery seeds
- `1/10` front-end regression

Unresolved seed:

- seed `1`: `front_end_regression` because it never collapsed (`drop_fraction = 0.0`)

This is the best "risky" configuration because it solves almost everything, but it does so by making the incumbent too sticky in one seed.

### 3. Best New Reward-Shape Candidate From the Latest Safe-Base Pass

This was the best reward-shaping candidate from the latest conservative sweep:

- `gamma=5`
- `reward_good_value=1.25`
- `reward_bad_value=0.1`
- `reward_order_gap=0.05`
- `reward_agent_sigma=0.05`
- `reward_clip_min=0.10`
- `reward_clip_max=2.5`
- `B_R=0.15`
- `B_F=0.10`
- `perturb_duration=18000`
- `perturb_strength=12`

Confirmed output:

- `experiments/outputs/exp_d/tuning/reward_shape_safebase_20260408/bestshape_all10`

Outcome:

- `8/10` full new-leader recoveries
- `1/10` same-leader recovery
- `1/10` no-stable-recovery
- `0/10` front-end regressions

Unresolved seeds:

- seed `1`: `same_leader_recovery`
- seed `4`: `no_stable_recovery`

This candidate was not accepted because it did not improve beyond the `8/10` safe baseline; it only changed which seeds failed.

## Why Seeds Fail Under Different D Configurations

### A. Risky config: seed 1 fails to collapse

Config:

- `gamma=7.5`
- `good=1.30`
- `gap=0.05`
- `B_R=0.15`
- `B_F=0.10`
- `dur=18000`
- `str=12`

Audit outputs:

- `experiments/outputs/exp_d/tuning/risky_seed1_collapse_audit_20260408`

What happens:

- preleader follower count stays `99` at every audited checkpoint
- `mean_rep_to_preleader_raw ≈ 0.149-0.156`
- `mean_estimated_reward_pu ≈ 0.855`
- with `gamma=7.5`, weighted reputation is about `1.12-1.17`

Paper / code gate:

- existing follower stays following if
  - `gamma * s_i(L_i,t) > max(B_F, J_hat_i^pu)`

So here:

- `1.12-1.17 > max(0.10, 0.855) = 0.855`

Result:

- collapse never starts
- this is a front-end failure, not a recovery failure

Interpretation:

- the incumbent is too sticky because gamma-weighted reputation stays above PU throughout the perturbation window

### B. Best safe config: seeds 2 and 3 collapse but never re-enter

Config:

- `gamma=5`
- `good=1.5`
- `gap=0.02`
- `B_R=0.15`
- `B_F=0.10`
- `dur=18000`

Output:

- `experiments/outputs/exp_d/tuning/recovery_longdur_goodlift_20260407/bestdur_good1p5_all10`

Final diagnostics for the failed seeds:

- seed `2`
  - `mean_gamma_selected_rep = -0.026`
  - `mean_estimated_reward_pu = 0.100`
  - `mean_threshold = 0.150`
  - `share_positive_step1_margin = 0.0`
- seed `3`
  - `mean_gamma_selected_rep = -0.050`
  - `mean_estimated_reward_pu = 0.101`
  - `mean_threshold = 0.150`
  - `share_positive_step1_margin = 0.0`

Result:

- full collapse happens
- but post-collapse selected reputation turns negative
- everyone remains in PU
- no successor gets through Step 1

Interpretation:

- this is a successor-vacuum failure
- gamma cannot rescue it because the post-collapse reputation signal is already negative

### C. Threshold-limited re-entry failures

In earlier D runs with larger `B_R`, seeds such as `0,5,6,7` were threshold-limited:

- they had positive reputation signal
- but it stayed below the re-entry gate

This is why lowering `B_R` helped those seeds.

### D. Seed 9: signal-limited behavior

Seed `9` repeatedly behaved differently from the threshold-limited group:

- in several runs, it was below both the gate and PU
- in the aggressive `good=5, gamma=10` branch, the saved selected raw reputation was still negative

Implication:

- gamma is not a general rescue lever for seed `9`
- when `selected_rep_raw < 0`, increasing gamma only multiplies the wrong sign

## Effect of Varying Each Important Parameter in Experiment D

### `gamma`

Observed effect:

- low / moderate gamma is necessary for collapse
- higher gamma helps only when post-collapse selected reputation is already positive
- too much gamma makes incumbents sticky and can prevent collapse

Concrete evidence:

- `gamma=5` is the best safe level so far
- `gamma=7.5` can fix hard recovery seeds under some reward shapes
- but `gamma=7.5` also caused seed `1` not to collapse

Rule of thumb:

- increase gamma only when the saved post-collapse `selected_rep_raw > 0`
- do not use gamma to rescue a seed with negative post-collapse selected reputation

### `reward_good_value`

Observed effect:

- this is strongly non-monotone

Examples under the `B_R=0.15`, `B_F=0.10`, `dur=18000` branch:

- `good=1.0`
  - seeds `2,3` re-entered, but the old leader came back
  - seed `9` still failed
- `good=1.5`
  - seed `9` improved
  - seeds `2,3` became no-stable-recovery
- `good=2.0`
  - some hard seeds improved
  - but other seeds regressed

Interpretation:

- increasing `good` can strengthen successor signal
- but it can also create a sharper vacuum after collapse or make other seeds unstable

### `reward_order_gap`

Observed effect:

- increasing the enforced good-vs-bad gap from `0.02` to `0.05` helped challenge seeds under moderate `good`
- it made conservative reward shaping more effective than just raising `good` alone

Interpretation:

- `gap` is a cleaner way to increase discrimination between good and bad behavior without lifting the whole scale as aggressively

### `reward_agent_sigma`

Observed effect:

- lowering sigma from `0.08` to `0.05` was important for getting full convergence + full collapse in the Exp D base
- lowering below `0.05` did not improve the best safe all-10 result

Specific result on the best safe branch:

- `sigma=0.04` fixed seed `2`
- but broke seed `5`
- net result stayed `8/10`

Interpretation:

- lower sigma can strengthen shared successor agreement
- but too little heterogeneity can just move failure to different seeds

### `reward_clip_min`

Observed effect:

- raising the floor from `0.01` to `0.05`, then `0.10`, improved the focused challenge screen under
  - `good=1.25`
  - `gap=0.05`
- the best focused result used `clip_min=0.10`

But:

- on all 10 seeds, that still only matched `8/10`

Interpretation:

- a higher floor can soften destructive post-collapse updates
- but it can also soften incumbent displacement and shift the failure set rather than eliminate failure

### `B_R`

Observed effect:

- `B_R` is the main re-entry lever
- lowering it helped threshold-limited seeds re-enter

Examples:

- `B_R=0.30` was too strict for many re-entry-failure seeds
- `B_R=0.20` fixed seed `0`
- `B_R=0.16` fixed `5,6,7`, but was judged too close to `B_F=0.15`
- `B_R=0.15`, `B_F=0.10` became the clean threshold pair used in the best D branches

Interpretation:

- lower `B_R` opens re-entry
- but once PU is the binding side of the gate, lowering `B_R` further will not help

### `B_F`

Observed effect:

- `B_F` controls continuation hysteresis for existing followers
- in many hard cases, `PU` rather than `B_F` was the binding comparator

Interpretation:

- `B_F` matters most when an agent is already following and `B_F > PU`
- in several risky-collapse failures, the real blocker was `gamma * rep > PU`, not `gamma * rep > B_F`

### `perturb_duration`

Observed effect:

- extending perturbation from `9000` to `12000` did not fix the hard seeds
- challenge-seed sweep over `18000`, `24000`, `30000` tied under the baseline reward shape

Interpretation:

- longer perturbation alone is not the decisive lever for the current D failures
- if the incumbent remains attractive enough, simply waiting longer under perturbation is not sufficient

### `perturb_strength`

Observed effect:

- increasing strength from `12` to `16` did not solve the key same-leader / no-recovery cases

Interpretation:

- in the explored range, strength was weaker than reward shaping and thresholds

### `reward_bad_value`

Observed effect:

- lowering the bad reward further was not helpful in earlier tests
- collapse was already strong, so this did not address the actual bottleneck

Interpretation:

- this is not currently a priority tuning lever for D

## Bottom Line

### Experiment B

Recommended working config:

- `gamma=5`
- `initial_actor_rate=0.7`
- `initial_participant_rate=0.7`
- `reward_base_sigma=0.15`
- `reward_agent_sigma=0.08`
- `delta=1e-6`

### Experiment D

Current best safe config:

- `gamma=5`
- `reward_good_value=1.5`
- `reward_bad_value=0.1`
- `reward_order_gap=0.02`
- `reward_agent_sigma=0.05`
- `reward_clip_min=0.01`
- `reward_clip_max=2.5`
- `B_R=0.15`
- `B_F=0.10`
- `perturb_duration=18000`
- `perturb_strength=12`

Current best risky config:

- `gamma=7.5`
- `reward_good_value=1.30`
- `reward_bad_value=0.1`
- `reward_order_gap=0.05`
- `reward_agent_sigma=0.05`
- `reward_clip_min=0.01`
- `reward_clip_max=2.5`
- `B_R=0.15`
- `B_F=0.10`
- `perturb_duration=18000`
- `perturb_strength=12`

Best conservative reward-shape follow-up candidate:

- `gamma=5`
- `reward_good_value=1.25`
- `reward_bad_value=0.1`
- `reward_order_gap=0.05`
- `reward_agent_sigma=0.05`
- `reward_clip_min=0.10`
- `reward_clip_max=2.5`
- `B_R=0.15`
- `B_F=0.10`
- `perturb_duration=18000`
- `perturb_strength=12`

Status:

- no all-10 Experiment D configuration has yet beaten `8/10` full new-leader recoveries without introducing a new failure mode
