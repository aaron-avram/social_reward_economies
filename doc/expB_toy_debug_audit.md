# Experiment B Toy Debug Audit

## Scope

This note is the current source of truth for the toy Experiment B gate. Scaled Exp B and Exp D reruns remain paused until this toy diagnosis is explained.

Frozen toy configuration:

- static Experiment B
- `8` agents, `3` states, `2` actions
- `5000` steps
- fixed role updates every `100` steps
- `kappa = 0`
- `initial_actor_rate = 0.7`
- `initial_participant_rate = 0.7`
- paper-compliance path frozen at:
  - scoped `B(t)` gossip update
  - `s_i(k,0) = 0`
  - Eq. `(9)` averaging over `all_agents`
  - `leader_update_mode = participants_only_post_eq9`

This pass answers four toy-only questions in one integrated audit:

1. what quantity `s_i(k,t)` appears to be tracking,
2. where the ranking first diverges,
3. how the selected target feeds into the Step-1 follow signal,
4. whether the earliest failure is in learning, selection, or follow-gate conversion.

## Evidence Base

The one-pass diagnosis uses the six seed-`0` toy trace runs below. Each directory now contains the derived role-update tables:

- `expB_toy_alignment_by_update.csv`
- `expB_toy_step1_by_update.csv`

Trace directories:

- simple, `gamma = 2`:
  [reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07)
- simple, `gamma = 3`:
  [reputation_scaling_toy_static_simple_gamma3_seed0_trace](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma3/trace/reputation_scaling_toy_static_simple_gamma3_seed0_trace)
- simple, `gamma = 5`:
  [reputation_scaling_toy_static_simple_gamma5_seed0_trace](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma5/trace/reputation_scaling_toy_static_simple_gamma5_seed0_trace)
- Gaussian, `gamma = 2`:
  [reputation_scaling_toy_static_gaussian_gamma2_seed0_trace](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/shared_base_gaussian/gamma2/trace/reputation_scaling_toy_static_gaussian_gamma2_seed0_trace)
- Gaussian, `gamma = 3`:
  [reputation_scaling_toy_static_gaussian_gamma3_seed0_trace](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/shared_base_gaussian/gamma3/trace/reputation_scaling_toy_static_gaussian_gamma3_seed0_trace)
- Gaussian, `gamma = 5`:
  [reputation_scaling_toy_static_gaussian_gamma5_seed0_trace](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/shared_base_gaussian/gamma5/trace/reputation_scaling_toy_static_gaussian_gamma5_seed0_trace)

Representative derived tables:

- alignment-by-update example:
  [expB_toy_alignment_by_update.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_alignment_by_update.csv)
- Step-1-by-update example:
  [expB_toy_step1_by_update.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_step1_by_update.csv)
- `v -> s` transition example:
  [expB_toy_v_to_s_by_update.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_v_to_s_by_update.csv)
- step-by-step `v -> s` paper-vs-code audit:
  [expB_toy_v_to_s_recurrence_audit_long.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_v_to_s_recurrence_audit_long.csv)
- `s -> highest` transition example:
  [expB_toy_s_to_highest_by_update.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_s_to_highest_by_update.csv)
- dense true-vs-estimate trace example:
  [expB_true_rep_vs_estimate_trace_long.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_true_rep_vs_estimate_trace_long.csv)
- dense true-reputation decomposition example:
  [expB_true_reputation_decomposition_long.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_true_reputation_decomposition_long.csv)

## Stage-by-Stage Verdict

| Audited stage | Evidence used | Verdict |
| --- | --- | --- |
| true-reputation oracle | existing deterministic helper test plus toy snapshot traces | `oracle validated` |
| learned target of `s_i(k,t)` | `expB_toy_alignment_by_update.csv` and `expB_toy_v_to_s_by_update.csv` | `does not robustly track paper true reputation` |
| `s -> highest` selection path | `expB_toy_s_to_highest_by_update.csv` | `mechanically consistent, but broad tie sets remain` |
| Step-1 conversion | `expB_toy_step1_by_update.csv` across `gamma = 2, 3, 5` and both reward models | `mechanically consistent, secondary scale issue` |
| follower formation | existing follow graphs and timeline figures | `downstream symptom` |

## Oracle Validation

The current true-reputation helper remains a valid reference target.

Evidence:

- [test_true_reputation_helper_uses_theta_and_expected_group_utility](/Users/xia/social_reward_economies/code_by_peter_tests/test_reputation_status_rate_allocation.py#L288) still validates the helper numerically on a hand-checkable toy case.
- The helper in [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py) uses:
  - role-consistent executed policy,
  - actor interaction rate in `\theta(\mu_k) = 1 - e^{-\mu_k}`,
  - self-exclusion in the group utility sum,
  - uniform state averaging.

Verdict:

- `oracle validated`

## What The Learned Reputation Appears To Track

The toy alignment tables show that mean observed reputation does not consistently track the paper target `R_k`.

What the six seed-`0` runs show:

- At the first role update (`t = 100`), all six runs are already labeled `learning_target_mismatch`.
- In the simple reward runs, the earliest alignment target is always `sum_expected_utility`, not `true_reputation`.
- In the Gaussian reward runs, the earliest alignment target is always `theta_mu`.
- By the final role update, the dominant alignment target is unstable:
  - `other_or_none` for simple `gamma = 2` and `5`,
  - `mean_incoming_v` for simple `gamma = 3`,
  - `sum_expected_utility` for Gaussian `gamma = 2` and `5`,
  - `other_or_none` for Gaussian `gamma = 3`.

So the learned reputation signal is not just weakly separated; it is often aligning more strongly with a non-`R_k` quantity, or with no stable positive target at all.

The new `v -> s` tables sharpen that further:

- in all six seed-`0` runs, `mean_incoming_v_top_agent != observed_top_agent` already at `t = 100`,
- the final correlation between mean observed reputation and mean incoming `v` is also unstable:
  - negative in simple `gamma = 2`, simple `gamma = 5`, Gaussian `gamma = 2`, and Gaussian `gamma = 3`,
  - only weakly positive in simple `gamma = 3` and Gaussian `gamma = 5`,
- the final `dominant_v_alignment_target` is not stable across runs either.

So the unresolved part is now very specific: the toy evidence still does not tell a clean story in which `s_i(k,t)` is stably tracking `R_k`, stably tracking mean incoming `v`, or stably tracking a single alternate quantity.

## Step-By-Step `v -> s` Paper-Vs-Code Audit

The new recurrence audit checks the actual Phase-4 update against the paper formula at the first failing toy role updates using the recorded:

- active actor set,
- active participant set,
- observed utility matrix,
- `eta_v(t)`,
- `avg_s` term used in Eq. `(9)`,
- and the actual post-step `v` and `s` matrices.

For the baseline toy run:

- [expB_toy_v_to_s_recurrence_audit_long.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_v_to_s_recurrence_audit_long.csv)

Observed result:

- at `t = 100`:
  - `64 / 64` rows satisfy `v_matches_paper = True`
  - `64 / 64` rows satisfy `s_matches_paper = True`
- at `t = 200`:
  - `64 / 64` rows satisfy `v_matches_paper = True`
  - `64 / 64` rows satisfy `s_matches_paper = True`

So for the current toy baseline, the code is implementing the audited `v -> s` recurrence exactly as expected at the first failing role updates:

- active actors update `v_i(k,t)` toward the observed utility,
- inactive actors decay as expected,
- active participants update only the gossip targets,
- and the resulting `s_i(k,t)` values match `avg_s + delta_v` exactly.

This rules out a remaining implementation bug in the `v -> s` recurrence itself for the current toy baseline.

## Cross-Run Comparison

| Reward model | Gamma | Final top followers | First `true_top != observed_top` | First `observed_top != modal_highest` | First positive Step 1 | `t=100` dominant target | Final dominant target | Overall diagnosis |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `simple_preferred_action` | `2` | `0` | `100` | `200` | never | `sum_expected_utility` | `other_or_none` | `learning_target_mismatch` |
| `simple_preferred_action` | `3` | `0` | `100` | `200` | `100` | `sum_expected_utility` | `mean_incoming_v` | `learning_target_mismatch` |
| `simple_preferred_action` | `5` | `2` | `100` | `200` | `100` | `sum_expected_utility` | `other_or_none` | `learning_target_mismatch` |
| `shared_base_gaussian` | `2` | `0` | `900` | `100` | never | `theta_mu` | `sum_expected_utility` | `learning_target_mismatch` |
| `shared_base_gaussian` | `3` | `3` | `200` | `100` | `100` | `theta_mu` | `other_or_none` | `learning_target_mismatch` |
| `shared_base_gaussian` | `5` | `1` | `100` | `100` | `100` | `theta_mu` | `sum_expected_utility` | `learning_target_mismatch` |

Interpretation:

- Raising `gamma` helps Step 1 turn positive, but it does not remove the upstream learning-target mismatch.
- The first mismatch is not always the same comparison:
  - sometimes `true_top` and `observed_top` diverge first,
  - sometimes `observed_top` and modal highest estimate diverge first.
- But the common pattern across all six runs is that the alignment tables still diagnose the earliest role-update failure as `learning_target_mismatch`, not as a pure Step-1 or follower-assignment problem.

## Where The Ranking First Goes Wrong

The toy alignment tables make the first divergence explicit.

Consistent observations:

- All six seed-`0` runs are already diagnosed as `learning_target_mismatch` at `t = 100`.
- In the simple reward runs:
  - `true_top_agent != observed_top_agent` already at `t = 100`,
  - `observed_top_agent != modal_highest_rep_agent_estimate` by `t = 200`.
- In the Gaussian runs:
  - the first mismatch can start as `observed_top_agent != modal_highest_rep_agent_estimate` at `t = 100`,
  - with `true_top_agent != observed_top_agent` appearing later at `t = 200` or `900`.

That matters because it means the failure is not only “the selected target is wrong later.” In some runs the learned mean ranking itself is already off, and in the others the learned ranking may look plausible briefly while the agent-level highest-target estimates already diverge from it.

## `s -> highest` Selection Audit

The new `s -> highest` tables rule out a basic implementation error in the highest-target selection path.

From [expB_toy_s_to_highest_by_update.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_s_to_highest_by_update.csv) and the corresponding files for the other five runs:

- `share_highest_within_delta_set = 1.0` at every role update in every run,
- so the stored `highest_rep_agent_estimate` is always inside the admissible `Delta` tie set,
- but `share_highest_equals_row_argmax` is often low:
  - final value `0.0` in simple `gamma = 2`,
  - `0.125` in simple `gamma = 3`,
  - `0.0` in simple `gamma = 5`,
  - `0.375`, `0.375`, and `0.25` in the three Gaussian runs.

Interpretation:

- the selector itself is behaving consistently with the `Delta` tie rule,
- but the tie sets remain broad enough that the stored highest target often differs from the exact row argmax,
- so there is still a secondary `Delta`-breadth issue,
- but it is not the earliest failure stage because the upstream learned ranking is already misaligned before that tie-handling matters.

## Step-1 Conversion

The Step-1 tables show that signal scale still matters, but it is not the earliest consistent blocker.

Key facts:

- At `gamma = 2`, Step 1 never becomes positive in either reward model.
- At `gamma = 3` and `5`, Step 1 becomes positive immediately at `t = 100` in both reward models.
- Even then, the overall diagnosis remains `learning_target_mismatch` and seed `0` still fails to recover the old toy single-leader behavior.

Selected final Step-1 values from [expB_toy_step1_by_update.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_rates07/expB_toy_step1_by_update.csv) and the corresponding files in the other five trace directories:

- simple, `gamma = 2`:
  - final mean Step-1 margin `-0.6287`
  - final positive-share `0.0`
- simple, `gamma = 5`:
  - final mean Step-1 margin `-0.2028`
  - final positive-share `0.25`
- Gaussian, `gamma = 2`:
  - final mean Step-1 margin `-0.2700`
  - final positive-share `0.0`
- Gaussian, `gamma = 5`:
  - final mean Step-1 margin `-0.0080`
  - final positive-share `0.375`

So the signal can be made strong enough to start crossing PU, but the upstream ranking/use path is still not stable enough to produce the expected toy outcome.

The new Step-1 transition audit also rules out a basic signal-construction bug:

- `share_selected_reputation_matches_highest_row_value = 1.0` at every role update in all six runs,
- `share_weighted_signal_matches_gamma_times_selected = 1.0` at every role update in all six runs.

So the current code is mechanically doing this correctly:

1. take the current `highest_rep_agent_estimate`,
2. read the corresponding row value `s_i(L_i,t)`,
3. multiply by `gamma`,
4. compare against PU.

That means the remaining problem is not a broken plumbing link between selected target and Step 1. The problem is earlier: which target is being selected, and what quantity the underlying `s_i(k,t)` rows represent.

## Follower Formation

Follower-network failure is downstream of the earlier ranking/use mismatch.

Evidence:

- At `gamma = 2`, no toy follower graph forms meaningfully in either reward model.
- At higher `gamma`, follower edges do appear in some runs, but they still do not restore reliable single-leader convergence in seed `0`.
- That is consistent with the diagnosis above: Step 1 eventually starts to fire, but it is firing on a reputation signal whose alignment to `R_k` is still unstable.

## Post-Fix Status Of The Python `v_i(k,t)` Activity Bug

The Python Section `6.4.2` activity-check bug was real and is fixed, but it is no longer the lead explanation for the toy failure.

Recheck artifacts:

- fast-path rerun:
  [reputation_scaling_toy_static_simple_gamma2_seed0_trace_vfix_fast](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_vfix_fast)
- Python-path rerun:
  [reputation_scaling_toy_static_simple_gamma2_seed0_trace_vfix_python](/Users/xia/social_reward_economies/experiments/outputs/exp_b/toy/simple_preferred_action/gamma2/trace/reputation_scaling_toy_static_simple_gamma2_seed0_trace_vfix_python)

Observed result:

- the corrected Python path and the fast path remain behaviorally identical on the toy `gamma = 2` rerun,
- seed `0` still ends with `0` followers,
- and the earliest role-update diagnosis is still not moved off the learning-target bucket.

So that bug was a valid compliance fix, but not the dominant remaining cause.

## Final Root-Cause Label

`learning_target_mismatch`

Why this is the right label:

- it is the earliest failure-stage label at `t = 100` in all six seed-`0` trace runs,
- it continues to dominate the per-update diagnosis counts even when `gamma` is large enough for Step 1 to turn positive,
- and the learned signal does not robustly align with the paper-defined true-reputation target.

Secondary qualifier:

- this is still a `ranking + scale` problem in the broad sense,
- but the one-pass toy audit narrows the dominant first failure to the **learning target / ranking path**, not the Step-1 gate itself.

What has now been ruled out mechanically:

- a basic `v -> s` implementation bug,
- a basic `s -> highest` implementation bug,
- a basic `highest -> Step 1` signal-construction bug.

What remains open:

- what quantity the paper-compliant `v -> s` recurrence is actually converging toward,
- whether the paper-defined true-reputation oracle `R_k` is the right comparison target for `s_i(k,t)`,
- and whether the old repo expectation of toy single-leader convergence is compatible with the current paper-compliant interpretation.

## Next Code Target

The next audit should stay in the toy setting, but it is now a target-interpretation question rather than another low-level code-path question:

1. identify whether `s_i(k,t)` is converging toward a specific alternate target or an unstable mixture,
2. decide whether `R_k` is truly the paper quantity that `s_i(k,t)` should be expected to approximate here,
3. only after that reopen `Delta` calibration or parameter sweeps if still needed.

Still paused:

- `delta` tuning
- reward-structure sweeps
- scaled Exp B reruns
- scaled Exp D reruns
