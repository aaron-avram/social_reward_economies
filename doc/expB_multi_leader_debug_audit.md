# Experiment B Multi-Leader Debug Audit

Note:

- The scaled `100`-agent audit below is no longer the earliest failure point in the current code path.
- A toy-first gate on the original `8`-agent, `3`-state Experiment B baseline now fails earlier, before follower formation begins at all.
- See [expB_toy_debug_audit.md](/Users/xia/social_reward_economies/doc/expB_toy_debug_audit.md) for the current small-N gate result. The latest one-pass toy audit identifies `learning_target_mismatch` as the dominant earliest failure stage, with Step-1 scale issues remaining secondary, so the scaling ladder and Gaussian reward-structure sweep remain paused pending a deeper toy-level ranking/use audit.

## Scope

This audit checks the high-`gamma` multi-leader issue in static Experiment B against the paper specification in [learning_paper_newest_ver.pdf](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver.pdf) only.

Audited configuration:

- static Experiment B
- `gamma = 5`
- `kappa = 0`
- `100` agents, `10` states, `2` actions
- fixed role updates every `3000` steps
- `50000` total steps
- seeds `0, 2, 7, 9`

Artifacts:

- main run summary: [reputation_scaling_runs_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9/reputation_scaling_runs_static.csv)
- seed diagnostic summary: [reputation_scaling_seed_diagnostic_summary_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9/reputation_scaling_seed_diagnostic_summary_static.csv)
- true-reputation checkpoints: [expB_true_reputation_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9/expB_true_reputation_checkpoints.csv)
- estimate-consensus checkpoints: [expB_estimate_consensus_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9/expB_estimate_consensus_checkpoints.csv)
- rate audit checkpoints: [expB_rate_audit_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9/expB_rate_audit_checkpoints.csv)

Observed outcomes in the audited seeds:

- seed `0`: full convergence to `99` followers
- seed `2`: partial convergence to `57` followers
- seed `7`: partial convergence to `41` followers
- seed `9`: fragmented outcome with `38` followers and `3` leader switches

## Issue 1: True-Reputation Ties

Paper reference:

- Section `7.3.1`
- `R_k(\pi_k,\mu_k)=\theta(\mu_k)\sum_{i\neq k} U_i(\pi_k)`
- `\theta(\mu)=1-e^{-\mu}`

Diagnostic:

- Added checkpoint-only true-reputation snapshots at every role-update epoch and at `t=50000`.
- Computed paper-aligned true reputation using each agent's current role-consistent policy and current actor interaction rate.

Observed result:

- In all `68` checkpoint snapshots (`4` seeds x `17` checkpoints), the true-top agent was unique.
- There were no exact top ties and no near-top ties under the implemented tolerances.
- The smallest positive final top-gap across the audited seeds was about `0.003774` and the smallest role-update top-gap was about `0.000111`.

Examples at the final checkpoint:

- seed `0`: true-top agent `75`, next-gap `0.004535`
- seed `2`: true-top agent `30`, next-gap `0.006229`
- seed `7`: true-top agent `11`, next-gap `0.005351`
- seed `9`: true-top agent `26`, next-gap `0.003774`

Verdict:

- `ruled out`

Interpretation:

- The multi-leader outcomes in this audit are not explained by ties in the paper-defined true reputation.
- Increasing heterogeneity is not the first move to make from this evidence.

## Issue 2: Estimate Consensus and Gossip Correctness

Paper reference:

- Section `7.3.3`
- observed-reputation learning should update only agents in
  `B(t) = \bigcup_{i \in A(t)} \{L_i(t)\}`

Diagnostic:

- At each role-update checkpoint and at `t=50000`, recorded for every observer:
  - current selected highest-reputation target
  - current top estimated agent
  - top-two estimate gap
  - candidate-set size within `Delta`
  - current follower root
  - whether selected target matches the paper-defined true-top agent
  - whether the current root leader matches the paper-defined true-top agent
- Also inspected the current gossip implementation directly in [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py).

Observed result:

- The observers do not behave as if they are all learning completely different rankings.
- At role-update checkpoints, almost all observers agree on the same `top_estimate_agent`:
  - seed `0`: about `0.9899`
  - seed `2`: about `0.9896`
  - seed `7`: about `0.9896`
  - seed `9`: about `0.9479` to `0.9897`
- But that common top-estimate agent is never the paper-defined true-top agent in any audited checkpoint.
- The mean share of observers whose selected target matched the true-top agent stayed near `0.0` to `0.04`.
- The share whose current root leader matched the true-top agent was essentially `0.0` throughout.

This means the estimate problem is not simply "everyone disagrees about the ranking." The stronger finding is:

- observers mostly agree on a top estimated agent
- but the agreed estimated top does not line up with the paper-defined true-top

Code inspection also shows a paper/code mismatch in the gossip update:

- vectorized path: [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L1178) to [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L1179)
- loop path: [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L1357) to [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L1361)

Both paths update all reputation columns, not just the paper's `B(t)` columns.

Verdict:

- `confirmed`

Interpretation:

- There is a real mismatch between the learned reputation ordering and the paper-defined true reputation.
- The current implementation also deviates from Section `7.3.3` by gossip-updating all columns instead of only `B(t)`.
- The audit does not yet isolate how much of the ranking mismatch is caused by the gossip-scope deviation versus other learning dynamics, but the mismatch is real even in the successful control seed.

## Issue 3: Tie-Handling

Paper reference:

- Section `7.3.4`
- choose uniformly at random from the `Delta`-candidate set

Diagnostic:

- Added checkpoint tracking of `candidate_count_within_delta` and selected targets.
- Added deterministic characterization tests for both selection paths:
  - `Agent.identify_highest_reputation_agent`
  - `_identify_highest_reputation_agent_from_matrix`

Observed result:

- Exact estimate ties are not the main story here.
- The dominant effect is persistent `Delta`-ties:
  - for every observer
  - at every audited checkpoint
  - in every audited seed
  - `candidate_count_within_delta = 99`
- Meanwhile, the actual top-two estimate gaps are tiny:
  - usually around `0.0002` to `0.0157`
- With `Delta = 0.1`, that means effectively all non-self agents remain admissible candidates.

This creates the following pattern:

- almost everyone agrees on the same `top_estimate_agent`
- but because all `99` non-self agents lie within `Delta` of that top value, the actually selected target is spread almost uniformly across the population
- final selected-target concentration is only about `0.04` to `0.05` for the most common target

The tie-handling implementation itself appears correct:

- the characterization tests showed approximately uniform selection over tied candidates in both code paths

Verdict:

- exact-tie bug: `ruled out`
- `Delta`-tie mechanism: `confirmed`

Interpretation:

- The present multi-leader behavior is strongly consistent with a `Delta` scale problem.
- Under the current estimate scale, `Delta = 0.1` is so large that the admissible candidate set never narrows.
- This does not point to biased random choice; it points to persistent over-broad tie sets.

## Issue 4: Interaction-Rate Audit

Paper reference:

- Section `6.6`
- Section `6.7`, Eq. `(13)`
- `\hat H_i(t) = \max\{\hat J_i^{pu}(t), \gamma \hat J_i^r(t), \kappa \hat J_i^s(t)\}`

Diagnostic:

- Recorded per-agent checkpoint rows for:
  - role
  - follower count
  - `\hat J_i^{pu}`, `\hat J_i^r`, `\hat J_i^s`
  - actor interaction rate
  - paper-side driver value
  - code-side driver value
  - whether they matched exactly
- Added deterministic tests for:
  - paper-aligned driver terms
  - the status-estimate update path for PU agents with followers

Observed result:

- All `6800` checkpoint rows matched the paper-side Eq. `(13)` driver exactly.
- In these four seeds, there were no runtime rows where a PU agent simultaneously had followers.
- So the specific Section `6.6` follower-status runtime case did not arise in this audited sample, but the code path is covered by tests and the actor-rate driver itself matches the newest paper.

Relevant implementation:

- actor-rate driver helper: [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L394) to [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L414)
- actor-rate update: [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L418) to [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L440)
- non-status agents with followers still updating status estimates: [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L1278) to [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L1290)

Verdict:

- `ruled out` as a paper-compliance bug for the audited `kappa = 0` Experiment B setting

Interpretation:

- Under the newest paper, followers influence interaction rates only through `\kappa \hat J_i^s`.
- Since `kappa = 0` here, the rate driver should not be status-driven.
- The current implementation matches that reading.

## Bottom Line

The audit does not support the view that the high-`gamma` multi-leader issue is caused by true-reputation ties or by incorrect Eq. `(13)` interaction-rate logic.

The strongest findings are:

1. The paper-defined true-top agent is unique at every audited checkpoint.
2. Observers mostly agree on a top estimated agent, so the problem is not simply lack of estimate consensus.
3. That estimated top agent never matches the paper-defined true-top agent in the audited checkpoints.
4. The implementation deviates from Section `7.3.3` by gossip-updating all reputation columns rather than only `B(t)`.
5. The `Delta`-candidate set is always maximal (`99` agents), so uniform tie-breaking acts over the whole population and directly supports fragmented following.

## Recommended Next Step

The next clean debugging step is:

1. Fix the Section `7.3.3` gossip-scope mismatch so only `B(t)` columns are updated.
2. Rerun the same audit on seeds `0, 2, 7, 9`.
3. Recheck:
   - whether the estimated top agent now aligns with the true-top agent
   - whether the `Delta`-candidate set shrinks below `99`
   - whether the failed seeds still fragment

If the `B(t)` fix still leaves `candidate_count_within_delta = 99`, then the next issue to debug is the scale of `Delta` relative to the learned reputation differences.

## Follow-up: Section 7.3.3 Scope Fix Rerun

After the audit above, the Section `7.3.3` gossip-scope mismatch was fixed in [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py) so that only `B(t)` columns are gossip-updated.

Follow-up artifacts:

- rerun folder: [reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix)
- rerun summary: [reputation_scaling_runs_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix/reputation_scaling_runs_static.csv)
- rerun estimate checkpoints: [expB_estimate_consensus_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix/expB_estimate_consensus_checkpoints.csv)
- rerun true-reputation checkpoints: [expB_true_reputation_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix/expB_true_reputation_checkpoints.csv)

### What changed

The fix materially changed the dynamics:

- seed `0`: `99 -> 8` followers, `0 -> 13` leader switches
- seed `2`: `57 -> 27` followers, `0 -> 3` leader switches
- seed `7`: `41 -> 10` followers, `0 -> 11` leader switches
- seed `9`: `38 -> 88` followers, `3 -> 0` leader switches

So the Section `7.3.3` change is not cosmetic. It changes the qualitative convergence behavior.

### What improved

The `Delta`-candidate set clearly shrank.

Before the fix:

- every final checkpoint had `candidate_count_within_delta = 99`

After the fix:

- seed `0`: mean final candidate count `19.82`
- seed `2`: mean final candidate count `12.95`
- seed `7`: mean final candidate count `22.92`
- seed `9`: mean final candidate count `19.45`

So the original audit conclusion about over-broad `Delta`-tie sets was correct: the Section `7.3.3` mismatch was indeed helping keep the candidate set artificially huge.

### What did not improve

Even after the scope fix, the estimated top agent still did not align with the paper-defined true-top agent.

At the final checkpoint, for every audited seed:

- share with `top_estimate_agent == unique_true_top_agent`: `0.0`
- share with `highest_rep_agent_estimate == unique_true_top_agent`: `0.0`

So the Section `7.3.3` fix alone does not solve the ranking mismatch identified earlier.

### Updated interpretation

The follow-up rerun changes the diagnosis in an important way:

1. The Section `7.3.3` mismatch really was affecting the scale of the candidate set.
2. Fixing that mismatch shrinks the `Delta`-candidate set substantially.
3. But the learned estimate ranking still fails to identify the paper-defined true-top agent.
4. And the system becomes more unstable in several seeds rather than more reliably convergent.

In other words:

- the scope fix addresses one real bug,
- but it does not by itself recover the expected paper outcome of a single leader with `99` followers across seeds.

### Revised next step

The next clean debugging question is no longer "does `B(t)` matter?" That is now answered: yes, it matters.

The next question is:

- why does the learned top-estimate ranking remain inconsistent with the paper-defined true reputation even after the Section `7.3.3` fix?

The most likely next levers to inspect are:

1. whether the current definition or timing of `L_i(t)` used in the implementation still differs from the paper
2. whether the initialization of `s_i(k,0)` matches the paper
3. whether the `Delta` scale should be calibrated to the post-fix estimate scale
4. whether some remaining detail in the observed-reputation update or selection timing still departs from the newest paper

## Additional Follow-up: Initialization Mismatch in `s_i(k,0)`

Paper reference:

- Section `6.4.3`
- `s_i(k,0) = 0` for all `k in C \\ {i}`

Current code:

- [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L172) to [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py#L175)

The implementation currently initializes:

- `v_i(k,0) = 0`
- but `s_i(k,0)` as `N(0, 0.1)` noise for every agent pair

This is a direct paper/code mismatch.

Why this matters:

- the paper-defined true-top gaps at the final checkpoint are small
  - for example, after the Section `7.3.3` scope fix they were:
    - seed `0`: about `0.00599`
    - seed `2`: about `0.00099`
    - seed `7`: about `0.00627`
    - seed `9`: about `0.00760`
- but the random initial column averages induced by `s_i(k,0) ~ N(0,0.1)` are often around `0.01` to `0.03`

So the initial random observed-reputation bias is on the same scale as, or larger than, the true-top gaps the algorithm is later trying to identify.

This does not perfectly determine the final `top_estimate_agent` in every seed, but it is large enough to create strong path dependence and is therefore a plausible contributor to the persistent mismatch between learned ranking and paper-defined true reputation.

Updated interpretation:

- The Section `7.3.3` scope fix explains why the `Delta`-candidate set was artificially huge.
- The nonzero random initialization of `s_i(k,0)` is a separate paper/code mismatch that can bias the learned ranking itself.

Clean next move:

1. make `s_i(k,0) = 0` paper-consistent
2. keep the Section `7.3.3` scope fix in place
3. rerun the same four-seed audit again
4. then re-evaluate:
   - whether `top_estimate_agent` moves closer to the true-top agent
   - whether the candidate set remains moderate after the scope fix
   - whether the single-leader outcome becomes more robust

## Additional Follow-up: Scope Fix Plus `s_i(k,0)=0`

After identifying the initialization mismatch above, the code was updated so that:

- `s_i(k,0) = 0` for all agent pairs, matching Section `6.4.3`
- the Section `7.3.3` scope fix remained in place

Follow-up artifacts:

- rerun folder: [reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit)

### What changed

Compared with the scope-fix-only rerun:

- seed `0`: `8 -> 12` followers
- seed `2`: `27 -> 9`
- seed `7`: `10 -> 12`
- seed `9`: `88 -> 88`

So zero-initializing `s_i(k,0)` changes the trajectories again, but it does not restore the original `99`-follower convergence pattern.

### Estimate alignment result

The key result is that the ranking mismatch still remains.

At the final checkpoint, for all four audited seeds:

- share with `top_estimate_agent == unique_true_top_agent`: `0.0`

Selected-target alignment improved only slightly in one seed:

- seed `2`: `selected_matches_true_top_share = 0.06`
- all other seeds remained at `0.0`

So zero-initializing `s_i(k,0)` by itself is not enough to make the learned estimate ranking track the paper-defined true reputation.

### Candidate-set result

The candidate sets remain substantially smaller than in the original pre-fix audit, but larger than in the scope-fix-only rerun:

- seed `0`: mean candidate count `23.80`
- seed `2`: mean candidate count `31.71`
- seed `7`: mean candidate count `29.71`
- seed `9`: mean candidate count `22.82`

This means:

- the Section `7.3.3` scope fix is still the main reason the candidate set shrank away from `99`
- zero initialization does not collapse the `Delta` tie set to a small number of candidates

### Updated interpretation

The current picture is now:

1. The original code had at least two real paper/code mismatches:
   - gossip scope
   - nonzero initialization of `s_i(k,0)`
2. Fixing those mismatches does change the system materially.
3. But even after both fixes, the learned top estimate still does not identify the paper-defined true-top agent in the audited seeds.

So the remaining mismatch between `s_i(k,t)` and `R_k` is not explained solely by:

- over-broad gossip scope, or
- random nonzero initialization of observed-reputation estimates

### Revised next step

The next clean target is now the update/selection logic itself:

1. inspect whether the implementation of Eq. `(9)` is averaging over exactly the right active set and with the right timing
2. inspect whether `L_i(t)` is being updated at the same moments and with the same information set as in the paper
3. only after that, revisit whether `Delta` should be calibrated to the new post-fix estimate scale

## Isolated Gossip Audit

To make the remaining diagnosis more systematic, the code now includes a deterministic Phase-4-only audit hook for the reputation-learning path. This isolated harness applies only the Section `6.4` recurrence:

- `v_i(k,t)` updates from supplied observed utilities
- `B(t)` construction from the current `L_i(t)` of active participants
- `s_i(k,t)` gossip updates only over the current paper-scope targets
- `L_i(t+1)` updates after the reputation update

It does not sample actions, update policies, or run role switching.

New artifacts and checks:

- new checkpoint rollup: [expB_rank_alignment_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_gossipiso/expB_rank_alignment_checkpoints.csv)
- isolated deterministic tests in [test_pu_gossip_role_switching.py](/Users/xia/social_reward_economies/code_by_peter_tests/test_pu_gossip_role_switching.py)

The isolated gossip tests cover:

- Python Phase-4 path vs paper-oracle step-by-step
- vectorized Phase-4 path vs paper-oracle step-by-step
- fast path vs slow path on the same deterministic trace
- `L_i(t+1)` updating only for active participants
- current averaging behavior in Eq. `(9)` characterization

Result:

- the isolated gossip audit passes
- the fast and slow Phase-4 implementations agree with each other
- the remaining problem is therefore not just a hidden fast-path bug

This sharpens the diagnosis: the unresolved problem is now much more likely to be a remaining paper/code mismatch in the specification details of the gossip algorithm itself, rather than an implementation discrepancy between the two code paths.

## Eq. (9) Averaging-Set Audit

The next paper-compliance check was the averaging set in Eq. `(9)` of [learning_paper_newest_ver.pdf](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver.pdf). For this pass, the paper was read literally: the observed-reputation average was treated as averaging over all agents in `A(t)`, not only the active participants.

That interpretation was implemented as a new explicit averaging mode:

- `participants_only`: average `s_j(k,t)` over active participants only
- `all_agents`: average `s_j(k,t)` over all agents in `A(t)`, with inactive agents contributing their unchanged current `s_j(k,t)`

The isolated gossip audit was extended to support both variants, and the deterministic oracle tests now verify:

- Eq. `(9)` averaging over active participants only
- Eq. `(9)` averaging over all agents
- fast-path vs slow-path equivalence under both averaging modes
- unchanged `B(t)` scope under the new averaging mode

Result of the isolated tests:

- both averaging variants pass the isolated oracle audit
- both code paths agree under both variants

The four-seed Experiment B rerun for the paper-literal variant was written to:

- rerun folder: [reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents)
- run summary: [reputation_scaling_runs_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents/reputation_scaling_runs_static.csv)
- rank alignment rollup: [expB_rank_alignment_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents/expB_rank_alignment_checkpoints.csv)

What changed:

- follower concentration improved materially relative to `scopefix_zeroinit_gossipiso`
- candidate-count means stayed well below the original pre-fix `99`
- but `top_estimate_matches_true_top_share` remained `0.0` for all four seeds at the final checkpoint

Per-seed final checkpoint results for the `all_agents` variant:

- seed `0`: `eq9_averaging_mode = all_agents`, true-top `71`, top-estimate mode `92`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `11.13`, distinct roots `2`, final top followers `75`
- seed `2`: `eq9_averaging_mode = all_agents`, true-top `23`, top-estimate mode `2`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `27.74`, distinct roots `15`, final top followers `16`
- seed `7`: `eq9_averaging_mode = all_agents`, true-top `56`, top-estimate mode `0`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `17.93`, distinct roots `3`, final top followers `52`
- seed `9`: `eq9_averaging_mode = all_agents`, true-top `2`, top-estimate mode `54`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `22.90`, distinct roots `8`, final top followers `48`

So the Eq. `(9)` paper-literal averaging change improves behavioral concentration, but it does not fix the ranking-alignment failure. That means this variant is a useful paper-compliance check, but it does not satisfy the main acceptance criterion for the audit pass.

## Section 6.4.4 `L_i(t+1)` Timing Audit

The next paper-compliance check targeted Section `6.4.4` of [learning_paper_newest_ver.pdf](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver.pdf). The paper text is not ambiguous on the main timing points:

- only agents `i ∈ A_p(t)` that are active as a participant update `L_i(t+1)`
- the candidate set is built from `s_i(k,t+1)`, not from pre-update `s_i(k,t)`
- self is excluded from the candidate set
- `B(t)` for the current gossip step is still built from the previous `L_i(t)`, not the newly selected `L_i(t+1)`

To make that explicit, the isolated gossip audit now supports three leader-update modes:

- `participants_only_post_eq9`: paper-literal Section `6.4.4`
- `all_agents_post_eq9`: audit-only scope variant
- `participants_only_pre_eq9`: audit-only timing variant

The isolated deterministic oracle tests now verify:

- slow-path vs oracle under each leader-update mode
- fast-path vs oracle under each leader-update mode
- active-participant-only updating under the paper-literal mode
- all-agent updating under the audit-only scope variant
- pre-update vs post-update leader selection when the timing changes the winner
- `B(t)` staying tied to the previous `L_i(t)` without same-step leakage from `L_i(t+1)`

Result of the isolated tests:

- the Section `6.4.4` mode system passes the isolated oracle audit
- the fast and slow paths agree under all three leader-update modes

The four-seed Experiment B rerun for the paper-literal leader-update rule was written to:

- rerun folder: [reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents_lpostparticipants](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents_lpostparticipants)
- run summary: [reputation_scaling_runs_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents_lpostparticipants/reputation_scaling_runs_static.csv)
- rank alignment rollup: [expB_rank_alignment_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_eq9allagents_lpostparticipants/expB_rank_alignment_checkpoints.csv)

This rerun produced the same outcome as the previous `scopefix_zeroinit_eq9allagents` baseline:

- seed `0`: final top followers `75`, `top_estimate_matches_true_top_share = 0.0`
- seed `2`: final top followers `16`, `top_estimate_matches_true_top_share = 0.0`
- seed `7`: final top followers `52`, `top_estimate_matches_true_top_share = 0.0`
- seed `9`: final top followers `48`, `top_estimate_matches_true_top_share = 0.0`

It also reproduced the same final candidate-count means and distinct-root counts as the prior `eq9allagents` rerun. So the explicit Section `6.4.4` paper-literal mode does not narrow the ranking-alignment gap any further; it only confirms that the current `eq9allagents` baseline was already using the same leader-update timing in practice.

## Variant Comparison

The following table summarizes the main audited variants on the same four seeds (`0,2,7,9`).

| variant | change type | mean final followers | mean leader switches | mean final top-estimate match to true-top | mean final candidate count | mean final distinct roots | conclusion |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | pre-fix audit | `58.75` | `0.75` | `0.0` | `99.0` | `2.75` | broad `Delta` tie set dominates; ranking mismatch already present |
| `scopefix` | paper-compliance fix | `33.25` | `6.75` | `0.0` | `18.79` | `9.50` | fixing `B(t)` shrinks the tie set but does not recover the true-top ranking |
| `scopefix_zeroinit` | paper-compliance fix | `30.25` | `7.00` | `0.0` | `27.01` | `12.25` | zero-init changes trajectories but still leaves the ranking mismatch unresolved |
| `scopefix_zeroinit_gossipiso` | isolated gossip audit + system rerun | `30.25` | `7.00` | `0.0` | `27.01` | `12.25` | isolated tests pass and the rerun reproduces the same behavior, so the issue is still in the paper-level gossip specification details |
| `scopefix_zeroinit_eq9allagents` | paper-compliance fix | `47.75` | `4.00` | `0.0` | `19.92` | `7.00` | all-agents Eq. `(9)` averaging improves follower concentration, but the learned top estimate still does not align with the true-top |
| `scopefix_zeroinit_eq9allagents_lpostparticipants` | paper-compliance fix | `47.75` | `4.00` | `0.0` | `19.92` | `7.00` | explicit Section `6.4.4` paper-literal leader timing reproduces the same result exactly, so this timing question is effectively ruled out at this level |

The isolated-audit rerun artifacts are:

- rerun folder: [reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_gossipiso](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_gossipiso)
- run summary: [reputation_scaling_runs_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_gossipiso/reputation_scaling_runs_static.csv)
- rank alignment rollup: [expB_rank_alignment_checkpoints.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/audits/reputation_scaling_static_10states_gaussian_hysteresisfix_light_paper_audit_gamma5_seeds_0_2_7_9_scopefix_zeroinit_gossipiso/expB_rank_alignment_checkpoints.csv)

Per-seed final checkpoint results in the isolated-audit rerun:

- seed `0`: true-top `80`, top-estimate mode `92`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `23.80`, final top followers `12`
- seed `2`: true-top `23`, top-estimate mode `66`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `31.71`, final top followers `9`
- seed `7`: true-top `94`, top-estimate mode `7`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `29.71`, final top followers `12`
- seed `9`: true-top `2`, top-estimate mode `20`, `top_estimate_matches_true_top_share = 0.0`, candidate-count mean `22.82`, final top followers `88`

## Updated Bottom Line

At this point, the systematic audit supports the following conclusions:

1. True-reputation ties are not the source of the Experiment B multi-leader issue.
2. Eq. `(13)` interaction-rate logic is not the source of the issue in the audited `kappa = 0` regime.
3. Exact random tie-breaking is not broken, although `Delta` breadth still matters.
4. We have already fixed two real paper/code mismatches in the gossip path:
   - `B(t)` scope
   - `s_i(k,0) = 0`
5. After those fixes, and after passing the isolated fast-vs-slow gossip audit, the learned top estimate still does not identify the paper-defined true-top agent in the audited seeds.
6. Promoting the paper-literal Eq. `(9)` averaging set to `all_agents` improves follower concentration, but still leaves final true-top alignment at `0.0` in all four audited seeds.
7. Making the Section `6.4.4` leader-update timing explicit confirms that the current `eq9allagents` baseline was already using the paper-literal `participants_only_post_eq9` behavior in practice.

So the remaining debugging target is now narrow and well defined:

- the remaining paper-level details of the gossip / reputation-learning recurrence itself,
- but no longer the high-level `L_i(t+1)` timing/scope question from Section `6.4.4`,
- rather the deeper mismatch in how the learned observed-reputation ranking approximates the paper-defined true-reputation target.
