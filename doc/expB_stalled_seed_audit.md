# Exp B Tuned-Paper Stalled Seed Audit

Date: 2026-04-04

## Summary

This note audits the three scaled Exp B seeds that do not converge in the finished tuned paper-faithful run:

- output directory: `experiments/outputs/exp_b/scaled/factorial_gamma_3_5_10_15/tuned_paper`
- stalled seeds: `0`, `1`, `7`
- canonical diagnosis slice: `gamma=5`
- invariance check: for seeds `0`, `1`, and `7`, the checkpoint trajectories at `gamma=5`, `gamma=10`, and `gamma=15` are exactly identical in:
  - `reputation_scaling_role_update_diagnostics_static.csv`
  - `expB_rank_alignment_checkpoints.csv`
  - `reputation_scaling_seed_diagnostic_summary_static.csv`

The key conclusion is that these are **not weak-following seeds**. The follow gate is open throughout, but the system plateaus into persistent multi-root fragmentation.

## Seed Dossiers

| Seed | Final top / second | Dominant failure label | Earliest persistent final blocs | Evidence |
| --- | --- | --- | --- | --- |
| `0` | `44 / 43` | `late_fragmentation_after_partial_lock_in` | `t=6000` for the final pair `{27,53}` | Three roots persist throughout. The run nearly locks in at `t=18000` (`87 / 6`), then decays back into a `44 / 43 / 10` split by the end. |
| `1` | `50 / 48` | `stable_two_leader_split` | `t=3000` for the final pair `{65,81}` | Two roots persist from the first role update to the end. The run oscillates around a `65` vs `81` split and never absorbs the second bloc. |
| `7` | `52 / 46` | `stable_two_leader_split` | `t=24000` for the final pair `{4,66}` | Three roots exist early, but by `t=24000` the final competing pair is fixed. The run collapses to two roots by `t=30000` and stays split to the end. |

### Common properties across the stalled seeds

- `share_gate_margin_positive = 1.0` and `final_share_gate_margin_positive = 1.0`
  - the gate is open; this is not a Step-1 failure
- `top_estimate_mode_share = 0.99` at the final checkpoint for all three stalled seeds
  - the raw argmax of the learned reputation row is already highly concentrated
- `top_estimate_matches_true_top_share = 0.0` at the final checkpoint for all three stalled seeds
  - but this is **not** a useful discriminator, because the same is true for the converged seeds in this run
- `candidate_count_mean` remains very large late in the run:
  - seed `0`: `18.83`
  - seed `1`: `17.82`
  - seed `7`: `18.81`
- `mean_gap_top2` remains tiny at the final checkpoint:
  - seed `0`: `0.01185`
  - seed `1`: `0.00165`
  - seed `7`: `0.00642`
- all agents are already at the actor-rate cap:
  - at `t=3000`: actor rate mean `1.0`, cap share `1.0`
  - at `t=48000`: actor rate mean `1.0`, cap share `1.0`

### Cross-seed comparison: stalled vs converged (`gamma=5`, final checkpoint)

| Metric | Stalled seeds avg | Converged seeds avg | Interpretation |
| --- | --- | --- | --- |
| `candidate_count_mean` | `18.49` | `14.85` | stalled runs keep much wider within-`delta` candidate sets |
| `mean_gap_top2` | `0.0066` | `0.0109` | stalled runs have weaker separation between top reputation candidates |
| `distinct_root_count` | `2.33` | `1.00` | stalled runs fail specifically at root collapse |
| `top_estimate_mode_share` | `0.99` | `0.99` | raw top-estimate consensus does not separate success from failure here |
| `top_estimate_matches_true_top_share` | `0.0` | `0.0` | oracle alignment is not the operational bottleneck in this regime |
| actor-rate cap share | `1.0` | `1.0` | higher actor-rate tuning cannot add much under the current cap |

## Interpretation

The stalled seeds do not fail because agents remain in PU or because `gamma` is too low. They fail because:

1. the within-`delta` candidate set remains very wide late in the run,
2. the top reputation gap is small,
3. follower roots do not collapse from two (or three) blocs down to one,
4. actor interaction rates are already saturated at the clip ceiling, so rate-based reinforcement has little room to act.

This makes the plateau a **fragmentation problem under weak separation**, not a gate-opening problem.

## Tuning Verdict

| Lever | Verdict | Evidence | Recommendation |
| --- | --- | --- | --- |
| `gamma > 5` | `unlikely to help` | Seeds `0`, `1`, and `7` are exactly unchanged at `gamma=5`, `10`, and `15`. | Stop sweeping `gamma` above `5` for this regime. |
| Higher activation rate | `unlikely to help beyond a small effect` | Actor rates are already clipped at `1.0` from the first checkpoint onward, and raw top-estimate consensus is already `0.99`. | Do not prioritize further activation tuning unless `M` or the participant process changes. |
| Lower `reward_agent_sigma` | `strongest next tuning lever` | The plateau is driven by weak separation and multi-root persistence. Lower heterogeneity is the most direct way to sharpen cross-agent agreement and narrow the candidate set. | First targeted follow-up: seeds `0,1,7`, `gamma=5`, change `reward_agent_sigma: 0.05 -> 0.02`. |
| Higher `reward_base_sigma` | `strongest next tuning lever` | The stalled seeds have small true-top gaps on average and small learned top-two gaps. Stronger shared structure should widen leader separation. | Second targeted follow-up: seeds `0,1,7`, `gamma=5`, change `reward_base_sigma: 0.08 -> 0.12` and `0.15`. |
| Smaller `delta` | `may help modestly, but low priority` | The stalled seeds keep `candidate_count_mean ~ 18-19`, but some fully converged seeds still converge with very large candidate counts, so `delta` is not the sole blocker. | Only test `delta` after reward-structure tuning if selected-target diversity remains high. |
| Follower-driven actor-rate reinforcement | `structural change, not mere tuning` | Under the current `M=1.0` cap, actor rates are already maxed out, so status-driven rate reinforcement has no room to improve activity. | Only revisit if changing `M` or deliberately replacing Eq. `(13)` with a different actor-rate rule. |

## Final Verdict

- The plateau is **not** due to insufficient `gamma`.
- The plateau is **not** due to the follow gate staying closed.
- The plateau is **not** likely to be fixed by pushing activation rates further under the current actor-rate cap.
- The most promising remaining tuning levers are the **reward-structure levers**:
  - lower `reward_agent_sigma`
  - higher `reward_base_sigma`

No confirmatory reruns were needed to reach this verdict, because the stalled seeds already provide a clear signature:

- exact `gamma` invariance above `5`
- fully positive gate margins
- saturated actor rates
- persistent multi-root fragmentation with weak reputation separation
