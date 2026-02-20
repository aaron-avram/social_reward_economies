# Bug Report: `src/code_old.py` (Grouped + Canonical IDs)

Date: 2026-02-19

## Scope

Validated the baseline against:

- `doc/learning_paper_newest_ver_transcription.md`
  - Section 6.2 (active actor/participant probability)
  - Section 6.4.3 Eq. (9) (reputation update)
  - Section 6.4.4 (highest-reputation agent set excludes self)
  - Section 7.3 (Step-1 reputation-switch criterion)
  - Section 6.7 Eq. (13) (actor-rate update)
- `doc/Social_Reward_Economies_transcription.md` for conceptual consistency
- testing style from `single-reward-economy/reputation_tests/`

## Test Summary

- Test files: `test_reputation_behavior.py`, `test_interaction_rates.py`
- Result: **5 failed, 3 passed**
- Full output: `test_run_2026-02-19.txt`
- Cross-check file for legacy code comments: `test_existing_bug_comments.py`
- Cross-check result: **3 failed, 0 passed**
- Cross-check output: `test_bug_comments_run_2026-02-19.txt`

Passed checks:

- Eq. (13) one-step actor-rate formula implementation
- Eq. (13) clipping to `[0, M]`
- personal-benefit decay for inactive agents

## Canonical Bug IDs

- `IR-1`: Active-set sampling must use `theta(mu)=1-exp(-mu)`
- `REP-1`: Highest-reputation selection must exclude self (`C\\{i}`)
- `REP-2`: Reputation update must follow Eq. (9) additive form (`avg + delta_v`)
- `REP-3`: Remove extra pairwise gossip pass in the same timestep
- `ROLE-1`: Non-followers must bootstrap reputation-role entry from observed reputation signal
- `ROLE-2`: Remove extra `max_rep >= B_i` gate from Step-1 follow condition
- `ROLE-3`: Redirect if selected follow target is itself a follower

Legacy mapping from old inline labels:

- old `Bug 1` -> `ROLE-1`
- old `Bug 2` -> `ROLE-2`
- old `Bug 3` -> `REP-2`
- old `Bug 4` -> `REP-3`
- old `Bug 5` -> `ROLE-3`

## Findings By Group

### Interaction Rates

#### IR-1) Active-set sampling uses raw `mu` instead of `theta(mu)=1-exp(-mu)`

- Severity: **High**
- Code:
  - `src/code_old.py:406` actor inclusion uses `random < actor_interaction_rate`
  - `src/code_old.py:412` participant inclusion uses `random < participant_interaction_rate`
- Evidence:
  - `test_actor_activation_probability_uses_theta_of_mu`
  - `test_participant_activation_probability_uses_theta_of_mu`
  - For `mu=0.8`, expected `theta(mu)=0.5507`; observed about `0.7974` (actors) and `0.7997` (participants)
- Impact:
  - Over-activates agents and shifts payoff/reputation/role dynamics away from the paper process

### Reputation Learning

#### REP-1) Highest-reputation selection can choose self

- Severity: **High**
- Code:
  - `src/code_old.py:271`–`src/code_old.py:273` build candidates without self-filtering
- Evidence:
  - `test_identify_highest_reputation_excludes_self`
- Impact:
  - Breaks follower-target logic and can create invalid self-target behavior

#### REP-2) Reputation update does not implement Eq. (9) additive form

- Severity: **High**
- Spec (Section 6.4.3 Eq. 9):
  - `s_i(k,t+1) = avg_j s_j(k,t) + v_i(k,t+1)-v_i(k,t)`
- Code:
  - `src/code_old.py:245`–`src/code_old.py:246` EMA toward average
  - `src/code_old.py:254`–`src/code_old.py:255` EMA toward raw payoff
- Evidence:
  - `test_reputation_update_matches_eq9_additive_delta_structure` (`expected 0.8`, got `1.0`)
- Impact:
  - Distorts reputation estimation semantics and downstream ranking/convergence

#### REP-3) Extra pairwise gossip pass in Phase 5 (double gossip per step)

- Severity: **Medium**
- Code:
  - `src/code_old.py:481` onward
- Evidence:
  - `test_bug4_step_contains_extra_pairwise_gossip_after_participant_updates`
  - `test_bug_comments_run_2026-02-19.txt` shows estimates changed by Phase 5 even when Phase 4 reputation update is monkeypatched to no-op
- Impact:
  - Changes convergence speed/shape and can mask other reputation-learning defects

### Role Updates

#### ROLE-1) Non-followers cannot bootstrap reputation-role entry

- Severity: **High**
- Code:
  - `src/code_old.py:571` uses `gamma * estimated_reward_rep` in Step-1
- Evidence:
  - `test_bug1_non_followers_do_not_switch_even_with_high_reputation_signal`
- Impact:
  - Blocks or delays follower emergence and distorts Section 7.3 dynamics

#### ROLE-2) Extra `max_rep >= B_i` gate not in Section 7.3 criterion

- Severity: **High**
- Code:
  - `src/code_old.py:583` requires
  - `est_rep_weighted > max(B_i, est_pu) and max_rep >= B_i`
- Evidence:
  - `test_step1_reputation_switch_uses_role_criterion_without_extra_rep_gate`
- Impact:
  - Suppresses following transitions and leader emergence

#### ROLE-3) Indirect-follow redirect logic is broken

- Severity: **Medium**
- Code:
  - `src/code_old.py:587`
- Evidence:
  - `test_bug5_indirect_following_redirect_not_applied`
- Impact:
  - Allows inconsistent follower chains and incorrect influence topology

## Replication Risk for `Norm__Working_Copy__2_-16.pdf`

These issues directly affect interaction exposure, reputation formation, and role transitions. Together they are likely to prevent qualitative replication of expected follower emergence, influencer concentration, and convergence trajectories.

## Recommended Next Step

Patch high-severity issues first (`IR-1`, `REP-1`, `REP-2`, `ROLE-1`, `ROLE-2`), then medium-severity structural issues (`REP-3`, `ROLE-3`), and rerun the test package before full-trajectory replication against `doc/Norm__Working_Copy__2_-16.pdf`.
