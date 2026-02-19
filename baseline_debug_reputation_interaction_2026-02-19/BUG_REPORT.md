# Bug Report: `doc/code_old.py` (Reputation + Interaction Rates)

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

Additional cross-check (comment-labeled bugs in `code_old.py`):

- Test file: `test_existing_bug_comments.py`
- Result: **3 failed, 0 passed**
- Full output: `test_bug_comments_run_2026-02-19.txt`

Passed checks:

- Eq. (13) one-step actor-rate formula implementation
- Eq. (13) clipping to `[0, M]`
- personal-benefit decay for inactive agents

## Findings

### 1) Active-set sampling uses `mu` directly instead of `theta(mu)=1-exp(-mu)`

- Severity: **High**
- Spec: Section 6.2 defines inclusion probabilities via `theta(mu)=1-exp(-mu)`.
- Code:
  - `doc/code_old.py:406` actor inclusion uses `random < actor_interaction_rate`
  - `doc/code_old.py:412` participant inclusion uses `random < participant_interaction_rate`
- Evidence:
  - `test_actor_activation_probability_uses_theta_of_mu`
  - `test_participant_activation_probability_uses_theta_of_mu`
  - For `mu=0.8`, expected `theta(mu)=0.5507`; observed about `0.7974` (actors) and `0.7997` (participants).
- Impact:
  - Over-activates agents, altering observed payoffs, gossip frequency, and downstream reward/role dynamics.
  - Makes replication against the paper’s stochastic process inconsistent.

### 2) Highest-reputation selection can choose self

- Severity: **High**
- Spec: Section 6.4.4 candidate set is `C\{i}`.
- Code:
  - `doc/code_old.py:271`–`doc/code_old.py:273` builds candidates from all keys in `reputation_estimates` with no self-filter.
- Evidence:
  - `test_identify_highest_reputation_excludes_self` fails; agent 0 selects itself as top reputation agent.
- Impact:
  - Breaks follower-target logic and can create invalid self-follow/preference loops in role updates.

### 3) Reputation update does not implement Eq. (9) additive form

- Severity: **High**
- Spec: Section 6.4.3 Eq. (9):
  - `s_i(k,t+1) = avg_j s_j(k,t) + v_i(k,t+1)-v_i(k,t)`
- Code:
  - `doc/code_old.py:245`–`doc/code_old.py:246` first EMA toward average
  - `doc/code_old.py:254`–`doc/code_old.py:255` second EMA toward raw payoff
- Evidence:
  - `test_reputation_update_matches_eq9_additive_delta_structure`:
    - expected `0.8`, got `1.0` in a deterministic setup.
- Impact:
  - Changes estimator semantics (mixes two EMAs instead of avg + delta), distorting reputation ranking and gossip convergence behavior.

### 4) Step-1 reputation switching adds an extra gate not in Section 7.3 criterion

- Severity: **High**
- Spec: Section 7.3 switch condition is
  - `gamma * J_r > max(B_i, J_pu)`
- Code:
  - `doc/code_old.py:583` requires
  - `est_rep_weighted > max(B_i, est_pu) and max_rep >= B_i`
- Evidence:
  - `test_step1_reputation_switch_uses_role_criterion_without_extra_rep_gate` fails:
    - `gamma*J_r` satisfied, but switch blocked because `max_rep < B_i`.
- Impact:
  - Suppresses follower formation and can prevent the expected leader-follower emergence needed for reproducing `Norm__Working_Copy__2_-16.pdf` dynamics.

## Replication Risk for `Norm__Working_Copy__2_-16.pdf`

These issues directly affect reputation formation, role transitions, and interaction exposure rates. Together they are likely to prevent qualitative replication of expected phenomena such as follower emergence, influencer concentration, and stable consensus trajectories.

## Recommended Next Step

Patch the four high-severity issues above first, then rerun this test package before attempting full-trajectory replication against `doc/Norm__Working_Copy__2_-16.pdf`.
