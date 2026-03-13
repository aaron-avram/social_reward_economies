# Bug Catalog (Canonical IDs)

This catalog uses grouped, stable IDs for the bugs originally identified in `src/code_old.py`.

## Canonical IDs

- `IR-1`: Active-set sampling must use `theta(mu)=1-exp(-mu)`
- `REP-1`: Highest-reputation selection must exclude self (`C\\{i}`)
- `REP-2`: Reputation update must follow Eq. (9) additive form (`avg + delta_v`)
- `REP-3`: Remove extra pairwise gossip pass in the same timestep
- `REP-4`: Personal-benefit estimates `v_i(k,t)` must update for all agents each step (not only active participants)
- `REP-5`: Reputation followers must emulate the leader's active-role policy (`w_k(t)`, PU vs STATUS), not always `w_k^pu`
- `REP-6`: Reputation reward estimate `\hat J_i^r(t)` must match `s_i(k,t)` for followed agent `k` (Section 6.6)
- `REP-7`: Personal-benefit learning must use observer-specific utility `u_i(s(t), x_k(t))`, not actor `k`'s self-payoff
- `ROLE-1`: Non-followers must bootstrap reputation-role entry from observed reputation signal
- `ROLE-2`: Remove extra `max_rep >= B_i` gate from Step-1 follow condition
- `ROLE-3`: Redirect if selected follow target is itself a follower
- `ROLE-4`: Prevent self-following when redirect chain points back to agent itself
- `ROLE-5`: Clear and redirect followers when agent becomes a follower
- `STATUS-1`: Status entry gate used `estimated_reward_status`, but this signal was only updated while already in STATUS

Legacy mapping from old inline labels:

- old `Bug 1` -> `ROLE-1`
- old `Bug 2` -> `ROLE-2`
- old `Bug 3` -> `REP-2`
- old `Bug 4` -> `REP-3`
- old `Bug 5` -> `ROLE-3`

## By Group

### Interaction Rates

#### IR-1 (High)
Active actor/participant sampling used raw `mu` instead of `theta(mu)=1-exp(-mu)`.

### Reputation Learning

#### REP-1 (High)
Highest-reputation selection allowed self as candidate.

#### REP-2 (High)
Reputation update deviated from Eq. (9): implementation used EMA-style updates instead of `avg + delta_v`.

#### REP-3 (Medium)
A second pairwise gossip update occurred in Phase 5 after Phase 4 already applied gossip updates.

#### REP-4 (High)
Section 6.4.2 specifies that each agent updates personal-benefit estimates `v_i(k,t)` for all agents every step (active agents via observed payoff; inactive agents via zero-payoff decay). Current `src/code_debugged.py` updates `v_i` only for agents active as participants.

#### REP-5 (High)
Section 6.4.5 specifies followers emulate leader behavior using `w_k(t)`, which is `w_k^pu(t)` if leader is in `P(t)` and `w_k^s(t)` otherwise. Current implementation routes reputation followers through leader `weights_pu` only, so they do not follow a STATUS leader's status policy.

#### REP-6 (High)
Section 6.6 specifies that active reputation-optimizing agents update reward estimate as `\hat J_i^r(t)=s_i(k,t)` for the followed agent `k`. Current implementation updates `estimated_reward_rep` as an EMA of the followed agent's realized payoff instead.

#### REP-7 (High)
Section 6.4.2 defines `v_i(k,t)` using the observer-specific utility `u_i(s(t), x_k(t))`. The old implementation reused actor `k`'s own realized payoff for every observer, so all agents learned essentially the same personal-benefit signal for a given actor action. That compressed observer disagreement and made action-based reputation shocks weaker than the paper intends.

### Role Updates

#### ROLE-1 (High)
Step-1 follow decision used `estimated_reward_rep` for non-followers, preventing reputation-role bootstrap.

#### ROLE-2 (High)
Step-1 follow condition added extra gate `max_rep >= B_i`, which is not in Section 7.3.

#### ROLE-3 (Medium)
Indirect-follow redirect logic failed to redirect to the leader of a follower target.

#### ROLE-4 (High)
After ROLE-3 redirect, no check prevented `best_k == i` (agent following itself). Redirect chains could point back to the agent, causing impossible follower counts (n followers with only n agents).

#### ROLE-5 (High)
When agent becomes a follower, existing followers were not redirected. Caused multi-level chains (A→B→C) and invalid states (agent following someone but still having followers). ROLE-3 alone insufficient due to processing order dependency.

### Status Updates

#### STATUS-1 (High)
Step-2 status entry checks `kappa * estimated_reward_status > estimated_reward_pu`, but `estimated_reward_status` was only updated inside the STATUS-actor branch. In practice, agents could reach follower eligibility yet never accumulate enough status-reward signal to enter STATUS. Fix: update `estimated_reward_status` for follower-holding active actors before role-specific branching so Step-2 can evaluate a meaningful signal.

## Newly Found Bugs (2026-03-05)

The following bugs were identified on **March 5, 2026** from new reputation-coverage tests in
`code_by_peter_tests/test_bug_report_fixes.py`:

- `REP-4` via `test_rep642_all_agents_update_personal_benefit_each_step`
- `REP-5` via `test_rep645_follower_tracks_status_policy_of_status_leader`
- `REP-6` via `test_rep66_reputation_reward_estimate_matches_followed_agent_reputation`

The following additional bug was identified on **March 12, 2026** from further reputation-learning
audit and Experiment D debugging:

- `REP-7` via `test_rep642_personal_benefit_is_observer_specific_in_step`
- `REP-7` via `test_rep642_numpy_fast_path_uses_observer_specific_utilities`

## Note on Fixed Code

`src/code_debugged.py` uses these canonical IDs in inline comments (e.g., `[IR-1]`, `[REP-2]`, `[ROLE-3]`, `[STATUS-1]`) so each fix point maps back to this catalog unambiguously.
