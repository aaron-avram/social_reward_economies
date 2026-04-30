# Bug Report: `code_old.py` → `code_debugged.py`

All 14 bugs listed here were found in `src/code_old.py` and fixed in `src/code_debugged.py`.
Fix points in `code_debugged.py` are annotated with the canonical ID (e.g. `# [REP-2]`).

---

## Overview

| ID | Group | Severity | One-line summary |
|----|-------|----------|-----------------|
| IR-1 | Interaction Rates | High | Active-set sampling used raw `μ` instead of `θ(μ)=1-exp(-μ)` |
| REP-1 | Reputation | High | Highest-reputation selection allowed self as candidate |
| REP-2 | Reputation | High | Reputation update used two EMAs instead of Eq. (9) `avg + Δv` structure |
| REP-3 | Reputation | Medium | Extra Phase 5 gossip pass doubled gossip per timestep |
| REP-4 | Reputation | High | Personal-benefit estimates updated only for active participants, not all agents |
| REP-5 | Reputation | High | Followers always copied leader's PU policy, ignoring STATUS policy |
| REP-6 | Reputation | High | Reputation reward estimate used EMA of leader payoff instead of `s_i(k,t)` |
| REP-7 | Reputation | High | Personal-benefit learning used actor's own payoff instead of observer-specific utility |
| ROLE-1 | Role Updates | High | Non-followers could never enter REPUTATION role (estimate stuck at 0) |
| ROLE-2 | Role Updates | High | Step-1 had an extra `max_rep >= B_i` gate not in Section 7.3 |
| ROLE-3 | Role Updates | Medium | Indirect-follow redirect logic was logically backwards |
| ROLE-4 | Role Updates | High | No self-follow prevention after redirect chain pointed back to agent |
| ROLE-5 | Role Updates | High | Agent becoming a follower did not redirect its own followers |
| STATUS-1 | Status | High | `estimated_reward_status` only updated inside STATUS branch, blocking entry |

**Legacy label mapping** (old inline BUG comments → canonical IDs):
`Bug 1` → ROLE-1 · `Bug 2` → ROLE-2 · `Bug 3` → REP-2 · `Bug 4` → REP-3 · `Bug 5` → ROLE-3

---

## Interaction Rates

### IR-1 — Active-set sampling used raw `μ` (High)

**Description.** Section 6.2 defines activation probability as `θ(μ) = 1 − exp(−μ)`, not `μ` itself. Using raw `μ` over-activates agents (especially for `μ > 1`) and breaks the link between the rate parameter and empirical frequency.

**code_old.py** (line 406):
```python
# A_a(t): Active actors
if np.random.random() < agent.state.actor_interaction_rate:   # raw μ, not θ(μ)
    active_actors.add(agent.agent_id)
```

**code_debugged.py** (line 1924):
```python
actor_prob = 1.0 - np.exp(-agent.state.actor_interaction_rate)   # [IR-1] θ(μ)
if np.random.random() < actor_prob:
    active_actors.add(agent.agent_id)
```
Same fix applied to participant sampling (line 1936).

**Test coverage:** `test_ir1_actor_activation_uses_theta_mu`, `test_ir1_participant_activation_uses_theta_mu`

---

## Reputation Learning

### REP-1 — Highest-reputation selection included self (High)

**Description.** Section 6.4.4 defines the candidate set as `C \ {i}`. The baseline built candidates from all keys, so an agent could select itself as its own opinion leader.

**code_old.py** (inferred — no exclusion before candidate selection):
```python
candidates = [
    k for k, rep in self.state.reputation_estimates.items()
    if rep >= max_rep - self.config.delta
]
# self (agent_id) is in the dict and could be selected
```

**code_debugged.py** (line ~356):
```python
# [REP-1] Section 6.4.4 defines candidates over C\{i}.
non_self_estimates = {
    k: rep for k, rep in self.state.reputation_estimates.items()
    if k != self.agent_id
}
```

**Test coverage:** `test_rep1_highest_reputation_selection_excludes_self`, `test_role_identify_highest_rep_excludes_self`

---

### REP-2 — Reputation update deviated from Eq. (9) (High)

**Description.** Eq. (9) requires `s_i(k,t+1) = avg_j s_j(k,t) + Δv_i(k,t)`. The baseline performed two sequential EMA updates (one toward the gossip average, then another toward the raw payoff), which does not produce this additive-delta structure.

**code_old.py** (lines 221–228, BUG 3):
```python
# BUG 3: two separate EMA updates instead of Eq. (9) avg + fresh_delta
s_i += eta * (avg_estimate - s_i)
s_i += eta * (payoff - s_i)
```

**code_debugged.py** (line ~278):
```python
# [REP-2] Eq. (9): return fresh delta so reputation can be updated as avg + Δv
delta_v = v_new - v_old
...
self.state.reputation_estimates[agent_k] = avg_estimate + delta_v
```

**Test coverage:** `test_rep2_reputation_update_matches_eq9_additive_structure`, `test_rep2_gossip_mean_only_helper`

---

### REP-3 — Extra Phase 5 gossip doubled updates (Medium)

**Description.** Phase 4 already applied gossip averaging for all active participants inside `update_reputation_estimates_gossip()`. Phase 5 then ran a second pairwise gossip on the same step, giving each agent two reputation updates per timestep.

**code_old.py** (lines 480–499, BUG 4):
```python
# BUG 4: Phase 4 already gossipped; this is a second pass
if np.random.random() < self.config.gossip_rate and len(active_participants) >= 2:
    indices = np.random.choice(len(active_participants), 2, replace=False)
    agent1 = active_participants[indices[0]]
    agent2 = active_participants[indices[1]]
    # pairwise averaging ...
```

**code_debugged.py:** Phase 5 removed entirely. Gossip occurs once, inside Phase 4 participant updates.

**Test coverage:** `test_rep3_no_extra_phase5_pairwise_gossip_strict_noop_phase4`

---

### REP-4 — Personal-benefit estimates updated only for active participants (High)

**Description.** Section 6.4.2 specifies that every agent updates `v_i(k,t)` each step — active actors via the observed payoff, all others via zero-payoff decay. The baseline called `update_personal_benefit_estimates` only inside the active-participant loop, so agents not sampled as participants never decayed their stale estimates.

**code_old.py** (lines 465–467):
```python
for agent in active_participants:          # only the sampled subset
    agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t)
```

**code_debugged.py** (vectorised Phase 4, lines 1563–1570):
```python
# [REP-4] update v_i(k,t) for all agents i and all k each step
new_v = prev_v * (1.0 - eta_v_t)          # decay all (inactive path)
if active_actor_ids.size > 0:
    new_v[:, active_actor_ids] = (         # EMA update for active actors
        prev_v[:, active_actor_ids]
        + eta_v_t * (observed_utility_matrix[:, active_actor_ids] - prev_v[:, active_actor_ids])
    )
```

**Test coverage:** `test_rep642_all_agents_update_personal_benefit_each_step`, `test_rep4_step_decays_all_v_estimates_when_no_active_actors`

---

### REP-5 — Followers always copied leader's PU policy (High)

**Description.** Section 6.4.5 says followers emulate the leader's current behavior `w_k(t)`, which is `w_k^pu` when the leader is in PU role and `w_k^s` when the leader is in STATUS role. The baseline always copied `weights_pu`, so followers never adopted a STATUS leader's status policy.

**code_old.py** (inferred — `adopt_leader_behavior` copies `weights_pu` unconditionally):
```python
self.state.weights_pu = np.copy(leader.state.weights_pu)  # ignores STATUS weights
```

**code_debugged.py** (line 419):
```python
# [REP-5] Copy role-consistent leader behavior w_k(t), not always w_k^pu.
self.state.weights_pu = np.copy(leader.get_behavior_weights())
# get_behavior_weights() returns weights_status when leader.state.role == STATUS
```

**Test coverage:** `test_rep645_follower_tracks_status_policy_of_status_leader`

---

### REP-6 — Reputation reward estimate used EMA of leader payoff (High)

**Description.** Section 6.6 defines `Ĵ_i^r(t) = s_i(k,t)` — the current reputation estimate of the followed agent, not an EMA of their realised payoff. The baseline passed `leader_payoff` into the EMA update, so the reputation reward estimate tracked payoff noise rather than the accumulated reputation signal.

**code_old.py** (line 447):
```python
leader_payoff = this_step_payoffs.get(agent.state.following, 0.0)
agent.update_reputation_reward_estimate(leader_payoff, eta_J_t)  # EMA of payoff
```

**code_debugged.py** (line ~1996):
```python
# [REP-6] Use current followed-agent reputation estimate s_i(k,t).
followed_rep_estimate = agent.state.reputation_estimates.get(agent.state.following, 0.0)
agent.update_reputation_reward_estimate(followed_rep_estimate, eta_J_t)
```

**Test coverage:** `test_rep66_reputation_reward_estimate_matches_followed_agent_reputation`

---

### REP-7 — Personal-benefit learning used actor's own payoff (High)

**Description.** Section 6.4.2 defines `v_i(k,t)` using the observer-specific utility `u_i(s,x_k)`. The baseline computed a single payoff for actor `k` and reused it for every observer `i`. This compressed observer disagreement and made reputation differences between actions weaker than the paper intends.

**code_old.py** (lines 430–431):
```python
observed_payoffs = {i: this_step_payoffs.get(i, 0.0) for i in range(self.config.num_agents)}
# this_step_payoffs[k] = actor k's own reward — not observer-specific
```

**code_debugged.py** (lines 1954–1957):
```python
# [REP-7] each observer i evaluates actor k's (state, action) via u_i(s, x_k)
observer_utilities = self.compute_observer_utility_vector(state, action)
observed_utility_matrix[:, agent_id] = observer_utilities
```

**Test coverage:** `test_rep642_personal_benefit_is_observer_specific_in_step`, `test_rep642_numpy_fast_path_uses_observer_specific_utilities`

---

## Role Updates

### ROLE-1 — Non-followers could never enter REPUTATION role (High)

**Description.** Step-1 used `estimated_reward_rep` as the reputation signal for non-followers. This value starts at 0 and is only updated inside the REPUTATION branch, so agents already in PU role could never accumulate enough signal to switch. Fix: use `γ · s_i(L_i,t)` (= `γ · max_rep`) as the bootstrap signal per Sections 6.6 and 7.3.

**code_old.py** (line 575, BUG 1):
```python
# BUG 1: estimated_reward_rep stays 0 for non-followers forever
est_rep_weighted = self.config.gamma * agent.state.estimated_reward_rep
```

**code_debugged.py** (Step-1 role update):
```python
max_rep = max(agent.state.reputation_estimates.values()) if agent.state.reputation_estimates else 0.0
est_rep_weighted = self.config.gamma * max_rep   # [ROLE-1] bootstrap from observed signal
```

**Test coverage:** `test_role1_bootstrap_non_follower_from_reputation_signal`, `test_bug1_non_followers_do_not_switch_even_with_high_reputation_signal` (baseline)

---

### ROLE-2 — Extra `max_rep >= B_i` gate blocked all following (High)

**Description.** The baseline added `and max_rep >= B_i` to the Step-1 follow condition. This clause is absent from Section 7.3. With payoffs in `[0,1]`, `max_rep` typically converges near 0.5 — below `B_R = 0.8` — so the gate blocked reputation-role entry even after fixing ROLE-1.

**code_old.py** (line 583, BUG 2):
```python
if est_rep_weighted > max(B_i, est_pu) and max_rep >= B_i:   # extra gate not in paper
```

**code_debugged.py**:
```python
if est_rep_weighted > max(B_i, est_pu):   # [ROLE-2] paper condition only
```

**Test coverage:** `test_role2_step1_reputation_switch_does_not_use_extra_max_rep_gate`

---

### ROLE-3 — Indirect-follow redirect logic was backwards (Medium)

**Description.** When a selected target `best_k` was itself a follower, the agent should redirect to `best_k`'s leader. The baseline checked whether `best_k` was in the follower-sets of agents who follow `i` — a check that is both logically backwards and nonsensical.

**code_old.py** (lines 587–594, BUG 5):
```python
# BUG 5: backwards check — looks at who follows i, not whether best_k is a follower
if best_k in [followers[f] for f in range(self.config.num_agents) if i in followers[f]]:
    best_k_follower = next(f for f in range(self.config.num_agents) if i in followers[f])
    best_k = best_k_follower   # wrong: re-assigns to a follower id, not a leader
```

**code_debugged.py** (lines ~2278–2283):
```python
redirect_target_is_follower = (best_k in R and self.agents[best_k].state.following is not None)
if redirect_target_is_follower:
    best_k = self.agents[best_k].state.following   # [ROLE-3] follow best_k's leader
```

**Test coverage:** `test_role3_redirects_if_best_agent_is_already_follower`, `test_bug5_indirect_following_redirect_not_applied` (baseline)

---

### ROLE-4 — No self-follow prevention after redirect (High)

**Description.** After the ROLE-3 redirect, if the redirect chain circled back to agent `i` (e.g., `i → k → i`), the agent would follow itself. The baseline had no guard against this, leading to impossible follower-count states.

**code_old.py:** No check for `best_k == i` after redirect.

**code_debugged.py** (lines 2291–2297):
```python
# [ROLE-4] After redirect, ensure we don't follow ourselves
if best_k == i:
    if audit_rows is not None:
        audit_rows[i]["decision_code"] = "SELF_REDIRECT_BLOCK"
    continue   # stay in PU instead
```

**Test coverage:** `test_role_4_no_self_following_after_redirect`, `test_async_partial_update_redirect_prevents_self_follow_after_chain`

---

### ROLE-5 — Followers not redirected when agent became a follower (High)

**Description.** When agent `i` switches from leader to follower (e.g., after losing followers or switching roles), its own existing followers remained pointing to `i`. This created multi-level chains (`A→B→C`) and invalid states. ROLE-3 alone was insufficient because of processing-order dependency within the same update pass.

**code_old.py:** No follower-redirect step when `agent.state.role` changes to REPUTATION.

**code_debugged.py** (lines 2299–2307):
```python
# [ROLE-5] When agent i becomes a follower, redirect i's own followers to i's new leader.
if len(followers[i]) > 0:
    for follower_id in list(followers[i]):
        self.agents[follower_id].state.following = best_k
        followers[best_k].add(follower_id)
    followers[i].clear()
```

**Test coverage:** `test_role_5_redirect_followers_when_leader_becomes_follower`, `test_async_partial_update_redirects_existing_followers_when_agent_becomes_follower`

---

## Status Updates

### STATUS-1 — `estimated_reward_status` only updated inside STATUS branch (High)

**Description.** Step-2 entry condition is `κ · Ĵ_i^s > Ĵ_i^pu`. But `estimated_reward_status` was only updated in Phase 3 when an agent was already in STATUS role. Agents with followers could accumulate social support yet never enter STATUS because the estimate stayed at its initial value of 0.

**code_old.py** (inferred — status update inside `elif role == STATUS` branch only):
```python
elif agent.state.role == AgentRole.STATUS:
    ...
    agent.update_status_optimization(...)   # only STATUS agents accumulate Ĵ^s
# Non-STATUS agents with followers never update estimated_reward_status
```

**code_debugged.py** (lines ~1971–1987):
```python
# [STATUS-1 FIXED] Keep Ĵ^s current even before the agent formally switches into STATUS,
# so Step-2 can compare κ·Ĵ^s against Ĵ^pu.
if len(agent.state.followers) > 0:
    ...
    if agent.state.role != AgentRole.STATUS:
        agent.state.estimated_reward_status += eta_J_t * (
            social_support_sum - agent.state.estimated_reward_status
        )
```

**Test coverage:** `test_status_entry_can_occur_after_status_reward_learning`

---

## Discovery Timeline

| Date | Bugs found | Method |
|------|-----------|--------|
| 2026-02-19 | REP-2, REP-3, ROLE-1, ROLE-2, ROLE-3 | Baseline test suite + inline BUG comments |
| 2026-02-19 | IR-1, REP-1, ROLE-4, ROLE-5, STATUS-1 | Extended test suite in `code_by_peter_tests/` |
| 2026-03-05 | REP-4, REP-5, REP-6 | New reputation-coverage tests |
| 2026-03-12 | REP-7 | Reputation-learning audit and Experiment D debugging |
