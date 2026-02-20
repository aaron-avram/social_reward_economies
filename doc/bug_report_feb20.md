# Bug Report

This report summarizes the issues found in the baseline implementation and the corresponding corrections in the debugged implementation.

---

## Interaction Rates

### Active actor/participant sampling used raw `mu` instead of `theta(mu)`

In the baseline, activity was sampled directly from the rate value:

```python
# src/code_old.py
if np.random.random() < agent.state.actor_interaction_rate:
    active_actors.add(agent.agent_id)
```

The model in the paper defines activation as `theta(mu) = 1 - exp(-mu)`, not `mu` itself.

In the debugged version, sampling is done with the paper formula:

```python
# src/code_debugged.py
actor_prob = 1.0 - np.exp(-agent.state.actor_interaction_rate)
if np.random.random() < actor_prob:
    active_actors.add(agent.agent_id)
```

(and the participant formula is updated the same way).

This change keeps the participation process aligned with the model assumptions and prevents over-activation.

---

## Reputation Learning

### Highest-reputation candidate selection included the agent itself

The baseline could select the current agent as its own highest-reputation candidate because candidate sets were built from all keys.

```python
# src/code_old.py
candidates = [
    k for k, rep in self.state.reputation_estimates.items()
    if rep >= max_rep - self.config.delta
]
```

In the debugged code, self is explicitly excluded before candidate selection:

```python
# src/code_debugged.py
non_self_estimates = {
    k: rep for k, rep in self.state.reputation_estimates.items()
    if k != self.agent_id
}
```

This avoids the situation where agent i follows itself.

### Reputation update did not match Eq. (9)

The baseline used two EMA-style updates (toward gossip average, then toward payoff), which is different from Eq. (9) in section 6.4.3.

```python
# src/code_old.py
s_i += eta * (avg_estimate - s_i)
s_i += eta * (payoff - s_i)
```

Eq. (9) requires an additive structure:

` s_i(k,t+1) = avg_j s_j(k,t) + (v_i(k,t+1)-v_i(k,t)) `

In the debugged implementation, the personal-benefit delta is computed explicitly and then added to the average estimate:

```python
# src/code_debugged.py
personal_benefit_deltas[agent_k] = new_val - prev_val
...
self.state.reputation_estimates[agent_k] = avg_estimate + delta_v
```

This fix keeps reputation updates consistent with the paper formula.

### Gossip was applied twice in one timestep

After participant updates, the baseline applied an additional pairwise gossip phase in the same step.

```python
# src/code_old.py
# Phase 4: participant reputation updates
...
# Phase 5: additional pairwise gossip
```

In the debugged code, the extra pairwise pass is removed, so gossip is applied once per step through participant updates.

This prevents double-counting the same social signal and keeps convergence behaviour more tractable.

---

## Role Updates

### Reputation-role entry for non-followers was blocked

Agents not already following someone could not switch into the reputation role, even when another agent had high observed reputation. In Step-1, the baseline used `estimated_reward_rep`, which typically stayed at 0 for non-followers, instead of using a signal derived from observed reputation.

The baseline used `estimated_reward_rep` in Step-1, which is typically zero for non-followers:

```python
# src/code_old.py
est_rep_weighted = self.config.gamma * agent.state.estimated_reward_rep
```

The debugged version uses the observed best candidate signal (`s_i(L_i,t)`, implemented as `max_rep`) for role switching:

```python
# src/code_debugged.py
max_rep = max(agent.state.reputation_estimates.values()) if agent.state.reputation_estimates else 0.0
est_rep_weighted = self.config.gamma * max_rep
```


### Step-1 follow decision had an extra gate not in Section 7.3

The baseline required an additional condition:

```python
# src/code_old.py
if est_rep_weighted > max(B_i, est_pu) and max_rep >= B_i:
```

The debugged code keeps only the paper condition:

```python
# src/code_debugged.py
if est_rep_weighted > max(B_i, est_pu):
```

We remove the extra condition that was not in the paper.

### Indirect follow redirection logic was incorrect

When a selected target was already a follower, baseline code's logic did not reliably redirect to that target’s leader.

In the debugged implementation:

```python
# src/code_debugged.py
if best_k in R and self.agents[best_k].state.following is not None:
    best_k = self.agents[best_k].state.following
```

This keeps follower chains coherent and avoids inconsistent influence paths.

---

## Validation and Tests

To verify the issues and fixes, we ran tests on both implementations.

For the baseline implementation (`code_old.py`), we ran:

```bash
pytest -q baseline_debug_reputation_interaction_2026-02-19/test_reputation_behavior.py \
         baseline_debug_reputation_interaction_2026-02-19/test_interaction_rates.py \
         baseline_debug_reputation_interaction_2026-02-19/test_existing_bug_comments.py
```

Result in this run: **8 failed, 3 passed**.  
These failures match the behavior described in this report.

For the debugged implementation (`code_debugged.py`), we ran:

```bash
pytest -q code_by_peter_tests/test_bug_report_fixes.py
```

Result in this run: **11 passed**.

The test outcomes are consistent with the baseline containing the identified bugs and the debugged implementation addressing them.
