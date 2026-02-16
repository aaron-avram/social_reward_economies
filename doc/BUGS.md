# Bugs in `code_by_peter.py`

Audit against `learning_paper_newest_ver_transcription.md`.

---

## Bug 1 (Critical): Wrong `Ĵ^r_i` in Follow Condition

**File/Line**: `_update_roles_sequential()`, line 561

**Paper (Section 7.3)**:
> If `γ·Ĵ^r_i(s_n) > max{B_i, Ĵ^{pu}_i(s_n)}`: agent i follows L_i(s_n).

**Section 6.6** defines:
> `Ĵ^r_i`: set equal to `s_i(k, t)` for the agent k being followed

At the role-update decision point for a **non-follower**, there is no "agent being followed" yet. The correct interpretation is: `Ĵ^r_i = s_i(L_i(s_n), s_n)` — the current reputation estimate of the highest-reputation candidate `L_i`. This is `max_rep` in Peter's code.

**What Peter's code does**:
```python
est_rep_weighted = self.config.gamma * agent.state.estimated_reward_rep
```
`estimated_reward_rep` is only updated via EMA when an agent is *already* in REPUTATION role (following someone). For all non-followers it stays at its initial value of 0.0.

**Effect**: `est_rep_weighted = 0` for every non-follower. The follow condition `0 > max(0.8, est_pu)` is never true. **No agent ever transitions to REPUTATION role.**

**Fix**:
```python
est_rep_weighted = self.config.gamma * max_rep  # s_i(L_i, t) per Section 6.6
```

---

## Bug 2 (Critical): Extra `max_rep >= B_i` Check Not in Paper

**File/Line**: `_update_roles_sequential()`, line 565

**Paper (Section 7.3)** condition (complete):
> `γ·Ĵ^r_i(s_n) > max{B_i, Ĵ^{pu}_i(s_n)}`

There is only **one inequality**. `B_i` appears on the RHS inside the `max{}`.

**What Peter's code does**:
```python
if est_rep_weighted > max(B_i, est_pu) and max_rep >= B_i:
```
Adds a second condition `max_rep >= B_i`. With `B_R = 0.8` and payoffs in [0,1] (meaning reputation estimates converge toward ~0.5), `max_rep >= 0.8` is essentially never true.

**Effect**: Even if Bug 1 were fixed (so `est_rep_weighted = gamma * max_rep ~1.0 > 0.8`), the extra check `max_rep >= 0.8` still blocks all following. Combined with Bug 1, no leader ever emerges.

**Fix**: Remove the `and max_rep >= B_i` clause:
```python
if est_rep_weighted > max(B_i, est_pu):
```

---

## Bug 3 (Minor): Gossip Formula Deviates from Eq. 9

**File/Line**: `update_reputation_estimates_gossip()`, lines ~210–240

**Paper (Eq. 9)**:
```
s_i(k, t+1) = [Σ_{j∈B(t)\{k}} s_j(k,t) / |A(t)|] + v_i(k, t+1) − v_i(k, t)
```
The formula is: **all active agents' average** of `s_j` PLUS the **fresh delta** `Δv_i = v_i(k,t+1) - v_i(k,t)`.

**What Peter's code does**:
1. EMA toward the average of other active participants' estimates: `s_i[k] += eta * (avg - s_i[k])`
2. Then separately, EMA toward the raw payoff: `s_i[k] += eta * (payoff - s_i[k])`

This is EMA-toward-payoff, not `avg + fresh_delta`. The `fresh_delta` (`v_i(k,t+1) - v_i(k,t)`) is never computed and added.

**Effect**: Reputation estimates converge more slowly and don't incorporate the incremental local observation correctly. This is a functional deviation but not a showstopper — the system still learns reputations, just less efficiently.

**Note**: Implementing Eq. 9 exactly requires saving `v_i(k,t)` before the update step, computing the delta, and adding it directly (not via EMA). Peter's implementation in Phase 4 overwrites estimates without the clean additive-delta structure.

**Partial Fix** (to exactly match Eq. 9): save `old_v` before `update_personal_benefit_estimates`, compute `delta_v = new_v - old_v`, then set `s_i(k) = avg_of_active + delta_v` (not EMA).

---

## Bug 4 (Minor): Double Gossip Per Step

**File/Line**: `step()` — Phase 4 (lines ~460–472) and Phase 5 (lines ~476–496)

Phase 4 already does gossip averaging in `update_reputation_estimates_gossip()` (all active participants). Phase 5 then does an additional pairwise gossip step. This is redundant and not matching the paper's single gossip update per timestep.

**Fix**: Either remove Phase 5 or remove the gossip averaging from Phase 4 (keep only Phase 5 pairwise gossip, which is simpler to implement correctly).

---

## Bug 5 (Minor): Indirect Follower Check is Broken

**File/Line**: `_update_roles_sequential()`, lines 570–573

**Paper (Section 7.3)**: "If `L_i(s_n)` is already a follower, follow the influencer of `L_i(s_n)` instead."

**What Peter's code does**:
```python
if best_k in [followers[f] for f in range(self.config.num_agents) if i in followers[f]]:
```
This checks if `best_k` is in the follower sets of agents who follow `i` — which makes no sense. It should check if `best_k` is itself a follower (i.e., `best_k in R`, or `self.agents[best_k].state.following is not None`).

**Fix**:
```python
if best_k in R:  # best_k is already following someone
    # Follow best_k's leader instead
    best_k = self.agents[best_k].state.following or best_k
```

---

## Bug 6 (Non-Bug): Plot Save Path

**File/Line**: `__main__`, line 915

```python
system.plot_results("/mnt/user-data/outputs/sections_6_7_corrected.png")
```

This path is a cloud storage path that doesn't exist locally. When running locally, this raises `FileNotFoundError`. Change to a local path like `"results_6_7.png"`.

---

## Summary

| # | Severity | Location | Description |
|---|----------|----------|-------------|
| 1 | **Critical** | `_update_roles_sequential` line 561 | Uses `estimated_reward_rep=0` instead of `max_rep` as `Ĵ^r_i` |
| 2 | **Critical** | `_update_roles_sequential` line 565 | Extra `max_rep >= B_i` gate not in paper |
| 3 | Minor | `update_reputation_estimates_gossip` | Not exactly Eq. 9: EMA-toward-payoff, not avg+fresh_delta |
| 4 | Minor | `step()` phases 4+5 | Double gossip per timestep |
| 5 | Minor | `_update_roles_sequential` line 570 | Indirect follower check logic is wrong |
| 6 | Env issue | `__main__` line 915 | Plot save path doesn't exist locally |

Bugs 1 and 2 together completely prevent leader emergence. All other bugs are secondary. Fixing Bugs 1 and 2 is sufficient to make the system produce meaningful results.
