# Toy Experiment Analysis: Reputation Scaling (n=20)

## Setup

The toy experiment is a minimal multi-agent simulation designed to isolate the reputation scaling mechanism from the full `norm/` codebase.

**Environment**: 2 states (sampled uniformly), 2 actions. Each agent has a state-dependent reward function: in state 0, action 0 is optimal (u=1.0); in state 1, action 1 is optimal (u=1.0). Reward functions are heterogeneous across agents (small random offsets around the optimal values).

**Agents**: Each agent maintains:
- `P[j]`: personal evaluation of agent j (EMA of own reward from j's actions)
- `R[j]`: accumulated reputation of agent j (sum of incremental P updates)
- A policy network trained by policy gradient

**Follower dynamics**: Every 3000 timesteps, each agent checks whether any other agent's scaled reputation `γ·R[j]` exceeds its own `P[i,i]` (personal evaluation of its own actions). If so, it follows that agent and copies its actions. Agents with followers cannot become followers (no chain-following). Gossip (global reputation averaging) starts at t=50.

**Scale**: n=20 agents. (n=100 results are covered separately in the scaling experiment.)

---

## Formula Verification Against Paper

The paper defines a two-index personal evaluation P_{j,i} — agent j's estimate of agent i's reputation — updated as:

```
P_{j,i}^{(t+1)} = (1−α) P_{j,i}^{(t)} + α · q_j(s(t), a_i(t))
R_{j,i}^{(t+1)} = R_{j,i}^{(t)} + (P_{j,i}^{(t+1)} − P_{j,i}^{(t)})
```

The follow condition is:
```
γ · R_{i,j*} > P_{i,i}     (follow j* if scaled reputation beats own personal evaluation)
```

### What Is Correctly Implemented

- **P update** (`update_reputation_paper`): `self.P[agent_id] = (1-α)*self.P[agent_id] + α*reward` where `reward = q_j(s, a_i)` — matches P_{j,i} update ✓
- **R increment**: `self.R[agent_id] += self.P[agent_id] - old_P` — matches R_{j,i} += ΔP_{j,i} ✓
- **Gossip** (`toy_experiment.py`): averages all agents' R vectors — matches paper formula ✓
- **Reputation update loop**: every observer j updates its estimate of every target i ✓

### Follow Condition Fix

**Paper**: `γ · R_{i,j*} > P_{i,i}` where `P_{i,i}` = agent i's personal evaluation of its own actions.

**Previous implementation** used `self.independent_beta` as the threshold — a separate EMA of own received reward, frozen when the agent is a follower.

**Current implementation** uses `self.P[self.id]` — the direct paper quantity P_{i,i}, continuously updated regardless of role. Since `rep` already equals `γ · R[j]` (scaled), the comparison is `γ·R[j] > P[i,i]`, i.e., `rep > self.P[self.id]`.

This aligns `toy_agent.py` exactly with the paper formula.

### Known Discrepancy in `norm_agent.py` (train.py)

`norm_agent.py` (used by the main training scripts, not the toy experiment) still uses `independent_beta` as the follow threshold. This is a reasonable approximation but technically deviates from the paper's `P_{i,i}`. It is documented here but **not changed** — only `toy_agent.py` is updated to match the paper.

---

## Results (Paper Approach, lr=0.003, n=20)

| γ | Max Followers | Stable? | Stability | Convergence Time |
|---|---------------|---------|-----------|------------------|
| 2 | 12/19 | ❌ | 0.48 | — |
| 3 | 19/19 | ✅ | 1.00 | 15000 |
| 5 | 19/19 | ✅ | 1.00 | 3000 |
| 10 | 19/19 | ✅ | 1.00 | 3000 |
| 20 | 19/19 | ✅ | 1.00 | 3000 |

---

## Key Findings

**1. The paper approach with lr=0.003 fails only at γ=2.**
γ=2 reaches only 12/19 followers with stability 0.48 — the reputation signal is too weak to produce stable leadership. All γ ≥ 3 converge stably.

**2. Convergence is faster at higher γ.**
γ=3 takes until t=15000 to converge; γ≥5 converges by t=3000. Higher γ amplifies the reputation signal more, allowing a dominant leader to emerge sooner.

---

## PDF Claim Assessment (n=20)

**PDF: "For γ > 2, a stable leader with 19 followers emerges"**

| γ | Verdict |
|---|---------|
| 2 | ❌ Fails — 12/19 followers, unstable |
| 3 | ✅ Confirmed — 19/19, stable by t=15000 |
| 5 | ✅ Confirmed — 19/19, stable by t=3000 |
| 10 | ✅ Confirmed — 19/19, stable by t=3000 |
| 20 | ✅ Confirmed — 19/19, stable by t=3000 |

The claim holds for γ ≥ 3. γ=2 is a borderline failure where the reputation signal is insufficient to produce stable leadership.
