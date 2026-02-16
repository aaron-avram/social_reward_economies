# Scale Experiment Analysis: Reputation Scaling (n=100)

## Setup

The scale experiment uses the same simulation as the toy experiment but at n=100 agents, testing whether leadership emergence holds at larger population size.

**Environment**: 2 states (sampled uniformly), 2 actions. Heterogeneous reward functions across agents.

**Agents**: Each agent maintains P_{j,i} (personal evaluation of each other agent) and R_{j,i} (accumulated reputation). Follow condition: `γ · R_{i,j*} > P_{i,i}`.

**Follower dynamics**: Influencer switching every 3000 timesteps. Gossip starts at t=50. Chain-following not allowed.

**Scale**: n=100 agents. (n=20 results are in the toy experiment.)

---

## Results (Paper Approach, lr=0.003, n=100)

| γ | Max Followers | Stable? | Stability | Convergence Time |
|---|---------------|---------|-----------|------------------|
| 3 | 45/99 | ❌ | 0.60 | — |
| 5 | 99/99 | ✅ | 1.00 | 18000 |
| 10 | 99/99 | ✅ | 1.00 | 6000 |
| 20 | 99/99 | ✅ | 1.00 | 12000 |

---

## Key Findings

**1. The convergence threshold shifts upward at n=100.**
At n=20, γ=3 was sufficient for stable leadership. At n=100, γ=3 fails — reaching only 45/99 followers with stability 0.60. A larger population requires a stronger reputation signal (γ ≥ 5) to produce stable leadership.

**2. γ=10 converges fastest.**
γ=10 converges by t=6000, faster than γ=5 (t=18000) and γ=20 (t=12000). The non-monotonic convergence time suggests that very high γ can overshoot, causing instability before settling.

**3. Full convergence (99/99) is achievable at n=100.**
For γ ≥ 5, a single leader with all 99 other agents as followers emerges and stabilizes, confirming the mechanism scales to larger populations with sufficient γ.

---

## Comparison: n=20 vs n=100 (lr=0.003)

| γ | n=20 Max Followers | n=20 Stable? | n=100 Max Followers | n=100 Stable? |
|---|-------------------|--------------|---------------------|---------------|
| 3 | 19/19 | ✅ | 45/99 | ❌ |
| 5 | 19/19 | ✅ | 99/99 | ✅ |
| 10 | 19/19 | ✅ | 99/99 | ✅ |
| 20 | 19/19 | ✅ | 99/99 | ✅ |

The minimum γ for stable leadership increases from γ=3 at n=20 to γ=5 at n=100, suggesting the required reputation scaling grows with population size.

---

## PDF Claim Assessment (n=100)

**PDF: "For γ > 2, a stable leader emerges"**

| γ | Verdict |
|---|---------|
| 3 | ❌ Fails — 45/99 followers, unstable |
| 5 | ✅ Confirmed — 99/99, stable by t=18000 |
| 10 | ✅ Confirmed — 99/99, stable by t=6000 |
| 20 | ✅ Confirmed — 99/99, stable by t=12000 |

The claim does not hold at n=100 for γ=3. The effective threshold appears to be γ ≥ 5 at this population size.
