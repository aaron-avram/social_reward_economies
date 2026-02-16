# Plan Implementation Summary

## What Was Done

### Task 1: Code Audit of norm/

**Location:** `/single-reward-economy/norm/norm_audit_report.md`

**Findings:**
- norm/ correctly implements paper's Equation 12
- Reputation updates: `R[i] += (P_new - P_old)` (no γ in update)
- Influencer selection: `γ * R[j] > beta` (γ in comparison)
- Status optimization: `S = κ * |F| * R * γ`
- No implementation bugs found

### Task 2: Toy Experiment — n=20 Gamma Sweep

**Location:** `/single-reward-economy/reputation_toy_experiment/`

Initial sweep comparing Test 4 (direct estimation) vs. Paper (Eq. 12) approaches at n=20.

**Note:** The original paper-approach results at n=20 had a bug causing all runs below γ=20 to report failure. Those results are superseded by the corrected sweep below.

### Task 3: Corrected n=20 Sweep

Re-ran paper approach with bug fixed. Result: paper approach with lr=0.003 converges stably for **all γ ≥ 2** at n=20. The original conclusion ("paper requires γ≈20") was wrong.

### Task 4: n=100 Scale Experiment

Ran paper approach at n=100 agents with γ ∈ {3, 5, 10, 20} and lr ∈ {0.003, 0.1}.

### Task 5: norm/ κ=0 Run

Ran norm/ with κ=0 (status optimization disabled), γ=9, n=100.

---

## Full Results Table

### Test 4 Approach (n=20, lr=0.1)

| γ | Max Followers | Stable? | Convergence Time |
|---|---------------|---------|------------------|
| 2–20 | 19/19 | ✅ Yes | 3000 |

Test 4 is uniformly robust at n=20.

### Paper Approach — n=20 Corrected (lr=0.003)

| γ | Max Followers | Stable? | Convergence Time |
|---|---------------|---------|------------------|
| 2 | 19/19 | ✅ Yes | 3000 |
| 3 | 19/19 | ✅ Yes | 3000 |
| 5 | 19/19 | ✅ Yes | 3000 |
| 10 | 19/19 | ✅ Yes | 3000 |
| 20 | 19/19 | ✅ Yes | 3000 |

### Paper Approach — n=100 Scale

| γ | lr | Max Followers | Stable? | Stability | Convergence Time |
|---|----|---------------|---------|-----------|------------------|
| 3 | 0.003 | 96/99 | ❌ False | 0.48 | 48000 |
| **5** | **0.003** | **99/99** | **✅ True** | **1.00** | **12000** |
| 10 | 0.003 | 54/99 | ❌ False | 0.76 | 3000 |
| **20** | **0.003** | **99/99** | **✅ True** | **1.00** | **3000** |
| 3 | 0.1 | 18/99 | ❌ False | 0.24 | — |
| 5 | 0.1 | 30/99 | ❌ False | 0.36 | — |
| 10 | 0.1 | 41/99 | ❌ False | 0.40 | 12000 |
| 20 | 0.1 | 49/99 | ❌ False | 0.48 | 12000 |

### norm/ with κ=0 (γ=9, n=100, lr=0.003)

- Agent 93 dominated t≈5K–46K; switched to Agent 94 near end
- Peak follower count ~94, not stably held
- Selfless count = 0 (κ=0 means status optimization never activates)
- Outcome: partial convergence, consistent with γ=9 being in the unstable zone at n=100

---

## Current Understanding

### What drives convergence at n=100

1. **lr=0.003 is required.** lr=0.1 fails for all γ at n=100. Fast learning erodes the leader's advantage because followers' β baselines rise in step with the leader's signal.

2. **Convergence is non-monotonic in γ.** γ=5 and γ=20 achieve clean 99/99 stable; γ=3 and γ=10 do not. The failure modes differ:
   - γ=3: nearly converges (96/99) but leadership unstable — likely insufficient signal margin
   - γ=10: only 54/99 peak — likely multi-leader competition (many agents simultaneously cross the threshold)

3. **Signal ratio is not the bottleneck.** γ·R_max / β_mean ≈ γ throughout all runs, stable regardless of convergence outcome. The issue is leader stability, not signal weakness.

4. **Hypothesis for non-monotonicity:** At intermediate γ (around 8–15), the threshold is low enough that multiple agents cross it simultaneously, creating competing leaders. At γ=5 the threshold is selectively crossed by only the best agent; at γ=20 the best agent's advantage is large enough to dominate quickly despite multiple crossings.

### norm/ status

| Configuration | Outcome |
|---------------|---------|
| κ=0, γ=9 | Partial convergence, unstable leader, ~94 peak followers |
| κ=1, γ=9 | Clean 99/99 stable convergence |

Status optimization (κ=1) stabilizes leader dynamics even in the unstable γ band, by giving the current leader an additional incentive to maintain high social welfare.

---

## Open Questions

1. **Where exactly is the unstable boundary?** γ=5 succeeds, γ=10 fails, γ=20 succeeds. Is there a second stable window above γ=15? Testing γ=7, 8, 9, 12, 15 at n=100 would characterize the boundary.

2. **Why does n=20 not show the unstable band?** All γ ≥ 2 succeed at n=20. The multi-leader competition effect may be suppressed at small n because there are fewer near-equal agents to compete.

3. **Is γ=3 a different failure mode than γ=10?** γ=3 gets 96/99 followers but stability 0.48; γ=10 gets only 54/99. These look like distinct mechanisms (weak signal vs. multi-leader competition).

---

## Next Steps

### Immediate

1. **Map the stable/unstable boundary at n=100:** Run γ ∈ {6, 7, 8, 9, 12, 15} with lr=0.003 to identify the exact range.

2. **Use γ=20 for norm/ production runs.** It reliably converges at n=100 with lr=0.003 and is the current setting that achieves 99-follower stable outcomes.

3. **Or use γ=5** if faster convergence is needed (12000 vs. 3000 timesteps, but still stable).

### Investigation

1. Test whether the unstable band shifts with n (e.g., does γ=10 become stable at n=50?).

2. Verify whether κ=1 rescues convergence across the entire unstable band (γ ∈ {3, 10} at n=100).

---

## Files Created

### Documentation
- `/single-reward-economy/norm/norm_audit_report.md`
- `/single-reward-economy/reputation_toy_experiment/ANALYSIS.md`
- `/single-reward-economy/reputation_toy_experiment/Results/SUMMARY.md`
- `/single-reward-economy/PLAN_IMPLEMENTATION_SUMMARY.md` (this file)

### Code
- `/single-reward-economy/reputation_toy_experiment/toy_env.py`
- `/single-reward-economy/reputation_toy_experiment/toy_agent.py`
- `/single-reward-economy/reputation_toy_experiment/toy_experiment.py`
- `/single-reward-economy/reputation_toy_experiment/run_gamma_sweep.py`

### Results
- All plots in `/single-reward-economy/reputation_toy_experiment/Results/`
- n=100 scale experiment results in `/single-reward-economy/reputation_scaling/Results/`
