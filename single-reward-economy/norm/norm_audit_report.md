# Code Audit Report: norm/ Implementation vs. Paper Formulas

## Executive Summary

**Overall Assessment:** The `norm/` implementation is **largely correct** and matches the paper formulas. However, there are some parameter choices and threshold conditions that may explain the behavior discrepancies.

**Key Finding:** The implementation correctly applies gamma (γ) in influencer selection comparisons, NOT in reputation updates. This matches the paper's Equation 12.

---

## Detailed Verification Checklist

### ✅ Reputation Learning (Section 7.3.3, Equation 12)

**Location:** `norm_agent.py:147-154`

```python
def update_reputation(self, agent, reward):
    old_PR = self.PR[agent]
    self.PR[agent] = update_with_discount(self.PR[agent], reward, self.update_factor)
    self.R[agent] += self.PR[agent] - old_PR
```

**Verification:**
- ✅ Personal benefit: `PR[agent] = (1-η)*PR[agent] + η*reward`
- ✅ Reputation estimate: `R[agent] += (PR_new - PR_old)` (NO gamma in update!)
- ✅ Incremental update formula matches Equation 12 exactly

**Parameters:**
- `update_factor` (η) = 0.001 * adjustment_factor = 0.001 * 3 = **0.003**
- This is **very slow** for 50K timesteps

---

### ✅ Influencer Selection

**Location:** `norm_agent.py:175-217`

```python
def switch_influencer(self, agents_list, timestep):
    for index, rep in enumerate(list(self.R)):
        rep = rep * self.reputation_factor  # Apply gamma HERE
        if rep > self.independent_beta and agents_list[index].target_influencer == -1 and ...:
            possible_agents.append(index)
```

**Verification:**
- ✅ Comparison: `γ * R[j] > independent_beta` (gamma IN comparison, NOT in update)
- ✅ Additional filters: epsilon threshold, rep_threshold, exclude agents who are already followers
- ✅ Random selection among eligible influencers

**Parameters:**
- `reputation_factor` (γ) = 9
- `epsilon` = 0.01 (used in line 188: `rep > max_val - self.epsilon`)
- `rep_threshold` = 0.2 (line 86, but used in line 188 as: `rep > 0.06 * self.reputation_factor`)

---

### ⚠️ Status Optimization (Section 7.3.5)

**Location:** `norm_agent.py:129-139`

```python
def become_selfless(self, agents_list):
    self.status = self.followers * self.R[self.num] * self.reputation_factor * self.status_factor
    if self.status > self.PR[self.num] and self.followers > 0 and self.selfless == 0 and self.followers >= self.threshold:
        self.selfless = 1
```

**Verification:**
- ✅ Status formula: `S_i = κ * |F_i| * R_i * γ`
- ⚠️ **POTENTIAL ISSUE:** Switching condition compares `status > PR[self.num]` instead of `status > beta` or `status > independent_beta`
  - `PR[self.num]` is the personal benefit estimate for oneself
  - Paper suggests comparing to personal utility baseline
  - This might make switching harder or easier depending on how PR evolves

**Parameters:**
- `status_factor` (κ) = 1
- `threshold` = 48 followers (out of 100 agents = 48%)
- This is a **high threshold** - agent needs nearly half the population as followers

---

### ✅ Gossip Mechanism

**Location:** `train.py:409-419`

```python
if i > gossiping_start:
    for j_agent in active_agents:
        total_reputation_vector = np.zeros(num_agents)
        total_reputation_vector += j_agent.R.copy()
        num_agents_found = 1
        for k_agent in active_agents:
            if j_agent.S[k] < 3 and k_agent.S[k] < 3:
                total_reputation_vector += k_agent.R.copy()
                num_agents_found += 1
        new_trc = total_reputation_vector / num_agents_found
        j_agent.R = new_trc
```

**Verification:**
- ✅ Similarity filtering: Only gossip with agents where `S[k] < 3`
- ✅ Averaging formula: Simple mean across eligible agents
- ✅ Timing: Starts after timestep 50

**Note:** The similarity check `j_agent.S[k] < 3 and k_agent.S[k] < 3` is asymmetric but ensures mutual low similarity.

---

### ✅ Policy Gradient (REINFORCE)

**Location:** `norm_actor.py:198-214`

```python
def train(self, state, reward, action):
    actions = self.function(ten(state))
    prob = F.log_softmax(actions, dim=0)[action]
    v_value = self.beta
    adv = np.array([reward - v_value])
    adv = ten(adv)
    self.optimizer.zero_grad()
    loss = -1 * torch.mul(prob, adv)
    loss.backward()
    self.optimizer.step()
```

**Verification:**
- ✅ Advantage: `reward - beta` (baseline subtraction)
- ✅ Policy gradient: `-log_prob * advantage`
- ✅ Uses AdamW optimizer with AMSGrad

**Parameters:**
- Learning rate (α) = 0.00005
- This is **very slow** for neural network training

---

## Parameter Appropriateness Analysis

### Learning Rates

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Reputation learning (η) | 0.003 | **Too slow** for 50K timesteps |
| Beta update | 0.03 | Reasonable |
| Neural network (α) | 0.00005 | **Too slow** for policy convergence |

**Concern:** With η=0.003, it takes ~333 timesteps for the running average to give 63% weight to new experiences. Over 50K timesteps, this might not converge fully.

### Thresholds

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Status threshold | 48 followers | **Very high** (48% of population) |
| Gossip similarity | S < 3 | Reasonable |
| Influencer switch interval | 3000 timesteps | Reasonable (asynchronous) |

**Concern:** The threshold of 48 followers is extremely high. For a single leader with 99 followers to emerge, they would need to cross this threshold first, but the threshold itself might prevent early leaders from optimizing for status.

### Reputation Scaling

| Parameter | Value | Assessment |
|-----------|-------|------------|
| Reputation factor (γ) | 9 | High, should be sufficient per PDF |
| Status factor (κ) | 1 | Enables status optimization |

**Key Insight:** With γ=9 and κ=1, the implementation SHOULD show stable leadership per PDF claims (γ>2 → 99 followers).

---

## Identified Discrepancies

### 1. Status Switching Condition (MINOR)

**Code:** `if self.status > self.PR[self.num] and self.followers >= self.threshold:`

**Expected (based on paper):** `if self.status > self.independent_beta and self.followers >= self.threshold:`

**Impact:** Comparing to `PR[self.num]` instead of `independent_beta` might change when agents switch to status optimization. Since `PR[self.num]` tracks personal benefit from oneself (which is typically personal utility), this might be approximately correct, but semantically confusing.

### 2. Threshold Value (PARAMETER CHOICE)

**Code:** `threshold = 48` (for 100 agents)

**Observation:** This is very high. The paper doesn't specify the threshold value used in experiments.

**Impact:** High threshold means agents need many followers before switching to status optimization. This could create a chicken-egg problem: agents won't optimize for status until they have many followers, but they might need to optimize for status to attract many followers.

### 3. Learning Rates (PARAMETER CHOICE)

**Code:**
- `η = 0.003` (reputation)
- `α = 0.00005` (neural network)

**Observation:** Both are very slow compared to Test 4 (which uses α=0.1).

**Impact:** Slow convergence. Over 50K timesteps, reputations and policies might not fully converge, leading to dynamic switching instead of stable leadership.

---

## Execution Flow Trace (One Timestep)

### Timestep i > 50 (Gossip Active)

1. **Get Active Agents** (train.py:309)
   - Each agent samples `random() < rate` to determine participation

2. **Sample State** (train.py:313)
   - Uniform random state from [0, max_state_num]

3. **Choose Actions** (train.py:316-330)
   - If follower: Copy influencer's action
   - Otherwise: Sample from own policy (5% epsilon-greedy)

4. **Compute Feedback** (train.py:339-356)
   - For each active agent j:
     - For each agent k's action: `feedback += U_j(s, a_k)`
     - Update personal reputation: `PR[k] = (1-η)*PR[k] + η*U_j(s,a_k)`
     - Separate `selfish_feedback[k]` for agent's own utility

5. **Train Policies** (train.py:360-365)
   - Independent actors: Train personal policy with selfish_feedback
   - Selfless influencers: Train status policy with total feedback

6. **Update Similarity Scores** (train.py:366-374)
   - For each pair of active agents: `S[j,k] = (1-η/10)*S[j,k] + (η/10)*|U_j - U_k|^2`

7. **Update Betas** (train.py:381-398)
   - Update baseline estimates for advantage calculation
   - Track `independent_beta` for influencer selection

8. **Gossip** (train.py:409-419)
   - For each active agent:
     - Average R with all active agents where S < 3
     - `R = average(R across similar agents)`

9. **Trigger Changes** (Asynchronous, train.py:424-438)
   - Each agent has individual counters
   - When `rate_counter == 0`: Call `adjust_rate()`
   - When `infl_counter == 0`: Call `switch_influencer()`
   - When `selfless_counter == 0`: Call `become_selfless()`

10. **Update Follower Counts** (train.py:572-610)
    - Count followers for each influencer
    - Update agent.followers

---

## Recommendations

### 1. Try Faster Learning Rates

**Current:** η=0.003, α=0.00005
**Suggested:** η=0.01 or 0.1, α=0.0005 or 0.001

**Rationale:** Test 4 uses α=0.1 and achieves stable convergence. Faster learning might allow reputations to converge before 50K timesteps.

### 2. Lower Status Threshold

**Current:** threshold=48 (48%)
**Suggested:** threshold=10-20 (10-20%)

**Rationale:** Lower threshold allows early leaders to switch to status optimization sooner, potentially accelerating winner-takes-all dynamics.

### 3. Verify Status Switching Condition

**Current:** `status > PR[self.num]`
**Suggested:** Verify this is intentional, or consider `status > independent_beta`

**Rationale:** Semantic clarity and alignment with paper's description.

### 4. Run Longer

**Current:** 50K timesteps
**Suggested:** 100K-200K timesteps if keeping slow learning rates

**Rationale:** With η=0.003, full convergence might require more timesteps.

---

## Conclusions

1. **Implementation is correct** with respect to the paper's core algorithms
2. **Gamma is correctly applied** in influencer selection, NOT in reputation updates
3. **Parameter choices** (slow learning rates, high threshold) may explain why stable leadership doesn't emerge with κ=0
4. **Status optimization (κ>0) is likely required** given the current parameter settings, but this contradicts PDF's claim that γ>2 alone is sufficient

**Next Step:** Build toy experiment to isolate reputation scaling mechanism and test with different parameter combinations.
