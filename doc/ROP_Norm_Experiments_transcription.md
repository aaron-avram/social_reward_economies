# Norm Experiments

**Author:** Kashish Mittal  
**Date:** January 2026

## 1. Personal Utility

Personal Utility of an agent \(i\) is:

\[
U_i(\pi_i) = \sum_{s \in S} p(s)\,u_i(s,\pi_i(s)).
\]

### 1.1 Test 1

If \(u_i(s,\pi_i(s)) = 1\) for all agents \(i\) and all states \(s\), then the utility of each agent under policy \(\pi_i\) is

\[
U_i(\pi_i) = \sum_{s\in S} p(s)\,u_i(s,\pi_i(s))
= \sum_{s\in S} p(s)\cdot 1
= \sum_{s\in S} p(s)
= 1,
\]

so \(U_i(\pi_i)=1\) for all agents \(i\).

#### 1.1.1 Results from Simulation

Parameters: (not specified)

**Figure 1 (page 1): Self Utility Overall For all 10 Agents**  
Description: A line plot titled *“Self-Utility Over Time”* showing self-utility for Agents 0–9 over timesteps up to about 50,000. All agent curves quickly rise and converge near 1.0, then remain stable.

---

### 1.2 Test 2

Let the state space be

\[
S=\{0\},
\]

with probability

\[
p(0)=1.
\]

Let the action space be

\[
A=\{0,1\}.
\]

For all agents \(i\), there are two possible policies:

\[
\pi_i(0)=0 \quad 	ext{or} \quad \pi_i(0)=1.
\]

Utilities are defined as

\[
u_i(0,0)=0,\quad u_i(0,1)=1.
\]

**Policy 1:** \(\pi_i(0)=0\)

\[
U_i(\pi_i)=\sum_{s\in S}p(s)u_i(s,\pi_i(s))
= p(0)u_i(0,0)
=1\cdot 0
=0.
\]

**Policy 2:** \(\pi_i(0)=1\)

\[
U_i(\pi_i)=\sum_{s\in S}p(s)u_i(s,\pi_i(s))
= p(0)u_i(0,1)
=1\cdot 1
=1.
\]

**Expected Outcome.** Since agents choose policies that maximize personal utility, all agents select policy

\[
\pi_i(0)=1,
\]

resulting in observed personal utility

\[
U_i(\pi_i)=1.
\]

#### 1.2.1 Results from Simulation

**Figure 2 (page 2): Actions Chosen by Agents Over Time**  
Description: A line plot of chosen actions versus time. Most agent trajectories rapidly settle at Action 1 and stay there, with a brief deviation/spike for one trajectory before returning to Action 1.

**Figure 3 (page 3): Self Utility Overall For all 10 Agents**  
Description: A line plot titled *“Self-Utility Over Time”* where all agents’ self-utility quickly approaches ~1.0 and stays close to that level for the full horizon.

---

### 1.3 Test 3

Let the state space be

\[
S=\{0,1\},
\]

with state probabilities

\[
p(0)=0.6,\quad p(1)=0.4.
\]

Let the action space consist of a single action,

\[
A=\{0\}.
\]

For all agents \(i\), the only policy is

\[
\pi_i(0)=\pi_i(1)=0.
\]

For all agents \(i\), utilities are:

\[
u_i(0,0)=1,\quad u_i(0,1)=3.
\]

Using the personal utility definition,

\[
U_i(\pi_i)=\sum_{s\in S}p(s)u_i(s,\pi_i(s))
=0.6\cdot 1 + 0.4\cdot 3
=1.8.
\]

#### 1.3.1 Results from Simulation

**Figure 4 (page 4): Self Utility Overall For all 10 Agents**  
Description: A noisy line plot where all agents’ self-utility jumps up early and then fluctuates around ~1.8 over time (up to roughly 200,000 timesteps).

---

### 1.4 Test 4

Let the state space be

\[
S=\{0,1\},
\]

with state probabilities

\[
p(0)=0.5,\quad p(1)=0.5.
\]

Let the action space be

\[
A=\{0,1\}.
\]

\[
u_i(0,0)=1,\quad u_i(0,1)=0,\
u_i(1,0)=0,\quad u_i(1,1)=2.
\]

Expected policy agents choose: \(\pi_i(0)=0,\ \pi_i(1)=1\).

Personal utility from this policy:

\[
U_i(\pi_i)
= \sum_{s\in S} p(s)u_i(s,\pi_i(s))
= p(0)u_i(0,0)+p(1)u_i(1,1)
= 0.5\cdot 1 + 0.5\cdot 2
= 1.5.
\]

#### 1.4.1 Results from Simulation

**Figure 5 (page 5): Self Utility Overall For all 10 Agents**  
Description: Curves for all 10 agents rise from low values and converge around ~1.45–1.5, with mild ongoing variation after convergence.

---

## 2. Gossip Network

Reputation of an agent \(i\):

\[
R_i(\pi_i)=\gamma \sum_{j\in C\{i}} U_j(\pi_i).
\]

Test Cases: (Remember to fix behavior)

### 2.1 Test 1

Let

\[
u_j(s,\pi_i(s))=1 \quad 	ext{for all agents } j
eq i,\ 	ext{all states } s,\ 	ext{and all actions}.
\]

Let the reputation scaling factor be

\[
\gamma=1.
\]

Let the total number of agents be

\[
|C|=10.
\]

Assume all agents participate in every timestep.

**Expected Reputation.** Reputation of agent \(i\) is

\[
R_i(\pi_i)=\gamma\sum_{j\in C\{i}}U_j(\pi_i).
\]

Since each of the remaining \(|C|-1=9\) agents receives utility 1,

\[
R_i(\pi_i)=1\cdot \sum_{j\in C\{i}}1=9,
\]

for all agents \(i\).

#### 2.1.1 Results from Simulation

Text states that reputation estimates \(R_{i,j}\) for all \(i
eq j\) converge to:

\[
R_{i,j}=1.0.
\]

Total reputation of Agent 0:

\[
R_0=\sum_{i\in C\{0}}R_{i,0}=1\cdot(10-1)=9.
\]

**Figure 6 (page 6): Reputation calculated from estimates**  
Description: A line plot titled *“Reputations of Agents Over Time”* where all agent reputation curves quickly converge to about 9 and remain flat thereafter.

---

### 2.2 Test 2

Let the reputation scaling factor be

\[
\gamma=1.
\]

Let the total number of agents be

\[
|C|=3.
\]

Assume one state (0) and one action (0), so each agent has fixed behavior:

\[
\pi_1(0)=0,\ \pi_2(0)=0,\ \pi_3(0)=0,\ \pi_4(0)=0.
\]

Utilities:

\[
u(0,0)=1 	ext{ for Agent 0},\quad
u(0,0)=2 	ext{ for Agent 1},\quad
u(0,0)=3 	ext{ for Agent 2}.
\]

Now,

\[
U_j(\pi_i)=u(0,0)\ orall j
eq i.
\]

Reputation of each agent:

\[
R_i=\sum_{j\in C\{i}}U_j(\pi_i).
\]

\[
R_1=2+3=5,\quad R_2=1+3=4,\quad R_3=1+2=3.
\]

### 2.3 Results from the simulation

The algorithm in code averages all estimates, so everyone ends up having the same estimate (2.333333).  
Because of this, all agents also end up with the same reputation, which is noted as unexpected.
