# Replicating Reputation Scaling and Leader Perturbation

## Background

This report documents two experiments drawn from the social reward economy framework developed in [Paper_I_Draft (2).pdf](/Users/xia/Downloads/Paper_I_Draft%20(2).pdf). Together, the experiments address two connected claims in the paper's broader argument about decentralized norm formation. The first claim is that sufficiently strong reputation incentives can transform a heterogeneous population of self-directed agents into a leader-centered coordination regime, in which a large share of the population imitates a single influential agent and thereby converges on a common norm. The second claim is that once such a norm exists, it may be vulnerable to rupture if the leader's behavior ceases to be socially beneficial, and that the resulting normless period may or may not be followed by reconvergence around a new leader.

Experiment B is designed to evaluate the first claim. In the language of the paper draft, it asks whether increasing the reputation scaling parameter `gamma` produces the threshold effect associated with norm emergence: weak or absent followership at low `gamma`, followed by increasingly concentrated followership and, eventually, near-universal imitation of a single leader at higher `gamma`. This experiment therefore serves as the bridge between decentralized local learning and system-level norm formation.

Experiment D is designed to evaluate the second claim. It starts from a regime in which a leader and corresponding norm have already emerged, then perturbs the leader and observes whether the system proceeds through the expected sequence of collapse, temporary normlessness, and possible reconvergence. In the paper's terms, this is the transition from norm emergence to norm rupture and potential re-stabilization. For the present draft, the main substantive focus is Experiment B; Experiment D is included only as a structured placeholder for later expansion.

## Setup

Both experiments are implemented in the same shared simulation environment described in [Norm__Working_Copy__2_-16.pdf](/Users/xia/social_reward_economies/doc/Norm__Working_Copy__2_-16.pdf). The environment consists of a finite set of social states and a finite set of actions. At each timestep, the community encounters one state, and agents either act directly or react to the actions of others. A state-action pair therefore represents a contextually situated behavioral choice: what an agent does depends both on the current situation and on the policy it has learned.

The community contains heterogeneous agents. Each agent has a reward structure over state-action pairs, so agents do not value all behaviors identically even when they observe the same event. In the replicated Experiment B environment used here, this heterogeneity is generated through a shared-base Gaussian reward model: each state-action pair has a common baseline value, and each agent's realized reward is sampled around that shared base. This preserves a common underlying environment while allowing meaningful agent-level variation in preferences and perceived benefits.

Agents maintain two distinct but interacting evaluative systems. First, they learn a personal-utility estimate from the rewards they themselves receive. Second, they maintain reputational estimates of other agents, updating those estimates through direct observation and gossip. The system therefore combines private reinforcement learning with social learning. An agent's choice of role depends on how these signals compare: it may remain self-directed, follow a reputationally attractive leader, or, when status incentives are present, optimize behavior for its own followers. In the present Experiment B replication, `kappa = 0`, so the relevant competition is between personal utility and reputation.

Role updates occur more slowly than action updates, which is important for interpretation. Actions and observed rewards evolve continuously, while role changes occur at scheduled update epochs. This separation of time scales means that social influence does not change at every timestep; instead, agents accumulate evidence and periodically reevaluate whether following someone else is worthwhile. In this framework, `gamma` scales the strength of reputation as a motive for following, while `kappa` scales the value of status. Experiment B isolates the effect of `gamma`, whereas Experiment D later reintroduces richer leader dynamics on top of this same environment.

## Experiment B

This section reports whether increasing reputation incentives in the replicated large-scale Gaussian environment still produces the threshold-style transition from self-directed behavior to leader-centered norm emergence.

- State the canonical report-facing configuration from [reputation_scaling_static_10states_gaussian_hysteresisfix_light_report](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_light_report): static mode, `N=100`, `10` states, `2` actions, `50000` steps, Gaussian shared-base rewards, hysteresis-fixed role logic, light tracking, and `seed=0`.
- State the exact gamma grid used in the report figures: `0`, `1`, `1.5`, `2`, `2.25`, `2.5`, `2.75`, `3`, `3.5`, `4`, `4.5`, `5`.
- Summarize the expected qualitative result from the paper and the Norm Working Copy: low `gamma` should yield little or no followership, while sufficiently high `gamma` should produce a dominant leader and a common norm.
- Summarize the actual seed-0 replication results from [reputation_scaling_runs_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_light_report/reputation_scaling_runs_static.csv):
  - `gamma <= 2.5`: no leader (`0` followers for the top agent)
  - `gamma = 2.75, 3, 3.5`: small leader blocs (`2`, `3`, `9`)
  - `gamma = 4`: intermediate concentration (`42`)
  - `gamma = 4.5, 5`: full consolidation (`99`)
- Note that the seed-0 run therefore preserves the paper's qualitative threshold effect, but at a higher transition range than the simpler setting suggested in the draft text.
- Add a robustness note based on [reputation_scaling_static_10states_gaussian_hysteresisfix_light_gammas_0_4_4p5_5_10seeds](/Users/xia/social_reward_economies/experiments/outputs/exp_b/static/gaussian/sweeps/reputation_scaling_static_10states_gaussian_hysteresisfix_light_gammas_0_4_4p5_5_10seeds):
  - `gamma=4`: mean final top followers `42.9`
  - `gamma=4.5`: mean final top followers `81.4`
  - `gamma=5`: mean final top followers `82.5`
  - note explicitly that some high-`gamma` seeds still stall below full consolidation, so robustness is substantial but not perfect.
- Explain why the threshold appears right-shifted in the replicated environment:
  - the replicated system is larger (`100` agents), uses more social states, and includes Gaussian agent heterogeneity around a shared base;
  - these changes make it harder for one leader to dominate early, so stronger reputation incentives are required before the system collapses into near-universal followership.
- Add a short note that the report-facing plots are intentionally single-seed and seed-agnostic, while seed robustness is handled in prose rather than by overlaying many trajectories.
- List the figures/tables to insert later:
  - progression figure from [reputation_scaling_progression_static.png](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_light_report/reputation_scaling_progression_static.png)
  - paper-style summary from [reputation_scaling_paper_style_static.png](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_light_report/reputation_scaling_paper_style_static.png)
  - top-followers curve from [reputation_scaling_top_followers_static.png](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_light_report/reputation_scaling_top_followers_static.png)
  - gamma-followers table from [reputation_scaling_table_values_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_light_report/reputation_scaling_table_values_static.csv)
  - per-agent trace figure for `gamma=5` from [reputation_scaling_agent_traces_full_g5_seed0_static.png](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_full_gamma5_report/reputation_scaling_agent_traces_full_g5_seed0_static.png)

### Agent 0 Trajectory Note

The `gamma=5` per-agent trace figure provides a useful mechanism-level illustration of how reputation competes with personal utility in the replicated environment. For agent `0`, the blue line is the agent's learned personal-utility estimate, while the red line is `gamma * s_i(L_i,t)`, the current reputational value of the agent's highest-reputation target. In the trace CSV [reputation_scaling_agent_traces_g5_seed0_static.csv](/Users/xia/social_reward_economies/experiments/outputs/exp_b/report/reputation_scaling_static_10states_gaussian_hysteresisfix_full_gamma5_report/reputation_scaling_agent_traces_g5_seed0_static.csv), agent `0` begins with `PU = 0.0221` and `gamma * reputation = 0.3175` at `t=1`, corresponding to a raw selected reputation of `0.0635`. By `t=50000`, the same agent has `PU = 0.4833` and `gamma * reputation = 0.7353`, corresponding to a raw selected reputation of `0.1471`. Across the full run, `gamma * reputation` exceeds `PU` for about `95.2%` of timesteps. This scale is sensible under the chosen reward model: personal utility converging near `0.48` is consistent with rewards centered near `0.5`, while a modest raw reputation signal becomes behaviorally decisive once multiplied by `gamma = 5`. The red line can continue to move late in the run even though agent `0` ends up following agent `38`, because the trace records the reputational value of the agent's *currently selected* highest-reputation target rather than the reputation of the already-followed leader alone. Early negative or volatile red values are therefore not evidence of a bug; they reflect the fact that reputation estimates are learned social signals that can begin near zero or below zero even when realized rewards are clipped to remain positive.

## Experiment D

This section will explain how the perturbation experiment tests whether an established leader-centered norm can collapse and, under some conditions, reconverge around a new leader.

### Objective

Describe the purpose of the perturbation experiment: to test whether a leader-centered norm remains stable under targeted disruption and whether collapse is followed by normlessness, reconvergence, or persistent fragmentation.

### Configuration

Summarize the final Experiment D configuration to be reported, including the scaled environment, perturbation mechanism, timing parameters, and any deviations from the simpler Gaussian replication baseline.

### Expected Qualitative Sequence

State the expected sequence clearly: pre-perturbation convergence to a dominant leader, post-perturbation collapse, a temporary normless interval, and eventual reconvergence under a new leader.

### Actual Results

Summarize the actual outcomes across the reported Experiment D runs, including which configurations produced collapse, which produced normlessness, and how often recovery occurred.

### Explanation and Open Issues

Explain the main interpretation of the Experiment D results, especially the tension between reliable collapse and incomplete recovery, and identify the remaining open issues that the final writeup should address.
