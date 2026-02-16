# Modeling Revised

## 1 Modeling

In this section, we formalize the modeling framework of norm emergence used in this study. Our objective is to bridge sociological accounts of norms with a multi-agent systems perspective by specifying a set of mechanisms through which collective behavioral regularities arise from decentralized interaction. Rather than treating norms as exogenously imposed constraints, the model conceptualizes norms as stable patterns of behavior that emerge endogenously as agents adapt to social rewards over time.

To this end, we introduce the social environment in which agents interact, the behavioral representations agents employ, the social reward structure guiding adaptation, and the social dynamics, such as gossip, role switching, and imitation, that govern learning and coordination.

### 1.1 Social Environment

The social environment is represented by a finite set of social situations, or states, denoted by `S`. Time evolves discretely and is indexed by `t = 1, 2, ...`. At each time step `t`, the community occupies a single state `s(t) ∈ S`, corresponding to a particular social or physical context.

For each state `s ∈ S`, there exists a finite set of admissible actions `A(s)`, representing the socially relevant behaviors agents may perform in that context. All agents face the same action set in a given state, although they may select different actions. This formulation captures the idea that agents repeatedly encounter similar coordination problems across contexts, while allowing individual behavior to vary.

### 1.2 Agents and Behaviors

The community consists of a finite set of agents `C = {1, ..., N}`. Agents respond to social situations using behavioral mappings `π_i: S → A(s)`, such that the action taken by agent `i` at time `t` is given by:

`a_i(t) = π_i(s(t))`.

The mapping `π_i` represents the externally observable manifestation of an agent’s internal schemas — cognitive structures that guide interpretation and action. Modeling behavior at the level of mappings, rather than isolated actions, allows norms to be defined as persistent regularities across social situations rather than as one-off behavioral choices.

### 1.3 Social Reward Structure

Agents adapt their behavior in response to social rewards. In this model, the social reward structure consists of three interacting components: personal utility, reputation, and status. Personal utility captures individual evaluations of actions based on agents’ own preferences. Reputation aggregates these evaluations at the social level, reflecting how beneficial an agent’s behavior is to others. Status captures structural influence, arising from being followed or imitated by other agents.

Together, these components determine how agents evaluate behavior and decide whether to act independently, imitate others, or seek influence.

#### 1.3.1 Personal Utility

Each agent `i ∈ C` has individual preferences over actions in different states, represented by a utility function:

`q_i(s, a): S × A(s) → ℝ`.

The value `q_i(s, a)` measures how much agent `i` benefits from action `a` being performed in state `s`, regardless of whether the action is performed by the agent itself or by another agent.

When agent `j` performs an action `a_j(t)` in state `s(t)`, and agent `i` is active and observing, agent `i` experiences utility:

`U_{i,j}^{(t)} = q_i(s(t), a_j(t))` if `j` is active, and `0` otherwise.

This experienced utility serves as the raw evaluative signal from which higher-level social evaluations are learned.

#### 1.3.2 Reputation

Reputation captures the social value of an agent’s behavior, understood as the extent to which that behavior benefits others across social situations. Importantly, reputation is treated as a conceptual, group-level quantity rather than as a directly observable attribute.

Formally, the reputation of agent `i` is defined as:

`R_i(π_i) = γ Σ_{j∈C\{i}} U_j(π_i),`

where `γ > 0` is a scaling factor that augments the attractiveness of socially beneficial behavior. This definition formalizes reputation as symbolic capital: agents are reputable to the extent that their behavior systematically generates benefits for others.

#### 1.3.3 Status

Status reflects an agent’s structural position within the community. Each agent `i` has a follower count `f_i`, representing the number of agents currently following `i`. Status is defined as:

`S_i(π_i) = κ f_i R_i(π_i),`

where `κ > 0` is a scaling factor that augments incentives associated with influence. While reputation reflects evaluation of behavior, status captures visibility and social power arising from being followed.

### 1.4 Gossip and Reputation Aggregation

Because agents cannot directly observe reputation, they must approximate it using local experience and social communication. Each agent `j` maintains two evaluative variables for every other agent `i`: a personal evaluation `P_{j,i}` and a reputation estimate `R_{j,i}`.

Personal evaluations summarize experienced utility and are updated via an exponential moving average:

`P_{j,i}^{(t+1)} = (1 − α)P_{j,i}^{(t)} + α q_j(s(t), a_i(t)).`

Reputation estimates track accumulated changes in personal evaluations:

`R_{j,i}^{(t+1)} = R_{j,i}^{(t)} + (P_{j,i}^{(t+1)} − P_{j,i}^{(t)}).`

These variables do not define reputation itself; rather, they serve as learning instruments that allow agents to approximate the underlying reputation function.

Reputation learning is further shaped by gossip. Let `C_active(t)` denote the set of active agents at time `t`. After each time step, active agents exchange reputation estimates, which are aggregated by averaging:

`R_{j,i}(t) = (1 / |C_active(t)|) Σ_{k∈C_active(t)} R_{k,i}(t).`

Through gossip, dispersed and noisy local experiences are transformed into a socially shared signal without centralized coordination.

### 1.5 Community Actions

#### 1.5.1 Active Rate Allocation

Agents are not continuously active. Each agent `i` has an activity rate `λ_i ∈ [0, 1]`, interpreted as the probability that the agent is active at a given time step. When inactive, an agent performs no action, observes nothing, and does not update beliefs. When active, an agent performs an action, observes others, updates evaluative beliefs, and participates in gossip.

There exists an outside utility `b_0`, representing the baseline payoff from allocating time outside the community. Each agent reallocates its activity rate at an average interval of `δ` time steps based on a comparison between `b_0` and the agent’s perceived utility from participation. Let `P_{i,i}` denote agent `i`’s personal estimate of the utility it receives from its own behavior. The activity rate is determined by:

`λ_i = 1 − exp(− P_{i,i} / (P_{i,i} + b_0)).`

This formulation ensures that participation increases as the community becomes more rewarding relative to the outside option, while remaining bounded.

#### 1.5.2 Role Switching and Imitation

Agents may occupy one of three roles in the community: Actors, Followers, and Influencers. These roles determine how agents select actions and respond to social rewards. Actors do not follow others and choose behavior to maximize personal utility. Followers align their behavior with that of a selected role model, while Influencers are followed by others and choose behavior to benefit their followers rather than themselves.

Roles are not fixed. All agents begin as Actors and periodically re-evaluate their role at an average interval of `δ` time steps, allowing role transitions to reflect accumulated experience rather than transient fluctuations.

At each role-evaluation step, agent `i` compares the expected benefits of acting independently, imitating others, and exerting influence. Let:

`j* = arg max_j R_{i,j},`

denote the agent perceived by `i` to have the highest reputation.

Agent `i` becomes a Follower if the reputational benefit of imitating the most reputable agent exceeds the expected utility of acting independently, that is, if:

`γ R_{i,j*} > P_{i,i},`

where `γ > 0` is a reputation modifier capturing the salience of reputational incentives.

Alternatively, agent `i` becomes an Influencer if its perceived standing within the community—captured by the product of its reputation estimate and its number of followers—exceeds the utility derived from self-directed behavior. Formally, this occurs when:

`f_i R_{i,i} > P_{i,i}.`

If neither imitation nor influence provides greater perceived value than independent action, the agent remains (or returns to) the Actor role. Specifically, agent `i` stays an Actor when:

`P_{i,i} > f_i R_{i,i}` and `P_{i,i} > R_{i,j} ∀ j`.

Together, these rules allow agents to move endogenously between roles based on their evolving assessments of personal utility, reputation, and influence.

Followers adopt the behavioral mapping of their selected role model `j*`. Through imitation, small differences in perceived reputation are amplified: agents with higher reputation attract more followers, increasing their visibility and social influence. This positive feedback accelerates behavioral convergence across the population and plays a central role in the emergence of shared norms.

### 1.6 Norm Emergence

A behavioral norm is defined as a behavior that becomes widely adopted across agents. Formally, a norm emerges when:

`π_i(s) = π_j(s) ∀ i, j ∈ C, ∀ s ∈ S`.

Norm emergence arises from the interaction of heterogeneous preferences, reputation learning via gossip, imitation, endogenous participation, and role differentiation.

### 1.7 Modeling Philosophy

This model prioritizes mechanistic clarity over realism. It assumes bounded rationality, local information, decentralized evaluation, and adaptive behavior driven by social rewards. Within these constraints, norm emergence arises endogenously from social interaction rather than being imposed externally.
