"""
Simplified agent for toy experiment.
Supports two reputation update approaches:
1. 'test4': Direct estimation (γ * Σ U_j)
2. 'paper': Incremental updates (Equation 12)
"""
import numpy as np
import random


class ToyAgent:
    def __init__(self, agent_id, num_agents, reward_function, gamma, approach='test4', learning_rate=0.1):
        self.id = agent_id
        self.num_agents = num_agents
        self.reward_function = reward_function
        self.gamma = gamma
        self.approach = approach

        # Reputation estimates (one per agent, from this agent's perspective)
        self.R = np.zeros(num_agents)

        # Personal benefit estimates (for paper approach only)
        self.P = np.zeros(num_agents)

        # Personal utility baseline — only updated when acting independently.
        # Used both as the policy gradient baseline and the influencer-selection threshold.
        # Mirrors norm's `independent_beta`.
        self.independent_beta = 0.0

        # Follower / influencer tracking
        self.is_follower = False
        self.target_influencer = -1
        self.followers = 0

        # Learning rates
        self.alpha_r = learning_rate       # Reputation update rate (η)
        self.alpha_beta = 0.1             # Baseline update rate

        # Tabular policy: probability of each action given each state
        self.policy = np.ones((2, 2)) * 0.5   # [state][action], uniform init
        self.alpha_policy = 0.01

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def get_action(self, state):
        probs = self.policy[state] / np.sum(self.policy[state])
        return np.random.choice([0, 1], p=probs)

    # ------------------------------------------------------------------
    # Policy update (only called when acting independently)
    # ------------------------------------------------------------------

    def update_policy(self, state, action, reward):
        advantage = reward - self.independent_beta
        for a in range(2):
            if a == action:
                self.policy[state][a] += self.alpha_policy * advantage * (1 - self.policy[state][a])
            else:
                self.policy[state][a] -= self.alpha_policy * advantage * self.policy[state][a]
        self.policy[state] = np.clip(self.policy[state], 0.01, 0.99)
        self.policy[state] /= np.sum(self.policy[state])

    # ------------------------------------------------------------------
    # Reputation updates
    # ------------------------------------------------------------------

    def update_reputation_test4(self, agent_id, actions, state, all_agents):
        """
        Approach A — Test 4: directly estimate γ * Σ_{j≠i} U_j(s, a_i).
        Gamma is folded into the running average, so comparisons use R directly.
        """
        social_feedback = sum(
            other.reward_function[state][actions[agent_id]]
            for j, other in enumerate(all_agents)
            if j != agent_id
        )
        target = self.gamma * social_feedback
        self.R[agent_id] = (1 - self.alpha_r) * self.R[agent_id] + self.alpha_r * target

    def update_reputation_paper(self, agent_id, reward):
        """
        Approach B — Paper Eq 12: incremental update, NO gamma in the update.
        Gamma is applied only at comparison time in switch_influencer.

        P[i] = (1-η)*P[i] + η*reward
        R[i] += P_new - P_old
        """
        old_P = self.P[agent_id]
        self.P[agent_id] = (1 - self.alpha_r) * self.P[agent_id] + self.alpha_r * reward
        self.R[agent_id] += self.P[agent_id] - old_P

    def update_reputation(self, agent_id, actions, state, all_agents):
        """
        Called once per (observer=self, target=agent_id) pair each timestep.
        `reward` here is self's utility from agent_id's action — i.e. how much
        self benefits from agent_id's behaviour.
        """
        if self.approach == 'test4':
            self.update_reputation_test4(agent_id, actions, state, all_agents)
        else:
            reward = self.reward_function[state][actions[agent_id]]
            self.update_reputation_paper(agent_id, reward)

    # ------------------------------------------------------------------
    # Baseline update (Issue 2 fix: only when acting independently)
    # ------------------------------------------------------------------

    def update_independent_beta(self, reward):
        """Update baseline only when this agent is acting on its own policy."""
        self.independent_beta = (1 - self.alpha_beta) * self.independent_beta + self.alpha_beta * reward

    # ------------------------------------------------------------------
    # Influencer selection (Issues 3 & 4 fixes)
    # ------------------------------------------------------------------

    def switch_influencer(self, all_agents):
        """
        Select the best independent agent to follow, if their scaled reputation
        exceeds this agent's independent baseline.

        Fixes applied vs. original toy:
          Issue 3 — near-maximum filter: candidate must be within `epsilon` of
                    the best scaled reputation seen; also must clear an absolute
                    floor so stale-zero reputations are never followed.
          Issue 4 — chain-following prevented: only follow agents who are
                    themselves independent (target_influencer == -1), and only
                    if this agent has no followers of its own.
        """
        # Issue 4: influencers cannot become followers
        if self.followers > 0:
            self.is_follower = False
            self.target_influencer = -1
            return

        epsilon = 0.01       # near-max window
        abs_floor = 0.01     # ignore near-zero or negative scaled reputations

        # Compute scaled reputation for every agent
        if self.approach == 'test4':
            scaled_R = self.R.copy()          # gamma already baked in
        else:
            scaled_R = self.gamma * self.R    # apply gamma at comparison time

        max_val = np.max(scaled_R)

        # Gather eligible candidates
        candidates = []
        for j in range(self.num_agents):
            if j == self.id:
                continue
            rep = scaled_R[j]
            if (rep > self.P[self.id]                    # beats P_{i,i}: own personal evaluation (paper formula)
                    and all_agents[j].target_influencer == -1   # Issue 4: not a follower
                    and rep > max_val - epsilon           # Issue 3: near the best
                    and rep > abs_floor):                 # Issue 3: absolute floor
                candidates.append(j)

        if candidates:
            self.is_follower = True
            self.target_influencer = random.choice(candidates)
        else:
            self.is_follower = False
            self.target_influencer = -1
