"""
Simplified NormAgent for reputation testing.

This version supports flexible initialization for progressive testing.
"""

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


class NormAgent:
    """
    Agent for social reward economy simulations.

    Simplified version for testing that supports:
    - Personal reputation (PR) tracking
    - Reputation (R) estimation via gossip
    - Similarity scores (S)
    - Influencer/follower dynamics (optional)
    - Status optimization (optional)
    """

    def __init__(self, agent_number, num_agents, reward_function, device,
                 enable_gossip=False, enable_influencer=False, enable_status=False):
        """
        Initialize agent.

        Args:
            agent_number: Agent ID
            num_agents: Total number of agents
            reward_function: Reward function u_i(s, a) as 2D array [state][action]
            device: torch device
            enable_gossip: Whether to enable gossip mechanism
            enable_influencer: Whether to enable influencer/follower dynamics
            enable_status: Whether to enable status optimization
        """
        self.num = agent_number
        self.num_agents = num_agents
        self.reward_function = reward_function
        self.device = device

        # Feature flags for progressive testing
        self.enable_gossip = enable_gossip
        self.enable_influencer = enable_influencer
        self.enable_status = enable_status

        # Reputation tracking
        self.PR = np.zeros(num_agents)  # Personal reputation estimates
        self.R = np.zeros(num_agents)   # Reputation estimates (updated via gossip)
        self.S = np.zeros(num_agents)   # Similarity scores

        # Influencer/follower dynamics
        self.is_follower = False
        self.target_influencer = -1
        self.followers = 0

        # Utility baselines
        self.beta = 0.0              # Current utility baseline
        self.selfish_beta = 0.0      # Personal utility baseline
        self.independent_beta = 0.0  # Baseline for influencer switching
        self.status = 0.0            # Status reward

        # Update factors
        self.update_factor = 0.001
        self.beta_update_factor = 0.01

        # Rate dynamics
        self.rate = 0.2
        self.alpha = 0.0

        # Status optimization
        self.selfless = False

        # Counters
        self.timestep = 0
        self.counter = 0

        # Thresholds
        self.rep_threshold = 0.2
        self.epsilon = 0.05

        # Configuration parameters (for compatibility)
        self.k = 1.0              # Kappa (status weight)
        self.b0 = 0.4             # Outside utility
        self.reputation_factor = 1.0  # Gamma
        self.status_factor = 1.0      # Status multiplier

    def get_utility(self, s, a):
        """
        Get utility for state-action pair.

        Args:
            s: State (can be int or list)
            a: Action

        Returns:
            Utility value
        """
        if isinstance(s, list):
            state = sum(val * (2 ** idx) for idx, val in enumerate(reversed(s)))
        else:
            state = s

        return self.reward_function[state][a]

    def update_beta(self, selfish_reward, status_reward):
        """
        Update utility baseline (beta) using exponential moving average.

        Args:
            selfish_reward: Personal utility
            status_reward: Status reward (if selfless)
        """
        if self.selfless and self.enable_status:
            # Status optimizer: use status reward
            self.beta = (1 - self.beta_update_factor) * self.beta + \
                       self.beta_update_factor * status_reward
        else:
            # Personal utility optimizer: use selfish reward
            self.beta = (1 - self.beta_update_factor) * self.beta + \
                       self.beta_update_factor * selfish_reward

        # Always update selfish_beta (for influencer switching)
        self.selfish_beta = (1 - self.beta_update_factor) * self.selfish_beta + \
                           self.beta_update_factor * selfish_reward

    def update_reputation(self, agent_id, utility_value, gamma=1.0):
        """
        Update reputation estimate for an agent.

        Args:
            agent_id: ID of agent whose reputation to update
            utility_value: Observed utility from that agent's action
            gamma: Reputation scaling factor
        """
        # Update personal reputation (PR)
        self.PR[agent_id] = utility_value

        # Update reputation (R) with exponential moving average
        alpha = 0.1
        new_r = gamma * utility_value
        self.R[agent_id] = (1 - alpha) * self.R[agent_id] + alpha * new_r

    def update_similarity(self, agent_id, similarity_value):
        """
        Update similarity score with another agent.

        Args:
            agent_id: ID of other agent
            similarity_value: Similarity score
        """
        self.S[agent_id] = similarity_value

    def switch_to_follower(self, influencer_id):
        """
        Become a follower of the specified influencer.

        Args:
            influencer_id: ID of agent to follow
        """
        if not self.enable_influencer:
            return

        self.is_follower = True
        self.target_influencer = influencer_id

    def switch_to_independent(self):
        """
        Stop following and become independent.
        """
        self.is_follower = False
        self.target_influencer = -1

    def become_selfless(self):
        """
        Switch to status optimization mode.
        """
        if not self.enable_status:
            return

        if not self.selfless:
            self.selfless = True
            self.timestep = 0

    def become_selfish(self):
        """
        Switch back to personal utility optimization.
        """
        self.selfless = False

    def should_follow_influencer(self, influencer_id, agents):
        """
        Decide whether to follow an influencer based on reputation.

        Args:
            influencer_id: ID of potential influencer
            agents: List of all agents

        Returns:
            True if should follow, False otherwise
        """
        if not self.enable_influencer:
            return False

        # Follow if influencer's reputation > our independent_beta
        return self.R[influencer_id] > self.independent_beta

    def rate_from_alpha(self, alpha, a=5.0):
        """
        Convert alpha (expected utility) to participation rate.

        θ(μ) = 1 - e^(-aμ)

        Args:
            alpha: Expected utility
            a: Scaling parameter

        Returns:
            Participation rate
        """
        return 1 - np.exp(-a * alpha)

    def update_rate(self):
        """
        Update participation rate based on current utility baseline.
        """
        self.rate = self.rate_from_alpha(max(0, self.beta - self.b0))
        self.alpha = max(0, self.beta - self.b0)

    def increment_timestep(self):
        """
        Increment internal timestep counter.
        """
        self.timestep += 1
        self.counter += 1
