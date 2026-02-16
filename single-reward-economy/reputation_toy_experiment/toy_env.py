"""
Minimal environment for reputation scaling toy experiment.
2 states, 2 actions - simplest possible setup.
"""
import numpy as np


class ToyEnv:
    def __init__(self):
        self.num_states = 2
        self.num_actions = 2

    def sample_state(self):
        """Sample a random state uniformly."""
        return np.random.choice([0, 1])

    def generate_state_dependent_rewards(self, agent_id, seed_offset=0):
        """
        Generate state-dependent reward function:
        - In state 0: action 0 gives higher reward
        - In state 1: action 1 gives higher reward

        Add some heterogeneity across agents but maintain same optimal structure.

        Returns: reward_table[state][action]
        """
        saved_state = np.random.get_state()
        np.random.seed(agent_id + seed_offset)

        # Base rewards
        reward_table = np.zeros((self.num_states, self.num_actions))

        # State 0: action 0 optimal
        reward_table[0][0] = 1.0 + np.random.uniform(-0.2, 0.2)  # High reward
        reward_table[0][1] = 0.3 + np.random.uniform(-0.1, 0.1)  # Low reward

        # State 1: action 1 optimal
        reward_table[1][0] = 0.3 + np.random.uniform(-0.1, 0.1)  # Low reward
        reward_table[1][1] = 1.0 + np.random.uniform(-0.2, 0.2)  # High reward

        np.random.set_state(saved_state)
        return reward_table
