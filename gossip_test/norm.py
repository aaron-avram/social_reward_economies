import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

"""

"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def ten(c):
    return torch.from_numpy(c).to(device).double()

class TemporaryReward(nn.Module):
    def __init__(self, oned_size): # we work with 1-d size here.
        super(TemporaryReward, self).__init__()
        self.layer1 = nn.Linear(oned_size + 1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)

    def forward(self, s, a):
        x = torch.cat((s, a))
        x = x.view(1, -1)
        x = F.leaky_relu(self.layer1(x))
        x = x.flatten()
        x = F.leaky_relu(self.layer2(x))
        return torch.exp(self.layer3(x))

class NormEnv:

    def __init__(self, state_dim, action_space):
        self.state_dim = state_dim
        self.action_space = action_space
        self.total_states = np.power(2, state_dim)

    def state_to_list(self, state_num):
        next = np.array([int(x) for x in list('{0:0b}'.format(state_num))])
        if len(next) < self.state_dim:
            repl = self.state_dim - len(next)
            for i in range(repl):
                next = np.concatenate((np.array([0]), next))
        return next

    def generate_reward_table(self, seed):
        reward_dict = {}
        np.random.seed(seed)
        fair_value = np.random.random()
        fair_value2 = np.random.random()
        for state in range(self.total_states):
            # vals = np.random.random(self.action_space)
            vals = np.random.random(self.action_space) + seed/20
            # vals = np.random.normal(0.4 + seed / 10, 0.5, self.action_space)
            vals = np.where(vals > 0, vals, 0)

            vals = (vals/2) + 0.5
            reward_dict[state] = vals

        return reward_dict

    """
    Generates a reward function based on a "dummy" Neural Network nonlinear function.
    """
    def generate_reward_table_procedural(self, seed):
        reward_dict = {}

        np.random.seed(seed)
        temp_reward = TemporaryReward(self.state_dim)
        temp_reward.to(device)

        max_reward = 0
        min_reward = 0
        for state in range(self.total_states):
            reward_dict[state] = []
            state1 = ten(np.array(self.state_to_list(state)))
            for i in range(self.action_space):

                a = ten(np.array([i]))
                reward = temp_reward(state1, a).detach().cpu().numpy()[0]
                reward_dict[state].append(reward)
                if reward > max_reward:
                    max_reward = reward
                elif reward < min_reward:
                    min_reward = reward

        for state in range(self.total_states):
            for i in range(self.action_space):
                if min_reward < 0:
                    reward_dict[state][i] -= min_reward
                    reward_dict[state][i] = reward_dict[state][i] / (max_reward - min_reward)

        return reward_dict

    """
    Generates a reward function based on the "Content Market" style distance-vector utility
    """
    def generate_reward_table_cmstyle(self, seed, top, bottom):
        reward_dict = {}

        max_reward = 2
        for state in range(self.total_states):
            reward_dict[state] = []

            best_action = random.randint(0, self.action_space)

            for i in range(self.action_space):

                reward = max_reward * (1 / ((1 + np.abs(i-best_action)) * top))
                reward_dict[state].append(reward)

        return reward_dict

    """
    Generates a reward function based on top/bottom rewards to simulate a Variance/Std based reward function.
    """
    def generate_reward_table_variance(self, seed, top, bottom):
        reward_dict = {}

        max_reward = 0
        min_reward = 0
        for state in range(self.total_states):
            reward_dict[state] = []


            for i in range(self.action_space):



                reward = np.random.normal(0.5, (top - bottom) / 3)
                # if reward < bottom:
                #     reward = bottom
                # if reward > top:
                #     reward = top
                reward_dict[state].append(reward)
                if reward > max_reward:
                    max_reward = reward
                elif reward < min_reward:
                    min_reward = reward
        return reward_dict

    """
    Generates a reward function with Variance and a "best state" (for debugging purposes).
    """
    def generate_reward_table_variance_separation(self, seed, top, bottom):
        reward_dict = {}

        # np.random.seed(seed)
        #temp_reward = TemporaryReward(self.state_dim)
        #temp_reward.to(device)

        best_states = [0, 0, 1, 1, 2, 2, 3, 3]

        max_reward = 0
        min_reward = 0
        for state in range(self.total_states):
            best_action = random.randint(0, self.action_space-1)
            if best_action == best_states[state]:
                best_action = self.action_space-1
            reward_dict[state] = []
            for i in range(self.action_space):

                reward = np.random.normal(0.5, (top - bottom) / 3)

                if best_action == i:
                    reward = min(top, np.random.normal(top, (top - bottom) / 5))
                if best_states[state] == i:
                    reward = min(top, np.random.normal(0.5 + (top - bottom) / 3, (top - bottom) / 5))
                reward_dict[state].append(reward)
                if reward > max_reward:
                    max_reward = reward
                elif reward < min_reward:
                    min_reward = reward
        return reward_dict

    """
    Generates a reward function with Variance but with the top/bottom rewards being the two separate means.
    """
    def generate_reward_table_polarized_variance(self, agent_number, top, bottom):
        reward_dict = {}

        group = agent_number % 2

        good_mean = top
        bad_mean = bottom

        max_reward = 0
        min_reward = 0
        for state in range(self.total_states):
            reward_dict[state] = []
            for i in range(self.action_space):

                if group == 0:
                    if i < self.action_space / 2:
                        reward = np.random.normal(good_mean, 0.5)
                    else:
                        reward = np.random.normal(bad_mean, 0.5)

                else:
                    if i < self.action_space / 2:
                        reward = np.random.normal(bad_mean, 0.5)
                    else:
                        reward = np.random.normal(good_mean, 0.5)

                reward_dict[state].append(reward)
                if reward > max_reward:
                    max_reward = reward
                elif reward < min_reward:
                    min_reward = reward



        return reward_dict

    def constant_reward_table(self, state_size, action_space, value=1.0):
        """
        Generate a constant reward table where all utilities are the same value.
        
        Args:
            state_size: Number of state bits
            action_space: Number of actions
            value: Constant reward value (default: 1.0)
        
        Returns:
            Dictionary mapping states to lists of constant rewards
        """
        total_states = 2 ** state_size
        reward = {}
        for s in range(total_states):
            reward[s] = [value] * action_space
        return reward