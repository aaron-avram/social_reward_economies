import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_dtype(torch.float64)

from torch import linalg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ten(c):
    return torch.from_numpy(c).to(device).double()

def unwrap_state(state):
    return state["agents"].copy(), state["apples"].copy()


N = 5
"""
Not currently used
"""
class ArbitraryCritic(nn.Module):
    def __init__(self, oned_size): # we work with 1-d size here.
        super(ArbitraryCritic, self).__init__()
        self.layer1 = nn.Linear(oned_size + N, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 1)

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

class ACWrapper():
    def __init__(self, oned_size):
        self.function = ArbitraryCritic(oned_size)
        self.function.to(device)

    def get_function_output(self, s, a):
        aa = np.ones(N)
        aa = aa * a

        return self.function(ten(s), ten(aa)).detach().cpu().numpy()


class CopierNetwork(nn.Module):
    def __init__(self, oned_size, action_space):  # we work with 1-d size here.
        super(CopierNetwork, self).__init__()
        self.layer1 = nn.Linear(oned_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, action_space)  # [0] is move left, [1] is move right, [2] is don't move

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, s):
        x = s
        x = x.view(1, -1)
        x = F.leaky_relu(self.layer1(x))
        x = x.flatten()
        x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        return self.layer4(x)

class Copier():
    def __init__(self, oned_size, action_space, alpha, discount=0.99, beta=None, avg_alpha=None, num=0):
        self.function = CopierNetwork(oned_size, action_space)
        self.function.to(device)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.num = num

        self.states = []
        self.new_states = []
        self.rewards = []
        self.poses = []
        self.actions = []

        self.beta = 0

    def get_function_output(self, s):
        return F.softmax(self.function(ten(s)), dim=0).detach().cpu().numpy()

    def get_learned_action(self, s, action_space):
        output = self.get_function_output(s)


        weights = output[action_space]
        if np.sum(weights) <= 10e-3:
            weights = np.ones(len(action_space))

        return random.choices(action_space, weights)


    def get_function_output_v(self, s):
        return self.function(ten(s)).detach().cpu().numpy()

    def train(self, state, action):
        actions = self.function(ten(state))

        action_onehot = np.zeros(len(actions))

        self.optimizer.zero_grad()

        action_onehot[action] = 1

        action_onehot = ten(action_onehot)
        # action_onehot = ten(np.array([action]))

        # actions = torch.reshape(actions, (1, 10))

        # action_onehot = torch.reshape(action_onehot, (1, 1))

        loss1 = nn.CrossEntropyLoss()

        loss = loss1(actions, action_onehot)

        loss.backward()

        self.optimizer.step()



class SimpleConnectedMultiple(nn.Module):
    def __init__(self, oned_size, action_space): # we work with 1-d size here.
        super(SimpleConnectedMultiple, self).__init__()
        self.layer1 = nn.Linear(oned_size, 64)
        #self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, action_space)

        torch.nn.init.xavier_uniform_(self.layer1.weight)
        #torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, s):
        x = s
        x = x.view(1, -1)
        x = F.leaky_relu(self.layer1(x))
        x = x.flatten()
        #x = F.leaky_relu(self.layer2(x))
        x = F.leaky_relu(self.layer3(x))
        return self.layer4(x)

counter = 0
total_reward = 0
class ActorNetwork():
    def __init__(self, oned_size, action_space, alpha, discount=0.99, beta=None, avg_alpha=None, num=0):
        self.function = SimpleConnectedMultiple(oned_size, action_space)
        self.function.to(device)
        self.optimizer = optim.AdamW(self.function.parameters(), lr=alpha, amsgrad=True)
        self.alpha = alpha
        self.discount = discount
        self.num = num

        self.states = []
        self.new_states = []
        self.rewards = []
        self.poses = []
        self.actions = []

        self.beta = 0

    def get_function_output(self, s):
        return F.softmax(self.function(ten(s)), dim=0).detach().cpu().numpy()

    def get_learned_action(self, s, action_space):
        output = self.get_function_output(s)


        weights = output[action_space]
        if np.sum(weights) <= 10e-3:
            weights = np.ones(len(action_space))

        return random.choices(action_space, weights)


    def get_function_output_v(self, s):
        return self.function(ten(s)).detach().cpu().numpy()

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


