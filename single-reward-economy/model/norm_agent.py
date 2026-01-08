import numpy as np

from norm_actor import ActorNetwork, ACWrapper, Copier

import random
import torch

torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

b0 = 0.4



def update_with_discount(old_val, new_val, disc):
    return old_val * (1 - disc) + new_val * disc

args = {"status_factor": 1,
        "reputation_factor": 1,
        "b0": 7.5}
class NormAgent:
    def __init__(self, num_agents, num, budget, k, th_factor, arguments, actornet: ActorNetwork, criticnet, copier: Copier, inflnet: ActorNetwork, adjustment_factor=1, dynamic_change=True, rho=1, kappa=1):
        self.num = num
        self.budget = budget

        self.actor_network = actornet
        self.critic_network = criticnet
        self.copier_network = copier
        self.influencer_network = inflnet

        self.copier_network_static = None

        self.dynamic_change = dynamic_change

        self.reward_function = None

        self.threshold = 0

        self.alpha = 0
        self.rate = 0.2
        self.status = 0

        self.adjustment_factor = adjustment_factor

        self.update_factor = 0.001 * self.adjustment_factor
        self.beta_update_factor = 0.01 * self.adjustment_factor

        self.num_agents = num_agents
        self.PR = np.zeros(num_agents)
        self.R = np.zeros(num_agents)
        self.S = np.zeros(num_agents)

        self.is_follower = False
        self.target_influencer = -1
        self.beta = 0
        self.selfish_beta = 0
        self.independent_beta = 0
        self.k = k
        self.th_factor = th_factor

        self.followers = 0

        self.selfless = 0
        self.counter = 0
        self.b0 = arguments["b0"]
        self.reputation_factor = arguments["reputation_factor"]
        self.status_factor = arguments["status_factor"]

        self.time_counter1 = 0
        self.time_counter2 = 0

        self.selfless_constant = 500

        self.cons = arguments["cons"]
        self.rate_counter = 3000 * rho + int(np.random.randint(0, self.cons * rho)) # int(np.random.normal(self.cons * rho, 30))
        self.infl_counter = 3000 * rho + int(np.random.randint(0, self.cons * rho))
        self.selfless_counter = 3000 * rho + int(np.random.randint(0, self.selfless_constant))
        self.rho = rho
        self.kappa = kappa

        self.timestep = 0
        self.selfless_times = []

        self.group = self.num % 2

        self.rep_threshold = 0.2
        self.epsilon = 0.01


    def get_utility(self, s, a):
        if self.reward_function is not None:
            state = sum(val*(2**idx) for idx, val in enumerate(reversed(s)))
            return self.reward_function[state][a]
        return self.critic_network.get_function_output(s, a)

    def get_action(self, s, action_space):
        if self.selfless:
            random_var = np.random.random()
            if random_var < 0.05:
                return random.choices(action_space)
            return self.influencer_network.get_learned_action(s, action_space)
        else:
            random_var = np.random.random()
            if random_var < 0.05:
                return random.choices(action_space)
            return self.actor_network.get_learned_action(s, action_space)

    def train_actor(self, state, reward, action):
        self.actor_network.train(state, reward, action)

    def adjust_rate(self):

        self.status = self.followers * self.R[self.num] * self.reputation_factor * self.status_factor # 0?
        # print(self.status)
        # self.rate = self.rate_from_alpha(self.beta, self.status)
        if self.selfless == 1:
            usable_beta = max(self.influencer_network.beta, self.R[self.num] * self.reputation_factor, self.status)
            self.rate = self.rate_from_alpha(usable_beta, self.status)
        else:
            if self.target_influencer != -1 and self.followers == 0:
                usable_beta = max(self.selfish_beta, self.R[self.target_influencer] * self.reputation_factor, self.status)
                self.rate = self.rate_from_alpha(usable_beta, 0)
            else:
                usable_beta = self.selfish_beta
                # if self.followers > 0:
                #     usable_beta = max(self.selfish_beta) , self.R[self.num] * self.reputation_factor)
                self.rate = self.rate_from_alpha(usable_beta, self.status)

    def become_selfless(self, agents_list):

        self.status = self.followers * self.R[self.num] * self.reputation_factor * self.status_factor
        if self.status > self.PR[self.num] and self.followers > 0 and self.selfless == 0 and self.followers >= self.threshold:
            self.selfless = 1
            print("Agent " + str(self.num) + " became selfless @ ")
            self.influencer_network.beta = self.actor_network.beta
        elif self.status <= self.PR[self.num] or self.followers < self.threshold:
            if self.selfless == 1:
                print("Agent " + str(self.num) + " is no longer selfless @ timestep")
            self.selfless = 0



    def get_active(self):
        val = random.random()
        return val < self.rate

    def update_reputation(self, agent, reward):
        #if self.num == agent:
        #    reward = 0
        old_PR = self.PR[agent]
        self.PR[agent] = update_with_discount(self.PR[agent], reward, self.update_factor)

        self.R[agent] += self.PR[agent] - old_PR

    def update_generic(self, old, reward):
        return update_with_discount(old, reward, self.update_factor)

    def update_s(self, old, reward):
        return update_with_discount(old, reward, self.update_factor / 10)

    def update_beta(self, old, reward):
        return update_with_discount(old, reward, self.beta_update_factor)

    def share_reputation(self, agent, active_agents):
        for a_agent in active_agents:
            for i in range(self.num_agents):
                combined = (self.R[i] + a_agent.R[i]) / 2
                self.R[i] = combined
                a_agent.R[i] = combined

    def check_your_influencer(self, agents_list):
        if self.target_influencer != -1:
            if agents_list[self.target_influencer].target_influencer != -1:
                self.target_influencer = agents_list[self.target_influencer].target_influencer
    def switch_influencer(self, agents_list, timestep):
        max_rep = 0
        infl = -1
        # self.beta = self.actor_network.beta
        if self.followers >= self.threshold:
            self.become_selfless(agents_list)
        else:
            possible_agents = []
            max_val = np.max(self.R) * self.reputation_factor
            for index, rep in enumerate(list(self.R)):
                # print(self.beta)
                # print(rep)
                rep = rep * self.reputation_factor
                if rep > self.independent_beta and agents_list[index].target_influencer == -1 and rep > max_val - self.epsilon and rep > 0.06 * self.reputation_factor:
                    possible_agents.append(index)

            if len(possible_agents) != 0:
                infl = random.choice(possible_agents)

            if infl != -1 and infl != self.num and agents_list[infl].target_influencer == -1 and self.followers == 0:
                if self.is_follower == False:
                    if timestep != 0:
                        print("Agent", self.num, "became a follower to Agent", infl, "(Timestep", str(timestep) + ")")
                    else:
                        if timestep != 0:
                            print("Agent", self.num, "became a follower to Agent", infl)
                else:
                    agents_list[self.target_influencer].followers -= 1
                self.is_follower = True
                self.target_influencer = infl
                agents_list[self.target_influencer].followers += 1
                agents_list[self.target_influencer].is_follower = False
                agents_list[self.target_influencer].target_influencer = -1

            else:
                if self.target_influencer != -1:
                    agents_list[self.target_influencer].followers -= 1

                self.target_influencer = -1
                if self.target_influencer == -1:
                    self.is_follower = False


    def trigger_change(self, agents_list, timestep=0):
        rho = self.rho
        kappa = self.kappa
        cons = self.cons
        """
        Synchronous
        """
        if not self.dynamic_change:
            if self.time_counter2 == 0:
                self.check_your_influencer(agents_list)
            self.time_counter1 += 1
            self.time_counter2 += 1
            if self.time_counter1 > 0:
                if self.time_counter2 > (cons * rho * kappa):
                    self.switch_influencer(agents_list, timestep)
                    self.become_selfless(agents_list)
                    self.time_counter2 = 0
                if self.time_counter1 > (cons * rho):
                    self.adjust_rate()
                    self.time_counter1 = 0
            return

        """
        Asynchronous
        """
        if self.rate_counter == -1:
            self.rate_counter = self.cons * rho
            self.infl_counter = self.cons * rho
        if self.selfless_counter < 0:
            self.selfless_counter = self.selfless_constant

        self.rate_counter -= 1
        self.infl_counter -= 1
        self.selfless_counter -= 1

        if self.rate_counter == 0:
            self.adjust_rate()
            self.rate_counter = self.cons * rho

            if self.rate_counter < 0:
                self.rate_counter = 1
            self.counter += 1


        # choose influencer
        if self.infl_counter == 0:
            self.switch_influencer(agents_list, timestep)
            self.infl_counter = self.cons * rho
            if self.infl_counter < 0:
                self.infl_counter = 1
        if self.selfless_counter == 0:
            self.selfless_counter = self.selfless_constant
            if self.selfless_counter < 0:
                self.selfless_counter = 1
            self.become_selfless(agents_list)



    def rate_from_alpha(self, alpha, followers_modification):
        # if self.followers > 0:
        #     print("Foll:", followers_modification)
        alpha = alpha + followers_modification

        rate = ((alpha) / (alpha + self.b0)) * 3

        if rate > 100:
            return 1

        return max(0.08, 1 - np.exp(-1 * rate)) + 0.02




