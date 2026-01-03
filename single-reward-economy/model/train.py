import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt

from norm import NormEnv
from itertools import combinations

np.random.seed(34243)

# torch.set_default_dtype(torch.float64)
torch.backends.cudnn.benchmark = True

from norm_actor import ActorNetwork
from norm_agent import NormAgent


def compute_overall_policy(max_state, agents_list, p_network_list, i_network_list):
    top_agent = 0
    max_foll = 0
    for ag in agents_list:
        if ag.followers > max_foll:
            top_agent = ag.num
            max_foll = ag.followers

    tot_reward = 0
    bits = max(1, max_state.bit_length())
    numbers = np.arange(max_state + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    for _, statevec in enumerate(pos_states):
        if agents_list[top_agent].selfless:
            output = i_network_list[top_agent].get_function_output(np.array(statevec))
        else:
            output = p_network_list[top_agent].get_function_output(np.array(statevec))
        for actnum, actchance in enumerate(output):
            for ag in agents_list:
                tot_reward += ag.get_utility(statevec, actnum) * actchance

    print("FINAL OVERALL POLICY REWARD:", tot_reward)


def compute_overall_policy_return(max_state, agents_list, p_network_list, i_network_list):
    top_agent = 0
    max_foll = 0
    for ag in agents_list:
        if ag.followers > max_foll:
            top_agent = ag.num
            max_foll = ag.followers

    print("Interim Policy |", top_agent, "| Selfless:", agents_list[top_agent].selfless)

    tot_reward = 0
    bits = max(1, max_state.bit_length())
    numbers = np.arange(max_state + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    for statenum, statevec in enumerate(pos_states):
        if agents_list[top_agent].selfless: # or top_agent == 51:
            output = i_network_list[top_agent].get_function_output(np.array(statevec))
        else:
            output = p_network_list[top_agent].get_function_output(np.array(statevec))
        for actnum, actchance in enumerate(output):
            for ag in agents_list:
                tot_reward += ag.get_utility(statevec, actnum) * actchance
    print(tot_reward)
    return tot_reward


def compute_policy_interim(max_state, agents_list, p_network_list, i_network_list):
    bits = max(1, max_state.bit_length())
    numbers = np.arange(max_state + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    reward_list = []
    for ag in agents_list:
        reward = 0
        for statenum, statevec in enumerate(pos_states):
            if ag.selfless:
                output = i_network_list[ag.num].get_function_output(np.array(statevec))
            else:
                output = p_network_list[ag.num].get_function_output(np.array(statevec))
            for actnum, actchance in enumerate(output):
                reward += ag.get_utility(statevec, actnum) * actchance
        reward_list.append(reward / 8)
    return reward_list

def compute_policy_interim_reputation(max_state, agents_list, p_network_list, i_network_list):
    bits = max(1, max_state.bit_length())
    numbers = np.arange(max_state + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    reward_list = []
    for ag in agents_list:
        reward = 0
        for statenum, statevec in enumerate(pos_states):
            if ag.selfless:
                output = i_network_list[ag.num].get_function_output(np.array(statevec))
            else:
                output = p_network_list[ag.num].get_function_output(np.array(statevec))
            for actnum, actchance in enumerate(output):
                for ag1 in agents_list:
                    reward += ag1.get_utility(statevec, actnum) * actchance
        reward_list.append((reward / 8) / len(agents_list))
    return reward_list


def get_arbitrary_state(state_size, max_state):
    state_num = np.random.randint(0, max_state+1)
    bits = state_size
    state = np.array([int(d) for d in f'{state_num:0{bits}b}'])

    action_space = np.array([0, 1, 2, 3, 4, 5]) 
    return state, action_space


def get_active_agents(agents_list):
    active_list = []
    inactive_list = []
    for agent in agents_list:
        if agent.get_active():
            active_list.append(agent.num)
        else:
            inactive_list.append(agent.num)
    return active_list, inactive_list


def get_gossip_with_probability(agent1, agent2, epsilon):

    if agent1.group == agent2.group:
        return 1
    else:
        random_val = random.random()
        return random_val < epsilon


def training_loop(num_agents, max_state_num, name, arguments, k=1.0, th_factor=0.5, discount=0.99, adjustment_factor=1.0,
                  dynamic_change=True, timesteps=50000):
    

    state_size = int(np.ceil(np.log2(max_state_num)))
    sample_state, sample_action_space = get_arbitrary_state(state_size, max_state_num)
    possible_action_size = len(sample_action_space)

    
    p_network_list = [] # Agents Optimising for Personal Utility
    # f_network_list = []
    i_network_list = [] # Agents Optimising for Status

    agents_list = []
    alpha = arguments["learning_rate"]  # 0.0001 #0.0008
    # alpha2 = arguments["learning_rate"]  # 0.0001

    norm_env = NormEnv(state_size, possible_action_size) # This is needed for reward functions

    for agn in range(num_agents):
        p_network_list.append(ActorNetwork(state_size, possible_action_size, alpha, num=agn))
        #i_network_list.append(ActorNetwork(state_size, possible_action_size, alpha, num=agn))
        i_network_list.append(p_network_list[agn])
        # f_network_list.append(Copier(state_size, possible_action_size, alpha2, num=agn))
        agents_list.append(
            NormAgent(num_agents, agn, 8, k, th_factor, arguments, p_network_list[agn], None, None,
                      i_network_list[agn], adjustment_factor=adjustment_factor, dynamic_change=dynamic_change,
                      rho=arguments["rho"], kappa=arguments["kappa"]))

        # Configure reward function based on reward_type argument
        reward_type = arguments.get("reward_type", "unimodal_variance")
        
        if reward_type == "polarized_variance":
            agents_list[agn].reward_function = norm_env.generate_reward_table_polarized_variance(
                agn, arguments["top_reward"], arguments["bottom_reward"])
        elif reward_type == "unimodal_variance":
            agents_list[agn].reward_function = norm_env.generate_reward_table_variance(
                agn, arguments["top_reward"], arguments["bottom_reward"])
        elif reward_type == "procedural":
            agents_list[agn].reward_function = norm_env.generate_reward_table_procedural(agn)
        elif reward_type == "cmstyle":
            agents_list[agn].reward_function = norm_env.generate_reward_table_cmstyle(
                agn, arguments["top_reward"], arguments["bottom_reward"])
        elif reward_type == "variance_separation":
            agents_list[agn].reward_function = norm_env.generate_reward_table_variance_separation(
                agn, arguments["top_reward"], arguments["bottom_reward"])
        elif reward_type == "default":
            agents_list[agn].reward_function = norm_env.generate_reward_table(agn)
        else:
            # Default to unimodal_variance if unknown type
            agents_list[agn].reward_function = norm_env.generate_reward_table_variance(
                agn, arguments["top_reward"], arguments["bottom_reward"])
        
        if "threshold" in arguments: # Set threshold for switching to status optimization if provided
            agents_list[agn].threshold = arguments["threshold"]

    """ Setup for Variable Tracking """
    if True:
        # Finds the action that maximizes total utility across all agents (used as a baseline for comparison).
        max_feedback = 0
        best_act = 0
        for act in sample_action_space:
            feedback = 0
            for j in range(num_agents):
                # feedback += v_network_list[j].get_function_output(sample_state, act)[0]
                feedback += agents_list[j].get_utility(sample_state, act)
            if feedback > max_feedback:
                max_feedback = feedback
                best_act = act
       
       # Reward Tracking
        sample_state_chance = [] # List of probabilities of the best action for the sample state
        ma_epoch_reward = 0 # Moving average of epoch rewards
        epoch_reward = 0 # Total reward for the current epoch
        optimal_epoch_reward = 0 # Total reward for the best action for the sample state
        epoch_rewards = [] # List of epoch rewards
        optimal_epoch_rewards = [] # List of optimal epoch rewards
        # random_rewards = [] # List of random rewards # NOT USED

        # Reputation Tracking
        reputations = []
        for ii in agents_list:
            reputations.append([])

        personals = []
        for ii in agents_list:
            personals.append([])

        top_reputations = []
        for ii in agents_list:
            top_reputations.append([])

        all_reputations = []
        for ii in agents_list:
            all_reputations.append([])
            for ii2 in agents_list:
                all_reputations[ii.num].append([])

        all_similars = []
        for ii in agents_list:
            all_similars.append([])
            for ii2 in agents_list:
                all_similars[ii.num].append([])


       real_actions = []     # Proportion of B1 actions
       real_actions2 = []    # Proportion of B2 actions # NOT USED
       b1_actions = 0        # Counter for actions [0,1,2]
       b2_actions = 0        # Counter for actions [3,4,5]
       all_actions = 1       # Total action counter

        all_personals = []
        for ii in agents_list:
            all_personals.append([])
            for ii2 in agents_list:
                all_personals[ii.num].append([])

        # Example:
        # behaviours[5][3][2] = list of probabilities that agent 5 takes action 2 in state 3 over time
        behaviours = []
        for whatever in agents_list:
            behaviours.append([])
            for ii in range(max_state_num+1):
                behaviours[whatever.num].append([])
                for ii2 in range(len(sample_action_space)):
                    behaviours[whatever.num][ii].append([])

        roles = []  # [0]=influencers, [1]=followers, [2]=actors
        for ii in range(3):
            roles.append([])

        infl_roles = [] # [0]=status-based influencers (selfless), [1]=utility-based influencers (have followers but are not optimizing for status)
        for ii in range(2):
            infl_roles.append([])

        self_update_timesteps = []

        rates = []
        for ii in agents_list:
            rates.append([])

        infl_actionsx = []
        for ii in agents_list:
            infl_actionsx.append([])
        infl_actionsy = []
        for ii in agents_list:
            infl_actionsy.append([])

        ma_epoch_rewards_tracker = []

        infl_actionsy_no = []
        for ii in agents_list:
            infl_actionsy_no.append([])

        infl_matchx = []
        infl_matchy = []

        followers = []
        for ii in agents_list:
            followers.append([])

        delta_followers = []
        for ii in agents_list:
            delta_followers.append([])

        infl_rep = []
        self_util = []
        ind_self_util = []
        self_util_kd = []
        for ii in agents_list:
            infl_rep.append([])
            self_util.append([])
            ind_self_util.append([])
            self_util_kd.append([])

        self_policy_e_util = []
        for ii in agents_list:
            self_policy_e_util.append([])

        self_rep_e_util = []
        for ii in agents_list:
            self_rep_e_util.append([])

        self_overall_policy = []

        # cho_act = [] # NOT USED
        # cho_self = [] # NOT USED
        bes_act = []
        best_self = []

        agent_correct = np.zeros(num_agents)
        agent_follownum = np.zeros(num_agents)
        correct_infl = 0
        total_num = 0
    """ End Setup for Variable Tracking """

    update_interval = 250 #num_agents * 5
    print("Beginning Time Loop |", name)
    for i in range(timesteps):
        active_list, inactive_list = get_active_agents(agents_list)  # get list of agents that are participating
        if len(active_list) == 0:
            continue

        state, actions = get_arbitrary_state(state_size, max_state_num)  # Choose a random state

        # For every agent, we choose an action
        crowd_actions = np.zeros(num_agents)
        for j in active_list:
            if agents_list[j].target_influencer != -1:
                # If you follow someone, copy them
                crowd_actions[j] = agents_list[agents_list[j].target_influencer].get_action(state, actions)[0]
            else:
                # Otherwise, get your own action
                crowd_actions[j] = agents_list[j].get_action(state, actions)[0]  # p_network_list[j].get_learned_action(state, actions)[0]
                # If you're an influencer, tally whether you're in B1 or B2
                if agents_list[j].followers > 20:
                    if crowd_actions[j] in [0, 1, 2]:
                        b1_actions += 1
                    if crowd_actions[j] in [3, 4, 5]:
                        b2_actions += 1
                    all_actions += 1

        num_agents_feedback = 0
        selfish_feedback = np.zeros(num_agents)
        feedback = np.zeros(num_agents)
        true_feedback = np.zeros(num_agents)
        crowd_actions = crowd_actions.astype(np.int32)

        """ Feedback Tallying """
        for j in active_list:
            # If agent is active, record their utility to you, and give them feedback if you're following them
            for agent in active_list:
                agent_feedback = agents_list[j].get_utility(state, crowd_actions[agent])
                if j == agent:
                    selfish_feedback[agent] = agent_feedback  # If you are that agent, then keep it for later
                    feedback[j] += agent_feedback
                else:
                    if agents_list[j].target_influencer == agent:  # If you are following, then give it to your infl
                        feedback[j] += agent_feedback
                        num_agents_feedback += 1
                true_feedback[j] += agent_feedback
                # Update your Personal Reputation for them
                agents_list[j].update_reputation(agent, agent_feedback)
            # If inactive, it's like they provided no utility
            for other in inactive_list:
                agents_list[j].update_reputation(other, 0)

        usable_feedback = feedback # Used for training influencer networks

        """ Training Off Feedback """
        for agent in active_list:
            if agents_list[agent].target_influencer == -1 and not agents_list[agent].selfless:
                p_network_list[agent].train(state, selfish_feedback[agent], crowd_actions[agent])
            if agents_list[agent].selfless:
                i_network_list[agent].train(state, usable_feedback[agent], crowd_actions[agent])

        """ Similarity Scores Calculations """
        act_pairs = list(combinations(active_list, 2))
        for j, k in act_pairs:
            for agent in active_list:
                agent_feedback1 = agents_list[j].get_utility(state, crowd_actions[agent])
                agent_feedback2 = agents_list[k].get_utility(state, crowd_actions[agent])
                diff = np.square(agent_feedback1 - agent_feedback2)
                agents_list[j].S[k] = agents_list[j].update_s(agents_list[j].S[k], diff)
                agents_list[k].S[j] = agents_list[k].update_s(agents_list[k].S[j], diff)

        """ Beta Updates
            - betas are mostly used for neural network training
            - 'independent beta' is the beta acquired only from your own actions (i.e. your personal utility), which 
                is used to compare against influencer reputation when deciding whether to follow
        """
        for ag in agents_list:
            if ag.num in active_list:
                if not agents_list[ag.num].selfless:
                    p_network_list[ag.num].beta = ag.update_generic(p_network_list[ag.num].beta,
                                                                    selfish_feedback[ag.num])
                else:
                    i_network_list[ag.num].beta = ag.update_generic(i_network_list[ag.num].beta,
                                                                    usable_feedback[ag.num])
                if ag.selfless:
                    ag.beta = ag.update_generic(ag.beta, feedback[ag.num] + selfish_feedback[ag.num])
                    ag.selfish_beta = ag.update_generic(ag.selfish_beta, selfish_feedback[ag.num])
                    ag.independent_beta = ag.update_generic(ag.beta, feedback[ag.num] + selfish_feedback[ag.num])
                else:
                    ag.beta = ag.update_generic(ag.beta, selfish_feedback[ag.num])
                    ag.selfish_beta = ag.update_generic(ag.selfish_beta, selfish_feedback[ag.num])
                    if ag.target_influencer == -1:
                        ag.independent_beta = ag.update_generic(ag.independent_beta, selfish_feedback[ag.num])

        """
        Sharing Reputation (Gossiping)
        
        The similarity upper limit used here is 3.
        Gossiping starts at 50 steps.
        """
        active_agents = []
        gossiping_start = 50
        for j in active_list:
            active_agents.append(agents_list[j])
        if i > gossiping_start:
            for j_agent in active_agents:
                total_reputation_vector = np.zeros(num_agents)
                total_reputation_vector += j_agent.R.copy()
                num_agents_found = 1
                for k_agent in active_agents:
                    if j_agent.S[k] < 3 and k_agent.S[j] < 3:
                        total_reputation_vector += k_agent.R.copy()
                        num_agents_found += 1
                new_trc = total_reputation_vector / num_agents_found
                j_agent.R = new_trc

        """
        Changing Rate / Influencer
        """
        if dynamic_change:
            for j_agent in active_agents:
                if i > gossiping_start:
                    j_agent.trigger_change(agents_list, i)
            for j_agent in inactive_list:
                if agents_list[j_agent].rate_counter > 1:
                    agents_list[j_agent].rate_counter -= 1
                if agents_list[j_agent].infl_counter > 1:
                    agents_list[j_agent].infl_counter -= 1
                if agents_list[j_agent].selfless_counter > 1:
                    agents_list[j_agent].selfless_counter -= 1
        else:
            for j_agent in agents_list:
                if i > gossiping_start:
                    j_agent.trigger_change(agents_list)


        # if i == 30000:
        #    for network in p_network_list:
        #        for g in network.optimizer.param_groups:
        #            g['lr'] = alpha / 2

        epoch_reward += np.sum(feedback) + np.sum(selfish_feedback)
        upda = agents_list[0].update_factor / 10
        ma_epoch_reward = (np.sum(true_feedback)) * upda + ma_epoch_reward * (1 - upda)
        if i % 1000 == 0:
            print("At timestep:", i)
        if i > gossiping_start and i % 100 == 0:
            for agent0 in agents_list:
                for agent1 in agents_list:
                    all_reputations[agent0.num][agent1.num].append(agent1.R[agent0.num])
                    all_similars[agent0.num][agent1.num].append(agent1.S[agent0.num])
                    all_personals[agent0.num][agent1.num].append(agent1.PR[agent0.num])
                if agent0.selfless:
                    agent0.selfless_times.append([1])
                else:
                    agent0.selfless_times.append([0])
        if i % update_interval == 0 and i > 0:
            infl = -1
            print("Epoch Reward:", epoch_reward, "/", optimal_epoch_reward)
            epoch_rewards.append(epoch_reward)
            ma_epoch_rewards_tracker.append(ma_epoch_reward)
            optimal_epoch_rewards.append(optimal_epoch_reward)
            epoch_reward = 0
            optimal_epoch_reward = 0
            ba_chance = p_network_list[infl].get_function_output(sample_state)[best_act]
            sample_state_chance.append(ba_chance)
            # random_rewards.append(random_reward)
            self_update_timesteps.append(i)
            # print("Chance for Best Act:", ba_chance)
            top_agent = -1

            bits = max(1, max_state_num.bit_length())
            numbers = np.arange(max_state_num + 1)[:, np.newaxis]
            powers = np.arange(bits - 1, -1, -1)
            pos_states = list((numbers >> powers) & 1)
            for agentnum in range(num_agents):
                for statenum, statevec in enumerate(pos_states):
                    output = p_network_list[agentnum].get_function_output(np.array(statevec))
                    for actnum, actchance in enumerate(output):
                        behaviours[agentnum][statenum][actnum].append(actchance)

            """ Role Graphing """
            max_folls = 0
            num_folls = 0
            num_infls = 0
            num_infls_selfl = 0
            num_infls_selfi = 0
            for ag in agents_list:
                if ag.selfless:
                    num_infls_selfl += 1
                    num_infls += 1
                elif ag.target_influencer != -1:
                    num_folls += 1
                elif ag.followers > 0:
                    num_infls_selfi += 1
                    num_infls += 1

            num_actors = 100 - num_folls - num_infls

            roles[0].append(num_infls)
            roles[1].append(num_folls)
            roles[2].append(num_actors)

            infl_roles[0].append(num_infls_selfl)
            infl_roles[1].append(num_infls_selfi)

            for ag in agents_list:
                if ag.is_follower == False:
                    # top_choice = p_network_list[ag.num].get_learned_action(state, actions)[0]
                    if ag.followers > max_folls:
                        top_agent = ag.num
                        max_folls = ag.followers

            """ Reputation Graphing (Kepeing Track) """

            for ii in agents_list:
                sums = 0
                for jj in agents_list:
                    sums += jj.R[ii.num]
                reputations[ii.num].append(sums / num_agents)
                sums1 = 0
                for jj in agents_list:
                    sums1 += jj.PR[ii.num]
                personals[ii.num].append(sums1 / num_agents)
                top_reputations[ii.num].append(ii.R[top_agent])

            for ii in agents_list:
                rates[ii.num].append(ii.rate)


            """ Comparing Best Action for Self v. Best Action for Community """
            max_feedback = 0
            best_act = 0
            max_self_feedback = 0
            max_self_feedback2 = 0
            for act in actions:
                this_feedback = 0
                self_feedback = 0
                # aa = v_network_list[infl].get_function_output(state, act)[0]
                aa = agents_list[infl].get_utility(state, act)
                this_feedback += aa
                self_feedback += aa
                for j in active_list:
                    # aa = v_network_list[j].get_function_output(state, act)[0]
                    aa = agents_list[j].get_utility(state, act)

                    if j == infl:
                        pass
                    else:
                        this_feedback += aa
                if this_feedback > max_feedback:
                    max_feedback = this_feedback
                    best_act = act
                if self_feedback > max_self_feedback:
                    max_self_feedback = self_feedback
                    max_self_feedback2 = this_feedback
            bes_act.append(max_feedback)
            best_self.append(max_self_feedback2)

            if all_actions == 0:
                all_actions = 1
            real_actions.append(b1_actions / all_actions)
            real_actions2.append(b2_actions / all_actions)
            all_actions = 0
            b1_actions = 0
            b2_actions = 0

        if i % (update_interval) == 0 and i > 0:
            acts = np.zeros(num_agents)
            folls = np.zeros(num_agents)

            top_agent = -1
            max_folls = 0

            for ag in agents_list:
                if ag.is_follower == False:
                    # top_choice = p_network_list[ag.num].get_learned_action(state, actions)[0]
                    if ag.followers > max_folls:
                        top_agent = ag.num
                        max_folls = ag.followers
                    top_choice = ag.get_action(state, actions)[0]
                    acts[ag.num] = top_choice

            overall_infl = top_agent
            overall_infl_choice = int(acts[overall_infl])
            if crowd_actions[0] == overall_infl_choice:
                correct_infl += 1
            total_num += 1

            for ag in agents_list:
                if ag.is_follower:
                    infl_choice = int(acts[ag.target_influencer])
                    # choices = f_network_list[ag.num].get_function_output(state)
                    # choices = agents_list[ag.num].copier_network_static.get_function_output(state)
                    # for ac in actions:
                    #     if ac == 0:
                    #         choices[ac] = 0
                    my_choice = 0  # int(np.argmax(choices))

                    if infl_choice == my_choice:
                        agent_correct[ag.num] += 1
                    agent_follownum[ag.num] += 1
                    folls[ag.target_influencer] += 1

            for ag in agents_list:
                ag.followers = folls[ag.num]


            if i % update_interval == 0 and i > 0:
                """ Record Follower Counts """
                for ag in agents_list:
                    if len(followers[ag.num]) == 0:
                        old_followers = 0
                    else:
                        old_followers = followers[ag.num][-1]
                    delta_followers[ag.num].append(folls[ag.num] - old_followers)
                    followers[ag.num].append(folls[ag.num])

                """ Record Self Utility """
                for ag in agents_list:
                    self_util[ag.num].append(ag.selfish_beta)
                    ind_self_util[ag.num].append(ag.independent_beta)
                    self_util_kd[ag.num].append(ag.selfish_beta * ag.k)

                    maxrep = np.max(ag.R)
                    infl_rep[ag.num].append(maxrep)

                """ Compute Expected Policy Rewards (Self Utility & Reputation) """

                self_policy_reward = compute_policy_interim(max_state_num, agents_list, p_network_list, i_network_list)
                for ag in agents_list:
                    self_policy_e_util[ag.num].append(self_policy_reward[ag.num])

                self_reputation_reward = compute_policy_interim_reputation(max_state_num, agents_list, p_network_list, i_network_list)
                for ag in agents_list:
                    self_rep_e_util[ag.num].append(self_reputation_reward[ag.num])

                self_overall_policy.append(
                    compute_overall_policy_return(max_state_num, agents_list, p_network_list, i_network_list))

                for ag in agents_list:
                    if ag.is_follower:
                        infl_actionsy[ag.num].append(agent_correct[ag.num] / agent_follownum[ag.num])
                        infl_actionsx[ag.num].append(i)

                infl_matchx.append(i)
                infl_matchy.append(correct_infl / total_num)

                agent_correct = np.zeros(num_agents)
                agent_follownum = np.zeros(num_agents)
                total_num = 0
                correct_infl = 0


    print("Final Overall Best Policy |", name)
    compute_overall_policy(max_state_num, agents_list, p_network_list, i_network_list)
    # Plotting
    if True:
        root = "Results/" + name + "/"
        if not os.path.exists("Results"):
            os.mkdir("Results")
        if not os.path.exists("Results/" + name):
            os.mkdir("Results/" + name)
        root2 = root + "parts"
        root += name + "_"

        plt.figure(name + "chosebest")
        plt.plot(bes_act, linewidth=1.1, label="Best Action for Community", linestyle="dashed")
        plt.plot(best_self, label="Best Action for Self", linestyle="dotted")
        plt.title("Chosen Action vs. Best Action (Utility to Community)")
        plt.legend()
        plt.savefig(root + "_best_action.png")
        plt.close()

        new_plot_embed_withx(agents_list, self_update_timesteps, reputations, root, "reputations.png",
                             "Reputations of Agents Over Time", top_agent=top_agent)

        new_plot_embed_withx(agents_list, self_update_timesteps, personals, root, "personals.png",
                             "Averaged Personal Estimates of Agents Over Time", top_agent=top_agent)

        new_plot_embed_withx(agents_list, self_update_timesteps, rates, root, "rates.png",
                             "Internal Rate of Agents Over Time", top_agent=top_agent)


        # plt.figure(name + "copy logits")
        # for ii in agents_list:
        #     plt.plot(infl_actionsx[ii.num], infl_actionsy[ii.num], label="Agent " + str(ii.num))
        # plt.title("Rate of Agent-Follower Action Synchronicity in Previous Epoch")
        # plt.legend()
        # plt.savefig(root + "zz20_copying.png")
        # plt.close()


        plt.figure(name + "b1 actions")
        plt.plot(real_actions2, label="Proportion of B1 Actions")
        plt.plot(real_actions, label="Proportion of B2 Actions")
        plt.title("Proportion of B1 Actions By Influencer")
        plt.legend()
        plt.savefig(root + "_b1b2.png")
        plt.close()


        plt.figure(name + " roles")
        plt.plot(infl_matchx, roles[0], label="Influencers")
        plt.plot(infl_matchx, roles[1], label="Followers")
        plt.plot(infl_matchx, roles[2], label="Actors")
        plt.title("Count of Each Agent Role")
        np.save(root + "agent_role_count" + ".npy", np.array(roles))
        plt.legend()
        plt.savefig(root + "_roles.png")
        plt.close()


        plt.figure(name + " roles2")
        plt.plot(infl_matchx, infl_roles[0], label="Influencers (Status)")
        plt.plot(infl_matchx, infl_roles[1], label="Influencers (Utility)")
        plt.title("Number of Agents within Each Influencer Focus")
        plt.legend()
        plt.savefig(root + "_roles_infls.png")
        plt.close()

        plt.figure(name + "ma epoch")
        plt.plot(self_update_timesteps, ma_epoch_rewards_tracker)
        plt.title("Moving Average of Social Welfare")
        plt.legend()
        plt.savefig(root + "_social_welfare.png")
        np.save(root + "_socwel" + ".npy", ma_epoch_rewards_tracker)
        plt.close()

        plt.figure(name + "ma epoch22")
        plt.plot(self_update_timesteps, np.array(self_overall_policy))
        plt.title("Overall Policy Reward")
        plt.legend()
        plt.savefig(root + "_overall_policy.png")
        np.save(root + "_o_policy" + ".npy", np.array(self_overall_policy))
        plt.close()


        np.save(root + "update_timesteps" + ".npy", self_update_timesteps)

        new_plot_withx(agents_list, infl_matchx, infl_matchy, root, "community_copying.png",
                       "Total Proportion of Actions (per Epoch) under Influencer or Follower Policy")

        new_plot_embed_withx(agents_list, self_update_timesteps, followers, root, "followers.png",
                             "Followers Over Time", top_agent=top_agent)

        new_plot_embed_withx(agents_list, self_update_timesteps, delta_followers, root, "delta_followers.png",
                             "Change in Followers Over Time", top_agent=top_agent)

        new_plot_embed_withx(agents_list, self_update_timesteps, self_util, root, "self_utility_overall.png",
                             "Self-Utility Over Time", top_agent=top_agent)
        new_plot_embed_withx(agents_list, self_update_timesteps, self_policy_e_util, root, "self_utility_policy.png",
                             "Self-Utility Over Time", top_agent=top_agent)
        new_plot_embed_withx(agents_list, self_update_timesteps, self_rep_e_util, root, "self_rep_policy.png",
                             "True Reputation Over Time", top_agent=top_agent)
        new_plot_embed_withx(agents_list, self_update_timesteps, ind_self_util, root, "ind_self_utility_overall.png",
                             "Independent Self-Utility Over Time",
                             top_agent=top_agent)

        np.save(root + "self_util_full" + ".npy", np.array(self_util))
        np.save(root + "self_util_policy" + ".npy", np.array(self_policy_e_util))
        np.save(root + "self_rep_policy" + ".npy", np.array(self_rep_e_util))

        if not os.path.exists(root2 + "/per_agent_rep"):
            os.makedirs(root2 + "/per_agent_rep")

        for agent in agents_list:
            plt.figure("Reputations4" + str(agent.num))
            plt.plot(all_reputations[agent.num][agent.num], label="Reputation")
            plt.plot(agent.selfless_times, label="Influencer Indicator")

            plt.legend()
            plt.title("Agent Reputations and Role, Agent " + str(agent.num))
            plt.savefig(root2 + "/per_agent_rep/" + name + "_" + str(agent.num) + "_repalls.png")
            plt.close()

        reputations2 = np.copy(np.array(reputations))
        reputations2 = np.delete(reputations2, top_agent, 0)

        if not os.path.exists(root2 + "/per_agent_pr_indiv"):
            os.makedirs(root2 + "/per_agent_pr_indiv")

        for agent in agents_list:
            plt.figure("Reputations3" + str(agent.num))
            for ag1 in agents_list:
                if ag1.num != agent.num:
                    plt.plot(all_personals[agent.num][ag1.num], alpha=1)
                else:
                    plt.plot([])

            plt.legend()
            plt.title("Agent Specific Personal, Agent " + str(agent.num))
            plt.savefig(root2 + "/per_agent_pr_indiv/" + name + "_" + str(agent.num) + "_prs_avg.png")
            plt.close()


        if not os.path.exists(root2 + "/per_agent_sim"):
            os.makedirs(root2 + "/per_agent_sim")
        for agent in agents_list:
            plt.figure("similars" + str(agent.num))
            for ag1 in agents_list:
                if ag1.num != agent.num:
                    plt.plot(np.array(range(len(all_similars[agent.num][ag1.num]))) * 100,
                             all_similars[agent.num][ag1.num], alpha=0.5)
                else:
                    plt.plot([])

            plt.legend()
            plt.title("Agent Specific Similarity Scores, Agent " + str(agent.num))
            plt.savefig(root2 + "/per_agent_sim/" + name + "_" + str(agent.num) + "_sims.png")
            plt.close()

        if not os.path.exists(root2 + "/per_agent_rep_indiv"):
            os.makedirs(root2 + "/per_agent_rep_indiv")
        for agent in agents_list:
            plt.figure("Reputations4" + str(agent.num))
            for ag1 in agents_list:
                if ag1.num != agent.num:
                    plt.plot(np.array(range(len(all_reputations[agent.num][ag1.num]))) * 100,
                             all_reputations[agent.num][ag1.num], alpha=0.5)
                else:
                    plt.plot([])

            plt.legend()
            plt.title("Agent Specific Reputation, Agent " + str(agent.num))
            plt.savefig(root2 + "/per_agent_rep_indiv/" + name + "_" + str(agent.num) + "_reps_avg.png")
            plt.close()

        if not os.path.exists(root2 + "/per_state_space"):
            os.makedirs(root2 + "/per_state_space")
        for agent_num in range(num_agents):
            # for statenum in range(2 ** state_size):
            plt.figure("behaviours" + str(agent_num))
            total_array2 = np.zeros(len(infl_matchx))
            total_array = np.zeros(len(infl_matchx))
            for state in range(len(behaviours[agent_num])):
                total_array = np.zeros(len(infl_matchx))
                for actnum in range(len(sample_action_space)):
                    this_one = np.abs(np.diff(behaviours[agent_num][state][actnum]))
                    this_one = np.append(this_one, this_one[-1])
                    total_array += this_one
            total_array2 += np.copy(total_array)
            plt.plot(infl_matchx, total_array2)  # , label="Behaviour Vector Difference")
            plt.title("Change in Weights of Agent Action Probabilities in States, Agent " + str(agent_num))
            plt.legend()
            plt.savefig(root2 + "/per_state_space/stateactions" + str(agent_num) + ".png")
            plt.close()

        if not os.path.exists(root2 + "/per_state_space2"):
            os.makedirs(root2 + "/per_state_space2")
        for agent_num in range(num_agents):
            plt.figure("behaviours3232" + str(agent_num))
            for actnum in range(len(sample_action_space)):

                total_array = np.zeros(len(infl_matchx))
                for state in range(len(behaviours[agent_num])):
                    this_one = behaviours[agent_num][state][actnum]
                    total_array += this_one

                plt.plot(infl_matchx, total_array)  # , label="Behaviour Vector Difference")
            plt.title("Weights of Agent Action Probabilities (Aggregated over States), Agent " + str(agent_num))
            plt.legend()
            plt.savefig(root2 + "/per_state_space2/agg_state_act_" + str(agent_num) + ".png")
            plt.close()


def new_plot_embed(agents_list, series, root, name, title, top_agent):
    plt.figure(name + title, figsize=(8, 5))
    ax = plt.subplot(111)
    box = ax.get_position()
    for ii in agents_list:
        if ii.num == top_agent or ii.target_influencer != top_agent:
            ax.plot(series[ii.num], label="Agent " + str(ii.num))
        else:
            ax.plot(series[ii.num])
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.savefig(root + name)
    np.save(root + name.split(".")[0] + ".npy", series)
    plt.close()


def new_plot_withx(agents_list, xseries, yseries, root, name, title):
    plt.figure(name + title, figsize=(8, 5))
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.plot(xseries, yseries)
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.savefig(root + name)
    np.save(root + name.split(".")[0] + ".npy", yseries)
    plt.close()


def new_plot_embed_withx(agents_list, xseries, yseries, root, name, title, top_agent):
    plt.figure(name + title, figsize=(8, 5))
    ax = plt.subplot(111)
    box = ax.get_position()
    # for ii in agents_list:
    #     ax.plot(xseries[ii.num], yseries[ii.num], label="Agent " + str(ii.num))
    for ii in agents_list:
        if ii.num == top_agent or ii.target_influencer != top_agent:
            ax.plot(xseries, yseries[ii.num], label="Agent " + str(ii.num))
        else:
            ax.plot(xseries, yseries[ii.num])
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.savefig(root + name)
    np.save(root + name.split(".")[0] + ".npy", yseries)
    plt.close()


#state_size = 3
# max_state_num = 9
# state_size = int(np.ceil(np.log2(max_state_num)))
# print(state_size)
# name = "test"
# agents_list = []

# base_rep = 1
# base_b0 = 10
# base_scale = 1


# scale_factor = 1
# rep_factor = 1
# alpha = 0.00005

# kappa = 1
# rho = 1

# """
# "status_factor" and "reputation_factor" are kappa and gamma, respectively.
# "b0" is the outside utility (that governs rate allocation)
# "cons" is the update interval for infl switch / rate allocation
# "top_reward" and "bottom_reward" are the max/min reward for unimodal variance; they are the means for bimodal variance.
# "epsilon" is the epsilon chance that special gossiping interactions will happen at each check.
# "threshold" is the threshold at which an agent will consider optimizing for status.

# """

# args = {"status_factor": base_scale * scale_factor,
#         "reputation_factor": base_rep * rep_factor,
#         "b0": base_b0,
#         "learning_rate": alpha,
#         "kappa": kappa,
#         "rho": rho,
#         "cons": 3000}

# import gc

# for splits in [(-1.25, 2.25)]:  # , (-29, 30)]:
#         for rep_factor in [9]:  # , 3, 5, 999, 0, 0.5, 0.8, 0.3]:
#             for scale_factor in [0.1]:
#                 for threshold in [48]:
#                     for epsilon in [1]:
#                         random.seed(5)
#                         np.random.seed(5)
#                         torch.manual_seed(5)
#                         args["status_factor"] = base_scale * scale_factor
#                         args["reputation_factor"] = base_rep * rep_factor
#                         args["top_reward"] = splits[1]
#                         args["bottom_reward"] = splits[0]
#                         args["threshold"] = threshold
#                         args["b0"] = base_b0 * splits[1] * 4
#                         args["cons"] = 3000
#                         args["epsilon"] = epsilon
#                         dyn_change = True
#                         timesteps1 = 15000 #int(cons * 3.5)
#                         sync_label = "static"
#                         if dyn_change:
#                             sync_label = "async"
#                         training_loop(100, max_state_num,
#                                       "sep5_" + sync_label + "_" + str(args["cons"]) + "_" + str(scale_factor) + "K_" + str(
#                                           rep_factor) + "G_" + str(
#                                           args["b0"]) + "_b0_" + str(splits[1]-splits[0]) + "_sep_" + str(epsilon),
#                                       arguments=args, k=0, th_factor=0.6, adjustment_factor=3, dynamic_change=dyn_change,
#                                       timesteps=timesteps1)
#                         gc.collect()
