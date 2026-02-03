import itertools
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
    for statenum, statevec in enumerate(pos_states):
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
    num_states = len(pos_states)  # Number of states: max_state_num + 1
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
        reward_list.append(reward / num_states)
    return reward_list

def compute_policy_interim_reputation(max_state, agents_list, p_network_list, i_network_list):
    bits = max(1, max_state.bit_length())
    numbers = np.arange(max_state + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    num_states = len(pos_states)  # Number of states: max_state_num + 1
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
        reward_list.append((reward / num_states) / len(agents_list))
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


# def get_gossip_with_probability(agent1, agent2, epsilon):

#     if agent1.group == agent2.group:
#         return 1
#     else:
#         random_val = random.random()
#         return random_val < epsilon


def training_loop(num_agents, max_state_num, name, arguments, k=1.0, adjustment_factor=1.0, timesteps=50000):
    state_size = int(np.ceil(np.log2(max_state_num)))
    sample_state, sample_action_space = get_arbitrary_state(state_size, max_state_num)
    possible_action_size = len(sample_action_space)

    p_network_list = []
    # f_network_list = []

    i_network_list = []

    agents_list = []
    alpha = arguments["learning_rate"]  # 0.0001 #0.0008
    # alpha2 = arguments["learning_rate"]  # 0.0001
    norm_env = NormEnv(state_size, possible_action_size)

    for agn in range(num_agents):

        p_network_list.append(ActorNetwork(state_size, possible_action_size, alpha, num=agn))
        #i_network_list.append(ActorNetwork(state_size, possible_action_size, alpha, num=agn))
        i_network_list.append(p_network_list[agn])
        # f_network_list.append(Copier(state_size, possible_action_size, alpha2, num=agn))
        agents_list.append(
            NormAgent(num_agents, agn, k, arguments, p_network_list[agn], adjustment_factor=adjustment_factor))

        # agents_list[agn].reward_function = norm_env.generate_reward_table_procedural(agn)
        agents_list[agn].reward_function = norm_env.constant_reward_table(state_size, possible_action_size, value=1.0)
        # agents_list[agn].reward_function = norm_env.generate_reward_table_polarized_variance(agn, arguments["top_reward"],
        #                                                                            arguments["bottom_reward"])
        # if "threshold" in arguments:
        #     agents_list[agn].threshold = arguments["threshold"]

    """ Setup for Variable Tracking """
    if True:
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
        sample_state_chance = []

        ma_epoch_reward = 0
        epoch_reward = 0
        optimal_epoch_reward = 0
        epoch_rewards = []
        optimal_epoch_rewards = []
        # random_rewards = []

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
        
        weights_over_time = []  # list of lists: weights_over_time[agent_num][timestep] = flattened weights
        for ag in agents_list:
            weights_over_time.append([])  # empty list for each agent


        real_actions = []
        real_actions2 = []
        b1_actions = 0
        b2_actions = 0
        all_actions = 1

        all_personals = []
        for ii in agents_list:
            all_personals.append([])
            for ii2 in agents_list:
                all_personals[ii.num].append([])

        behaviours = []
        for whatever in agents_list:
            behaviours.append([])
            for ii in range(max_state_num+1):
                behaviours[whatever.num].append([])
                for ii2 in range(len(sample_action_space)):
                    behaviours[whatever.num][ii].append([])

        roles = []
        for ii in range(3):
            roles.append([])

        infl_roles = []
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

        # cho_act = []
        # cho_self = []
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
        # active_list, inactive_list = get_active_agents(agents_list)  # get list of agents that are participating
        active_list = [j for j in range(num_agents)]
        inactive_list = []
        if len(active_list) == 0:
            continue

        state, actions = get_arbitrary_state(state_size, max_state_num)  # Choose a random state

        # Every Agent chooses their own actions
        crowd_actions = np.zeros(num_agents)
        for j in active_list:
            # if agents_list[j].target_influencer != -1:
            #     # If you follow someone, copy them
            #     crowd_actions[j] = agents_list[agents_list[j].target_influencer].get_action(state, actions)[0]
                # Otherwise, get your own action
                crowd_actions[j] = agents_list[j].get_action(state, actions)[0]  # p_network_list[j].get_learned_action(state, actions)[0]
                # If you're an influencer, tally whether you're in B1 or B2
                # if agents_list[j].followers > 20:
                #     if crowd_actions[j] in [0, 1, 2]:
                #         b1_actions += 1
                #     if crowd_actions[j] in [3, 4, 5]:
                #         b2_actions += 1
                #     all_actions += 1

        # num_agents_feedback = 0
        selfish_feedback = np.zeros(num_agents)
        # feedback = np.zeros(num_agents)
        true_feedback = np.zeros(num_agents)
        crowd_actions = crowd_actions.astype(np.int32)

        """ Feedback Tallying """
        for j in active_list:
            # If agent is active, record their utility to you, and give them feedback if you're following them
            for agent in active_list:
                agent_feedback = agents_list[j].get_utility(state, crowd_actions[agent])
                if j == agent:
                    selfish_feedback[agent] = agent_feedback  # If you are that agent, then keep it for later
                    # feedback[j] += agent_feedback
                # else:
                #     if agents_list[j].target_influencer == agent:  # If you are following, then give it to your infl
                #         feedback[j] += agent_feedback
                #         num_agents_feedback += 1
                true_feedback[j] += agent_feedback
                # Update your Personal Reputation for them
                agents_list[j].update_reputation(agent, agent_feedback)
            # If inactive, it's like they provided no utility
            for other in inactive_list:
                agents_list[j].update_reputation(other, 0)

        # usable_feedback = feedback

        """ Training Off Feedback """
        for agent in active_list:
            # if agents_list[agent].target_influencer == -1 and not agents_list[agent].selfless:
            p_network_list[agent].train(state, selfish_feedback[agent], crowd_actions[agent])
            # if agents_list[agent].selfless:
            #     i_network_list[agent].train(state, usable_feedback[agent], crowd_actions[agent])

        """ Similarity Scores Calculations """
        # act_pairs = list(combinations(active_list, 2))
        # for j, k in act_pairs:
        #     for agent in active_list:
        #         agent_feedback1 = agents_list[j].get_utility(state, crowd_actions[agent])
        #         agent_feedback2 = agents_list[k].get_utility(state, crowd_actions[agent])
        #         diff = np.square(agent_feedback1 - agent_feedback2)
        #         agents_list[j].S[k] = agents_list[j].update_s(agents_list[j].S[k], diff)
        #         agents_list[k].S[j] = agents_list[k].update_s(agents_list[k].S[j], diff)

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
                # else:
                #     i_network_list[ag.num].beta = ag.update_generic(i_network_list[ag.num].beta,
                #                                                     usable_feedback[ag.num])
                # if ag.selfless:
                #     ag.beta = ag.update_generic(ag.beta, feedback[ag.num] + selfish_feedback[ag.num])
                #     ag.selfish_beta = ag.update_generic(ag.selfish_beta, selfish_feedback[ag.num])
                #     ag.independent_beta = ag.update_generic(ag.beta, feedback[ag.num] + selfish_feedback[ag.num])
                # else:
                ag.beta = ag.update_generic(ag.beta, selfish_feedback[ag.num])
                ag.selfish_beta = ag.update_generic(ag.selfish_beta, selfish_feedback[ag.num])
                # if ag.target_influencer == -1:
                ag.independent_beta = ag.update_generic(ag.independent_beta, selfish_feedback[ag.num])

        """
        Sharing Reputation (Gossiping)
        
        The similarity upper limit used here is 3.
        Gossiping starts at 50 steps.
        """
        # active_agents = []
        # gossiping_start = 50
        # for j in active_list:
        #     active_agents.append(agents_list[j])
        # if i > gossiping_start:
        #     for j_agent in active_agents:
        #         total_reputation_vector = np.zeros(num_agents)
        #         total_reputation_vector += j_agent.R.copy()
        #         num_agents_found = 1
        #         for k_agent in active_agents:
        #             if j_agent.S[k] < 3 and k_agent.S[k] < 3:
        #                 total_reputation_vector += k_agent.R.copy()
        #                 num_agents_found += 1
        #         new_trc = total_reputation_vector / num_agents_found
        #         j_agent.R = new_trc

        """
        Changing Rate / Influencer
        """
        # if dynamic_change:
        #     for j_agent in active_agents:
        #         if i > gossiping_start:
        #             j_agent.trigger_change(agents_list, i)
        #     for j_agent in inactive_list:
        #         if agents_list[j_agent].rate_counter > 1:
        #             agents_list[j_agent].rate_counter -= 1
        #         if agents_list[j_agent].infl_counter > 1:
        #             agents_list[j_agent].infl_counter -= 1
        #         if agents_list[j_agent].selfless_counter > 1:
        #             agents_list[j_agent].selfless_counter -= 1
        # else:
        #     for j_agent in agents_list:
        #         if i > gossiping_start:
        #             j_agent.trigger_change(agents_list)


        # if i == 30000:
        #    for network in p_network_list:
        #        for g in network.optimizer.param_groups:
        #            g['lr'] = alpha / 2

        epoch_reward += np.sum(selfish_feedback)
        upda = agents_list[0].update_factor / 10
        ma_epoch_reward = (np.sum(true_feedback)) * upda + ma_epoch_reward * (1 - upda)
        if i % 1000 == 0:
            print("At timestep:", i)

        if i % update_interval == 0 and i > 0:
            print("Epoch Reward:", epoch_reward, "/", optimal_epoch_reward)
            epoch_rewards.append(epoch_reward)
            ma_epoch_rewards_tracker.append(ma_epoch_reward)
            optimal_epoch_rewards.append(optimal_epoch_reward)
            epoch_reward = 0
            optimal_epoch_reward = 0
            self_update_timesteps.append(i)

            for agent_num in range(num_agents):
                weights_flat = np.concatenate([
                    p.detach().cpu().numpy().flatten()
                    for p in p_network_list[agent_num].function.parameters()
                ])
                weights_over_time[agent_num].append(weights_flat)




            bits = max(1, max_state_num.bit_length())
            numbers = np.arange(max_state_num + 1)[:, np.newaxis]
            powers = np.arange(bits - 1, -1, -1)
            pos_states = list((numbers >> powers) & 1)
            for agentnum in range(num_agents):
                for statenum, statevec in enumerate(pos_states):
                    output = p_network_list[agentnum].get_function_output(np.array(statevec))
                    for actnum, actchance in enumerate(output):
                        behaviours[agentnum][statenum][actnum].append(actchance)

            """ Reputation Graphing """
            for ii in agents_list:
                sums = 0
                for jj in agents_list:
                    sums += jj.R[ii.num]
                reputations[ii.num].append(sums / num_agents)
                sums1 = 0
                for jj in agents_list:
                    sums1 += jj.PR[ii.num]
                personals[ii.num].append(sums1 / num_agents)

            for ii in agents_list:
                rates[ii.num].append(ii.rate)

            """ Record Self Utility """
            for ag in agents_list:
                self_util[ag.num].append(ag.selfish_beta)
                ind_self_util[ag.num].append(ag.independent_beta)
                self_util_kd[ag.num].append(ag.selfish_beta * ag.k)
                maxrep = np.max(ag.R)
                infl_rep[ag.num].append(maxrep)

            """ Compute Expected Policy Rewards (Self Utility) """
            self_policy_reward = compute_policy_interim(max_state_num, agents_list, p_network_list, i_network_list)
            for ag in agents_list:
                self_policy_e_util[ag.num].append(self_policy_reward[ag.num])

    print("Final Overall Best Policy |", name)
    compute_overall_policy(max_state_num, agents_list, p_network_list, i_network_list)
    
    # Plotting
    if True:
        root = "Results/" + name + "/"
        if not os.path.exists("Results"):
            os.mkdir("Results")
        if not os.path.exists("Results/" + name):
            os.mkdir("Results/" + name)
        root += name + "_"


        for agent in agents_list:
            plot_agent_weights(agent.num, weights_over_time, self_update_timesteps, root)

            plot_agent_utility_reputation(
            agent.num,
            self_update_timesteps,
            reputations,
            personals,
            self_util,
            self_policy_e_util,
            root
        )

        new_plot_embed_withx(agents_list, self_update_timesteps, reputations, root, "reputations.png",
                             "Reputations of Agents Over Time")

        new_plot_embed_withx(agents_list, self_update_timesteps, personals, root, "personals.png",
                             "Averaged Personal Estimates of Agents Over Time")

        # new_plot_embed_withx(agents_list, self_update_timesteps, rates, root, "rates.png",
        #                      "Internal Rate of Agents Over Time")

        plt.figure(name + "ma epoch")
        plt.plot(self_update_timesteps, ma_epoch_rewards_tracker)
        plt.title("Moving Average of Social Welfare")
        plt.legend()
        plt.savefig(root + "_social_welfare.png")
        np.save(root + "_socwel" + ".npy", ma_epoch_rewards_tracker)
        plt.close()

        np.save(root + "update_timesteps" + ".npy", self_update_timesteps)

        new_plot_embed_withx(agents_list, self_update_timesteps, self_util, root, "self_utility_overall.png",
                             "Self-Utility Over Time")
        new_plot_embed_withx(agents_list, self_update_timesteps, self_policy_e_util, root, "self_utility_policy.png",
                             "Self-Utility Over Time")
        new_plot_embed_withx(agents_list, self_update_timesteps, ind_self_util, root, "ind_self_utility_overall.png",
                             "Independent Self-Utility Over Time")

        np.save(root + "self_util_full" + ".npy", np.array(self_util))
        np.save(root + "self_util_policy" + ".npy", np.array(self_policy_e_util))

        # # Plot PR, R , stuff related to policy for each agent 
       
        # # Per-Agent Reputation Plots
        # for agent in agents_list:
        #     plt.figure("Reputations4" + str(agent.num))
        #     plt.plot(all_reputations[agent.num][agent.num], label="Reputation")
        #     # plt.plot(agent.selfless_times, label="Influencer Indicator")

        #     plt.legend()
        #     plt.title("Agent Reputations and Role, Agent " + str(agent.num))
        #     plt.savefig(root2 + "/per_agent_rep/" + name + "_" + str(agent.num) + "_repalls.png")
        #     plt.close()
            
def plot_agent_utility_reputation(
    agent_num,
    xseries,
    reputations,
    personals,
    self_util,
    policy_util,
    root
):
    fig, axs = plt.subplots(4, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(xseries, reputations[agent_num])
    axs[0].set_ylabel("Reputation")
    axs[0].set_title(f"Agent {agent_num}")

    axs[1].plot(xseries, personals[agent_num])
    axs[1].set_ylabel("Personal Estimate")

    axs[2].plot(xseries, self_util[agent_num])
    axs[2].set_ylabel("Self Utility")

    axs[3].plot(xseries, policy_util[agent_num])
    axs[3].set_ylabel("Policy Utility")
    axs[3].set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(f"{root}agent_{agent_num}_utility_reputation.png")
    plt.close()

def plot_agent_weights(agent_num, weights_over_time, xseries, root):
    W = np.array(weights_over_time[agent_num])  # (T, N)

    if len(W) < 2:
        return  # not enough data

    # Weight updates between snapshots
    deltas = np.diff(W, axis=0)                  # (T-1, N)
    mean_delta = np.mean(np.abs(deltas), axis=1)

    plt.figure(figsize=(10,5))
    plt.plot(xseries[1:], mean_delta)
    plt.xlabel("Timestep")
    plt.ylabel("Mean |Δweight|")
    plt.title(f"Agent {agent_num} Weight Update Magnitude")
    plt.tight_layout()
    plt.savefig(f"{root}agent_{agent_num}_weight_deltas.png")
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


def new_plot_embed_withx(agents_list, xseries, yseries, root, name, title):
    plt.figure(name + title, figsize=(8, 5))
    ax = plt.subplot(111)
    box = ax.get_position()
    for ii in agents_list:
        ax.plot(xseries, yseries[ii.num], label="Agent " + str(ii.num))
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

"""
"status_factor" and "reputation_factor" are kappa and gamma, respectively.
"b0" is the outside utility (that governs rate allocation)
"cons" is the update interval for infl switch / rate allocation
"top_reward" and "bottom_reward" are the max/min reward for unimodal variance; they are the means for bimodal variance.
"epsilon" is the epsilon chance that special gossiping interactions will happen at each check.
"threshold" is the threshold at which an agent will consider optimizing for status.

"""

args = {
        "learning_rate": 0.00005,
    }

import gc

training_loop(
        num_agents=10,
        max_state_num=3,
        name="personal_utility_test_1_rewards_as_1",
        arguments=args,
        k=1.0,
        adjustment_factor=3.0,
    )
gc.collect()
