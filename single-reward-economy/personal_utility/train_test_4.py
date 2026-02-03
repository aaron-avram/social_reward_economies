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


def compute_overall_policy(max_state, agents_list, p_network_list, i_network_list, state_probs=None):
    """
    Compute overall policy reward, optionally weighted by state probabilities.
    """
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
    num_states = len(pos_states)
    
    # Default to uniform probabilities if not specified
    if state_probs is None:
        state_probs = np.ones(num_states) / num_states
    else:
        state_probs = np.array(state_probs)
        state_probs = state_probs / np.sum(state_probs)
    
    for statenum, statevec in enumerate(pos_states):
        state_prob = state_probs[statenum]
        if agents_list[top_agent].selfless:
            output = i_network_list[top_agent].get_function_output(np.array(statevec))
        else:
            output = p_network_list[top_agent].get_function_output(np.array(statevec))
        for actnum, actchance in enumerate(output):
            for ag in agents_list:
                tot_reward += ag.get_utility(statevec, actnum) * actchance * state_prob

    print("FINAL OVERALL POLICY REWARD:", tot_reward)
    print(f"Expected utility (Test 4): 0.5 * 1 + 0.5 * 2 = {0.5 * 1 + 0.5 * 2}")


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


def compute_policy_interim(max_state, agents_list, p_network_list, i_network_list, state_probs=None):
    """
    Compute expected policy utility for each agent.
    If state_probs is provided, weight states by their probabilities.
    Otherwise, assume uniform distribution over states.
    """
    bits = max(1, max_state.bit_length())
    numbers = np.arange(max_state + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    num_states = len(pos_states)
    
    # Default to uniform probabilities if not specified
    if state_probs is None:
        state_probs = np.ones(num_states) / num_states
    else:
        # Ensure probabilities sum to 1
        state_probs = np.array(state_probs)
        state_probs = state_probs / np.sum(state_probs)
    
    reward_list = []
    for ag in agents_list:
        reward = 0
        for statenum, statevec in enumerate(pos_states):
            state_prob = state_probs[statenum]
            if ag.selfless:
                output = i_network_list[ag.num].get_function_output(np.array(statevec))
            else:
                output = p_network_list[ag.num].get_function_output(np.array(statevec))
            for actnum, actchance in enumerate(output):
                reward += ag.get_utility(statevec, actnum) * actchance * state_prob
        reward_list.append(reward)
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


def get_test4_state():
    """
    Test 4: Multiple States with Highest Utility Action Different in Each
    
    State space: S = {0, 1} with probabilities p(0) = 0.5, p(1) = 0.5
    Action space: A = {0, 1}
    Utilities: 
      u(0, 0) = 1, u(0, 1) = 0
      u(1, 0) = 0, u(1, 1) = 2
    Expected policy: π(0) = 0, π(1) = 1
    Expected utility: 0.5 * 1 + 0.5 * 2 = 1.5
    """
    # Sample state according to probabilities: p(0) = 0.5, p(1) = 0.5
    rand_val = np.random.random()
    if rand_val < 0.5:
        state_num = 0
    else:
        state_num = 1
    
    # Convert state number to binary vector
    state = np.array([int(d) for d in f'{state_num:01b}'])
    
    # Action space: A = {0, 1}
    action_space = np.array([0, 1])
    
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
    # For Test 4: max_state_num=1 (states 0 and 1)
    # Calculate state_size needed to represent max_state_num
    if max_state_num == 0:
        state_size = 1
    else:
        state_size = max(1, int(np.ceil(np.log2(max_state_num + 1))))
    
    sample_state, sample_action_space = get_test4_state()
    possible_action_size = len(sample_action_space)
    
    # State probabilities for Test 4: p(0) = 0.5, p(1) = 0.5
    state_probs = [0.5, 0.5]  # Probabilities for states 0 and 1

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
        agents_list[agn].reward_function = norm_env.test4_reward_table(state_size, possible_action_size)
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

        # Track state distribution to verify p(0) = 0.5, p(1) = 0.5
        state_distribution = {0: 0, 1: 0}
        state_distribution_over_time = []  # Track distribution at each update interval

        # Track actions chosen by each agent over time
        actions_chosen = []
        for ii in agents_list:
            actions_chosen.append([])  # actions_chosen[agent_num] will store actions at each update interval

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

        state, actions = get_test4_state()  # Sample state according to p(0)=0.5, p(1)=0.5
        
        # Track state distribution
        state_num = int(state[0])  # Convert state vector to state number
        state_distribution[state_num] = state_distribution.get(state_num, 0) + 1

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
            # Use state probabilities: p(0) = 0.5, p(1) = 0.5
            self_policy_reward = compute_policy_interim(max_state_num, agents_list, p_network_list, i_network_list, state_probs=state_probs)
            for ag in agents_list:
                self_policy_e_util[ag.num].append(self_policy_reward[ag.num])
            
            # Track state distribution at this interval
            total_states_seen = sum(state_distribution.values())
            if total_states_seen > 0:
                dist = {
                    0: state_distribution.get(0, 0) / total_states_seen,
                    1: state_distribution.get(1, 0) / total_states_seen
                }
                state_distribution_over_time.append(dist.copy())
                print(f"  State distribution: p(0)={dist[0]:.3f}, p(1)={dist[1]:.3f} (expected: 0.500, 0.500)")
            
            # Track actions chosen at this timestep
            state, actions = get_test4_state()
            action_summary = {0: 0, 1: 0}
            for agent_num in range(num_agents):
                chosen_action = agents_list[agent_num].get_action(state, actions)[0]
                actions_chosen[agent_num].append(chosen_action)
                action_summary[int(chosen_action)] = action_summary.get(int(chosen_action), 0) + 1
            print(f"  Actions chosen: {action_summary[0]} agents chose action 0, {action_summary[1]} agents chose action 1")
            
            # Print policy probabilities for verification
            bits = max(1, max_state_num.bit_length())
            numbers = np.arange(max_state_num + 1)[:, np.newaxis]
            powers = np.arange(bits - 1, -1, -1)
            pos_states = list((numbers >> powers) & 1)
            print(f"  Policy probabilities (expected: State 0 → action 0, State 1 → action 1):")
            for statenum, statevec in enumerate(pos_states):
                output = p_network_list[0].get_function_output(np.array(statevec))
                print(f"    State {statenum}: action probabilities = {[f'{p:.4f}' for p in output]}")
            
            # Print expected utility from policy
            if len(self_policy_e_util[0]) > 0:
                current_expected_util = self_policy_e_util[0][-1]
                print(f"  Expected utility from policy: {current_expected_util:.4f} (target: 1.5000)")
                print(f"  Difference from target: {abs(current_expected_util - 1.5):.4f}")

    print("Final Overall Best Policy |", name)
    compute_overall_policy(max_state_num, agents_list, p_network_list, i_network_list, state_probs=state_probs)
    
    # Print final state distribution summary
    total_final = sum(state_distribution.values())
    if total_final > 0:
        print(f"\nFinal State Distribution Summary:")
        print(f"  State 0: {state_distribution[0]} occurrences ({100*state_distribution[0]/total_final:.2f}%)")
        print(f"  State 1: {state_distribution[1]} occurrences ({100*state_distribution[1]/total_final:.2f}%)")
        print(f"  Expected: State 0 = 50%, State 1 = 50%")
    
    # Print final policy summary
    print(f"\nFinal Policy Summary (all agents should be identical):")
    bits = max(1, max_state_num.bit_length())
    numbers = np.arange(max_state_num + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    for statenum, statevec in enumerate(pos_states):
        # Check first agent's policy (all should be the same)
        output = p_network_list[0].get_function_output(np.array(statevec))
        print(f"  State {statenum}: action probabilities = {[f'{p:.4f}' for p in output]}")
        if statenum == 0:
            print(f"    Expected: [1.0, 0.0] (action 0 optimal)")
        else:
            print(f"    Expected: [0.0, 1.0] (action 1 optimal)")
    
    # Print final action summary
    if len(actions_chosen[0]) > 0:
        print("\nFinal Action Summary:")
        state, actions = get_test4_state()
        final_action_summary = {0: 0, 1: 0}
        for agent_num in range(num_agents):
            final_action = agents_list[agent_num].get_action(state, actions)[0]
            final_action_summary[int(final_action)] = final_action_summary.get(int(final_action), 0) + 1
        print(f"  Action 0: {final_action_summary[0]} agents")
        print(f"  Action 1: {final_action_summary[1]} agents")
        print(f"  Expected: Agents should choose action 0 in state 0, action 1 in state 1")
    
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
                             "Actual Rewards Received Over Time\n(Note: Variance expected - rewards fluctuate between 0, 1, and 2)")
        new_plot_embed_withx(agents_list, self_update_timesteps, self_policy_e_util, root, "self_utility_policy.png",
                             "Expected Utility from Policy Over Time\n(Should converge to 1.5)")
        new_plot_embed_withx(agents_list, self_update_timesteps, ind_self_util, root, "ind_self_utility_overall.png",
                             "Independent Self-Utility Over Time")

        np.save(root + "self_util_full" + ".npy", np.array(self_util))
        np.save(root + "self_util_policy" + ".npy", np.array(self_policy_e_util))
        
        # Save and plot state distribution
        if len(state_distribution_over_time) > 0:
            np.save(root + "state_distribution_over_time" + ".npy", state_distribution_over_time)
            plot_state_distribution(self_update_timesteps, state_distribution_over_time, root)
        
        # Save and plot actions chosen
        np.save(root + "actions_chosen" + ".npy", np.array(actions_chosen))
        plot_actions_chosen(agents_list, self_update_timesteps, actions_chosen, root)
        
        # Plot policy probabilities over time
        plot_policy_probabilities(agents_list, self_update_timesteps, behaviours, max_state_num, root)

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


def plot_state_distribution(xseries, state_distribution_over_time, root):
    """
    Plot the observed state distribution over time.
    Expected: p(0) = 0.5, p(1) = 0.5
    """
    if len(state_distribution_over_time) == 0:
        return
    
    plt.figure("State Distribution Over Time", figsize=(10, 6))
    ax = plt.subplot(111)
    
    state0_probs = [d[0] for d in state_distribution_over_time]
    state1_probs = [d[1] for d in state_distribution_over_time]
    
    ax.plot(xseries, state0_probs, 'b-o', label='State 0', markersize=3, alpha=0.7)
    ax.plot(xseries, state1_probs, 'r-s', label='State 1', markersize=3, alpha=0.7)
    ax.axhline(y=0.5, color='g', linestyle='--', alpha=0.3, label='Expected p(0) = p(1) = 0.5')
    
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Observed Probability")
    ax.set_title("State Distribution Over Time\n(Expected: p(0) = 0.5, p(1) = 0.5)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(root + "state_distribution.png")
    plt.close()


def plot_actions_chosen(agents_list, xseries, actions_chosen, root):
    """
    Plot the actions chosen by each agent over time.
    For Test 4: Expected: action 0 in state 0, action 1 in state 1.
    """
    plt.figure("Actions Chosen Over Time", figsize=(10, 6))
    ax = plt.subplot(111)
    box = ax.get_position()
    
    # Plot actions for each agent
    for agent in agents_list:
        ax.plot(xseries, actions_chosen[agent.num], 'o-', label=f"Agent {agent.num}", 
                markersize=3, alpha=0.7)
    
    # Add horizontal lines for action values
    ax.axhline(y=0, color='b', linestyle='--', alpha=0.3, label='Action 0 (optimal in state 0)')
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.3, label='Action 1 (optimal in state 1)')
    
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Action Chosen")
    ax.set_title("Actions Chosen by Agents Over Time\n(Expected: action 0 in state 0, action 1 in state 1)")
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Action 0', 'Action 1'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    plt.savefig(root + "actions_chosen.png")
    plt.close()
    
    # Also create a summary plot showing action distribution over time
    plt.figure("Action Distribution Over Time", figsize=(10, 6))
    action_counts = []
    for t_idx in range(len(xseries)):
        counts = {0: 0, 1: 0}
        for agent in agents_list:
            if t_idx < len(actions_chosen[agent.num]):
                action = int(actions_chosen[agent.num][t_idx])
                counts[action] = counts.get(action, 0) + 1
        action_counts.append(counts)
    
    action0_counts = [c[0] for c in action_counts]
    action1_counts = [c[1] for c in action_counts]
    
    ax = plt.subplot(111)
    ax.plot(xseries, action0_counts, 'b-o', label='Action 0', markersize=3, alpha=0.7)
    ax.plot(xseries, action1_counts, 'r-s', label='Action 1', markersize=3, alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Distribution of Actions Over Time\n(Expected: Mixed based on state)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig(root + "action_distribution.png")
    plt.close()


def plot_policy_probabilities(agents_list, xseries, behaviours, max_state_num, root):
    """
    Plot policy probabilities for each state over time.
    For Test 4: Should show probability 1.0 for action 0 in state 0, action 1 in state 1.
    """
    bits = max(1, max_state_num.bit_length())
    numbers = np.arange(max_state_num + 1)[:, np.newaxis]
    powers = np.arange(bits - 1, -1, -1)
    pos_states = list((numbers >> powers) & 1)
    
    for statenum in range(len(pos_states)):
        plt.figure(f"Policy Probabilities State {statenum}", figsize=(10, 6))
        ax = plt.subplot(111)
        
        # Plot action probabilities for each agent (should all be identical)
        for agent in agents_list:
            if len(behaviours[agent.num]) > statenum:
                if len(behaviours[agent.num][statenum]) > 0:
                    # Get probabilities for both actions
                    action0_probs = [b[0] if len(b) > 0 else 0 for b in behaviours[agent.num][statenum]]
                    action1_probs = [b[1] if len(b) > 1 else 0 for b in behaviours[agent.num][statenum]]
                    ax.plot(xseries, action0_probs, label=f"Agent {agent.num} - Action 0", alpha=0.5, linewidth=1)
                    ax.plot(xseries, action1_probs, '--', label=f"Agent {agent.num} - Action 1", alpha=0.5, linewidth=1)
        
        if statenum == 0:
            ax.axhline(y=1.0, color='b', linestyle='-', alpha=0.5, label='Expected Action 0 (1.0)')
            ax.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, label='Expected Action 1 (0.0)')
            expected_title = "Policy Probabilities for State 0\n(Expected: [1.0, 0.0] - action 0 optimal)"
        else:
            ax.axhline(y=0.0, color='b', linestyle='-', alpha=0.5, label='Expected Action 0 (0.0)')
            ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Expected Action 1 (1.0)')
            expected_title = "Policy Probabilities for State 1\n(Expected: [0.0, 1.0] - action 1 optimal)"
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Probability")
        ax.set_title(expected_title)
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.savefig(root + f"policy_probabilities_state_{statenum}.png")
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
        max_state_num=1,  # States: S = {0, 1}
        name="personal_utility_test_4",
        arguments=args,
        k=1.0,
        adjustment_factor=3.0,
    )
gc.collect()
