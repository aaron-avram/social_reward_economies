"""
Main toy experiment for isolating reputation scaling mechanism.
Tests both Test 4 approach and Paper approach.
n=20 agents, 2 states, 2 actions.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from toy_env import ToyEnv
from toy_agent import ToyAgent


def gossip(agents):
    """
    Average reputation vectors across all agents.
    Simplified from norm/ (no similarity filter) — the norm similarity bug
    (stale loop variable) means it effectively gossips with everyone anyway.
    """
    all_R = np.array([agent.R for agent in agents])
    avg_R = np.mean(all_R, axis=0)
    for agent in agents:
        agent.R = avg_R.copy()


def get_actions(agents, state):
    """
    Non-followers choose their own action; followers copy their influencer.
    Two-pass to guarantee influencer actions are resolved first.
    """
    actions = np.zeros(len(agents), dtype=int)
    for i, agent in enumerate(agents):
        if not agent.is_follower:
            actions[i] = agent.get_action(state)
    for i, agent in enumerate(agents):
        if agent.is_follower and agent.target_influencer >= 0:
            actions[i] = actions[agent.target_influencer]
    return actions


def update_follower_counts(agents):
    for agent in agents:
        agent.followers = 0
    for agent in agents:
        if agent.is_follower and agent.target_influencer >= 0:
            agents[agent.target_influencer].followers += 1


def run_experiment(num_agents=20, gamma=20, timesteps=50000,
                   approach='test4', learning_rate=0.1, seed=42):
    """
    Run a single experiment.

    Args:
        num_agents:    Number of agents (default 20 for baseline comparison).
        gamma:         Reputation scaling factor.
        timesteps:     Length of run.
        approach:      'test4' or 'paper'.
        learning_rate: η for reputation updates.
        seed:          RNG seed.

    Returns:
        dict of tracked metrics.
    """
    np.random.seed(seed)

    env = ToyEnv()
    agents = [
        ToyAgent(i, num_agents,
                 env.generate_state_dependent_rewards(i),
                 gamma, approach, learning_rate)
        for i in range(num_agents)
    ]

    max_followers       = []
    leader_ids          = []
    reputations         = [[] for _ in range(num_agents)]
    personal_benefits   = [[] for _ in range(num_agents)]
    num_followers_list  = []
    num_influencers_list = []
    num_independents_list = []

    for t in range(timesteps):
        state   = env.sample_state()
        actions = get_actions(agents, state)

        # ---------------------------------------------------------------
        # Issue 1 fix: every agent observes every other agent.
        # Outer loop = observer (j), inner loop = target (i).
        # agents[j].update_reputation(i, ...) computes j's utility from
        # i's action and updates j's estimate of i's reputation.
        # ---------------------------------------------------------------
        for j in range(num_agents):
            for i in range(num_agents):
                agents[j].update_reputation(i, actions, state, agents)

        # Gossip starts at timestep 50
        if t >= 50:
            gossip(agents)

        # ---------------------------------------------------------------
        # Issue 2 fix: independent_beta only updated when not following.
        # Policy also only trained when acting independently.
        # ---------------------------------------------------------------
        for i in range(num_agents):
            reward = agents[i].reward_function[state][actions[i]]
            if not agents[i].is_follower:
                agents[i].update_independent_beta(reward)
                agents[i].update_policy(state, actions[i], reward)

        # Influencer switching every 3000 timesteps (after gossip warm-up)
        if t % 3000 == 0 and t > 50:
            update_follower_counts(agents)
            for agent in agents:
                agent.switch_influencer(agents)
            update_follower_counts(agents)

        # Track metrics every 100 timesteps
        if t % 100 == 0:
            max_foll  = 0
            leader_id = -1
            for i, agent in enumerate(agents):
                if agent.followers > max_foll:
                    max_foll  = agent.followers
                    leader_id = i

            max_followers.append(max_foll)
            leader_ids.append(leader_id)

            for i, agent in enumerate(agents):
                # After gossip all agents share the same R vector; agent 0's
                # view is representative.
                reputations[i].append(agents[0].R[i])
                personal_benefits[i].append(agent.P[i] if approach == 'paper' else 0)

            num_followers     = sum(1 for a in agents if a.is_follower)
            num_influencers   = sum(1 for a in agents if a.followers > 0)
            num_independents  = num_agents - num_followers - num_influencers
            num_followers_list.append(num_followers)
            num_influencers_list.append(num_influencers)
            num_independents_list.append(num_independents)

    # Stability: fraction of last 25% of samples with the same leader
    last_quarter = leader_ids[len(leader_ids) * 3 // 4:]
    if last_quarter:
        top_leader = max(set(last_quarter), key=last_quarter.count)
        stability  = last_quarter.count(top_leader) / len(last_quarter)
    else:
        top_leader = -1
        stability  = 0.0

    conv_threshold = int(num_agents * 0.75)
    convergence_time = next(
        (i * 100 for i, mf in enumerate(max_followers) if mf >= conv_threshold),
        -1
    )

    return {
        'max_followers':          max_followers[-1] if max_followers else 0,
        'stable':                 stability > 0.8,
        'stability_score':        stability,
        'convergence_time':       convergence_time,
        'leader_id':              top_leader,
        'max_followers_over_time': max_followers,
        'leader_ids_over_time':   leader_ids,
        'reputations':            reputations,
        'personal_benefits':      personal_benefits,
        'num_followers':          num_followers_list,
        'num_influencers':        num_influencers_list,
        'num_independents':       num_independents_list,
        'timesteps':              list(range(0, timesteps, 100)),
    }


def plot_results(results, approach, gamma, learning_rate, output_dir):
    suffix = f"{approach}_gamma{gamma}_lr{learning_rate}"

    # Follower dynamics
    plt.figure(figsize=(10, 6))
    plt.plot(results['timesteps'], results['max_followers_over_time'], linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Max Followers')
    plt.title(f'Maximum Follower Count\n({approach}, γ={gamma}, lr={learning_rate})')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'followers_{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Role distribution
    plt.figure(figsize=(10, 6))
    plt.plot(results['timesteps'], results['num_followers'],    label='Followers',    linewidth=2)
    plt.plot(results['timesteps'], results['num_influencers'],  label='Influencers',  linewidth=2)
    plt.plot(results['timesteps'], results['num_independents'], label='Independents', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Count')
    plt.title(f'Agent Roles Over Time\n({approach}, γ={gamma}, lr={learning_rate})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'roles_{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Reputation convergence (top 5 agents by final reputation)
    final_reps    = [r[-1] if r else 0 for r in results['reputations']]
    top_5_indices = np.argsort(final_reps)[-5:]
    plt.figure(figsize=(10, 6))
    for idx in top_5_indices:
        plt.plot(results['timesteps'], results['reputations'][idx],
                 label=f'Agent {idx}', linewidth=2)
    plt.xlabel('Timestep')
    plt.ylabel('Reputation')
    plt.title(f'Reputation Values (Top 5 Agents)\n({approach}, γ={gamma}, lr={learning_rate})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'reputations_{suffix}.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    gammas     = [3, 5, 10, 20]
    num_agents = 20
    timesteps  = 50000
    lr         = 0.003
    approach   = 'paper'

    output_dir = os.path.join(os.path.dirname(__file__), "Results")
    os.makedirs(output_dir, exist_ok=True)

    colors     = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    all_results = {}

    for gamma in gammas:
        print(f"Running γ={gamma}...")
        results = run_experiment(num_agents=num_agents, gamma=gamma,
                                 timesteps=timesteps, approach=approach,
                                 learning_rate=lr, seed=42)
        all_results[gamma] = results
        print(f"  Max followers: {results['max_followers']}  Stable: {results['stable']}")
        plot_results(results, approach, gamma, lr, output_dir)

    # Max followers over time — one line per gamma
    plt.figure(figsize=(12, 5))
    for gamma, color in zip(gammas, colors):
        r = all_results[gamma]
        plt.plot(r['timesteps'], r['max_followers_over_time'],
                 label=f'γ={gamma}', color=color, linewidth=1.5)
    plt.axhline(y=num_agents - 1, color='black', linestyle='--',
                linewidth=0.8, label=f'Max possible ({num_agents - 1})')
    plt.xlabel('Timestep')
    plt.ylabel('Max Followers')
    plt.title(f'Max Followers Over Time by γ  (paper, lr={lr}, n={num_agents})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'max_followers_over_time_n{num_agents}_lr{lr}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}")
