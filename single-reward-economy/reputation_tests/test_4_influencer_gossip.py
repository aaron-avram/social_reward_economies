"""
Test 4: Influencer Dynamics WITH Gossip (Matches PDF Setup)

Objective: Reproduce preliminary results from PDF using gossip + influencer dynamics

Setup:
- 20 agents, two states (p=0.5 each), two actions
- State-dependent optimal: π(s=0)→a=0, π(s=1)→a=1
- ENABLE unrestricted gossip (after timestep 50)
- ENABLE influencer switching based on shared reputation estimates
- NO status optimization (matches reputation_scaling with become_selfless disabled)

Expected Outcome:
- Phase 1 (t<50): Agents learn independently
- Phase 2 (t>50): Gossip enables consensus on who has high reputation
- Phase 3: Followers emerge, copying high-reputation influencers
- Follower dynamics should match PDF graphs
- System utility → 1.5 (optimal policy)

Success: After 50K timesteps:
- ≥1 agent has followers > 5 (winner-takes-all effect)
- All agents have similar R[influencer] estimates (gossip convergence)
- Follower policies match influencer within 0.1
- Follower count trajectory matches PDF qualitatively
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from norm import NormEnv
from norm_agent import NormAgent
from norm_actor import ActorNetwork
import validation_utils as val_utils


def generate_reward_table_state_dependent(num_states, num_actions):
    """
    Generate state-dependent reward function.

    Optimal policy: π(s=0)→a=0, π(s=1)→a=1
    """
    reward_table = np.zeros((num_states, num_actions))

    for s in range(num_states):
        for a in range(num_actions):
            if s == a:
                # Matching state-action: high reward
                reward_table[s][a] = 1.0
            else:
                # Mismatched state-action: low reward
                reward_table[s][a] = 0.0

    return reward_table


def main():
    # Test configuration
    NUM_AGENTS = 20
    STATE_DIM = 1  # 2^1 = 2 states
    NUM_ACTIONS = 2
    NUM_STATES = 2
    GAMMA = 20.0  # Typical value for 20 agents (as per paper)
    KAPPA = 0.0  # NO status optimization
    NUM_TIMESTEPS = 50000
    GOSSIP_START = 50
    INFLUENCER_CHECK_INTERVAL = 3000  # Check for influencer switching every 3000 timesteps
    LEARNING_RATE = 0.00005

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Results directory
    results_dir = "Results/test_4_influencer_gossip"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize environment
    env = NormEnv(state_dim=STATE_DIM, action_space=list(range(NUM_ACTIONS)))

    # Initialize agents with state-dependent reward functions
    agents = []
    for i in range(NUM_AGENTS):
        # All agents have same reward function (homogeneous preferences for clearer dynamics)
        reward_function = generate_reward_table_state_dependent(NUM_STATES, NUM_ACTIONS)

        agent = NormAgent(
            agent_number=i,
            num_agents=NUM_AGENTS,
            reward_function=reward_function,
            device=device,
            enable_gossip=True,       # ENABLE gossip
            enable_influencer=True,   # ENABLE influencer dynamics
            enable_status=False       # NO status optimization (matches rep_scaling)
        )
        agent.reputation_factor = GAMMA
        agent.status_factor = KAPPA
        agents.append(agent)

    # Initialize actor networks
    actor_networks = []
    for i in range(NUM_AGENTS):
        network = ActorNetwork(
            oned_size=STATE_DIM,
            action_space=NUM_ACTIONS,
            alpha=LEARNING_RATE,
            num=i
        )
        actor_networks.append(network)

    # Initialize debug log
    debug_log = val_utils.create_debug_log()
    for i in range(NUM_AGENTS):
        debug_log['reputation_traces'][i] = []
        debug_log['beta_traces'][i] = []
        debug_log['utility_traces'][i] = []

    # Track follower dynamics
    follower_counts = {i: [] for i in range(NUM_AGENTS)}
    total_followers_over_time = []
    influencer_switches = []

    # Track roles
    role_counts = {'independent': [], 'follower': [], 'influencer': []}

    print(f"\n{'='*60}")
    print("Test 4: Influencer Dynamics WITH Gossip")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  States: {NUM_STATES}, Actions: {NUM_ACTIONS}")
    print(f"  Gamma: {GAMMA}, Kappa: {KAPPA}")
    print(f"  Timesteps: {NUM_TIMESTEPS}")
    print(f"  Gossip starts: t={GOSSIP_START}")
    print(f"  Influencer check interval: {INFLUENCER_CHECK_INTERVAL}")
    print(f"\nReward Function:")
    print(f"  π(s=0)→a=0 gives u=1.0, π(s=0)→a=1 gives u=0.0")
    print(f"  π(s=1)→a=1 gives u=1.0, π(s=1)→a=0 gives u=0.0")
    print(f"\nExpected Optimal Utility: 1.0 (always choose correct action)")
    print(f"{'='*60}\n")

    # Training loop
    for t in range(NUM_TIMESTEPS):
        # Sample state (uniform distribution over 2 states)
        state = np.random.choice([0, 1])
        state_list = env.state_to_list(state)

        # Get actions (followers copy their influencer)
        actions = []
        for i in range(NUM_AGENTS):
            if agents[i].is_follower and agents[i].target_influencer >= 0:
                # Follower: copy influencer's action
                influencer_id = agents[i].target_influencer
                action = actor_networks[influencer_id].get_learned_action(state_list, [0, 1])[0]
            else:
                # Independent or influencer: use own policy
                if np.random.random() < 0.05:  # 5% exploration
                    action = np.random.choice([0, 1])
                else:
                    action = actor_networks[i].get_learned_action(state_list, [0, 1])[0]
            actions.append(action)

        # Compute rewards
        rewards = []
        for i in range(NUM_AGENTS):
            reward = agents[i].reward_function[state][actions[i]]
            rewards.append(reward)

        # Update reputation (PR and R) - BEFORE gossip
        for i in range(NUM_AGENTS):
            # Update personal reputation
            agents[i].PR[i] = rewards[i]

            # Update reputation based on feedback from others
            social_feedback = 0.0
            for j in range(NUM_AGENTS):
                if j != i:
                    utility_j_from_i = agents[j].reward_function[state][actions[i]]
                    social_feedback += utility_j_from_i

            # Update R with exponential moving average
            alpha_r = 0.1
            new_r = GAMMA * social_feedback
            agents[i].R[i] = (1 - alpha_r) * agents[i].R[i] + alpha_r * new_r

        # GOSSIP PHASE (after timestep 50, unrestricted averaging)
        if t >= GOSSIP_START:
            active_agents = list(range(NUM_AGENTS))  # All agents active

            if len(active_agents) > 1:
                # Average reputation estimates across all active agents
                for j in range(NUM_AGENTS):
                    estimates = [agents[i].R[j] for i in active_agents]
                    avg_estimate = np.mean(estimates)

                    for i in active_agents:
                        agents[i].R[j] = avg_estimate

        # Update beta and independent_beta
        for i in range(NUM_AGENTS):
            agents[i].update_beta(rewards[i], rewards[i])
            # Update independent_beta (used for influencer switching)
            agents[i].independent_beta = agents[i].selfish_beta

        # Train networks (only if not following)
        for i in range(NUM_AGENTS):
            if not agents[i].is_follower:
                actor_networks[i].beta = agents[i].beta
                actor_networks[i].train(state_list, rewards[i], actions[i])

        # INFLUENCER SWITCHING (periodic check)
        if t > 0 and t % INFLUENCER_CHECK_INTERVAL == 0:
            # Count current followers
            current_follower_counts = {i: 0 for i in range(NUM_AGENTS)}

            for i in range(NUM_AGENTS):
                # Influencers cannot become followers
                if agents[i].followers > 0:
                    agents[i].switch_to_independent()
                    continue

                # Find best influencer based on reputation
                best_influencer = -1
                best_reputation = agents[i].independent_beta  # Threshold

                for j in range(NUM_AGENTS):
                    if j != i and agents[j].target_influencer == -1 and agents[j].R[j] > best_reputation:
                        best_reputation = agents[j].R[j]
                        best_influencer = j

                # Switch to follower if found better influencer
                if best_influencer >= 0:
                    if not agents[i].is_follower or agents[i].target_influencer != best_influencer:
                        agents[i].switch_to_follower(best_influencer)
                        influencer_switches.append({
                            'timestep': t,
                            'follower': i,
                            'influencer': best_influencer
                        })
                    current_follower_counts[best_influencer] += 1
                else:
                    # No good influencer, stay independent
                    agents[i].switch_to_independent()

            # Update follower counts
            for i in range(NUM_AGENTS):
                agents[i].followers = current_follower_counts[i]

        # Record traces
        for i in range(NUM_AGENTS):
            debug_log['reputation_traces'][i].append(agents[i].R[i])
            debug_log['beta_traces'][i].append(agents[i].beta)
            debug_log['utility_traces'][i].append(rewards[i])

            follower_counts[i].append(agents[i].followers)

        # Record role counts
        num_followers = sum(1 for a in agents if a.is_follower)
        num_influencers = sum(1 for a in agents if a.followers > 0)
        num_independent = NUM_AGENTS - num_followers - num_influencers

        role_counts['follower'].append(num_followers)
        role_counts['influencer'].append(num_influencers)
        role_counts['independent'].append(num_independent)

        total_followers_over_time.append(num_followers)

        # Periodic validation
        if (t + 1) % 10000 == 0:
            print(f"\nTimestep {t+1}/{NUM_TIMESTEPS}")

            avg_beta = np.mean([agents[i].beta for i in range(NUM_AGENTS)])
            avg_utility = np.mean(rewards)
            num_followers_now = sum(1 for a in agents if a.is_follower)
            num_influencers_now = sum(1 for a in agents if a.followers > 0)
            max_followers = max(a.followers for a in agents)

            print(f"  Average beta: {avg_beta:.6f}")
            print(f"  Average utility: {avg_utility:.6f}")
            print(f"  Followers: {num_followers_now}/{NUM_AGENTS}")
            print(f"  Influencers: {num_influencers_now}")
            print(f"  Max follower count: {max_followers}")

            if num_influencers_now > 0:
                print(f"  Top influencers:")
                top_influencers = sorted(range(NUM_AGENTS), key=lambda i: agents[i].followers, reverse=True)[:3]
                for rank, i in enumerate(top_influencers[:3]):
                    if agents[i].followers > 0:
                        print(f"    #{rank+1}: Agent {i} with {agents[i].followers} followers (R={agents[i].R[i]:.3f})")

    # Final validation
    print(f"\n{'='*60}")
    print("Final Validation")
    print(f"{'='*60}")

    final_utilities = np.array([debug_log['utility_traces'][i][-1] for i in range(NUM_AGENTS)])
    max_followers = max(a.followers for a in agents)
    num_influencers = sum(1 for a in agents if a.followers > 0)

    print(f"\nFollower Dynamics:")
    print(f"  Max followers: {max_followers}")
    print(f"  Number of influencers: {num_influencers}")
    print(f"  Winner-takes-all: {'YES ✓' if max_followers > 5 else 'NO ✗'}")

    print(f"\nSystem Performance:")
    print(f"  Average utility: {np.mean(final_utilities):.6f} (optimal: 1.0)")

    # Check gossip convergence
    convergence_result = val_utils.validate_gossip_convergence(agents, tolerance=0.05)
    print(f"\nGossip Convergence:")
    print(f"  Converged: {'YES ✓' if convergence_result['converged'] else 'NO ✗'}")
    print(f"  Max divergence: {convergence_result['max_divergence']:.6f}")

    # Test success criteria
    test_passed = (max_followers > 5 and convergence_result['converged'])

    print(f"\n{'='*60}")
    print(f"TEST 4 RESULT: {'PASSED ✓' if test_passed else 'FAILED ✗'}")
    print(f"{'='*60}\n")

    # Generate plots
    print("Generating plots...")

    # Plot 1: Follower counts over time (KEY PLOT - compare with PDF)
    plt.figure(figsize=(14, 6))
    top_influencers = sorted(range(NUM_AGENTS), key=lambda i: agents[i].followers, reverse=True)[:5]

    for i in top_influencers:
        if agents[i].followers > 0:
            plt.plot(follower_counts[i], label=f'Agent {i}', linewidth=2)

    plt.axvline(GOSSIP_START, color='red', linestyle='--', alpha=0.5, label='Gossip starts')
    plt.xlabel('Timestep')
    plt.ylabel('Follower Count')
    plt.title('Follower Dynamics Over Time (Top Influencers)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'follower_dynamics.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Total followers over time
    plt.figure(figsize=(14, 6))
    plt.plot(total_followers_over_time, color='blue', linewidth=2)
    plt.axvline(GOSSIP_START, color='red', linestyle='--', alpha=0.5, label='Gossip starts')
    plt.xlabel('Timestep')
    plt.ylabel('Total Number of Followers')
    plt.title('Total Followers Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'total_followers.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Role distribution over time
    plt.figure(figsize=(14, 6))
    plt.plot(role_counts['follower'], label='Followers', linewidth=2)
    plt.plot(role_counts['influencer'], label='Influencers', linewidth=2)
    plt.plot(role_counts['independent'], label='Independent', linewidth=2)
    plt.axvline(GOSSIP_START, color='red', linestyle='--', alpha=0.5, label='Gossip starts')
    plt.xlabel('Timestep')
    plt.ylabel('Number of Agents')
    plt.title('Agent Roles Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'role_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Reputation distributions (KEY PLOT - compare with PDF)
    plt.figure(figsize=(14, 6))
    for i in top_influencers[:5]:
        plt.plot(debug_log['reputation_traces'][i], label=f'Agent {i}', linewidth=2)
    plt.axvline(GOSSIP_START, color='red', linestyle='--', alpha=0.5, label='Gossip starts')
    plt.xlabel('Timestep')
    plt.ylabel('Reputation (R)')
    plt.title('Reputation Values Over Time (Top Influencers)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'reputation_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 5: Beta convergence
    plt.figure(figsize=(14, 6))
    for i in range(NUM_AGENTS):
        plt.plot(debug_log['beta_traces'][i], alpha=0.5)
    plt.axhline(1.0, color='red', linestyle='--', label='Optimal beta')
    plt.xlabel('Timestep')
    plt.ylabel('Beta (Utility Baseline)')
    plt.title('Beta Convergence Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'beta_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save debug log and data
    val_utils.save_debug_log(debug_log, results_dir)
    np.save(os.path.join(results_dir, 'follower_counts.npy'), np.array([follower_counts[i] for i in range(NUM_AGENTS)]))
    np.save(os.path.join(results_dir, 'influencer_switches.npy'), np.array(influencer_switches))

    print(f"\nResults saved to: {results_dir}")
    print("KEY PLOTS (compare with PDF):")
    print("  - follower_dynamics.png (Figure: Follower counts)")
    print("  - reputation_distribution.png (Figure: Reputation values)")
    print("\nOther plots:")
    print("  - total_followers.png")
    print("  - role_distribution.png")
    print("  - beta_convergence.png")
    print("  - debug_log.npz")
    print("  - follower_counts.npy")
    print("  - influencer_switches.npy")

    print(f"\n{'='*60}")
    print("NEXT STEP: Compare follower_dynamics.png and reputation_distribution.png")
    print("with corresponding figures from the PDF to validate results!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(34243)
    torch.manual_seed(34243)

    main()
