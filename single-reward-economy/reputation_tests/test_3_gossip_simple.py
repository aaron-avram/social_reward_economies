"""
Test 3: Gossip Mechanism (Unrestricted Averaging)

Objective: Verify unrestricted gossip averaging matches reputation_scaling implementation

Setup:
- 10 agents, single state, single action
- Heterogeneous rewards: u_i(s,a) varies by agent (use generate_reward_table_variance)
- ENABLE gossip after timestep 50 (unrestricted - all agents share)
- NO similarity filtering (matches reputation_scaling line 416)
- NO influencer dynamics

Expected Outcome:
- Before gossip (t<50): R[j] values diverge across agents (different personal estimates)
- After gossip (t>50): R[j] converges to same value across all agents
- Converged value = average of true reputations

Success: After 10K timesteps:
- max_i,k |R_i[j] - R_k[j]| < 0.05 (all agents agree on R[j])
- Verify averaging formula: R_new = (Σ R_k) / num_agents
"""

import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from norm import NormEnv
from norm_agent import NormAgent
from norm_actor import ActorNetwork
import validation_utils as val_utils


def generate_reward_table_variance(top_reward, bottom_reward, max_states, max_actions):
    """
    Generate heterogeneous reward functions using variance.

    Each agent gets different rewards for the same state-action pair.
    """
    reward_table = []

    for s in range(max_states):
        state_rewards = []
        for a in range(max_actions):
            # Random reward between bottom_reward and top_reward
            reward = np.random.uniform(bottom_reward, top_reward)
            state_rewards.append(reward)
        reward_table.append(state_rewards)

    return np.array(reward_table)


def main():
    # Test configuration
    NUM_AGENTS = 10
    STATE_DIM = 1
    NUM_ACTIONS = 1
    NUM_STATES = 1
    GAMMA = 1.0
    NUM_TIMESTEPS = 10000
    GOSSIP_START = 50  # Start gossip at timestep 50
    LEARNING_RATE = 0.00005

    # Reward heterogeneity
    TOP_REWARD = 2.0
    BOTTOM_REWARD = 0.5

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Results directory
    results_dir = "Results/test_3_gossip_simple"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize environment
    env = NormEnv(state_dim=STATE_DIM, action_space=list(range(NUM_ACTIONS)))

    # Initialize agents with heterogeneous reward functions
    agents = []
    true_rewards = []  # Store true rewards for each agent

    for i in range(NUM_AGENTS):
        reward_function = generate_reward_table_variance(
            TOP_REWARD, BOTTOM_REWARD, NUM_STATES, NUM_ACTIONS
        )
        true_rewards.append(reward_function[0][0])  # Store the single reward value

        agent = NormAgent(
            agent_number=i,
            num_agents=NUM_AGENTS,
            reward_function=reward_function,
            device=device,
            enable_gossip=True,  # ENABLE gossip
            enable_influencer=False,
            enable_status=False
        )
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
        debug_log['pr_traces'][i] = []
        debug_log['beta_traces'][i] = []
        debug_log['utility_traces'][i] = []

    # Track reputation matrix over time (for gossip convergence analysis)
    reputation_matrices = []
    reputation_variances = []

    # Snapshot before gossip
    reputation_matrix_before = None

    print(f"\n{'='*60}")
    print("Test 3: Gossip Mechanism (Unrestricted Averaging)")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  States: {NUM_STATES}, Actions: {NUM_ACTIONS}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Timesteps: {NUM_TIMESTEPS}")
    print(f"  Gossip starts at: t={GOSSIP_START}")
    print(f"\nReward Heterogeneity:")
    print(f"  Range: [{BOTTOM_REWARD}, {TOP_REWARD}]")
    print(f"  True rewards per agent:")
    for i, r in enumerate(true_rewards):
        print(f"    Agent {i}: {r:.3f}")
    print(f"\nExpected Behavior:")
    print(f"  Before gossip (t<{GOSSIP_START}): R[j] diverges across agents")
    print(f"  After gossip (t>{GOSSIP_START}): R[j] converges to consensus")
    print(f"{'='*60}\n")

    # Training loop
    for t in range(NUM_TIMESTEPS):
        state = 0
        state_list = env.state_to_list(state)

        # All agents take action 0 (only action available)
        actions = [0] * NUM_AGENTS

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
            alpha = 0.1
            new_r = GAMMA * social_feedback
            agents[i].R[i] = (1 - alpha) * agents[i].R[i] + alpha * new_r

        # GOSSIP PHASE (after timestep 50, unrestricted averaging)
        if t >= GOSSIP_START:
            # Snapshot before first gossip
            if t == GOSSIP_START:
                reputation_matrix_before = np.array([
                    [agents[i].R[j] for j in range(NUM_AGENTS)]
                    for i in range(NUM_AGENTS)
                ])

            # Unrestricted gossip: all active agents share reputation estimates
            # This matches reputation_scaling/train.py lines 410-420
            active_agents = list(range(NUM_AGENTS))  # All agents active for this test

            if len(active_agents) > 1:
                # Average reputation estimates across all active agents
                for j in range(NUM_AGENTS):
                    # Collect all estimates of agent j's reputation
                    estimates = [agents[i].R[j] for i in active_agents]

                    # Compute average
                    avg_estimate = np.mean(estimates)

                    # Update all active agents' estimates to the average
                    for i in active_agents:
                        agents[i].R[j] = avg_estimate

        # Update beta
        for i in range(NUM_AGENTS):
            agents[i].update_beta(rewards[i], rewards[i])

        # Train networks
        for i in range(NUM_AGENTS):
            actor_networks[i].beta = agents[i].beta
            actor_networks[i].train(state_list, rewards[i], actions[i])

        # Record traces
        for i in range(NUM_AGENTS):
            debug_log['reputation_traces'][i].append(agents[i].R[i])
            debug_log['pr_traces'][i].append(agents[i].PR[i])
            debug_log['beta_traces'][i].append(agents[i].beta)
            debug_log['utility_traces'][i].append(rewards[i])

        # Track reputation matrix
        if t % 10 == 0:  # Sample every 10 timesteps
            rep_matrix = np.array([
                [agents[i].R[j] for j in range(NUM_AGENTS)]
                for i in range(NUM_AGENTS)
            ])
            reputation_matrices.append(rep_matrix)

            # Compute variance in estimates for each agent j
            variances = np.var(rep_matrix, axis=0)  # Variance across estimators (rows)
            reputation_variances.append(np.mean(variances))

        # Periodic validation
        if (t + 1) % 2000 == 0:
            print(f"\nTimestep {t+1}/{NUM_TIMESTEPS}")

            # Check gossip convergence
            convergence_result = val_utils.validate_gossip_convergence(agents, tolerance=0.05)

            print(f"  Gossip convergence: {'YES ✓' if convergence_result['converged'] else 'NO ✗'}")
            print(f"  Max divergence: {convergence_result['max_divergence']:.6f}")
            print(f"  Mean variance: {convergence_result['mean_variance']:.6f}")

    # Final validation
    print(f"\n{'='*60}")
    print("Final Validation")
    print(f"{'='*60}")

    # Final convergence check
    final_convergence = val_utils.validate_gossip_convergence(agents, tolerance=0.05)

    print(f"\nGossip Convergence:")
    print(f"  Converged: {'YES ✓' if final_convergence['converged'] else 'NO ✗'}")
    print(f"  Max divergence: {final_convergence['max_divergence']:.6f}")
    print(f"  Mean variance: {final_convergence['mean_variance']:.6f}")

    # Check if all agents agree on reputation values
    reputation_matrix_after = final_convergence['reputation_matrix']

    print(f"\nReputation Estimates (sample):")
    print(f"  Agent 0's reputation as estimated by:")
    for i in range(min(5, NUM_AGENTS)):
        print(f"    Agent {i}: {agents[i].R[0]:.6f}")

    # Test passed if convergence achieved
    test_passed = final_convergence['converged']

    print(f"\n{'='*60}")
    print(f"TEST 3 RESULT: {'PASSED ✓' if test_passed else 'FAILED ✗'}")
    print(f"{'='*60}\n")

    # Generate plots
    print("Generating plots...")

    # Plot 1: Gossip convergence over time
    plt.figure(figsize=(12, 6))
    plt.plot(reputation_variances, color='blue', alpha=0.7)
    plt.axvline(GOSSIP_START / 10, color='red', linestyle='--', label=f'Gossip starts (t={GOSSIP_START})')
    plt.axhline(0.05, color='orange', linestyle='--', label='Convergence threshold')
    plt.xlabel('Timestep (sampled every 10)')
    plt.ylabel('Mean Variance in Reputation Estimates')
    plt.title('Gossip Convergence: Variance in R Estimates Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'gossip_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Reputation trajectories for a sample agent
    plt.figure(figsize=(12, 6))
    sample_agent = 0
    for i in range(NUM_AGENTS):
        # Plot agent i's estimate of agent 0's reputation
        estimates = [reputation_matrices[t][i][sample_agent] for t in range(len(reputation_matrices))]
        plt.plot(estimates, label=f'Agent {i}', alpha=0.7)
    plt.axvline(GOSSIP_START / 10, color='red', linestyle='--', label=f'Gossip starts')
    plt.xlabel('Timestep (sampled every 10)')
    plt.ylabel(f'R[{sample_agent}] - Reputation estimate for Agent {sample_agent}')
    plt.title(f'Gossip Effect: All Agents\' Estimates of Agent {sample_agent}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'reputation_estimates_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Gossip effect (before vs after)
    if reputation_matrix_before is not None:
        val_utils.plot_gossip_effect(
            reputation_matrix_before,
            reputation_matrix_after,
            results_dir,
            'gossip_effect.png'
        )

    # Plot 4: All reputation traces
    val_utils.plot_reputation_convergence(
        debug_log['reputation_traces'],
        results_dir,
        'reputation_convergence.png'
    )

    # Save debug log
    val_utils.save_debug_log(debug_log, results_dir)

    # Save reputation matrices for analysis
    np.save(os.path.join(results_dir, 'reputation_matrices.npy'), np.array(reputation_matrices))
    np.save(os.path.join(results_dir, 'reputation_variances.npy'), np.array(reputation_variances))

    print(f"\nResults saved to: {results_dir}")
    print("Plots generated:")
    print("  - gossip_convergence.png")
    print("  - reputation_estimates_convergence.png")
    print("  - gossip_effect.png")
    print("  - reputation_convergence.png")
    print("  - debug_log.npz")
    print("  - reputation_matrices.npy")
    print("  - reputation_variances.npy")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(34243)
    torch.manual_seed(34243)

    main()
