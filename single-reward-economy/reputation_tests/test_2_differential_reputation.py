"""
Test 2: Differential Reputation Learning

Objective: Verify agents distinguish high vs low reputation actions

Setup:
- 10 agents, single state, two actions
- u_i(s,0) = 0.0, u_i(s,1) = 1.0
- NO gossip, NO influencer dynamics

Expected Outcome:
- Agents learn action 1 (higher utility)
- R[j] → gamma * (num_agents-1) * 1.0 for agents choosing action 1
- Policy converges to π(a=1) ≈ 1.0

Success: After 15K timesteps, π(a=1) > 0.95, PR[j] > 0.9
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


def main():
    # Test configuration
    NUM_AGENTS = 10
    STATE_DIM = 1
    NUM_ACTIONS = 2  # Two actions: 0 (bad) and 1 (good)
    NUM_STATES = 1
    GAMMA = 1.0
    NUM_TIMESTEPS = 15000
    LEARNING_RATE = 0.0001  # Higher learning rate for faster convergence

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Results directory
    results_dir = "Results/test_2_differential_reputation"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize environment
    env = NormEnv(state_dim=STATE_DIM, action_space=list(range(NUM_ACTIONS)))

    # Initialize agents with binary reward function
    agents = []
    for i in range(NUM_AGENTS):
        # u(s,0) = 0.0, u(s,1) = 1.0
        reward_function = np.array([[0.0, 1.0]])  # Shape: [states, actions]
        agent = NormAgent(
            agent_number=i,
            num_agents=NUM_AGENTS,
            reward_function=reward_function,
            device=device,
            enable_gossip=False,  # No gossip in this test
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

    # Track policy evolution
    policy_action1_probs = {i: [] for i in range(NUM_AGENTS)}
    action_counts = {0: [], 1: []}  # Track how many agents choose each action

    # Expected values
    expected_r_good = GAMMA * (NUM_AGENTS - 1) * 1.0  # R when choosing action 1
    expected_r_bad = GAMMA * (NUM_AGENTS - 1) * 0.0   # R when choosing action 0

    print(f"\n{'='*60}")
    print("Test 2: Differential Reputation Learning")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  States: {NUM_STATES}, Actions: {NUM_ACTIONS}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Timesteps: {NUM_TIMESTEPS}")
    print(f"\nReward Function:")
    print(f"  u(s,0) = 0.0 (bad action)")
    print(f"  u(s,1) = 1.0 (good action)")
    print(f"\nExpected Outcomes:")
    print(f"  Policy: π(a=1) → 1.0 (choose good action)")
    print(f"  R[j] → {expected_r_good:.1f} (for good agents)")
    print(f"  Beta → 1.0")
    print(f"{'='*60}\n")

    # Training loop
    for t in range(NUM_TIMESTEPS):
        state = 0
        state_list = env.state_to_list(state)

        # Get actions from policy (with epsilon-greedy exploration)
        actions = []
        for i in range(NUM_AGENTS):
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

        # Update reputation (PR and R)
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

            # Record policy for action 1
            policy_probs = actor_networks[i].get_function_output(state_list)
            policy_action1_probs[i].append(policy_probs[1])

        # Track action distribution
        action_counts[0].append(sum(1 for a in actions if a == 0))
        action_counts[1].append(sum(1 for a in actions if a == 1))

        # Periodic validation
        if (t + 1) % 3000 == 0:
            print(f"\nTimestep {t+1}/{NUM_TIMESTEPS}")

            avg_beta = np.mean([agents[i].beta for i in range(NUM_AGENTS)])
            avg_action1_prob = np.mean([policy_action1_probs[i][-1] for i in range(NUM_AGENTS)])
            num_choosing_action1 = sum(1 for a in actions if a == 1)

            print(f"  Average beta: {avg_beta:.6f}")
            print(f"  Average π(a=1): {avg_action1_prob:.6f}")
            print(f"  Agents choosing action 1: {num_choosing_action1}/{NUM_AGENTS}")

    # Final validation
    print(f"\n{'='*60}")
    print("Final Validation")
    print(f"{'='*60}")

    final_beta = np.array([agents[i].beta for i in range(NUM_AGENTS)])
    final_action1_probs = np.array([policy_action1_probs[i][-1] for i in range(NUM_AGENTS)])
    final_pr = np.array([agents[i].PR[i] for i in range(NUM_AGENTS)])

    print(f"\nPolicy Statistics:")
    print(f"  Mean π(a=1): {np.mean(final_action1_probs):.6f}")
    print(f"  Min π(a=1): {np.min(final_action1_probs):.6f}")
    print(f"  Max π(a=1): {np.max(final_action1_probs):.6f}")
    print(f"  Convergence: {'PASSED ✓' if np.mean(final_action1_probs) > 0.95 else 'FAILED ✗'}")

    print(f"\nBeta Statistics:")
    print(f"  Mean: {np.mean(final_beta):.6f} (expected: ~1.0)")
    print(f"  Std:  {np.std(final_beta):.6f}")

    print(f"\nPR Statistics:")
    print(f"  Mean: {np.mean(final_pr):.6f} (expected: ~1.0)")
    print(f"  Min: {np.min(final_pr):.6f}")
    print(f"  Convergence: {'PASSED ✓' if np.mean(final_pr) > 0.9 else 'FAILED ✗'}")

    # Overall test result
    test_passed = (np.mean(final_action1_probs) > 0.95 and np.mean(final_pr) > 0.9)
    print(f"\n{'='*60}")
    print(f"TEST 2 RESULT: {'PASSED ✓' if test_passed else 'FAILED ✗'}")
    print(f"{'='*60}\n")

    # Generate plots
    print("Generating plots...")

    # Plot 1: Policy evolution
    plt.figure(figsize=(12, 6))
    for i in range(NUM_AGENTS):
        plt.plot(policy_action1_probs[i], label=f'Agent {i}', alpha=0.7)
    plt.axhline(1.0, color='r', linestyle='--', label='Optimal π(a=1)')
    plt.axhline(0.95, color='orange', linestyle='--', label='Success threshold')
    plt.xlabel('Timestep')
    plt.ylabel('π(a=1) - Probability of choosing action 1')
    plt.title('Policy Learning: Probability of Good Action')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'policy_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Action distribution over time
    plt.figure(figsize=(12, 6))
    plt.plot(action_counts[0], label='Action 0 (bad)', color='red', alpha=0.7)
    plt.plot(action_counts[1], label='Action 1 (good)', color='green', alpha=0.7)
    plt.xlabel('Timestep')
    plt.ylabel('Number of Agents')
    plt.title('Action Distribution Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'action_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Beta convergence
    plt.figure(figsize=(12, 6))
    for i in range(NUM_AGENTS):
        plt.plot(debug_log['beta_traces'][i], label=f'Agent {i}', alpha=0.7)
    plt.axhline(1.0, color='r', linestyle='--', label='Expected beta')
    plt.xlabel('Timestep')
    plt.ylabel('Beta (Utility Baseline)')
    plt.title('Beta Convergence Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'beta_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Reputation convergence
    val_utils.plot_reputation_convergence(
        debug_log['reputation_traces'],
        results_dir,
        'reputation_convergence.png'
    )

    # Save debug log
    val_utils.save_debug_log(debug_log, results_dir)

    print(f"\nResults saved to: {results_dir}")
    print("Plots generated:")
    print("  - policy_evolution.png")
    print("  - action_distribution.png")
    print("  - beta_convergence.png")
    print("  - reputation_convergence.png")
    print("  - debug_log.npz")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(34243)
    torch.manual_seed(34243)

    main()
