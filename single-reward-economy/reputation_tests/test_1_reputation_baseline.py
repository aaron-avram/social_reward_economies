"""
Test 1: Reputation Calculation Baseline

Objective: Verify reputation updates work in simplest case

Setup:
- 10 agents, single state, single action
- Constant reward: u_i(s,a) = 1.0 for all
- NO gossip, NO influencer dynamics, NO rate dynamics
- All agents always active

Expected Outcome:
- PR[j] → 1.0 (each agent gets 1.0 utility from each action)
- R[j] → gamma * (num_agents - 1) * 1.0 (reputation = gamma * sum of others' utilities)
- beta → 1.0 (personal utility baseline)

Success: After 10K timesteps, |PR[j] - 1.0| < 0.01 and |R[j] - expected_R| < 0.1
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
    STATE_DIM = 1  # Number of binary features (2^1 = 2 states, but we use only 1)
    NUM_ACTIONS = 1
    NUM_STATES = 1  # We'll only use state 0
    GAMMA = 1.0  # Reputation scaling factor
    NUM_TIMESTEPS = 10000
    LEARNING_RATE = 0.00005

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Results directory
    results_dir = "Results/test_1_reputation_baseline"
    os.makedirs(results_dir, exist_ok=True)

    # Initialize environment
    env = NormEnv(state_dim=STATE_DIM, action_space=list(range(NUM_ACTIONS)))

    # Initialize agents with constant reward function u(s,a) = 1.0
    agents = []
    for i in range(NUM_AGENTS):
        reward_function = np.ones((NUM_STATES, NUM_ACTIONS))  # All rewards = 1.0
        agent = NormAgent(
            agent_number=i,
            num_agents=NUM_AGENTS,
            reward_function=reward_function,
            device=device
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

    # Initialize traces for each agent
    for i in range(NUM_AGENTS):
        debug_log['reputation_traces'][i] = []
        debug_log['pr_traces'][i] = []
        debug_log['beta_traces'][i] = []
        debug_log['utility_traces'][i] = []

    # Expected values (theoretical)
    expected_pr = 1.0
    expected_r = GAMMA * (NUM_AGENTS - 1) * 1.0  # gamma * sum of others' utilities
    expected_beta = 1.0

    print(f"\n{'='*60}")
    print("Test 1: Reputation Baseline")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Agents: {NUM_AGENTS}")
    print(f"  States: {NUM_STATES}, Actions: {NUM_ACTIONS}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Timesteps: {NUM_TIMESTEPS}")
    print(f"\nExpected Values:")
    print(f"  PR[j] → {expected_pr:.3f}")
    print(f"  R[j] → {expected_r:.3f}")
    print(f"  beta → {expected_beta:.3f}")
    print(f"{'='*60}\n")

    # Training loop
    for t in range(NUM_TIMESTEPS):
        # Sample state (always state 0 in this test)
        state = 0

        # All agents take action 0 (only action available)
        actions = [0] * NUM_AGENTS

        # Compute rewards for each agent
        rewards = []
        for i in range(NUM_AGENTS):
            reward = agents[i].reward_function[state][actions[i]]
            rewards.append(reward)

        # Update reputation (PR and R)
        for i in range(NUM_AGENTS):
            # Update personal reputation (PR) - each agent's own utility tracking
            agents[i].PR[i] = rewards[i]

            # Update reputation (R) based on feedback from others
            # R_i = gamma * sum of utilities others get from i's action
            social_feedback = 0.0
            for j in range(NUM_AGENTS):
                if j != i:
                    # How much utility does j get from i's action?
                    utility_j_from_i = agents[j].reward_function[state][actions[i]]
                    social_feedback += utility_j_from_i

            # Update R with exponential moving average (alpha = 0.1)
            alpha = 0.1
            new_r = GAMMA * social_feedback
            agents[i].R[i] = (1 - alpha) * agents[i].R[i] + alpha * new_r

        # Update beta (personal utility baseline)
        for i in range(NUM_AGENTS):
            agents[i].update_beta(rewards[i], rewards[i])  # selfish_reward = status_reward in this test

        # Train networks (even though policy is deterministic with 1 action)
        for i in range(NUM_AGENTS):
            # Convert state to binary list format expected by network
            state_list = env.state_to_list(state)

            # Update network's beta for advantage calculation
            actor_networks[i].beta = agents[i].beta

            # Train with state, reward, action
            actor_networks[i].train(state_list, rewards[i], actions[i])

        # Record traces
        for i in range(NUM_AGENTS):
            debug_log['reputation_traces'][i].append(agents[i].R[i])
            debug_log['pr_traces'][i].append(agents[i].PR[i])
            debug_log['beta_traces'][i].append(agents[i].beta)
            debug_log['utility_traces'][i].append(rewards[i])

        # Periodic validation
        if (t + 1) % 2500 == 0:
            print(f"\nTimestep {t+1}/{NUM_TIMESTEPS}")

            # Check convergence
            avg_pr = np.mean([agents[i].PR[i] for i in range(NUM_AGENTS)])
            avg_r = np.mean([agents[i].R[i] for i in range(NUM_AGENTS)])
            avg_beta = np.mean([agents[i].beta for i in range(NUM_AGENTS)])

            print(f"  Average PR: {avg_pr:.6f} (expected: {expected_pr:.3f})")
            print(f"  Average R:  {avg_r:.6f} (expected: {expected_r:.3f})")
            print(f"  Average beta: {avg_beta:.6f} (expected: {expected_beta:.3f})")

            # Validate formula
            validation_result = val_utils.validate_reputation_formula(
                agents, state, actions, GAMMA, epsilon=0.5
            )
            print(f"  Formula validation: {'PASSED ✓' if validation_result['passed'] else 'FAILED ✗'}")
            print(f"    Max error: {validation_result['max_error']:.6f}")

    # Final validation
    print(f"\n{'='*60}")
    print("Final Validation")
    print(f"{'='*60}")

    final_pr = np.array([agents[i].PR[i] for i in range(NUM_AGENTS)])
    final_r = np.array([agents[i].R[i] for i in range(NUM_AGENTS)])
    final_beta = np.array([agents[i].beta for i in range(NUM_AGENTS)])

    pr_error = np.abs(final_pr - expected_pr)
    r_error = np.abs(final_r - expected_r)
    beta_error = np.abs(final_beta - expected_beta)

    print(f"\nPR Statistics:")
    print(f"  Mean: {np.mean(final_pr):.6f} (expected: {expected_pr:.3f})")
    print(f"  Std:  {np.std(final_pr):.6f}")
    print(f"  Max error: {np.max(pr_error):.6f}")
    print(f"  Convergence: {'PASSED ✓' if np.max(pr_error) < 0.01 else 'FAILED ✗'}")

    print(f"\nR Statistics:")
    print(f"  Mean: {np.mean(final_r):.6f} (expected: {expected_r:.3f})")
    print(f"  Std:  {np.std(final_r):.6f}")
    print(f"  Max error: {np.max(r_error):.6f}")
    print(f"  Convergence: {'PASSED ✓' if np.max(r_error) < 0.1 else 'FAILED ✗'}")

    print(f"\nBeta Statistics:")
    print(f"  Mean: {np.mean(final_beta):.6f} (expected: {expected_beta:.3f})")
    print(f"  Std:  {np.std(final_beta):.6f}")
    print(f"  Max error: {np.max(beta_error):.6f}")

    # Formula validation
    validation_result = val_utils.validate_reputation_formula(
        agents, state, actions, GAMMA, epsilon=0.5
    )
    val_utils.print_validation_summary(validation_result, "Reputation Formula")

    # Overall test result
    test_passed = (np.max(pr_error) < 0.01 and np.max(r_error) < 0.1)
    print(f"\n{'='*60}")
    print(f"TEST 1 RESULT: {'PASSED ✓' if test_passed else 'FAILED ✗'}")
    print(f"{'='*60}\n")

    # Generate plots
    print("Generating plots...")

    # Plot 1: Reputation convergence
    val_utils.plot_reputation_convergence(
        debug_log['reputation_traces'],
        results_dir,
        'reputation_convergence.png'
    )

    # Plot 2: PR vs R comparison
    val_utils.plot_pr_vs_r_comparison(
        debug_log['pr_traces'],
        debug_log['reputation_traces'],
        results_dir,
        'pr_vs_r_comparison.png'
    )

    # Plot 3: Formula validation
    val_utils.plot_formula_validation(
        validation_result['expected_R'],
        validation_result['actual_R'],
        results_dir,
        value_name='Reputation',
        filename='formula_validation.png'
    )

    # Plot 4: Beta convergence
    plt.figure(figsize=(12, 6))
    for i in range(NUM_AGENTS):
        plt.plot(debug_log['beta_traces'][i], label=f'Agent {i}', alpha=0.7)
    plt.axhline(expected_beta, color='r', linestyle='--', label='Expected beta')
    plt.xlabel('Timestep')
    plt.ylabel('Beta (Personal Utility Baseline)')
    plt.title('Beta Convergence Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'beta_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save debug log
    val_utils.save_debug_log(debug_log, results_dir)

    print(f"\nResults saved to: {results_dir}")
    print("Plots generated:")
    print("  - reputation_convergence.png")
    print("  - pr_vs_r_comparison.png")
    print("  - formula_validation.png")
    print("  - beta_convergence.png")
    print("  - debug_log.npz")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(34243)
    torch.manual_seed(34243)

    main()
