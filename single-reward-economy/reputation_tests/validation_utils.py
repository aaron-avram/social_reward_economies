"""
Validation utilities for reputation scaling experiments.

This module provides functions to:
1. Validate reputation formulas against mathematical definitions
2. Check gossip convergence
3. Compute policy distances
4. Calculate expected utilities
5. Generate diagnostic plots
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Tuple, Optional
import os


def validate_reputation_formula(agents, state: int, actions: List[int], gamma: float,
                                 epsilon: float = 0.1) -> Dict:
    """
    Verify R_i = γ Σ_{j≠i} U_j(π_i)

    For each agent i:
    - Calculate expected R_i based on other agents' utilities when taking i's action
    - Compare with actual R values stored in agents

    Args:
        agents: List of NormAgent instances
        state: Current state
        actions: List of actions taken by each agent
        gamma: Reputation scaling factor
        epsilon: Tolerance for error

    Returns:
        Dict with 'expected_R', 'actual_R', 'errors', 'passed'
    """
    num_agents = len(agents)
    expected_R = np.zeros(num_agents)
    actual_R = np.zeros(num_agents)

    for i in range(num_agents):
        # Calculate R_i = γ Σ_{j≠i} U_j(π_i)
        # This is the social feedback: how much utility does i's action give to others?
        total_utility_to_others = 0.0

        for j in range(num_agents):
            if j != i:
                # What utility does agent j get when agent i takes action actions[i]?
                utility_j = agents[j].reward_function[state][actions[i]]
                total_utility_to_others += utility_j

        expected_R[i] = gamma * total_utility_to_others
        actual_R[i] = agents[i].R[i]  # Agent i's own reputation

    errors = np.abs(expected_R - actual_R)
    passed = np.all(errors < epsilon)

    return {
        'expected_R': expected_R,
        'actual_R': actual_R,
        'errors': errors,
        'max_error': np.max(errors),
        'mean_error': np.mean(errors),
        'passed': passed
    }


def validate_gossip_convergence(agents, tolerance: float = 0.05) -> Dict:
    """
    Check if reputation estimates (R) have converged across all agents.

    After gossip, all agents should have similar estimates of each agent's reputation.
    For each agent j, check if all agents i have similar R_i[j] values.

    Args:
        agents: List of NormAgent instances
        tolerance: Maximum allowed variance in reputation estimates

    Returns:
        Dict with 'converged', 'max_divergence', 'variance_matrix', 'mean_variance'
    """
    num_agents = len(agents)

    # Matrix where reputation_matrix[i][j] = agent i's estimate of agent j's reputation
    reputation_matrix = np.zeros((num_agents, num_agents))

    for i in range(num_agents):
        for j in range(num_agents):
            reputation_matrix[i][j] = agents[i].R[j]

    # For each agent j, calculate variance in estimates across all agents i
    variances = np.var(reputation_matrix, axis=0)  # Variance across estimators (rows)
    max_divergence = np.max(np.ptp(reputation_matrix, axis=0))  # Max range across estimators

    converged = np.all(variances < tolerance)

    return {
        'converged': converged,
        'max_divergence': max_divergence,
        'variances': variances,
        'mean_variance': np.mean(variances),
        'reputation_matrix': reputation_matrix
    }


def compute_policy_distance(policy1: torch.Tensor, policy2: torch.Tensor,
                            metric: str = 'l1') -> float:
    """
    Compute distance between two policy distributions.

    Args:
        policy1: First policy distribution (action probabilities)
        policy2: Second policy distribution (action probabilities)
        metric: Distance metric ('l1', 'kl', 'expected_action')

    Returns:
        Distance value
    """
    if metric == 'l1':
        return torch.sum(torch.abs(policy1 - policy2)).item()

    elif metric == 'kl':
        # KL divergence: D_KL(p1 || p2) = Σ p1(a) log(p1(a)/p2(a))
        eps = 1e-10
        kl = torch.sum(policy1 * torch.log((policy1 + eps) / (policy2 + eps)))
        return kl.item()

    elif metric == 'expected_action':
        # Expected difference in action selection
        actions = torch.arange(len(policy1), dtype=torch.float64)
        exp_a1 = torch.sum(policy1 * actions)
        exp_a2 = torch.sum(policy2 * actions)
        return torch.abs(exp_a1 - exp_a2).item()

    else:
        raise ValueError(f"Unknown metric: {metric}")


def compute_expected_utility(agent, policy_net, state_probs: np.ndarray,
                             max_states: int, device) -> float:
    """
    Calculate E_π[U_i] = Σ_s p(s) Σ_a π(a|s) u(s,a)

    This should match the agent's beta value (running average of utility).

    Args:
        agent: NormAgent instance
        policy_net: ActorNetwork instance
        state_probs: Probability distribution over states
        max_states: Number of states
        device: torch device

    Returns:
        Expected utility under current policy
    """
    expected_utility = 0.0

    for state in range(max_states):
        state_tensor = torch.tensor([state], dtype=torch.int64).to(device)
        policy = policy_net.get_action_probabilities(state_tensor)

        # E_a[u(s,a)] = Σ_a π(a|s) u(s,a)
        state_utility = 0.0
        for action in range(len(policy)):
            state_utility += policy[action].item() * agent.reward_function[state][action]

        expected_utility += state_probs[state] * state_utility

    return expected_utility


def plot_reputation_convergence(reputation_traces: Dict, output_dir: str,
                                filename: str = 'reputation_convergence.png'):
    """
    Plot reputation values over time for all agents.

    Args:
        reputation_traces: Dict[agent_id] -> List of R values over time
        output_dir: Directory to save plot
        filename: Filename for plot
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    for agent_id, trace in reputation_traces.items():
        plt.plot(trace, label=f'Agent {agent_id}', alpha=0.7)

    plt.xlabel('Timestep')
    plt.ylabel('Reputation (R)')
    plt.title('Reputation Convergence Over Time')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_pr_vs_r_comparison(pr_traces: Dict, r_traces: Dict, output_dir: str,
                            filename: str = 'pr_vs_r_comparison.png'):
    """
    Compare Personal Reputation (PR) vs Reputation (R) estimates.

    Args:
        pr_traces: Dict[agent_id] -> List of PR values over time
        r_traces: Dict[agent_id] -> List of R values over time
        output_dir: Directory to save plot
        filename: Filename for plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot PR values
    for agent_id, trace in pr_traces.items():
        axes[0].plot(trace, label=f'Agent {agent_id}', alpha=0.7)
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Personal Reputation (PR)')
    axes[0].set_title('PR Over Time')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot R values
    for agent_id, trace in r_traces.items():
        axes[1].plot(trace, label=f'Agent {agent_id}', alpha=0.7)
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Reputation (R)')
    axes[1].set_title('R Over Time')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_policy_evolution(policy_snapshots: List[Dict], output_dir: str,
                         filename: str = 'policy_evolution.png'):
    """
    Plot how policy probabilities evolve over time.

    Args:
        policy_snapshots: List of dicts with 'timestep', 'agent_id', 'state', 'policy'
        output_dir: Directory to save plot
        filename: Filename for plot
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group by agent and state
    from collections import defaultdict
    policy_by_agent_state = defaultdict(lambda: defaultdict(list))

    for snapshot in policy_snapshots:
        agent_id = snapshot['agent_id']
        state = snapshot['state']
        timestep = snapshot['timestep']
        policy = snapshot['policy']

        policy_by_agent_state[agent_id][state].append({
            'timestep': timestep,
            'policy': policy
        })

    # Create subplot for each agent
    num_agents = len(policy_by_agent_state)
    fig, axes = plt.subplots(num_agents, 1, figsize=(12, 4 * num_agents))

    if num_agents == 1:
        axes = [axes]

    for idx, (agent_id, states_data) in enumerate(policy_by_agent_state.items()):
        for state, snapshots in states_data.items():
            timesteps = [s['timestep'] for s in snapshots]

            # Plot each action's probability
            num_actions = len(snapshots[0]['policy'])
            for action in range(num_actions):
                probs = [s['policy'][action] for s in snapshots]
                axes[idx].plot(timesteps, probs, label=f'State {state}, Action {action}',
                             marker='o', alpha=0.7)

        axes[idx].set_xlabel('Timestep')
        axes[idx].set_ylabel('Action Probability')
        axes[idx].set_title(f'Agent {agent_id} Policy Evolution')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_gossip_effect(reputation_matrix_before: np.ndarray,
                      reputation_matrix_after: np.ndarray,
                      output_dir: str, filename: str = 'gossip_effect.png'):
    """
    Visualize how gossip changes reputation estimates across agents.

    Args:
        reputation_matrix_before: R estimates before gossip [agent_i][agent_j]
        reputation_matrix_after: R estimates after gossip [agent_i][agent_j]
        output_dir: Directory to save plot
        filename: Filename for plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Before gossip
    im0 = axes[0].imshow(reputation_matrix_before, cmap='viridis', aspect='auto')
    axes[0].set_xlabel('Agent j (being estimated)')
    axes[0].set_ylabel('Agent i (estimator)')
    axes[0].set_title('Reputation Estimates Before Gossip')
    plt.colorbar(im0, ax=axes[0])

    # After gossip
    im1 = axes[1].imshow(reputation_matrix_after, cmap='viridis', aspect='auto')
    axes[1].set_xlabel('Agent j (being estimated)')
    axes[1].set_ylabel('Agent i (estimator)')
    axes[1].set_title('Reputation Estimates After Gossip')
    plt.colorbar(im1, ax=axes[1])

    # Difference (change due to gossip)
    diff = reputation_matrix_after - reputation_matrix_before
    im2 = axes[2].imshow(diff, cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(diff)),
                        vmax=np.max(np.abs(diff)))
    axes[2].set_xlabel('Agent j (being estimated)')
    axes[2].set_ylabel('Agent i (estimator)')
    axes[2].set_title('Change Due to Gossip')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def plot_formula_validation(expected_values: np.ndarray, actual_values: np.ndarray,
                           output_dir: str, value_name: str = 'Reputation',
                           filename: str = 'formula_validation.png'):
    """
    Plot expected vs actual values to validate formula implementation.

    Args:
        expected_values: Theoretically expected values
        actual_values: Actually observed values
        output_dir: Directory to save plot
        value_name: Name of the value being validated
        filename: Filename for plot
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: expected vs actual
    axes[0].scatter(expected_values, actual_values, alpha=0.6)
    axes[0].plot([expected_values.min(), expected_values.max()],
                [expected_values.min(), expected_values.max()],
                'r--', label='Perfect match')
    axes[0].set_xlabel(f'Expected {value_name}')
    axes[0].set_ylabel(f'Actual {value_name}')
    axes[0].set_title(f'{value_name}: Expected vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = actual_values - expected_values
    axes[1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--', label='Zero error')
    axes[1].set_xlabel('Error (Actual - Expected)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{value_name} Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


def create_debug_log() -> Dict:
    """
    Create a comprehensive debug log structure for tracking experiments.

    Returns:
        Dict with keys for different trace types
    """
    return {
        'reputation_traces': {},      # agent_id -> [R values over time]
        'pr_traces': {},               # agent_id -> [PR values over time]
        'beta_traces': {},             # agent_id -> [beta values over time]
        'policy_snapshots': [],        # List of {timestep, agent_id, state, policy}
        'gossip_events': [],           # List of {timestep, agents_involved, convergence_metric}
        'follower_events': [],         # List of {timestep, agent_id, influencer_id, action}
        'formula_validations': [],     # List of {timestep, validation_results}
        'action_distributions': {},    # timestep -> {agent_id -> action_counts}
        'utility_traces': {},          # agent_id -> [utility values over time]
    }


def save_debug_log(debug_log: Dict, output_dir: str, filename: str = 'debug_log.npz'):
    """
    Save debug log to disk for later analysis.

    Args:
        debug_log: Debug log dictionary
        output_dir: Directory to save to
        filename: Filename for saved log
    """
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, filename), **debug_log)


def load_debug_log(filepath: str) -> Dict:
    """
    Load a saved debug log.

    Args:
        filepath: Path to saved debug log

    Returns:
        Debug log dictionary
    """
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key].item() if data[key].shape == () else data[key]
            for key in data.keys()}


def print_validation_summary(validation_results: Dict, test_name: str = "Test"):
    """
    Print a formatted summary of validation results.

    Args:
        validation_results: Results from validation functions
        test_name: Name of the test being validated
    """
    print(f"\n{'='*60}")
    print(f"{test_name} Validation Summary")
    print(f"{'='*60}")

    if 'passed' in validation_results:
        status = "PASSED ✓" if validation_results['passed'] else "FAILED ✗"
        print(f"Status: {status}")

    if 'max_error' in validation_results:
        print(f"Max Error: {validation_results['max_error']:.6f}")
        print(f"Mean Error: {validation_results['mean_error']:.6f}")

    if 'converged' in validation_results:
        status = "CONVERGED ✓" if validation_results['converged'] else "NOT CONVERGED ✗"
        print(f"Convergence: {status}")
        print(f"Max Divergence: {validation_results['max_divergence']:.6f}")
        print(f"Mean Variance: {validation_results['mean_variance']:.6f}")

    print(f"{'='*60}\n")
