"""
Experiment 2: Reputation Scaling
Runs experiments for multiple gamma values and generates comparison graphs

This script:
1. Runs Experiment 2 for gamma values: 1, 1.25, 1.5, 1.75, 2, 3, 5
2. Runs both static and async modes
3. Generates comparison graphs showing "Maximum Followers for a Single Agent Over Time"
   with blips indicating top influencer switching
"""

import os
import sys
import random
import numpy as np
import torch
import gc
import matplotlib.pyplot as plt
import glob
import re

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'model'))

from train import training_loop

# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# Simulation Parameters
NUM_AGENTS = 100
MAX_STATE_NUM = 9
TIMESTEPS = 50000

# Random Seed (for reproducibility)
RANDOM_SEED = 5

# Gamma values to test (from the PDF images)
GAMMA_VALUES = [1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 5.0]

# Experiment 2 Parameters (from PDF)
KAPPA = 0.0  # No influencer benefits
THRESHOLD = 48
UPDATE_INTERVAL = 3000
LEARNING_RATE = 0.00005
BASE_B0 = 10
REWARD_TYPE = "unimodal_variance"
REWARD_BOTTOM = 0.1
REWARD_TOP = 1.0
EPSILON = 1.0  # Gossiping connection probability # IS THIS EVEN BEING USED?

# Advanced Parameters
K = 1.0
TH_FACTOR = 0.6
DISCOUNT = 0.99
ADJUSTMENT_FACTOR = 3.0


# ============================================================================
# EXPERIMENT RUNNING
# ============================================================================

def run_single_experiment(gamma, async_mode, seed_offset=0):
    """Run a single Experiment 2 run with given gamma and mode"""
    
    # Set random seeds for reproducibility (with offset for different runs)
    random.seed(RANDOM_SEED + seed_offset)
    np.random.seed(RANDOM_SEED + seed_offset)
    torch.manual_seed(RANDOM_SEED + seed_offset)
    
    # Determine sync label
    sync_label = "async" if async_mode else "static"
    
    # Create experiment name
    exp_name = f"exp2_reputation_scaling_{sync_label}_G{gamma:.2f}"
    
    print(f"\n{'='*60}")
    print(f"Running: {exp_name}")
    print(f"  Gamma: {gamma}")
    print(f"  Mode: {sync_label}")
    print(f"{'='*60}\n")
    
    # Build arguments dictionary
    args = {
        "status_factor": KAPPA,           # κ
        "reputation_factor": gamma,        # γ
        "b0": BASE_B0 * REWARD_TOP * 4,   # External utility
        "learning_rate": LEARNING_RATE,   # α
        "kappa": 1,
        "rho": 1,
        "cons": UPDATE_INTERVAL,
        "top_reward": REWARD_TOP,
        "bottom_reward": REWARD_BOTTOM,
        "threshold": THRESHOLD,
        "epsilon": EPSILON,  # Gossiping connection probability
        "follower_epsilon": 0.1,  # Epsilon for follower decision
        "reward_type": REWARD_TYPE
    }
    
    # Run training loop
    try:
        training_loop(
            num_agents=NUM_AGENTS,
            max_state_num=MAX_STATE_NUM,
            name=exp_name,
            arguments=args,
            k=K,
            th_factor=TH_FACTOR,
            discount=DISCOUNT,
            adjustment_factor=ADJUSTMENT_FACTOR,
            dynamic_change=async_mode,
            timesteps=TIMESTEPS
        )
        print(f"Completed: {exp_name}\n")
        return exp_name
    except Exception as e:
        print(f"Error in {exp_name}: {e}\n")
        return None


# ============================================================================
# DATA LOADING
# ============================================================================

def load_experiment_data(results_dir, experiment_name):
    """Load all data files from experiment results"""
    base_name = experiment_name + "_"
    root = os.path.join(results_dir, experiment_name)
    
    if not os.path.exists(root):
        print(f"Warning: Experiment directory not found: {root}")
        return None
    
    data = {}
    
    # Load timesteps
    timesteps_file = os.path.join(root, base_name + "update_timesteps.npy")
    if os.path.exists(timesteps_file):
        data['timesteps'] = np.load(timesteps_file)
    else:
        print(f"Warning: Could not find {timesteps_file}")
        return None
    
    # Load followers (key metric for Experiment 2)
    followers_file = os.path.join(root, base_name + "followers.npy")
    if os.path.exists(followers_file):
        data['followers'] = np.load(followers_file)
    
    # Extract gamma value from experiment name
    gamma_match = re.search(r'_G([0-9.]+)', experiment_name)
    if gamma_match:
        data['gamma'] = float(gamma_match.group(1))
    else:
        data['gamma'] = None
    
    # Extract mode (static/async) from experiment name
    if 'static' in experiment_name:
        data['mode'] = 'static'
    elif 'async' in experiment_name:
        data['mode'] = 'async'
    else:
        data['mode'] = 'unknown'
    
    return data, root


# ============================================================================
# GRAPH GENERATION
# ============================================================================

def compute_max_followers_over_time(followers_data, timesteps):
    """
    Compute maximum followers at each timestep and track when top influencer switches
    
    Returns:
        max_followers: list of maximum follower counts at each timestep
        switch_timesteps: list of timesteps where top influencer switched
        switch_values: list of follower counts at switch points
    """
    if len(followers_data) == 0 or len(followers_data[0]) == 0:
        return [], [], []
    
    num_agents = len(followers_data)
    num_timesteps = len(followers_data[0])
    
    max_followers = []
    switch_timesteps = []
    switch_values = []
    
    prev_top_agent = -1
    
    for t in range(num_timesteps):
        # Get follower counts for all agents at this timestep
        follower_counts = [followers_data[i][t] if len(followers_data[i]) > t else 0 
                          for i in range(num_agents)]
        
        # Find agent with maximum followers
        max_followers_at_t = max(follower_counts)
        top_agent_at_t = np.argmax(follower_counts)
        
        max_followers.append(max_followers_at_t)
        
        # Check if top influencer switched
        if prev_top_agent != -1 and top_agent_at_t != prev_top_agent:
            switch_timesteps.append(timesteps[t] if t < len(timesteps) else t * UPDATE_INTERVAL)
            switch_values.append(max_followers_at_t)
        
        prev_top_agent = top_agent_at_t
    
    return max_followers, switch_timesteps, switch_values


def plot_maximum_followers_comparison(all_experiments, output_dir):
    """
    Create "Maximum Followers for a Single Agent Over Time" graphs
    Matching the PDF images with blips for top influencer switching
    """
    # Separate by mode
    static_exps = {k: v for k, v in all_experiments.items() if v[0]['mode'] == 'static'}
    async_exps = {k: v for k, v in all_experiments.items() if v[0]['mode'] == 'async'}
    
    # Color map for different gamma values
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Plot static mode
    if static_exps:
        plt.figure(figsize=(12, 7))
        color_idx = 0
        
        for exp_name, (data, root) in sorted(static_exps.items(), key=lambda x: x[1][0].get('gamma', 999)):
            if 'followers' in data and data['gamma'] is not None:
                followers_data = data['followers']
                timesteps = data['timesteps']
                
                # Compute maximum followers over time and switch points
                max_followers, switch_timesteps, switch_values = compute_max_followers_over_time(
                    followers_data, timesteps
                )
                
                if len(max_followers) > 0:
                    # Plot the line
                    color = colors[color_idx % len(colors)]
                    plt.plot(timesteps[:len(max_followers)], max_followers, 
                            label=f'γ = {data["gamma"]:.2f}', 
                            linewidth=2, alpha=0.8, color=color)
                    
                    # Plot blips for influencer switching
                    if len(switch_timesteps) > 0:
                        plt.scatter(switch_timesteps, switch_values, 
                                  s=50, color=color, marker='o', 
                                  zorder=5, edgecolors='black', linewidths=0.5)
                    
                    color_idx += 1
        
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Max Number of Followers', fontsize=12)
        plt.title('Maximum Followers for a Single Agent Over Time\nSummary of experiments for γ for static', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, title='Gamma')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, TIMESTEPS)
        plt.ylim(0, NUM_AGENTS)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'exp2_max_followers_static.png'), dpi=300)
        plt.close()
        print(f"Created: exp2_max_followers_static.png")
    
    # Plot async mode
    if async_exps:
        plt.figure(figsize=(12, 7))
        color_idx = 0
        
        for exp_name, (data, root) in sorted(async_exps.items(), key=lambda x: x[1][0].get('gamma', 999)):
            if 'followers' in data and data['gamma'] is not None:
                followers_data = data['followers']
                timesteps = data['timesteps']
                
                # Compute maximum followers over time and switch points
                max_followers, switch_timesteps, switch_values = compute_max_followers_over_time(
                    followers_data, timesteps
                )
                
                if len(max_followers) > 0:
                    # Plot the line
                    color = colors[color_idx % len(colors)]
                    plt.plot(timesteps[:len(max_followers)], max_followers, 
                            label=f'γ = {data["gamma"]:.2f}', 
                            linewidth=2, alpha=0.8, color=color)
                    
                    # Plot blips for influencer switching
                    if len(switch_timesteps) > 0:
                        plt.scatter(switch_timesteps, switch_values, 
                                  s=50, color=color, marker='o', 
                                  zorder=5, edgecolors='black', linewidths=0.5)
                    
                    color_idx += 1
        
        plt.xlabel('Timestep', fontsize=12)
        plt.ylabel('Max Number of Followers', fontsize=12)
        plt.title('Maximum Followers for a Single Agent Over Time\nSummary of experiments for γ for async', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10, title='Gamma')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, TIMESTEPS)
        plt.ylim(0, NUM_AGENTS)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'exp2_max_followers_async.png'), dpi=300)
        plt.close()
        print(f"Created: exp2_max_followers_async.png")


def plot_gamma_sweep_summary(all_experiments, output_dir):
    """
    Create parameter sweep graph showing final follower counts vs gamma
    """
    # Collect data
    static_data = []
    async_data = []
    
    for exp_name, (data, root) in all_experiments.items():
        if 'followers' not in data or data.get('gamma') is None:
            continue
        
        followers_data = data['followers']
        if len(followers_data) > 0 and len(followers_data[0]) > 0:
            # Get maximum followers at end
            final_max = max(followers_data[i][-1] if len(followers_data[i]) > 0 else 0 
                          for i in range(len(followers_data)))
            
            if data['mode'] == 'static':
                static_data.append((data['gamma'], final_max))
            elif data['mode'] == 'async':
                async_data.append((data['gamma'], final_max))
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    if static_data:
        static_data.sort(key=lambda x: x[0])
        gammas_static = [x[0] for x in static_data]
        max_followers_static = [x[1] for x in static_data]
        plt.plot(gammas_static, max_followers_static, 'o-', label='Static', 
                linewidth=2, markersize=8, alpha=0.8)
    
    if async_data:
        async_data.sort(key=lambda x: x[0])
        gammas_async = [x[0] for x in async_data]
        max_followers_async = [x[1] for x in async_data]
        plt.plot(gammas_async, max_followers_async, 's-', label='Async', 
                linewidth=2, markersize=8, alpha=0.8)
    
    plt.xlabel('γ (Reputation Modifier)', fontsize=12)
    plt.ylabel('Maximum Followers at Convergence', fontsize=12)
    plt.title('Experiment 2: Maximum Followers vs γ\nReputation Scaling Parameter Sweep', 
             fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_gamma_sweep_summary.png'), dpi=300)
    plt.close()
    print(f"Created: exp2_gamma_sweep_summary.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run all Experiment 2 runs and generate graphs"""
    
    print("\n" + "="*60)
    print("Experiment 2: Reputation Scaling")
    print("="*60)
    print(f"Gamma values: {GAMMA_VALUES}")
    print(f"Timesteps per run: {TIMESTEPS}")
    print(f"Total runs: {len(GAMMA_VALUES) * 2} (static + async)")
    print("="*60 + "\n")
    
    results_dir = "Results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Run all experiments
    completed_experiments = []
    seed_offset = 0
    
    # Run static mode experiments
    print("\n" + "="*60)
    print("Running STATIC mode experiments...")
    print("="*60)
    for gamma in GAMMA_VALUES:
        exp_name = run_single_experiment(gamma, async_mode=False, seed_offset=seed_offset)
        if exp_name:
            completed_experiments.append(exp_name)
        seed_offset += 1
        gc.collect()
    
    # Run async mode experiments
    print("\n" + "="*60)
    print("Running ASYNC mode experiments...")
    print("="*60)
    for gamma in GAMMA_VALUES:
        exp_name = run_single_experiment(gamma, async_mode=True, seed_offset=seed_offset)
        if exp_name:
            completed_experiments.append(exp_name)
        seed_offset += 1
        gc.collect()
    
    # Generate comparison graphs
    print("\n" + "="*60)
    print("Generating comparison graphs...")
    print("="*60)
    
    # Load all experiment data
    all_experiments = {}
    for exp_name in completed_experiments:
        result = load_experiment_data(results_dir, exp_name)
        if result is not None:
            data, root = result
            all_experiments[exp_name] = (data, root)
            print(f"Loaded: {exp_name} (γ = {data.get('gamma', 'unknown')}, mode = {data.get('mode', 'unknown')})")
        else:
            print(f"Warning: Could not load {exp_name}")
    
    if not all_experiments:
        print("Error: No experiments loaded successfully")
        return
    
    # Create output directory
    output_dir = os.path.join(results_dir, "exp2_comparison_graphs")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nGenerating graphs...")
    print(f"Output directory: {output_dir}\n")
    
    # Generate graphs
    plot_maximum_followers_comparison(all_experiments, output_dir)
    plot_gamma_sweep_summary(all_experiments, output_dir)
    
    print("\n" + "="*60)
    print("Experiment 2 Complete!")
    print(f"Completed {len(completed_experiments)} experiments")
    print(f"Graphs saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()