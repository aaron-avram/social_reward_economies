"""
Run reputation_scaling experiment with PDF-matching parameters

This runs the SIMPLIFIED version (status optimization disabled, unrestricted gossip)
to match what we validated in Tests 1-4.
"""

import random
import numpy as np
import torch
import gc

# Import from local train.py
from train import training_loop

# Set random seed for reproducibility
RANDOM_SEED = 5
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("="*60)
print("Running reputation_scaling Experiment")
print("="*60)
print("\nConfiguration:")
print("  Implementation: SIMPLIFIED (status disabled, unrestricted gossip)")
print("  Agents: 100")
print("  States: 512 (max_state_num=9)")
print("  Timesteps: 50000")
print("  Gamma: 10")
print("  Kappa: 0.0 (status optimization DISABLED in code)")
print("  Threshold: 48")
print("  Reward function: variance(1, 0.1) [HARDCODED in train.py]")
print("  Gossip: Unrestricted (similarity filter DISABLED)")
print("  Random seed: 5")
print("="*60 + "\n")

# Configuration matching PDF experiments
NUM_AGENTS = 100
MAX_STATE_NUM = 9  # 2^9 = 512 states
TIMESTEPS = 50000
GAMMA = 10
KAPPA = 0.0  # Note: This is symbolic - status optimization is commented out
THRESHOLD = 48
CONS = 3000
B0_BASE = 10
EPSILON = 1.0

# Reward range (note: actual function is hardcoded in train.py as variance(1, 0.1))
TOP_REWARD = 1.0
BOTTOM_REWARD = 0.1

# Learning rate
LEARNING_RATE = 0.003

# Async mode
DYNAMIC_CHANGE = True
RHO = 1
KAPPA_TIME = 1

# Build args dictionary
args = {
    "status_factor": KAPPA,
    "reputation_factor": GAMMA,
    "b0": B0_BASE * TOP_REWARD * 4,  # 10 * 1.0 * 4 = 40
    "learning_rate": LEARNING_RATE,
    "kappa": KAPPA_TIME,
    "rho": RHO,
    "cons": CONS,
    "top_reward": TOP_REWARD,
    "bottom_reward": BOTTOM_REWARD,
    "threshold": THRESHOLD,
    "epsilon": EPSILON
}

# Experiment name
sync_label = "async" if DYNAMIC_CHANGE else "static"
exp_name = f"simplified_{sync_label}_G{GAMMA}_K{KAPPA}_T{TIMESTEPS}"

print(f"Starting experiment: {exp_name}\n")
print("NOTE: Despite args, the actual implementation uses:")
print("  - Hardcoded reward: generate_reward_table_variance(agn, 1, 0.1)")
print("  - Status optimization: DISABLED (become_selfless commented out)")
print("  - Gossip: UNRESTRICTED (similarity filter commented out)")
print("\n" + "="*60 + "\n")

# Run training loop
training_loop(
    NUM_AGENTS,
    MAX_STATE_NUM,
    exp_name,
    arguments=args,
    k=0,
    th_factor=0.6,
    adjustment_factor=3,
    dynamic_change=DYNAMIC_CHANGE,
    timesteps=TIMESTEPS
)

print("\n" + "="*60)
print("Experiment Complete!")
print("="*60)
print(f"\nResults saved to: Results/{exp_name}/")
print("\nKey plots to check:")
print(f"  - Results/{exp_name}/Followers.png")
print(f"  - Results/{exp_name}/Reputations.png")
print(f"  - Results/{exp_name}/Social_Welfare.png")
print("\nCompare these with PDF figures!")
print("="*60 + "\n")

gc.collect()
