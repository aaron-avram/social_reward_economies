# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project on **Social Reward Economies** - multi-agent reinforcement learning simulations that model how agents learn behavioral norms through social feedback, reputation, and influence dynamics. The project investigates emergent phenomena like cohesion, polarization, resistance, and manipulation in agent societies.

The codebase extends prior work from https://github.com/DeathByThermodynamics/orchard-action-market

## Repository Structure

- **`single-reward-economy/`**: Main experimental directory
  - `model/`: Core implementation with standard configuration
  - `personal_utility/`: Experiments focusing on personal utility dynamics
  - `reputation_scaling/`: Active experiment directory for reputation scaling (γ) sweeps; `run_experiment.py` is the batch runner
  - `reputation_toy_experiment/`: Minimal isolation experiment for reputation scaling; `toy_experiment.py` (n=20) and `scale_experiment.py` (n=100) are the entry points; results in `Results/` and `Results_scale/`
- **`gossip_test/`**: Experiments testing gossip/reputation-sharing mechanisms
- **`resistance-and-polarization/`**: Placeholder for polarization research
- **`subcultures/`**: Placeholder for subculture formation research
- **`doc/`**: Research papers and transcriptions; `doc/report` (no extension) is the LaTeX report file

## Core Architecture

### Key Components

**`norm.py`** - Environment and reward generation
- `NormEnv`: Manages state space and action space
- Multiple reward function generators: `generate_reward_table_variance()`, `generate_reward_table_polarized_variance()`, `generate_reward_table_procedural()`, etc.
- Each agent gets a different reward function (heterogeneous preferences)

**`norm_agent.py`** - Agent behavior and dynamics
- `NormAgent`: Tracks reputation (`R`), personal reputation estimates (`PR`), similarity scores (`S`)
- Implements influencer/follower dynamics and role switching
- Two optimization modes: personal utility (selfish) vs. status maximization (selfless)
- Rate adjustment based on reputation and follower count
- Gossip-based reputation sharing with similarity-based filtering

**`norm_actor.py`** - Neural network policies
- `ActorNetwork`: Policy network using policy gradient (advantage = reward - beta baseline)
- `SimpleConnectedMultiple`: 3-layer MLP with LeakyReLU activations
- Returns action probability distributions via softmax

**`train.py`** - Main training loop
- Initializes agents with heterogeneous reward functions
- Each timestep: sample state, get actions (with influencer copying), compute feedback, train networks, gossip reputations
- Tracks extensive metrics: roles (influencers/followers/actors), reputations, utilities, policy convergence
- Generates comprehensive plots and saves results to `Results/[experiment_name]/`

### Agent Dynamics

**Roles**: Agents can be influencers (have followers), followers (copy an influencer), or independent actors.

**Reputation System**:
- `PR[i]`: Agent's personal estimate of agent i's reputation
- `R[i]`: Averaged reputation of agent i (updated via gossip)
- `S[i]`: Similarity score with agent i (based on reward alignment)

**Influencer Selection**: The follow condition in the paper is `γ·R_{i,j*} > P_{i,i}` (scaled reputation of candidate beats agent's own personal evaluation of itself).
- `toy_agent.py` implements this correctly using `self.P[self.id]` as the threshold.
- `norm_agent.py` uses `independent_beta` instead — a separate EMA of own received reward that is **frozen when the agent is a follower**. This is a known deviation from the paper formula.

**Status Optimization**: Agents with followers above a threshold switch from optimizing personal utility to optimizing for social feedback (status). Controlled by `status_factor` (κ). Setting κ=0 disables status optimization.

## Common Development Commands

### Setup
```bash
# Install dependencies
pip install numpy torch matplotlib
```

### Running Experiments

Main training script:
```bash
cd single-reward-economy/model
python train.py
```

Reputation scaling (active):
```bash
cd single-reward-economy/reputation_scaling
python train.py              # single run with current args
python run_experiment.py     # batch γ sweep
```

Toy/scale isolation experiments:
```bash
cd single-reward-economy/reputation_toy_experiment
python toy_experiment.py     # n=20 γ sweep, results → Results/
python scale_experiment.py   # n=100 γ sweep, results → Results_scale/
```

Personal utility experiments:
```bash
cd single-reward-economy/personal_utility
python train_test_1.py  # or train_test_2.py, train_test_3.py, train_test_4.py
```

Gossip mechanism tests:
```bash
cd gossip_test
python train.py         # Main gossip experiment
python train_test_1.py  # or train_test_2.py
```

### Key Configuration Parameters

Located in the `args` dictionary at the bottom of training files:

- `status_factor` (κ): Weight on status rewards; set to 0 to disable status optimization
- `reputation_factor` (γ): Scaling factor applied to reputation at comparison time
- `b0`: Outside utility parameter (affects participation rate)
- `learning_rate`: Neural network learning rate (typically 0.00005)
- `cons`: Update interval for influencer switching and rate adjustment (typically 3000)
- `threshold`: Follower count needed to switch to status optimization
- `top_reward`, `bottom_reward`: Max/min rewards for variance-based reward functions (or means for bimodal)
- `reward_type`: One of "unimodal_variance", "polarized_variance", "procedural", "cmstyle", "variance_separation"
- `rho`, `kappa` (in `args`): Temporal scaling factors for asynchronous updates (distinct from the κ/`status_factor` parameter)

**`max_state_num` gotcha**: In `train.py` this is the maximum state *index* (states are 0..N, so 10 states if `max_state_num=9`). In `run_experiment.py` it is treated as a bit-width (2^N states, so 512 states if `max_state_num=9`). The two files use the same variable name with different semantics.

### Output and Analysis

Results are saved to `Results/[experiment_name]/`:
- PNG plots: reputations, roles, social welfare, policy convergence, follower dynamics
- NPY arrays: raw data for all tracked metrics
- `parts/` subdirectory: per-agent detailed breakdowns
- `SUMMARY.md`: auto-generated table of γ sweep results
- `ANALYSIS.md`: manually maintained analysis and PDF claim assessment (in `reputation_toy_experiment/Results/` and `Results_scale/`)

## Important Implementation Details

- **Device**: Auto-selects CUDA if available, otherwise CPU
- **Gossiping**: Only occurs between agents with similarity score < 3 (after timestep 50)
- **Training**: Agents only train their personal policy when not following anyone; influencers train status policy when selfless
- **Beta updates**: Running estimates of expected rewards, used as baselines for advantage calculation
- **Follower copying**: Followers copy their influencer's actions, not their own policy
- **Random exploration**: 5% epsilon-greedy exploration for all agents

## Analysis Utilities

`analyze_debug_logs.py` - Utility script for analyzing debug output from training runs

## Working with PyTorch Models

- Models use `torch.float64` as default dtype
- Xavier initialization for all layer weights
- AdamW optimizer with AMSGrad
- Loss for actor networks: `-log_prob * advantage` (policy gradient with advantage)
