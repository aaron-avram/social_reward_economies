"""
Scaling experiment: paper approach, n=100 agents.
Inherits all corrected logic from toy_experiment.py.
Sweeps gamma = [3, 5, 10, 20] at lr = [0.003, 0.1].

Diagnostic output is included to expose the reputation signal
strength (γ·R[j] vs independent_beta) so failures can be
attributed to specific mechanisms rather than guessed at.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from toy_experiment import run_experiment, plot_results, gossip, get_actions, update_follower_counts
from toy_env import ToyEnv
from toy_agent import ToyAgent

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Results_scale")


def run_scale_sweep():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gammas     = [3, 5, 10, 20]
    num_agents = 100
    timesteps  = 50000
    approach   = 'paper'

    all_results = {}

    print("=" * 70)
    print("SCALE EXPERIMENT  —  paper approach, n=100")
    print("=" * 70)

    for lr in [0.003]:
        for gamma in gammas:
            print(f"\n{'='*50}")
            print(f"  γ={gamma}  lr={lr}")
            print(f"{'='*50}")

            results = run_experiment_with_diagnostics(
                num_agents=num_agents, gamma=gamma,
                timesteps=timesteps, approach=approach,
                learning_rate=lr, seed=42,
            )
            key = f"paper_gamma{gamma}_lr{lr}"
            all_results[key] = results

            print(f"  Max followers : {results['max_followers']}/{num_agents - 1}")
            print(f"  Stable        : {results['stable']}")
            print(f"  Stability     : {results['stability_score']:.2f}")
            print(f"  Convergence   : {results['convergence_time']}")

            plot_results(results, approach, gamma, lr, OUTPUT_DIR)
            _plot_signal_ratio(results, gamma, lr)

    _comparison_plots(all_results, gammas)
    _summary_table(all_results, gammas, num_agents)
    print(f"\nAll results saved to {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Instrumented experiment — adds signal diagnostics
# ---------------------------------------------------------------------------

def run_experiment_with_diagnostics(num_agents=100, gamma=10, timesteps=50000,
                                     approach='paper', learning_rate=0.003, seed=42):
    """
    Identical logic to toy_experiment.run_experiment but also records:
      - mean γ·R[best_agent] across agents  (numerator of follow decision)
      - mean independent_beta               (denominator of follow decision)
      - ratio = signal / baseline           (> 1 means some agents should follow)
    These are sampled at each switch interval to diagnose why following does or
    doesn't happen.
    """
    np.random.seed(seed)

    env = ToyEnv()
    agents = [
        ToyAgent(i, num_agents,
                 env.generate_state_dependent_rewards(i),
                 gamma, approach, learning_rate)
        for i in range(num_agents)
    ]

    max_followers        = []
    leader_ids           = []
    reputations          = [[] for _ in range(num_agents)]
    personal_benefits    = [[] for _ in range(num_agents)]
    num_followers_list   = []
    num_influencers_list = []
    num_independents_list = []

    # Diagnostic tracking (recorded at each switch event)
    diag_timesteps   = []
    diag_signal      = []   # mean max γ·R[j] seen by each agent
    diag_baseline    = []   # mean independent_beta across agents
    diag_ratio       = []   # signal / baseline

    for t in range(timesteps):
        state   = env.sample_state()
        actions = get_actions(agents, state)

        # Cross-observation reputation update (Issue 1 fix)
        for j in range(num_agents):
            for i in range(num_agents):
                agents[j].update_reputation(i, actions, state, agents)

        if t >= 50:
            gossip(agents)

        # Independent beta + policy only when acting freely (Issue 2 fix)
        for i in range(num_agents):
            reward = agents[i].reward_function[state][actions[i]]
            if not agents[i].is_follower:
                agents[i].update_independent_beta(reward)
                agents[i].update_policy(state, actions[i], reward)

        # Influencer switching with Issues 3 & 4 fixes (inside switch_influencer)
        if t % 3000 == 0 and t > 50:
            update_follower_counts(agents)
            for agent in agents:
                agent.switch_influencer(agents)
            update_follower_counts(agents)

            # --- Diagnostics at switch time ---
            scaled_R = gamma * agents[0].R   # post-gossip: all agents share R
            max_signal   = np.max(scaled_R)
            mean_baseline = np.mean([a.independent_beta for a in agents])
            diag_timesteps.append(t)
            diag_signal.append(max_signal)
            diag_baseline.append(mean_baseline)
            diag_ratio.append(
                max_signal / mean_baseline if mean_baseline > 1e-9 else 0.0
            )
            print(f"    [t={t:>6}]  γ·R_max={max_signal:.4f}  "
                  f"β_mean={mean_baseline:.4f}  "
                  f"ratio={diag_ratio[-1]:.2f}")

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
                reputations[i].append(agents[0].R[i])
                personal_benefits[i].append(agent.P[i] if approach == 'paper' else 0)

            num_followers    = sum(1 for a in agents if a.is_follower)
            num_influencers  = sum(1 for a in agents if a.followers > 0)
            num_independents = num_agents - num_followers - num_influencers
            num_followers_list.append(num_followers)
            num_influencers_list.append(num_influencers)
            num_independents_list.append(num_independents)

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
        'max_followers':           max_followers[-1] if max_followers else 0,
        'stable':                  stability > 0.8,
        'stability_score':         stability,
        'convergence_time':        convergence_time,
        'leader_id':               top_leader,
        'max_followers_over_time': max_followers,
        'leader_ids_over_time':    leader_ids,
        'reputations':             reputations,
        'personal_benefits':       personal_benefits,
        'num_followers':           num_followers_list,
        'num_influencers':         num_influencers_list,
        'num_independents':        num_independents_list,
        'timesteps':               list(range(0, timesteps, 100)),
        # diagnostics
        'diag_timesteps':  diag_timesteps,
        'diag_signal':     diag_signal,
        'diag_baseline':   diag_baseline,
        'diag_ratio':      diag_ratio,
    }


# ---------------------------------------------------------------------------
# Signal-ratio diagnostic plot
# ---------------------------------------------------------------------------

def _plot_signal_ratio(results, gamma, lr):
    if not results['diag_timesteps']:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(results['diag_timesteps'], results['diag_signal'],
                 label='γ · R[best]', linewidth=2)
    axes[0].plot(results['diag_timesteps'], results['diag_baseline'],
                 label='mean β (independent)', linewidth=2, linestyle='--')
    axes[0].set_ylabel('Value')
    axes[0].set_title(f'Signal vs Baseline  (paper, γ={gamma}, lr={lr}, n=100)')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['diag_timesteps'], results['diag_ratio'],
                 color='green', linewidth=2)
    axes[1].axhline(y=1.0, color='red', linestyle='--', alpha=0.6,
                    label='ratio = 1  (signal = baseline)')
    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('Signal / Baseline')
    axes[1].set_title('Signal-to-Baseline Ratio (>1 required for following)')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'signal_ratio_paper_gamma{gamma}_lr{lr}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def _comparison_plots(all_results, gammas):
    series = [
        (0.003, 'lr=0.003', '^-'),
    ]

    def _extract(metric, lr):
        return [
            all_results.get(f"paper_gamma{g}_lr{lr}", {}).get(metric, 0)
            for g in gammas
        ]

    plt.figure(figsize=(10, 5))
    for lr, label, style in series:
        plt.plot(gammas, _extract('max_followers', lr),
                 style, label=f'Paper {label}', linewidth=2, markersize=8)
    plt.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='Target (75 of 100)')
    plt.xlabel('γ'); plt.ylabel('Final Max Followers')
    plt.title('Max Followers vs. γ  (paper, n=100)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_max_followers.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Max followers over time — one line per gamma, lr=0.003 only
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for lr in [0.003]:
        plt.figure(figsize=(12, 5))
        for gamma, color in zip(gammas, colors):
            r = all_results.get(f"paper_gamma{gamma}_lr{lr}")
            if r is None:
                continue
            ts  = r['timesteps']
            mft = r['max_followers_over_time']
            plt.plot(ts, mft, label=f'γ={gamma}', color=color, linewidth=1.5)
        plt.axhline(y=99, color='black', linestyle='--', linewidth=0.8, label='Max possible (99)')
        plt.xlabel('Timestep')
        plt.ylabel('Max Followers')
        plt.title(f'Max Followers Over Time by γ  (paper, lr={lr}, n=100)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f'max_followers_over_time_lr{lr}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 5))
    for lr, label, style in series:
        plt.plot(gammas, _extract('stability_score', lr),
                 style, label=f'Paper {label}', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('γ'); plt.ylabel('Stability Score')
    plt.title('Leadership Stability vs. γ  (paper, n=100)')
    plt.ylim(0, 1.05); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_stability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _summary_table(all_results, gammas, num_agents):
    threshold = int(num_agents * 0.75)

    lines = [
        f"# Scale Experiment — paper approach, n={num_agents}",
        "",
        "| lr | γ | Max Followers | Stable? | Stability | Convergence |",
        "|----|---|---------------|---------|-----------|-------------|",
    ]
    for lr in [0.003]:
        for gamma in gammas:
            r = all_results.get(f"paper_gamma{gamma}_lr{lr}")
            if r is None:
                continue
            stable = "✅" if r['stable'] else "❌"
            conv   = str(r['convergence_time']) if r['convergence_time'] > 0 else "—"
            lines.append(
                f"| {lr} | {gamma} | {r['max_followers']} "
                f"| {stable} | {r['stability_score']:.2f} | {conv} |"
            )

    lines += ["", "## Analysis", ""]
    for lr in [0.003]:
        wins = [g for g in gammas
                if all_results.get(f"paper_gamma{g}_lr{lr}", {}).get('stable')
                and all_results.get(f"paper_gamma{g}_lr{lr}", {}).get('max_followers', 0) >= threshold]
        if wins:
            lines.append(f"- **lr={lr}:** stable leadership at γ = {wins}")
        else:
            lines.append(f"- **lr={lr}:** no stable leadership — check signal_ratio plots")

    with open(os.path.join(OUTPUT_DIR, 'SUMMARY.md'), 'w') as f:
        f.write('\n'.join(lines))
    print("Summary written to Results_scale/SUMMARY.md")


if __name__ == "__main__":
    run_scale_sweep()
