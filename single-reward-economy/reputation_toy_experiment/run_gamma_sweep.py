"""
Gamma sweep: test4 vs paper approach, n=20 agents.
All reputation bugs fixed — results here are the corrected baseline.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from toy_experiment import run_experiment, plot_results

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "Results")


def run_gamma_sweep():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gammas     = [2, 3, 5, 10, 20]
    num_agents = 20
    timesteps  = 50000

    configs = [
        ('test4', [0.1]),
        ('paper', [0.003, 0.1]),
    ]

    all_results = {}

    print("=" * 70)
    print("GAMMA SWEEP  —  n=20, corrected reputation observation")
    print("=" * 70)

    for approach, learning_rates in configs:
        for lr in learning_rates:
            for gamma in gammas:
                print(f"\n{'='*50}")
                print(f"  approach={approach}  γ={gamma}  lr={lr}")
                print(f"{'='*50}")

                results = run_experiment(
                    num_agents=num_agents, gamma=gamma,
                    timesteps=timesteps, approach=approach,
                    learning_rate=lr, seed=42,
                )
                key = f"{approach}_gamma{gamma}_lr{lr}"
                all_results[key] = results

                print(f"  Max followers : {results['max_followers']}/{num_agents - 1}")
                print(f"  Stable        : {results['stable']}")
                print(f"  Stability     : {results['stability_score']:.2f}")
                print(f"  Convergence   : {results['convergence_time']}")

                plot_results(results, approach, gamma, lr, OUTPUT_DIR)

    _comparison_plots(all_results, gammas)
    _summary_table(all_results, gammas)
    print(f"\nAll results saved to {OUTPUT_DIR}")


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def _comparison_plots(all_results, gammas):
    series = [
        ('test4', 0.1,  'Test 4  (lr=0.1)',       'o-'),
        ('paper', 0.1,  'Paper   (lr=0.1)',        's-'),
        ('paper', 0.003,'Paper   (lr=0.003)',       '^-'),
    ]

    def _extract(metric, approach, lr):
        return [
            all_results.get(f"{approach}_gamma{g}_lr{lr}", {}).get(metric, 0)
            for g in gammas
        ]

    # Max followers vs gamma
    plt.figure(figsize=(10, 5))
    for approach, lr, label, style in series:
        plt.plot(gammas, _extract('max_followers', approach, lr),
                 style, label=label, linewidth=2, markersize=8)
    plt.axhline(y=15, color='red', linestyle='--', alpha=0.5, label='Target (75% of 20)')
    plt.xlabel('γ'); plt.ylabel('Final Max Followers')
    plt.title('Max Followers vs. γ  (n=20, corrected)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_max_followers.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Stability vs gamma
    plt.figure(figsize=(10, 5))
    for approach, lr, label, style in series:
        plt.plot(gammas, _extract('stability_score', approach, lr),
                 style, label=label, linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Stability threshold')
    plt.xlabel('γ'); plt.ylabel('Stability Score')
    plt.title('Leadership Stability vs. γ  (n=20, corrected)')
    plt.ylim(0, 1.05); plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_stability.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Convergence time vs gamma
    plt.figure(figsize=(10, 5))
    for approach, lr, label, style in series:
        raw = [
            all_results.get(f"{approach}_gamma{g}_lr{lr}", {}).get('convergence_time', -1)
            for g in gammas
        ]
        vals = [v if v > 0 else 50000 for v in raw]
        plt.plot(gammas, vals, style, label=label, linewidth=2, markersize=8)
    plt.xlabel('γ'); plt.ylabel('Convergence Time (steps)')
    plt.title('Time to Convergence vs. γ  (n=20, corrected)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _summary_table(all_results, gammas):
    threshold = 15  # 75% of 20

    lines = [
        "# Gamma Sweep — n=20 (corrected)",
        "",
        "| Approach | lr | γ | Max Followers | Stable? | Stability | Convergence |",
        "|----------|----|---|---------------|---------|-----------|-------------|",
    ]

    for approach, lrs in [('test4', [0.1]), ('paper', [0.003, 0.1])]:
        for lr in lrs:
            for gamma in gammas:
                key = f"{approach}_gamma{gamma}_lr{lr}"
                r   = all_results.get(key)
                if r is None:
                    continue
                stable  = "✅" if r['stable'] else "❌"
                conv    = str(r['convergence_time']) if r['convergence_time'] > 0 else "—"
                lines.append(
                    f"| {approach} | {lr} | {gamma} | {r['max_followers']} "
                    f"| {stable} | {r['stability_score']:.2f} | {conv} |"
                )

    lines += ["", "## Analysis", ""]

    for approach, lrs in [('test4', [0.1]), ('paper', [0.003, 0.1])]:
        for lr in lrs:
            wins = [g for g in gammas
                    if all_results.get(f"{approach}_gamma{g}_lr{lr}", {}).get('stable')
                    and all_results.get(f"{approach}_gamma{g}_lr{lr}", {}).get('max_followers', 0) >= threshold]
            label = f"**{approach} lr={lr}:**"
            if wins:
                lines.append(f"- {label} stable leadership at γ = {wins}")
            else:
                lines.append(f"- {label} no stable leadership in tested range")

    with open(os.path.join(OUTPUT_DIR, 'SUMMARY.md'), 'w') as f:
        f.write('\n'.join(lines))
    print("Summary written to Results/SUMMARY.md")


if __name__ == "__main__":
    run_gamma_sweep()
