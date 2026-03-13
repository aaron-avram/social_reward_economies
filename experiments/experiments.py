"""
Four experiments (A–D) using Peter's MultiAgentSystem after bug fixes.

Run from the project root:
    python3 experiments/experiments.py

Experiments:
  A: γ=0, κ=0 — Pure personal utility. Expect: no leader, all independent.
  B: γ>0, κ=0 — Reputation only. Expect: opinion leader may emerge (Proposition 1).
  C: γ>0, κ>0 — Full algorithm. Expect: welfare-optimal common norm (Propositions 2&3).
  D: Perturbation test. Expect: leader loses followers after perturbation.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make project root importable, then import debugged implementation from src/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.code_debugged import SystemConfig, MultiAgentSystem, AgentRole
try:
    from experiments.perturbation_recovery import run_experiment as run_perturbation_recovery
except ImportError:
    # Supports direct execution: python3 experiments/experiments.py
    from perturbation_recovery import run_experiment as run_perturbation_recovery

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Shared helper ───────────────────────────────────────────────────────────

def make_config(**kwargs) -> SystemConfig:
    """Return a SystemConfig with sensible defaults, overriding with kwargs."""
    defaults = dict(
        num_agents=8,
        num_states=3,
        num_actions=2,
        num_time_steps=5000,
        M=1.0,
        u_0=0.1,
        gamma=2.5,
        kappa=2.0,
        c_threshold=0.1,
        B_R=0.3,   # Lowered from 0.8: payoffs are in [0,1] with mean ~0.5
        B_F=0.15,  # Lowered from 0.6: must be < B_R
        delta=0.15,
        alpha_pu_base=0.05,
        beta_status_base=0.05,
        eta_v_base=0.1,
        eta_s_base=0.1,
        eta_J_base=0.05,
        role_update_base_interval=100,
        gossip_rate=0.5,
        gossip_alpha=0.5,
    )
    defaults.update(kwargs)
    return SystemConfig(**defaults)


def summarize(results: dict, label: str):
    """Print a brief summary of simulation results."""
    final_roles = results["final_roles"]
    final_followers = results["final_followers"]
    leader = results["opinion_leader"]

    n_pu = sum(1 for r in final_roles if r == AgentRole.PERSONAL_UTILITY)
    n_rep = sum(1 for r in final_roles if r == AgentRole.REPUTATION)
    n_st = sum(1 for r in final_roles if r == AgentRole.STATUS)

    welfare = np.mean(results["social_welfare"][-500:])  # last 500 steps

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  PU: {n_pu}  |  Reputation: {n_rep}  |  Status: {n_st}")
    if leader >= 0:
        print(f"  Opinion leader: Agent {leader}  ({final_followers[leader]} followers)")
    else:
        print(f"  No opinion leader emerged.")
    print(f"  Avg social welfare (last 500 steps): {welfare:.4f}")


def plot_follower_timeseries(results: dict, title: str, filename: str):
    """Save a plot of follower counts over time."""
    followers_array = np.array(results["follower_counts"])
    n_agents = followers_array.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for i in range(n_agents):
        ax.plot(followers_array[:, i], label=f"Agent {i}", color=colors[i], linewidth=1.5)
    ax.set_title(f"Follower Counts — {title}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Followers")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(results["social_welfare"], color="darkgreen", linewidth=1.5)
    ax2.set_title("Social Welfare")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Total payoff per step")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {filename}")


# ─── Experiment A: γ=0, κ=0 (Pure personal utility) ─────────────────────────

def experiment_A():
    """
    Experiment A: No reputation incentive, no status incentive.
    Expected: no leader emerges; all agents stay in personal utility role.
    Validates the null case (Proposition 1 complement: γ=0 means no follow condition fires).
    """
    print("\n" + "#"*60)
    print("# Experiment A: γ=0, κ=0 (Pure personal utility)")
    print("#"*60)

    config = make_config(gamma=0.0, kappa=0.0, num_time_steps=3000)
    system = MultiAgentSystem(config)
    results = system.simulate()

    summarize(results, "Experiment A: γ=0, κ=0")
    plot_follower_timeseries(results, "Experiment A: γ=0 κ=0 (No leader expected)",
                             str(OUTPUT_DIR / "exp_A_no_leader.png"))
    return results


# ─── Experiment B: γ>0, κ=0 (Reputation only) ───────────────────────────────

def experiment_B():
    """
    Experiment B: Reputation incentive only, no status optimization.
    Expected: opinion leader emerges based on γ threshold (Proposition 1).
    Leader optimizes personal utility; followers copy leader's policy.
    """
    print("\n" + "#"*60)
    print("# Experiment B: γ=2.0, κ=0 (Reputation only)")
    print("#"*60)

    config = make_config(gamma=2.0, kappa=0.0, num_time_steps=5000)
    system = MultiAgentSystem(config)
    results = system.simulate()

    summarize(results, "Experiment B: γ=2.0, κ=0")
    plot_follower_timeseries(results, "Experiment B: γ=2.0 κ=0 (Leader expected)",
                             str(OUTPUT_DIR / "exp_B_reputation_only.png"))
    return results


# ─── Experiment C: γ>0, κ>0 (Full algorithm) ────────────────────────────────

def experiment_C():
    """
    Experiment C: Both reputation and status incentives active.
    Expected: welfare-optimal common norm emerges (Propositions 2&3).
    Leader switches to STATUS role and optimizes social support from followers.
    Social welfare should be higher than Experiment B.
    """
    print("\n" + "#"*60)
    print("# Experiment C: γ=2.0, κ=2.0 (Full algorithm)")
    print("#"*60)

    config = make_config(gamma=2.0, kappa=2.0, num_time_steps=5000)
    system = MultiAgentSystem(config)
    results = system.simulate()

    summarize(results, "Experiment C: γ=2.0, κ=2.0")
    plot_follower_timeseries(results, "Experiment C: γ=2.0 κ=2.0 (Welfare-optimal norm expected)",
                             str(OUTPUT_DIR / "exp_C_full_algorithm.png"))
    return results


# ─── Experiment D: Perturbation test ────────────────────────────────────────

def experiment_D():
    """
    Experiment D wrapper around the reproducible perturbation/recovery harness.
    Produces the legacy PNG plus CSV summaries for traceability.
    """
    print("\n" + "#"*60)
    print("# Experiment D: Perturbation test")
    print("#"*60)

    result = run_perturbation_recovery(
        mode="static",
        num_agents=8,
        num_states=3,
        num_actions=2,
        num_steps_max=12000,
        gamma=2.5,
        kappa=2.0,
        B_R=0.3,
        B_F=0.15,
        c_threshold=0.1,
        seeds=1,
        seed_start=25,
        role_update_base_interval=100,
        fixed_role_update_interval=True,
        perturb_strength=12.0,
        perturb_duration=300,
        collapse_followers_on_perturb=False,
        reputation_shock_factor=1.0,
        post_window=3500,
        conv_threshold=1.0,  # strict n-1
        conv_hold_steps=120,
        recovery_threshold=1.0,  # strict n-1 recovery
        recovery_hold_steps=80,
        stable_tail_window=200,
        dominant_threshold=0.5,
        drop_fraction_threshold=0.5,
        output_dir=str(OUTPUT_DIR),
        run_label="exp_D_perturbation_seed25",
        auto_run_subdir=True,
        plot_sample_interval=100,
        tracking_mode="light",
        reward_model="shared_base_gaussian",
        reward_base_mu=0.5,
        reward_base_sigma=0.08,
        reward_agent_sigma=0.1,
        reward_clip_min=0.01,
        reward_clip_max=2.5,
        output_prefix="exp_D_perturbation",
    )

    record = result["run_records"][0]
    details = result["details_by_seed"][record.seed]
    follower_rows = details["follower_rows"]
    welfare_series = details["welfare_series"].tolist()

    if record.t_perturb_start > 0:
        pre_idx = record.t_perturb_start - 1
    else:
        pre_idx = len(follower_rows) - 1

    pre_followers = follower_rows[pre_idx] if follower_rows else [0] * 8
    post_followers = follower_rows[-1] if follower_rows else [0] * 8
    pre_leader = int(record.leader_pre)
    post_leader = int(record.final_leader)

    print("\n[Experiment D Summary]")
    print(
        f"  t_conv={record.t_conv}, t_perturb_start={record.t_perturb_start}, "
        f"t_perturb_end={record.t_perturb_end}, recovery_time={record.recovery_time}"
    )
    print(
        f"  leader_pre={pre_leader}, pre_followers={record.pre_followers}, "
        f"drop_fraction={record.drop_fraction:.3f}, normless_duration={record.normless_duration}"
    )
    print(
        f"  final_leader={post_leader}, final_top_followers={record.final_top_followers}, "
        f"leader_changed_after_recovery={int(record.leader_changed)}, "
        f"leader_changed_final={int(record.final_leader_changed)}, "
        f"stable_recovery_last_{record.stable_tail_window}={int(record.stable_recovery)}"
    )
    print(f"  runs csv → {result['runs_csv']}")
    print(f"  aggregate csv → {result['aggregate_csv']}")
    print(f"  plot → {result['plot_files'][record.seed]}")

    # Preserve old function contract for the comparison table in __main__.
    results_phase1 = {
        "opinion_leader": pre_leader if pre_leader >= 0 else -1,
        "final_followers": pre_followers,
        "social_welfare": welfare_series[: max(1, pre_idx + 1)],
    }
    results_phase2 = {
        "opinion_leader": post_leader if post_leader >= 0 else -1,
        "final_followers": post_followers,
        "social_welfare": welfare_series,
    }

    return results_phase1, results_phase2


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  Running Experiments A–D")
    print("="*60)

    res_A = experiment_A()
    res_B = experiment_B()
    res_C = experiment_C()
    res_D1, res_D2 = experiment_D()

    # Comparison table
    print("\n" + "="*60)
    print("  COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Exp':<6} {'γ':<6} {'κ':<6} {'Leader':>8} {'Followers':>10} {'Welfare':>10}")
    print("-"*50)

    def row(label, gamma, kappa, results):
        leader = results["opinion_leader"]
        followers = results["final_followers"][leader] if leader >= 0 else 0
        welfare = np.mean(results["social_welfare"][-500:])
        print(f"{label:<6} {gamma:<6.1f} {kappa:<6.1f} {str(leader):>8} {followers:>10} {welfare:>10.4f}")

    row("A", 0.0, 0.0, res_A)
    row("B", 2.0, 0.0, res_B)
    row("C", 2.0, 2.0, res_C)
    if res_D2:
        row("D-pre", 2.0, 2.0, res_D1)
        row("D-post", 2.0, 2.0, res_D2)

    print("\nDone.")
