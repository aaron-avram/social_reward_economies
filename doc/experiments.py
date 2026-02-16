"""
Four experiments (A–D) using Peter's MultiAgentSystem after bug fixes.

Run from the doc/ directory:
    python3 experiments.py

Experiments:
  A: γ=0, κ=0 — Pure personal utility. Expect: no leader, all independent.
  B: γ>0, κ=0 — Reputation only. Expect: opinion leader may emerge (Proposition 1).
  C: γ>0, κ>0 — Full algorithm. Expect: welfare-optimal common norm (Propositions 2&3).
  D: Perturbation test. Expect: leader loses followers after perturbation.
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import from code_by_peter (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from code_by_peter import SystemConfig, MultiAgentSystem, AgentRole


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
        gamma=2.0,
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
                             "exp_A_no_leader.png")
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
                             "exp_B_reputation_only.png")
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
                             "exp_C_full_algorithm.png")
    return results


# ─── Experiment D: Perturbation test ────────────────────────────────────────

def experiment_D():
    """
    Experiment D: Run until leader emerges (Phase 1), then perturb the leader's
    policy weights to be suboptimal (Phase 2), and continue simulation.

    Expected:
    - Leader loses followers as its reputation drops.
    - If perturbation is large enough, a new leader may emerge.
    - System is resilient: eventually self-corrects.
    """
    print("\n" + "#"*60)
    print("# Experiment D: Perturbation test")
    print("#"*60)

    # Phase 1: Run until leader emerges
    print("\n[Phase 1] Running until leader emerges (3000 steps)...")
    config = make_config(gamma=2.0, kappa=2.0, num_time_steps=3000)
    system = MultiAgentSystem(config)
    results_phase1 = system.simulate()

    leader_id = results_phase1["opinion_leader"]
    leader_followers_pre = results_phase1["final_followers"][leader_id] if leader_id >= 0 else 0
    print(f"\n[Phase 1] Leader: Agent {leader_id} with {leader_followers_pre} followers")

    # Track pre-perturbation welfare
    welfare_pre = np.mean(results_phase1["social_welfare"][-500:])
    print(f"[Phase 1] Avg welfare (last 500 steps): {welfare_pre:.4f}")

    if leader_id < 0:
        print("[Phase 1] No leader emerged — skipping perturbation.")
        return results_phase1, None

    # Phase 2: Perturb leader's policy weights to be anti-optimal
    print("\n[Phase 2] Perturbing leader's policy weights (random noise)...")
    leader_agent = system.agents[leader_id]
    old_weights = leader_agent.state.weights_pu.copy()

    # Reset policy weights to strong anti-preference (opposite of convergence)
    # This forces the leader to take suboptimal actions
    leader_agent.state.weights_pu = -5.0 * np.abs(old_weights)
    leader_agent.state.weights_status = -5.0 * np.abs(leader_agent.state.weights_status)

    # Also reset the leader's reward estimates to simulate reputation collapse
    leader_agent.state.estimated_reward_pu = 0.0
    leader_agent.state.estimated_reward_status = 0.0
    for k in system.agents[leader_id].state.reputation_estimates:
        system.agents[leader_id].state.reputation_estimates[k] *= 0.1

    # Continue simulation for another 3000 steps
    print("[Phase 2] Running for 3000 more steps after perturbation...")
    system.config.num_time_steps = 6000  # extend the total
    for _ in range(3000):
        system.step()

    results_phase2 = {
        "norm_consensus": results_phase1["norm_consensus"] + system.results["norm_consensus"][-3000:],
        "follower_counts": results_phase1["follower_counts"] + system.results["follower_counts"][-3000:],
        "social_welfare": results_phase1["social_welfare"] + system.results["social_welfare"][-3000:],
        "roles_history": results_phase1["roles_history"] + system.results["roles_history"][-3000:],
        "final_roles": [a.state.role for a in system.agents],
        "final_followers": [len(a.state.followers) for a in system.agents],
        "opinion_leader": int(np.argmax([len(a.state.followers) for a in system.agents])),
        "actor_counts": results_phase1["actor_counts"] + system.results["actor_counts"][-3000:],
        "participant_counts": results_phase1["participant_counts"] + system.results["participant_counts"][-3000:],
        "actor_rates": results_phase1["actor_rates"] + system.results["actor_rates"][-3000:],
        "expected_utilities": results_phase1["expected_utilities"] + system.results["expected_utilities"][-3000:],
        "actual_payoffs": results_phase1["actual_payoffs"] + system.results["actual_payoffs"][-3000:],
    }

    welfare_post = np.mean(system.results["social_welfare"][-500:])
    new_leader = results_phase2["opinion_leader"]
    new_followers = results_phase2["final_followers"][new_leader]

    print(f"\n[Phase 2] After perturbation:")
    print(f"  New leader: Agent {new_leader} ({new_followers} followers)")
    print(f"  Ex-leader (Agent {leader_id}) followers: {results_phase2['final_followers'][leader_id]}")
    print(f"  Avg welfare (last 500 steps): {welfare_post:.4f}")

    summarize(results_phase2, "Experiment D (post-perturbation)")

    # Save combined plot
    followers_array = np.array(results_phase2["follower_counts"])
    n_agents = followers_array.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    ax = axes[0]
    for i in range(n_agents):
        ax.plot(followers_array[:, i], label=f"Agent {i}", color=colors[i], linewidth=1.5)
    ax.axvline(x=3000, color="red", linestyle="--", linewidth=2, label="Perturbation")
    ax.set_title("Experiment D: Follower Counts (Perturbation at step 3000)")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Followers")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    welfare_series = results_phase2["social_welfare"]
    ax2.plot(welfare_series, color="darkgreen", linewidth=1.5)
    ax2.axvline(x=3000, color="red", linestyle="--", linewidth=2, label="Perturbation")
    ax2.set_title("Social Welfare")
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Total payoff per step")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Experiment D: Perturbation Test", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("exp_D_perturbation.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Plot saved → exp_D_perturbation.png")

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
