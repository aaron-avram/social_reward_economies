from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.code_debugged import SystemConfig, MultiAgentSystem, AgentRole


OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "experiment_c"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Phase1Record:
    seed: int
    gamma: float
    kappa: float
    leader_id: int
    final_top_followers: int
    final_pu: int
    final_rep: int
    final_status: int
    ever_status: int
    first_status_time: int
    tail_welfare: float


def make_config(**kwargs) -> SystemConfig:
    defaults = dict(
        num_agents=100,
        num_states=3,
        num_actions=2,
        num_time_steps=12000,
        M=1.0,
        u_0=0.1,
        gamma=5.0,
        kappa=2.0,
        c_threshold=0.1,
        B_R=0.1,
        B_F=0.05,
        delta=0.15,
        alpha_pu_base=0.05,
        beta_status_base=0.05,
        eta_v_base=0.1,
        eta_s_base=0.1,
        eta_J_base=0.05,
        role_update_base_interval=50,
        fixed_role_update_interval=True,
        tracking_mode="light",
        reward_model="shared_base_gaussian",
        reward_base_mu=0.5,
        reward_base_sigma=0.08,
        reward_agent_sigma=0.1,
        reward_clip_min=0.01,
        reward_clip_max=2.5,
        initial_actor_interaction_rate=0.7,
        initial_participant_interaction_rate=0.7,
        use_numpy_fast_path=True,
    )
    defaults.update(kwargs)
    return SystemConfig(**defaults)


def run_single(seed: int, gamma: float, kappa: float) -> Phase1Record:
    np.random.seed(seed)

    config = make_config(gamma=gamma, kappa=kappa)
    system = MultiAgentSystem(config)
    results = system.simulate()

    final_roles = results["final_roles"]
    final_followers = results["final_followers"]
    leader = results["opinion_leader"]

    status_counts = np.array(results["status_counts"])
    welfare = np.array(results["social_welfare"])

    ever_status = int(np.any(status_counts > 0))
    first_status_time = int(np.argmax(status_counts > 0) + 1) if ever_status else -1

    tail_window = min(500, len(welfare))
    tail_welfare = float(np.mean(welfare[-tail_window:]))

    return Phase1Record(
        seed=seed,
        gamma=gamma,
        kappa=kappa,
        leader_id=leader,
        final_top_followers=max(final_followers) if final_followers else 0,
        final_pu=sum(1 for r in final_roles if r == AgentRole.PERSONAL_UTILITY),
        final_rep=sum(1 for r in final_roles if r == AgentRole.REPUTATION),
        final_status=sum(1 for r in final_roles if r == AgentRole.STATUS),
        ever_status=ever_status,
        first_status_time=first_status_time,
        tail_welfare=tail_welfare,
    )


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_status_probability(records: list[Phase1Record]):
    kappas = sorted(set(r.kappa for r in records))
    probs = []
    for k in kappas:
        vals = [r.ever_status for r in records if r.kappa == k]
        probs.append(np.mean(vals))

    plt.figure(figsize=(7, 4))
    plt.plot(kappas, probs, marker="o")
    plt.xlabel("kappa")
    plt.ylabel("Fraction of runs with STATUS")
    plt.title("STATUS emergence vs kappa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp_c_status_probability.png", dpi=140)
    plt.close()


def plot_welfare_vs_kappa(records: list[Phase1Record]):
    kappas = sorted(set(r.kappa for r in records))
    means = []
    stds = []
    for k in kappas:
        vals = [r.tail_welfare for r in records if r.kappa == k]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    plt.figure(figsize=(7, 4))
    plt.errorbar(kappas, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("kappa")
    plt.ylabel("Tail welfare")
    plt.title("Tail welfare vs kappa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp_c_welfare_vs_kappa.png", dpi=140)
    plt.close()


def plot_final_status_vs_kappa(records: list[Phase1Record]):
    kappas = sorted(set(r.kappa for r in records))
    means = []
    stds = []
    for k in kappas:
        vals = [r.final_status for r in records if r.kappa == k]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    plt.figure(figsize=(7, 4))
    plt.errorbar(kappas, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("kappa")
    plt.ylabel("Final # STATUS agents")
    plt.title("Final STATUS count vs kappa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp_c_final_status_vs_kappa.png", dpi=140)
    plt.close()


def plot_top_followers_vs_kappa(records: list[Phase1Record]):
    kappas = sorted(set(r.kappa for r in records))
    means = []
    stds = []
    for k in kappas:
        vals = [r.final_top_followers for r in records if r.kappa == k]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    plt.figure(figsize=(7, 4))
    plt.errorbar(kappas, means, yerr=stds, marker="o", capsize=4)
    plt.xlabel("kappa")
    plt.ylabel("Final top followers")
    plt.title("leader concentration vs kappa")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "exp_c_top_followers_vs_kappa.png", dpi=140)
    plt.close()


def main():
    gamma = 5.0
    kappas = [0.0, 0.5, 1.0, 2.0, 3.0]
    seeds = [0, 1, 2, 3, 4]

    records = []
    for kappa in kappas:
        for seed in seeds:
            print(f"Running experiment c: gamma={gamma}, kappa={kappa}, seed={seed}")
            rec = run_single(seed=seed, gamma=gamma, kappa=kappa)
            records.append(rec)

    save_csv(OUTPUT_DIR / "runs.csv", [asdict(r) for r in records])
    plot_status_probability(records)
    plot_welfare_vs_kappa(records)
    plot_final_status_vs_kappa(records)
    plot_top_followers_vs_kappa(records)

    print("\nDone.")
    print(f"Outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()