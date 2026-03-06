"""
Phase-2 reputation scaling harness (Experiment B family).

This runner performs gamma sweeps with kappa fixed (default 0), collects
per-run metrics, aggregates across seeds, and writes plots/CSVs.

Example:
    python3 experiments/reputation_scaling.py \
      --mode static \
      --gammas "0,1,1.25,1.5,1.75,2,3,5" \
      --num-agents 100 \
      --num-states 5 \
      --num-actions 3 \
      --num-steps 50000 \
      --seeds 20 \
      --kappa 0
"""

from __future__ import annotations

import argparse
import csv
import io
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.code_debugged import MultiAgentSystem, SystemConfig  # noqa: E402


@dataclass
class RunRecord:
    mode: str
    gamma: float
    seed: int
    leader_id: int
    final_top_followers: int
    time_to_90pct_followers: int
    leader_switches: int
    tail_welfare: float


@dataclass
class AggregateRecord:
    mode: str
    gamma: float
    n_runs: int
    mean_final_top_followers: float
    std_final_top_followers: float
    ci95_final_top_followers: float
    mean_time_to_90pct_followers: float
    std_time_to_90pct_followers: float
    ci95_time_to_90pct_followers: float
    mean_leader_switches: float
    std_leader_switches: float
    ci95_leader_switches: float
    mean_tail_welfare: float
    std_tail_welfare: float
    ci95_tail_welfare: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reputation scaling gamma sweep harness.")
    parser.add_argument("--mode", choices=["static", "async"], required=True)
    parser.add_argument("--gammas", type=str, default="0,1,1.25,1.5,1.75,2,3,5")
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (inclusive).")
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--tail-window", type=int, default=500)
    parser.add_argument(
        "--role-update-s0",
        type=int,
        default=0,
        help="Paper notation s_0 for role-update epochs.",
    )
    parser.add_argument(
        "--role-update-T-seq",
        type=str,
        default="",
        help="Paper notation interval sequence T_n as comma-separated positive ints "
             "(e.g., \"2000,3000,6000\"); epochs are s_n = s_{n-1}+T_n. "
             "In async mode this is used as each agent's local interval progression.",
    )
    parser.add_argument("--role-update-base-interval", type=int, default=3000)
    parser.add_argument(
        "--fixed-role-update-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use constant role-update epochs T_n = const (Section 7.1.4) when enabled.",
    )
    parser.add_argument(
        "--plot-sample-interval",
        type=int,
        default=250,
        help="Downsample plotted time-series to every N steps (for paper-style visualization).",
    )
    parser.add_argument(
        "--role-update-epochs",
        type=str,
        default="",
        help="Optional direct role-update epochs s_n as comma-separated positive ints "
             "(e.g., \"2000,3000,6000\"). Used when role_update_T_seq is empty.",
    )
    parser.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    parser.add_argument("--initial-actor-rate", type=float, default=0.2)
    parser.add_argument("--initial-participant-rate", type=float, default=0.2)
    parser.add_argument(
        "--numpy-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vectorized reputation updates in code_debugged.",
    )
    parser.add_argument(
        "--async-role-update-prob",
        type=float,
        default=None,
        help="Optional per-step Bernoulli probability for role updates in async mode. "
             "If omitted, async uses independent per-agent clocks driven by "
             "role_update_T_seq/role_update_epochs (or role_update_base_interval as fallback).",
    )
    return parser.parse_args()


def parse_gammas(gamma_text: str) -> List[float]:
    parts = [p.strip() for p in gamma_text.split(",") if p.strip()]
    return [float(x) for x in parts]


def parse_role_update_epochs(epoch_text: str) -> List[int]:
    if not epoch_text.strip():
        return []
    parts = [p.strip() for p in epoch_text.split(",") if p.strip()]
    epochs = [int(x) for x in parts]
    return sorted(set(e for e in epochs if e > 0))


def parse_role_update_T_seq(t_text: str) -> List[int]:
    if not t_text.strip():
        return []
    parts = [p.strip() for p in t_text.split(",") if p.strip()]
    seq = [int(x) for x in parts]
    return [t for t in seq if t > 0]


def _interval_seq_from_epochs(s0: int, epochs: Sequence[int]) -> List[int]:
    """
    Convert epoch list s_n into interval sequence T_n, using paper relation:
    s_n = s_{n-1} + T_n with provided s_0.
    """
    prev = max(0, int(s0))
    intervals: List[int] = []
    for epoch in sorted(set(int(e) for e in epochs if int(e) > 0)):
        if epoch > prev:
            intervals.append(int(epoch - prev))
            prev = int(epoch)
    return intervals


def _build_async_interval_sequence(args: argparse.Namespace) -> Tuple[List[int], int, str]:
    """
    Build async per-agent interval sequence.

    Priority:
    1) --role-update-T-seq (paper T_n),
    2) --role-update-epochs converted to T_n from s_0,
    3) constant interval from --role-update-base-interval.
    """
    s0 = max(0, int(args.role_update_s0))
    t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    if t_seq:
        return t_seq, s0, "T_sequence"

    epochs = parse_role_update_epochs(args.role_update_epochs)
    if epochs:
        from_epochs = _interval_seq_from_epochs(s0=s0, epochs=epochs)
        if from_epochs:
            return from_epochs, s0, "epochs"

    return [max(1, int(args.role_update_base_interval))], s0, "base_interval"


def make_config(args: argparse.Namespace, gamma: float, mode: str) -> SystemConfig:
    # Keep defaults aligned with experiments/experiments.py unless explicitly overridden.
    role_interval = args.role_update_base_interval
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)
    if mode == "async":
        # Disable built-in synchronized periodic update; async mode runs external
        # stochastic update events after each step.
        role_interval = args.num_steps + 1_000_000
        role_s0 = 0
        role_t_seq = []
        role_epochs = []

    return SystemConfig(
        num_agents=args.num_agents,
        num_states=args.num_states,
        num_actions=args.num_actions,
        num_time_steps=args.num_steps,
        M=1.0,
        u_0=0.1,
        gamma=gamma,
        kappa=args.kappa,
        c_threshold=0.1,
        B_R=0.3,
        B_F=0.15,
        delta=0.15,
        alpha_pu_base=0.05,
        beta_status_base=0.05,
        eta_v_base=0.1,
        eta_s_base=0.1,
        eta_J_base=0.05,
        role_update_s0=role_s0,
        role_update_T_sequence=role_t_seq,
        role_update_base_interval=role_interval,
        fixed_role_update_interval=args.fixed_role_update_interval,
        role_update_epochs=role_epochs,
        gossip_rate=0.5,
        gossip_alpha=0.5,
        tracking_mode=args.tracking_mode,
        use_numpy_fast_path=args.numpy_fast_path,
        initial_actor_interaction_rate=args.initial_actor_rate,
        initial_participant_interaction_rate=args.initial_participant_rate,
    )


def _finalize_results(system: MultiAgentSystem) -> Dict:
    final_roles = [a.state.role for a in system.agents]
    final_followers = [len(a.state.followers) for a in system.agents]
    opinion_leader = int(np.argmax(final_followers)) if max(final_followers) > 0 else -1

    results = dict(system.results)
    results["final_roles"] = final_roles
    results["final_followers"] = final_followers
    results["opinion_leader"] = opinion_leader
    return results


def _leader_series_from_follower_counts(follower_counts: np.ndarray) -> np.ndarray:
    leaders = np.full(shape=(follower_counts.shape[0],), fill_value=-1, dtype=int)
    for t in range(follower_counts.shape[0]):
        row = follower_counts[t]
        m = int(np.max(row))
        if m > 0:
            candidates = np.where(row == m)[0]
            leaders[t] = int(candidates[0])  # deterministic tie-break
    return leaders


def _leader_switches(leader_series: np.ndarray) -> int:
    non_null = [x for x in leader_series.tolist() if x >= 0]
    if len(non_null) <= 1:
        return 0
    return sum(1 for a, b in zip(non_null[:-1], non_null[1:]) if a != b)


def _time_to_threshold(series: np.ndarray, threshold: int) -> int:
    idx = np.where(series >= threshold)[0]
    return int(idx[0] + 1) if idx.size > 0 else -1


def run_single(
    args: argparse.Namespace,
    mode: str,
    gamma: float,
    seed: int,
) -> Tuple[RunRecord, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    config = make_config(args, gamma=gamma, mode=mode)
    system = MultiAgentSystem(config)

    if mode == "async":
        with redirect_stdout(io.StringIO()):
            if args.async_role_update_prob is None:
                # Paper-faithful async relaxation:
                # independent per-agent role clocks with random phase and per-agent T_n progression.
                interval_seq, async_s0, _ = _build_async_interval_sequence(args)
                first_interval = int(interval_seq[0])
                role_timers = np.random.randint(1, first_interval + 1, size=args.num_agents, dtype=int)
                if async_s0 > 0:
                    role_timers = role_timers + async_s0
                interval_indices = np.zeros(args.num_agents, dtype=int)
            else:
                # Optional Bernoulli async mode: each agent independently reevaluates
                # with probability p each step.
                async_update_prob = float(args.async_role_update_prob)

            for _ in range(args.num_steps):
                system.step()
                if args.async_role_update_prob is None:
                    role_timers -= 1
                    update_ids = np.where(role_timers <= 0)[0]
                    if update_ids.size > 0:
                        update_list = update_ids.tolist()
                        system._update_roles_sequential(update_list)

                        if len(interval_seq) == 1:
                            role_timers[update_ids] += int(interval_seq[0])
                        else:
                            # Each agent advances independently through the provided T_n sequence.
                            # Once the sequence is exhausted, keep using the final interval.
                            for agent_id in update_list:
                                idx = int(interval_indices[agent_id])
                                next_interval = int(interval_seq[idx if idx < len(interval_seq) else -1])
                                role_timers[agent_id] += next_interval
                                if idx < len(interval_seq) - 1:
                                    interval_indices[agent_id] = idx + 1
                        system.role_update_epoch += 1
                else:
                    update_mask = np.random.random(args.num_agents) < async_update_prob
                    update_ids = np.where(update_mask)[0]
                    if update_ids.size > 0:
                        system._update_roles_sequential(update_ids.tolist())
                        # Track async update events for diagnostics.
                        system.role_update_epoch += 1
            results = _finalize_results(system)
    else:
        with redirect_stdout(io.StringIO()):
            for _ in range(args.num_steps):
                system.step()
            results = _finalize_results(system)

    follower_counts = np.array(results["follower_counts"], dtype=float)
    top_follower_series = follower_counts.max(axis=1)
    leader_series = _leader_series_from_follower_counts(follower_counts)

    threshold_90 = int(np.ceil(0.90 * (args.num_agents - 1)))
    time_to_90 = _time_to_threshold(top_follower_series, threshold_90)
    leader_switches = _leader_switches(leader_series)
    tail_window = min(args.tail_window, len(results["social_welfare"]))
    tail_welfare = float(np.mean(results["social_welfare"][-tail_window:]))

    record = RunRecord(
        mode=mode,
        gamma=float(gamma),
        seed=int(seed),
        leader_id=int(results["opinion_leader"]),
        final_top_followers=int(max(results["final_followers"])),
        time_to_90pct_followers=time_to_90,
        leader_switches=int(leader_switches),
        tail_welfare=tail_welfare,
    )
    return record, top_follower_series, leader_series


def _mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    n = arr.size
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci95


def aggregate(records: Sequence[RunRecord]) -> List[AggregateRecord]:
    grouped: Dict[Tuple[str, float], List[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.mode, rec.gamma), []).append(rec)

    rows: List[AggregateRecord] = []
    for (mode, gamma), recs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        final_top_vals = [r.final_top_followers for r in recs]
        time_vals = [r.time_to_90pct_followers for r in recs]
        # Exclude "not reached" marker from mean time-to-90 stats.
        reached_vals = [v for v in time_vals if v >= 0]
        if not reached_vals:
            reached_vals = [-1.0]
        switch_vals = [r.leader_switches for r in recs]
        welfare_vals = [r.tail_welfare for r in recs]

        m1, s1, c1 = _mean_std_ci(final_top_vals)
        m2, s2, c2 = _mean_std_ci(reached_vals)
        m3, s3, c3 = _mean_std_ci(switch_vals)
        m4, s4, c4 = _mean_std_ci(welfare_vals)

        rows.append(
            AggregateRecord(
                mode=mode,
                gamma=gamma,
                n_runs=len(recs),
                mean_final_top_followers=m1,
                std_final_top_followers=s1,
                ci95_final_top_followers=c1,
                mean_time_to_90pct_followers=m2,
                std_time_to_90pct_followers=s2,
                ci95_time_to_90pct_followers=c2,
                mean_leader_switches=m3,
                std_leader_switches=s3,
                ci95_leader_switches=c3,
                mean_tail_welfare=m4,
                std_tail_welfare=s4,
                ci95_tail_welfare=c4,
            )
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_progression(
    mode: str,
    gammas: Sequence[float],
    top_series_by_gamma: Dict[float, List[np.ndarray]],
    output_file: Path,
    sample_interval: int,
) -> None:
    n = len(gammas)
    cols = 4 if n >= 4 else n
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)

    for idx, gamma in enumerate(gammas):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        series_list = top_series_by_gamma[gamma]
        stack = np.stack(series_list, axis=0)  # [seeds, T]
        mean = np.mean(stack, axis=0)
        std = np.std(stack, axis=0)
        x = np.arange(1, stack.shape[1] + 1)
        if sample_interval > 1:
            sample_mask = (x % sample_interval) == 0
            if not np.any(sample_mask):
                sample_mask[-1] = True
            x = x[sample_mask]
            mean = mean[sample_mask]
            std = std[sample_mask]

        ax.plot(x, mean, linewidth=1.6, label=f"gamma={gamma:g}")
        ax.fill_between(x, mean - std, mean + std, alpha=0.25)
        ax.set_title(f"gamma={gamma:g}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Top followers")
        ax.grid(True, alpha=0.3)

    # Hide unused axes if subplot grid has spare cells.
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    plt.suptitle(
        f"Reputation Scaling Progression ({mode})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def plot_top_followers_curve(
    mode: str,
    aggregate_rows: Sequence[AggregateRecord],
    output_file: Path,
) -> None:
    rows = sorted(aggregate_rows, key=lambda r: r.gamma)
    gammas = np.array([r.gamma for r in rows], dtype=float)
    means = np.array([r.mean_final_top_followers for r in rows], dtype=float)
    cis = np.array([r.ci95_final_top_followers for r in rows], dtype=float)

    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(gammas, means, yerr=cis, fmt="-o", capsize=4, linewidth=1.8)
    plt.title(f"Final Top Followers vs Gamma ({mode})", fontsize=12, fontweight="bold")
    plt.xlabel("gamma")
    plt.ylabel("Final top followers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def _format_gamma(gamma: float) -> str:
    if float(gamma).is_integer():
        return str(int(gamma))
    return f"{gamma:g}"


def write_table_values_csv(
    gammas: Sequence[float],
    aggregate_rows: Sequence[AggregateRecord],
    output_file: Path,
) -> None:
    """Write table-style summary values (gamma, followers) to CSV."""
    agg_by_gamma = {row.gamma: row for row in aggregate_rows}
    rows = []
    for gamma in gammas:
        row = agg_by_gamma[gamma]
        rows.append(
            {
                "gamma": _format_gamma(gamma),
                "followers": int(round(row.mean_final_top_followers)),
            }
        )
    write_csv(output_file, rows)


def plot_paper_style_summary(
    mode: str,
    gammas: Sequence[float],
    top_series_by_gamma: Dict[float, List[np.ndarray]],
    leader_series_by_gamma: Dict[float, List[np.ndarray]],
    output_file: Path,
    num_agents: int,
    sample_interval: int,
) -> None:
    """
    Produce a figure styled close to the paper screenshot's chart section:
    max-follower-over-time curves by gamma.
    """
    gamma_order = list(gammas)
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    x_max = 0
    marker_every = 1

    # Screenshot focuses on gamma>0 in the legend; keep gamma=0 in the table only.
    plot_gammas = [g for g in gamma_order if g > 0.0]
    for gamma in plot_gammas:
        if not top_series_by_gamma.get(gamma):
            continue
        # Use first seed trajectory for paper-like single-trajectory visualization.
        y_full = np.array(top_series_by_gamma[gamma][0], dtype=float)
        leader_full = np.array(leader_series_by_gamma[gamma][0], dtype=int)

        y = y_full.copy()
        x = np.arange(1, len(y_full) + 1)
        if sample_interval > 1:
            sample_mask = (x % sample_interval) == 0
            if not np.any(sample_mask):
                sample_mask[-1] = True
            x = x[sample_mask]
            y = y[sample_mask]

        x_max = max(x_max, int(x[-1]) if len(x) > 0 else 0)
        marker_every = max(marker_every, len(y) // 14)

        line, = ax.plot(
            x,
            y,
            label=f"Gamma={_format_gamma(gamma)}",
            linewidth=1.7,
            drawstyle="steps-post",
        )

        # Mark only top-influencer switches so visible blips correspond to leader changes.
        switch_mask = np.zeros_like(leader_full, dtype=bool)
        for t in range(1, len(leader_full)):
            if leader_full[t] >= 0 and leader_full[t - 1] >= 0 and leader_full[t] != leader_full[t - 1]:
                switch_mask[t] = True
        switch_idx = np.where(switch_mask)[0]
        if switch_idx.size > 0:
            ax.plot(
                switch_idx + 1,
                y_full[switch_idx],
                "o",
                color=line.get_color(),
                markersize=6,
                alpha=0.95,
            )

    ax.set_title("Maximum Followers for a Single Agent Over Time", fontsize=16)
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Max Number of Followers", fontsize=12)
    ax.set_xlim(0, max(1, x_max))
    ax.set_ylim(-1, num_agents + 1)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    gammas = parse_gammas(args.gammas)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("#" * 72, flush=True)
    print("Reputation Scaling Run", flush=True)
    print(f"mode={args.mode}", flush=True)
    print(f"gammas={gammas}", flush=True)
    print(f"num_agents={args.num_agents}, num_states={args.num_states}, num_actions={args.num_actions}", flush=True)
    print(f"num_steps={args.num_steps}, seeds={len(seeds)} ({seeds[0]}..{seeds[-1]})", flush=True)
    print(f"kappa={args.kappa}", flush=True)
    print(
        f"role_update_base_interval={args.role_update_base_interval}, "
        f"fixed_role_update_interval={args.fixed_role_update_interval}",
        flush=True,
    )
    if args.mode != "async" and role_t_seq:
        print(f"role_update_schedule: s0={int(args.role_update_s0)}, T_n={role_t_seq}", flush=True)
    elif args.mode != "async" and role_epochs:
        print(f"role_update_epochs={role_epochs}", flush=True)
    elif args.mode != "async" and args.role_update_T_seq.strip():
        print("role_update_T_seq parsed empty (check values).", flush=True)
    print(f"initial_rates=(actor={args.initial_actor_rate}, participant={args.initial_participant_rate})", flush=True)
    print(f"plot_sample_interval={args.plot_sample_interval}", flush=True)
    print(f"tracking_mode={args.tracking_mode}, numpy_fast_path={args.numpy_fast_path}", flush=True)
    if args.mode == "async":
        if args.async_role_update_prob is None:
            async_t_seq, async_s0, async_src = _build_async_interval_sequence(args)
            print(
                f"async_mode=independent_agent_clocks(source={async_src}, s0={async_s0}, T_n={async_t_seq}, random_phase_in=[1,{int(async_t_seq[0])}], activity_coupled=False)",
                flush=True,
            )
        else:
            print(f"async_mode=bernoulli_per_agent(p={float(args.async_role_update_prob):.6f})", flush=True)
    print("#" * 72, flush=True)

    all_records: List[RunRecord] = []
    top_series_by_gamma: Dict[float, List[np.ndarray]] = {g: [] for g in gammas}
    leader_series_by_gamma: Dict[float, List[np.ndarray]] = {g: [] for g in gammas}
    total_jobs = len(gammas) * len(seeds)
    job = 0
    runs_csv = output_dir / f"reputation_scaling_runs_{args.mode}.csv"
    agg_csv = output_dir / f"reputation_scaling_aggregate_{args.mode}.csv"
    table_csv = output_dir / f"reputation_scaling_table_values_{args.mode}.csv"
    prog_png = output_dir / f"reputation_scaling_progression_{args.mode}.png"
    curve_png = output_dir / f"reputation_scaling_top_followers_{args.mode}.png"
    paper_png = output_dir / f"reputation_scaling_paper_style_{args.mode}.png"

    for gamma in gammas:
        for seed in seeds:
            job += 1
            print(f"[{job:03d}/{total_jobs:03d}] mode={args.mode} gamma={gamma:g} seed={seed}", flush=True)
            rec, top_series, leader_series = run_single(args=args, mode=args.mode, gamma=gamma, seed=seed)
            all_records.append(rec)
            top_series_by_gamma[gamma].append(top_series)
            leader_series_by_gamma[gamma].append(leader_series)
            # Incremental checkpoint for long sweeps.
            write_csv(runs_csv, [asdict(r) for r in all_records])

    agg_records = aggregate(all_records)

    write_csv(runs_csv, [asdict(r) for r in all_records])
    write_csv(agg_csv, [asdict(r) for r in agg_records])
    write_table_values_csv(gammas=gammas, aggregate_rows=agg_records, output_file=table_csv)
    plot_progression(
        args.mode,
        gammas,
        top_series_by_gamma,
        prog_png,
        sample_interval=max(1, int(args.plot_sample_interval)),
    )
    plot_top_followers_curve(args.mode, agg_records, curve_png)
    plot_paper_style_summary(
        mode=args.mode,
        gammas=gammas,
        top_series_by_gamma=top_series_by_gamma,
        leader_series_by_gamma=leader_series_by_gamma,
        output_file=paper_png,
        num_agents=args.num_agents,
        sample_interval=max(1, int(args.plot_sample_interval)),
    )

    print("\nCompleted.", flush=True)
    print(f"Per-run CSV:     {runs_csv}", flush=True)
    print(f"Aggregate CSV:   {agg_csv}", flush=True)
    print(f"Table CSV:       {table_csv}", flush=True)
    print(f"Progression PNG: {prog_png}", flush=True)
    print(f"Curve PNG:       {curve_png}", flush=True)
    print(f"Paper PNG:       {paper_png}", flush=True)


if __name__ == "__main__":
    main()
