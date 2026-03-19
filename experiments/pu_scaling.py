"""
Phase-2 baseline harness for Experiment A (pure personal utility).

Goal:
- Scale up Experiment A under the same large-scale settings used for B
- Verify that with gamma=0 and kappa=0, no opinion leader emerges
- Use as null/control baseline for later comparison with B and C

Example:
    python3 experiments/personal_utility_scaling.py \
      --mode static \
      --num-agents 100 \
      --num-states-list "3,10" \
      --num-actions 2 \
      --num-steps 50000 \
      --seeds 10 \
      --reward-models "simple_preferred_action,shared_base_gaussian"
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

from src.code_debugged import MultiAgentSystem, SystemConfig, AgentRole  # noqa: E402


@dataclass
class RunRecord:
    mode: str
    num_agents: int
    num_states: int
    num_actions: int
    reward_model: str
    seed: int
    leader_id: int
    final_top_followers: int
    time_to_50pct_followers: int
    tail_welfare: float
    final_pu: int
    final_rep: int
    final_status: int


@dataclass
class AggregateRecord:
    mode: str
    num_agents: int
    num_states: int
    num_actions: int
    reward_model: str
    n_runs: int
    mean_final_top_followers: float
    std_final_top_followers: float
    ci95_final_top_followers: float
    mean_time_to_50pct_followers: float
    std_time_to_50pct_followers: float
    ci95_time_to_50pct_followers: float
    mean_tail_welfare: float
    std_tail_welfare: float
    ci95_tail_welfare: float
    mean_final_pu: float
    mean_final_rep: float
    mean_final_status: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale-up harness for Experiment A baseline.")
    parser.add_argument("--mode", choices=["static", "async"], default="static")
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--num-states-list", type=str, default="3,10")
    parser.add_argument("--num-actions", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--reward-models",
        type=str,
        default="simple_preferred_action,shared_base_gaussian",
        help='Comma-separated subset of {simple_preferred_action, shared_base_gaussian}',
    )
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "outputs"))
    parser.add_argument("--tail-window", type=int, default=500)

    parser.add_argument("--role-update-s0", type=int, default=0)
    parser.add_argument("--role-update-T-seq", type=str, default="")
    parser.add_argument("--role-update-base-interval", type=int, default=3000)
    parser.add_argument(
        "--fixed-role-update-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--role-update-epochs", type=str, default="")

    parser.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    parser.add_argument("--initial-actor-rate", type=float, default=0.2)
    parser.add_argument("--initial-participant-rate", type=float, default=0.2)

    parser.add_argument("--reward-base-mu", type=float, default=0.5)
    parser.add_argument("--reward-base-sigma", type=float, default=0.08)
    parser.add_argument("--reward-agent-sigma", type=float, default=0.1)
    parser.add_argument("--reward-clip-min", type=float, default=0.01)
    parser.add_argument("--reward-clip-max", type=float, default=2.5)

    parser.add_argument(
        "--numpy-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--async-role-update-prob",
        type=float,
        default=None,
        help="Optional Bernoulli async role-update probability. If omitted, async uses independent per-agent clocks.",
    )
    parser.add_argument(
        "--plot-sample-interval",
        type=int,
        default=1,
    )

    return parser.parse_args()


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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
    prev = max(0, int(s0))
    intervals: List[int] = []
    for epoch in sorted(set(int(e) for e in epochs if int(e) > 0)):
        if epoch > prev:
            intervals.append(int(epoch - prev))
            prev = int(epoch)
    return intervals


def _build_async_interval_sequence(args: argparse.Namespace) -> Tuple[List[int], int, str]:
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


def make_config(
    args: argparse.Namespace,
    mode: str,
    num_states: int,
    reward_model: str,
) -> SystemConfig:
    role_interval = args.role_update_base_interval
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)

    if mode == "async":
        # disable built-in global periodic role updates; external async updates will be used
        role_interval = args.num_steps + 1_000_000
        role_s0 = 0
        role_t_seq = []
        role_epochs = []

    return SystemConfig(
        num_agents=args.num_agents,
        num_states=num_states,
        num_actions=args.num_actions,
        num_time_steps=args.num_steps,
        M=1.0,
        u_0=0.1,
        gamma=0.0,   # Experiment A
        kappa=0.0,   # Experiment A
        c_threshold=0.1,
        B_R=0.3,
        B_F=1_000_000.0,
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
        reward_model=reward_model,
        reward_base_mu=args.reward_base_mu,
        reward_base_sigma=args.reward_base_sigma,
        reward_agent_sigma=args.reward_agent_sigma,
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
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
            leaders[t] = int(candidates[0])
    return leaders


def _time_to_threshold(series: np.ndarray, threshold: int) -> int:
    idx = np.where(series >= threshold)[0]
    return int(idx[0] + 1) if idx.size > 0 else -1


def role_counts_from_results(results: Dict) -> Tuple[int, int, int]:
    final_roles = results["final_roles"]
    n_pu = sum(r == AgentRole.PERSONAL_UTILITY for r in final_roles)
    n_rep = sum(r == AgentRole.REPUTATION for r in final_roles)
    n_st = sum(r == AgentRole.STATUS for r in final_roles)
    return n_pu, n_rep, n_st


def run_single(
    args: argparse.Namespace,
    mode: str,
    num_states: int,
    reward_model: str,
    seed: int,
) -> Tuple[RunRecord, np.ndarray]:
    np.random.seed(seed)
    config = make_config(args=args, mode=mode, num_states=num_states, reward_model=reward_model)
    system = MultiAgentSystem(config)

    if mode == "async":
        with redirect_stdout(io.StringIO()):
            if args.async_role_update_prob is None:
                interval_seq, async_s0, _ = _build_async_interval_sequence(args)
                first_interval = int(interval_seq[0])
                role_timers = np.random.randint(1, first_interval + 1, size=args.num_agents, dtype=int)
                if async_s0 > 0:
                    role_timers = role_timers + async_s0
                interval_indices = np.zeros(args.num_agents, dtype=int)
            else:
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
                        system.role_update_epoch += 1

            results = _finalize_results(system)
    else:
        with redirect_stdout(io.StringIO()):
            for _ in range(args.num_steps):
                system.step()
            results = _finalize_results(system)

    follower_counts = np.array(results["follower_counts"], dtype=float)
    top_follower_series = follower_counts.max(axis=1)

    threshold_50 = int(np.ceil(0.50 * (args.num_agents - 1)))
    time_to_50 = _time_to_threshold(top_follower_series, threshold_50)

    tail_window = min(args.tail_window, len(results["social_welfare"]))
    tail_welfare = float(np.mean(results["social_welfare"][-tail_window:]))

    final_pu, final_rep, final_status = role_counts_from_results(results)

    record = RunRecord(
        mode=mode,
        num_agents=args.num_agents,
        num_states=num_states,
        num_actions=args.num_actions,
        reward_model=reward_model,
        seed=seed,
        leader_id=int(results["opinion_leader"]),
        final_top_followers=int(max(results["final_followers"])),
        time_to_50pct_followers=time_to_50,
        tail_welfare=tail_welfare,
        final_pu=final_pu,
        final_rep=final_rep,
        final_status=final_status,
    )
    return record, top_follower_series


def _mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    n = arr.size
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci95


def aggregate(records: Sequence[RunRecord]) -> List[AggregateRecord]:
    grouped: Dict[Tuple[str, int, int, str], List[RunRecord]] = {}
    for rec in records:
        key = (rec.mode, rec.num_agents, rec.num_states, rec.reward_model)
        grouped.setdefault(key, []).append(rec)

    rows: List[AggregateRecord] = []
    for (mode, num_agents, num_states, reward_model), recs in sorted(grouped.items()):
        top_vals = [r.final_top_followers for r in recs]
        time_vals = [r.time_to_50pct_followers for r in recs]
        reached_vals = [v for v in time_vals if v >= 0]
        if not reached_vals:
            reached_vals = [-1.0]
        welfare_vals = [r.tail_welfare for r in recs]
        pu_vals = [r.final_pu for r in recs]
        rep_vals = [r.final_rep for r in recs]
        st_vals = [r.final_status for r in recs]

        m1, s1, c1 = _mean_std_ci(top_vals)
        m2, s2, c2 = _mean_std_ci(reached_vals)
        m3, s3, c3 = _mean_std_ci(welfare_vals)

        rows.append(
            AggregateRecord(
                mode=mode,
                num_agents=num_agents,
                num_states=num_states,
                num_actions=recs[0].num_actions,
                reward_model=reward_model,
                n_runs=len(recs),
                mean_final_top_followers=m1,
                std_final_top_followers=s1,
                ci95_final_top_followers=c1,
                mean_time_to_50pct_followers=m2,
                std_time_to_50pct_followers=s2,
                ci95_time_to_50pct_followers=c2,
                mean_tail_welfare=m3,
                std_tail_welfare=s3,
                ci95_tail_welfare=c3,
                mean_final_pu=float(np.mean(pu_vals)),
                mean_final_rep=float(np.mean(rep_vals)),
                mean_final_status=float(np.mean(st_vals)),
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
    num_states: int,
    reward_model: str,
    series_list: List[np.ndarray],
    output_file: Path,
    sample_interval: int,
) -> None:
    stack = np.stack(series_list, axis=0)
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

    plt.figure(figsize=(9, 4.8))
    plt.plot(x, mean, linewidth=1.8, label="Mean over seeds")
    plt.fill_between(x, mean - std, mean + std, alpha=0.25, label="±1 std")
    plt.title(
        f"Experiment A Baseline — max followers over time\n"
        f"mode={mode}, states={num_states}, reward={reward_model}",
        fontsize=12,
        fontweight="bold",
    )
    plt.xlabel("Timestep")
    plt.ylabel("Max followers of a single agent")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def plot_top_followers_bar(
    aggregate_rows: Sequence[AggregateRecord],
    output_file: Path,
) -> None:
    labels = [
        f"{row.mode}\nstates={row.num_states}\n{row.reward_model}"
        for row in aggregate_rows
    ]
    means = [row.mean_final_top_followers for row in aggregate_rows]
    cis = [row.ci95_final_top_followers for row in aggregate_rows]

    x = np.arange(len(labels))
    plt.figure(figsize=(10, 5))
    plt.bar(x, means, yerr=cis, capsize=4)
    plt.xticks(x, labels, rotation=0)
    plt.ylabel("Final top followers")
    plt.title("Experiment A baseline: final top followers across settings", fontsize=12, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    num_states_list = parse_int_list(args.num_states_list)
    reward_models = parse_str_list(args.reward_models)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("#" * 72)
    print("Experiment A scale-up baseline run")
    print(f"mode={args.mode}")
    print(f"num_agents={args.num_agents}")
    print(f"num_states_list={num_states_list}")
    print(f"reward_models={reward_models}")
    print(f"num_steps={args.num_steps}")
    print(f"seeds={len(seeds)} ({seeds[0]}..{seeds[-1]})")
    print("gamma=0, kappa=0")
    print("#" * 72)

    all_records: List[RunRecord] = []
    all_series: Dict[Tuple[str, int, str], List[np.ndarray]] = {}

    total_jobs = len(num_states_list) * len(reward_models) * len(seeds)
    job = 0

    runs_csv = output_dir / f"expA_scaling_runs_{args.mode}.csv"
    agg_csv = output_dir / f"expA_scaling_aggregate_{args.mode}.csv"
    bar_png = output_dir / f"expA_scaling_top_followers_{args.mode}.png"

    for num_states in num_states_list:
        for reward_model in reward_models:
            key = (args.mode, num_states, reward_model)
            all_series[key] = []

            for seed in seeds:
                job += 1
                print(
                    f"[{job:03d}/{total_jobs:03d}] mode={args.mode} "
                    f"states={num_states} reward={reward_model} seed={seed}",
                    flush=True,
                )
                rec, series = run_single(
                    args=args,
                    mode=args.mode,
                    num_states=num_states,
                    reward_model=reward_model,
                    seed=seed,
                )
                all_records.append(rec)
                all_series[key].append(series)

                write_csv(runs_csv, [asdict(r) for r in all_records])

            prog_png = output_dir / f"expA_progression_{args.mode}_states{num_states}_{reward_model}.png"
            plot_progression(
                mode=args.mode,
                num_states=num_states,
                reward_model=reward_model,
                series_list=all_series[key],
                output_file=prog_png,
                sample_interval=max(1, int(args.plot_sample_interval)),
            )

    agg_records = aggregate(all_records)
    write_csv(runs_csv, [asdict(r) for r in all_records])
    write_csv(agg_csv, [asdict(r) for r in agg_records])
    plot_top_followers_bar(agg_records, bar_png)

    print("\nCompleted.")
    print(f"Per-run CSV:   {runs_csv}")
    print(f"Aggregate CSV: {agg_csv}")
    print(f"Summary PNG:   {bar_png}")
    print("Per-setting progression PNGs were also written to outputs/.")


if __name__ == "__main__":
    main()
