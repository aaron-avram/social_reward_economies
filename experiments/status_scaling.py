from __future__ import annotations

import argparse
import csv
import io
from contextlib import redirect_stdout
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple, Optional

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
    gamma: float
    kappa: float
    seed: int
    leader_id: int
    final_top_followers: int
    mean_tail_top_followers: float
    time_to_90pct_followers: int
    leader_switches: int
    tail_welfare: float
    final_status_count: int
    mean_tail_status_count: float
    max_status_count: int
    leader_is_status_at_end: int
    final_pu_count: int
    final_rep_count: int
    final_norm_detected: int
    mean_tail_norm_detected: float
    final_norm_mean_consensus: float
    mean_tail_norm_mean_consensus: float
    min_tail_state_consensus: float


@dataclass
class AggregateRecord:
    mode: str
    gamma: float
    kappa: float
    n_runs: int

    mean_final_top_followers: float
    std_final_top_followers: float
    ci95_final_top_followers: float

    mean_mean_tail_top_followers: float
    std_mean_tail_top_followers: float
    ci95_mean_tail_top_followers: float

    mean_time_to_90pct_followers: float
    std_time_to_90pct_followers: float
    ci95_time_to_90pct_followers: float

    mean_leader_switches: float
    std_leader_switches: float
    ci95_leader_switches: float

    mean_tail_welfare: float
    std_tail_welfare: float
    ci95_tail_welfare: float

    mean_final_status_count: float
    std_final_status_count: float
    ci95_final_status_count: float

    mean_mean_tail_status_count: float
    std_mean_tail_status_count: float
    ci95_mean_tail_status_count: float

    mean_max_status_count: float
    std_max_status_count: float
    ci95_max_status_count: float

    mean_leader_is_status_at_end: float
    std_leader_is_status_at_end: float
    ci95_leader_is_status_at_end: float

    mean_final_pu_count: float
    std_final_pu_count: float
    ci95_final_pu_count: float

    mean_final_rep_count: float
    std_final_rep_count: float
    ci95_final_rep_count: float

    mean_final_norm_detected: float
    std_final_norm_detected: float
    ci95_final_norm_detected: float

    mean_mean_tail_norm_detected: float
    std_mean_tail_norm_detected: float
    ci95_mean_tail_norm_detected: float

    mean_final_norm_mean_consensus: float
    std_final_norm_mean_consensus: float
    ci95_final_norm_mean_consensus: float

    mean_mean_tail_norm_mean_consensus: float
    std_mean_tail_norm_mean_consensus: float
    ci95_mean_tail_norm_mean_consensus: float

    mean_min_tail_state_consensus: float
    std_min_tail_state_consensus: float
    ci95_min_tail_state_consensus: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Status scaling kappa sweep harness (Experiment C family)."
    )

    parser.add_argument("--mode", choices=["static", "async"], required=True)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--kappas", type=str, default="0,0.05,0.1,0.2,0.5,1")

    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)

    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (inclusive).")

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "outputs"),
    )
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
        help=(
            "Paper notation interval sequence T_n as comma-separated positive ints "
            '(e.g., "2000,3000,6000"); epochs are s_n = s_(n-1)+T_n. '
            "In async mode this is used as each agent's local interval progression."
        ),
    )
    parser.add_argument("--role-update-base-interval", type=int, default=3000)
    parser.add_argument(
        "--fixed-role-update-interval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use constant role-update epochs T_n = const when enabled.",
    )
    parser.add_argument(
        "--plot-sample-interval",
        type=int,
        default=1,
        help="Downsample plotted time-series to every N steps (1 records every timestep).",
    )
    parser.add_argument(
        "--role-update-epochs",
        type=str,
        default="",
        help='Optional direct role-update epochs s_n as comma-separated positive ints (e.g., "2000,3000,6000").',
    )

    parser.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    parser.add_argument("--initial-actor-rate", type=float, default=0.2)
    parser.add_argument("--initial-participant-rate", type=float, default=0.2)

    parser.add_argument(
        "--reward-model",
        choices=[
            "simple_preferred_action",
            "shared_base_gaussian",
            "shared_good_bad_heterogeneous",
            "consensus_welfare_gaussian",
        ],
        default="simple_preferred_action",
        help="Reward model for payoff generation.",
    )
    parser.add_argument("--reward-base-mu", type=float, default=0.5)
    parser.add_argument("--reward-base-sigma", type=float, default=0.08)
    parser.add_argument("--reward-agent-sigma", type=float, default=0.1)
    parser.add_argument("--reward-clip-min", type=float, default=0.01)
    parser.add_argument("--reward-clip-max", type=float, default=2.5)
    parser.add_argument("--reward-good-value", type=float, default=1.0)
    parser.add_argument("--reward-bad-value", type=float, default=0.1)
    parser.add_argument("--reward-order-gap", type=float, default=0.02)
    parser.add_argument("--reward-consensus-high", type=float, default=0.85)
    parser.add_argument("--reward-consensus-low", type=float, default=0.65)
    parser.add_argument("--reward-welfare-high", type=float, default=0.82)
    parser.add_argument("--reward-welfare-low", type=float, default=0.60)
    parser.add_argument("--reward-lambda-min", type=float, default=0.55)
    parser.add_argument("--reward-lambda-max", type=float, default=0.85)

    parser.add_argument(
        "--numpy-fast-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable vectorized reputation updates in code_debugged.",
    )

    parser.add_argument(
        "--c-threshold",
        type=float,
        default=0.1,
        help="Status-entry follower threshold c in |F_i| >= cN.",
    )
    parser.add_argument(
        "--B-R",
        dest="B_R",
        type=float,
        default=0.3,
        help="Reputation start-follow threshold.",
    )
    parser.add_argument(
        "--B-F",
        dest="B_F",
        type=float,
        default=1000000.0,
        help="Reputation continue-follow threshold. Large default effectively disables hysteresis.",
    )

    parser.add_argument(
        "--async-role-update-prob",
        type=float,
        default=None,
        help="Optional per-step Bernoulli probability for role updates in async mode.",
    )

    parser.add_argument(
        "--role-update-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write lightweight role-update-only diagnostics for convergence/fragmentation analysis.",
    )

    return parser.parse_args()


def parse_kappas(kappa_text: str) -> List[float]:
    parts = [p.strip() for p in kappa_text.split(",") if p.strip()]
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


def make_config(args: argparse.Namespace, gamma: float, kappa: float, mode: str) -> SystemConfig:
    role_interval = args.role_update_base_interval
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)

    if mode == "async":
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
        kappa=kappa,
        c_threshold=args.c_threshold,
        B_R=args.B_R,
        B_F=args.B_F,
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
        reward_model=args.reward_model,
        reward_base_mu=args.reward_base_mu,
        reward_base_sigma=args.reward_base_sigma,
        reward_agent_sigma=args.reward_agent_sigma,
        reward_clip_min=args.reward_clip_min,
        reward_clip_max=args.reward_clip_max,
        reward_good_value=args.reward_good_value,
        reward_bad_value=args.reward_bad_value,
        reward_order_gap=args.reward_order_gap,
        reward_consensus_high=args.reward_consensus_high,
        reward_consensus_low=args.reward_consensus_low,
        reward_welfare_high=args.reward_welfare_high,
        reward_welfare_low=args.reward_welfare_low,
        reward_lambda_min=args.reward_lambda_min,
        reward_lambda_max=args.reward_lambda_max,
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


def _leader_switches(leader_series: np.ndarray) -> int:
    non_null = [x for x in leader_series.tolist() if x >= 0]
    if len(non_null) <= 1:
        return 0
    return sum(1 for a, b in zip(non_null[:-1], non_null[1:]) if a != b)


def _time_to_threshold(series: np.ndarray, threshold: int) -> int:
    idx = np.where(series >= threshold)[0]
    return int(idx[0] + 1) if idx.size > 0 else -1


def _mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=float)
    n = arr.size
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci95


def aggregate(records: Sequence[RunRecord]) -> List[AggregateRecord]:
    grouped: Dict[Tuple[str, float, float], List[RunRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.mode, rec.gamma, rec.kappa), []).append(rec)

    rows: List[AggregateRecord] = []
    for (mode, gamma, kappa), recs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        def vals(name: str) -> List[float]:
            return [getattr(r, name) for r in recs]

        final_top_vals = vals("final_top_followers")
        mean_tail_top_vals = vals("mean_tail_top_followers")
        reached_vals = [v for v in vals("time_to_90pct_followers") if v >= 0]
        if not reached_vals:
            reached_vals = [-1.0]
        switch_vals = vals("leader_switches")
        welfare_vals = vals("tail_welfare")
        final_status_vals = vals("final_status_count")
        mean_tail_status_vals = vals("mean_tail_status_count")
        max_status_vals = vals("max_status_count")
        leader_status_vals = vals("leader_is_status_at_end")
        final_pu_vals = vals("final_pu_count")
        final_rep_vals = vals("final_rep_count")
        final_norm_detected_vals = vals("final_norm_detected")
        mean_tail_norm_detected_vals = vals("mean_tail_norm_detected")
        final_norm_mean_consensus_vals = vals("final_norm_mean_consensus")
        mean_tail_norm_mean_consensus_vals = vals("mean_tail_norm_mean_consensus")
        min_tail_state_consensus_vals = vals("min_tail_state_consensus")

        m1, s1, c1 = _mean_std_ci(final_top_vals)
        m1b, s1b, c1b = _mean_std_ci(mean_tail_top_vals)
        m2, s2, c2 = _mean_std_ci(reached_vals)
        m3, s3, c3 = _mean_std_ci(switch_vals)
        m4, s4, c4 = _mean_std_ci(welfare_vals)
        m5, s5, c5 = _mean_std_ci(final_status_vals)
        m6, s6, c6 = _mean_std_ci(mean_tail_status_vals)
        m7, s7, c7 = _mean_std_ci(max_status_vals)
        m8, s8, c8 = _mean_std_ci(leader_status_vals)
        m9, s9, c9 = _mean_std_ci(final_pu_vals)
        m10, s10, c10 = _mean_std_ci(final_rep_vals)
        m11, s11, c11 = _mean_std_ci(final_norm_detected_vals)
        m12, s12, c12 = _mean_std_ci(mean_tail_norm_detected_vals)
        m13, s13, c13 = _mean_std_ci(final_norm_mean_consensus_vals)
        m14, s14, c14 = _mean_std_ci(mean_tail_norm_mean_consensus_vals)
        m15, s15, c15 = _mean_std_ci(min_tail_state_consensus_vals)

        rows.append(
            AggregateRecord(
                mode=mode,
                gamma=gamma,
                kappa=kappa,
                n_runs=len(recs),
                mean_final_top_followers=m1,
                std_final_top_followers=s1,
                ci95_final_top_followers=c1,
                mean_mean_tail_top_followers=m1b,
                std_mean_tail_top_followers=s1b,
                ci95_mean_tail_top_followers=c1b,
                mean_time_to_90pct_followers=m2,
                std_time_to_90pct_followers=s2,
                ci95_time_to_90pct_followers=c2,
                mean_leader_switches=m3,
                std_leader_switches=s3,
                ci95_leader_switches=c3,
                mean_tail_welfare=m4,
                std_tail_welfare=s4,
                ci95_tail_welfare=c4,
                mean_final_status_count=m5,
                std_final_status_count=s5,
                ci95_final_status_count=c5,
                mean_mean_tail_status_count=m6,
                std_mean_tail_status_count=s6,
                ci95_mean_tail_status_count=c6,
                mean_max_status_count=m7,
                std_max_status_count=s7,
                ci95_max_status_count=c7,
                mean_leader_is_status_at_end=m8,
                std_leader_is_status_at_end=s8,
                ci95_leader_is_status_at_end=c8,
                mean_final_pu_count=m9,
                std_final_pu_count=s9,
                ci95_final_pu_count=c9,
                mean_final_rep_count=m10,
                std_final_rep_count=s10,
                ci95_final_rep_count=c10,
                mean_final_norm_detected=m11,
                std_final_norm_detected=s11,
                ci95_final_norm_detected=c11,
                mean_mean_tail_norm_detected=m12,
                std_mean_tail_norm_detected=s12,
                ci95_mean_tail_norm_detected=c12,
                mean_final_norm_mean_consensus=m13,
                std_final_norm_mean_consensus=s13,
                ci95_final_norm_mean_consensus=c13,
                mean_mean_tail_norm_mean_consensus=m14,
                std_mean_tail_norm_mean_consensus=s14,
                ci95_mean_tail_norm_mean_consensus=c14,
                mean_min_tail_state_consensus=m15,
                std_min_tail_state_consensus=s15,
                ci95_min_tail_state_consensus=c15,
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


def build_static_role_update_times(args: argparse.Namespace, horizon: int) -> List[int]:
    if args.mode == "async":
        return []

    horizon = max(0, int(horizon))
    if horizon == 0:
        return []

    t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    if t_seq:
        cursor = max(0, int(args.role_update_s0))
        epochs = []
        for interval in t_seq:
            cursor += int(interval)
            if 0 < cursor <= horizon:
                epochs.append(int(cursor))
        return epochs

    epochs = parse_role_update_epochs(args.role_update_epochs)
    if epochs:
        return [int(t) for t in epochs if 0 < int(t) <= horizon]

    if args.fixed_role_update_interval:
        interval = max(1, int(args.role_update_base_interval))
        return list(range(interval, horizon + 1, interval))

    out: List[int] = []
    epoch = 0
    next_time = max(1, int(args.role_update_base_interval))
    while next_time <= horizon:
        out.append(int(next_time))
        epoch += 1
        next_interval = max(
            int(args.role_update_base_interval),
            int(args.role_update_base_interval * (1.0 + epoch * 0.1)),
        )
        next_time += next_interval
    return out


def run_single(
    args: argparse.Namespace,
    mode: str,
    gamma: float,
    kappa: float,
    seed: int,
) -> Tuple[RunRecord, np.ndarray, np.ndarray, np.ndarray, Optional[List[Dict[str, object]]]]:
    np.random.seed(seed)
    config = make_config(args, gamma=gamma, kappa=kappa, mode=mode)
    system = MultiAgentSystem(config)

    role_update_diagnostics_enabled = bool(mode == "static" and getattr(args, "role_update_diagnostics", False))
    if role_update_diagnostics_enabled:
        system.enable_role_update_diagnostics()

    if mode == "async":
        with redirect_stdout(io.StringIO()):
            if args.async_role_update_prob is None:
                interval_seq, async_s0, _ = _build_async_interval_sequence(args)
                first_interval = int(interval_seq[0])
                role_timers = np.random.randint(1, first_interval + 1, size=args.num_agents, dtype=int)
                if async_s0 > 0:
                    role_timers = role_timers + async_s0
                interval_indices = np.zeros(args.num_agents, dtype=int)

                for _ in range(args.num_steps):
                    system.step()
                    role_timers -= 1
                    update_ids = np.where(role_timers <= 0)[0]
                    if update_ids.size > 0:
                        update_list = update_ids.tolist()
                        system._update_roles_sequential(update_list)
                        system.refresh_last_tracked_state()
                        system.results.setdefault("role_update_times", []).append(int(system.time_step))
                        system.role_update_epoch += 1

                        if len(interval_seq) == 1:
                            role_timers[update_ids] += int(interval_seq[0])
                        else:
                            for agent_id in update_list:
                                idx = int(interval_indices[agent_id])
                                next_interval = int(interval_seq[idx if idx < len(interval_seq) else -1])
                                role_timers[agent_id] += next_interval
                                if idx < len(interval_seq) - 1:
                                    interval_indices[agent_id] = idx + 1
            else:
                async_update_prob = float(args.async_role_update_prob)
                for _ in range(args.num_steps):
                    system.step()
                    update_mask = np.random.random(args.num_agents) < async_update_prob
                    update_ids = np.where(update_mask)[0]
                    if update_ids.size > 0:
                        system._update_roles_sequential(update_ids.tolist())
                        system.refresh_last_tracked_state()
                        system.results.setdefault("role_update_times", []).append(int(system.time_step))
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
    mean_tail_top_followers = float(np.mean(top_follower_series[-tail_window:]))

    status_counts_series = np.array(results.get("status_counts", []), dtype=float)
    pu_counts_series = np.array(results.get("pu_counts", []), dtype=float)
    rep_counts_series = np.array(results.get("rep_counts", []), dtype=float)

    norm_detected_series = np.array(results.get("norm_detected_history", []), dtype=float)
    norm_mean_consensus_series = np.array(results.get("norm_mean_consensus_history", []), dtype=float)
    norm_state_consensus_history = np.array(results.get("norm_state_consensus_history", []), dtype=float)

    if norm_detected_series.size > 0:
        final_norm_detected = int(norm_detected_series[-1])
        mean_tail_norm_detected = float(np.mean(norm_detected_series[-tail_window:]))
    else:
        final_norm_detected = 0
        mean_tail_norm_detected = 0.0

    if norm_mean_consensus_series.size > 0:
        final_norm_mean_consensus = float(norm_mean_consensus_series[-1])
        mean_tail_norm_mean_consensus = float(np.mean(norm_mean_consensus_series[-tail_window:]))
    else:
        final_norm_mean_consensus = 0.0
        mean_tail_norm_mean_consensus = 0.0

    if norm_state_consensus_history.size > 0:
        tail_state_consensus = norm_state_consensus_history[-tail_window:]
        min_tail_state_consensus = float(np.min(np.mean(tail_state_consensus, axis=0)))
    else:
        min_tail_state_consensus = 0.0

    if status_counts_series.size > 0:
        final_status_count = int(status_counts_series[-1])
        mean_tail_status_count = float(np.mean(status_counts_series[-tail_window:]))
        max_status_count = int(np.max(status_counts_series))
    else:
        final_status_count = 0
        mean_tail_status_count = 0.0
        max_status_count = 0

    final_pu_count = int(pu_counts_series[-1]) if pu_counts_series.size > 0 else 0
    final_rep_count = int(rep_counts_series[-1]) if rep_counts_series.size > 0 else 0

    leader_is_status_at_end = 0
    if results["opinion_leader"] >= 0:
        leader_role = system.agents[results["opinion_leader"]].state.role
        leader_is_status_at_end = int(leader_role == AgentRole.STATUS)

    record = RunRecord(
        mode=mode,
        gamma=float(gamma),
        kappa=float(kappa),
        seed=int(seed),
        leader_id=int(results["opinion_leader"]),
        final_top_followers=int(max(results["final_followers"])),
        mean_tail_top_followers=mean_tail_top_followers,
        time_to_90pct_followers=time_to_90,
        leader_switches=int(leader_switches),
        tail_welfare=tail_welfare,
        final_status_count=final_status_count,
        mean_tail_status_count=mean_tail_status_count,
        max_status_count=max_status_count,
        leader_is_status_at_end=leader_is_status_at_end,
        final_pu_count=final_pu_count,
        final_rep_count=final_rep_count,
        final_norm_detected=final_norm_detected,
        mean_tail_norm_detected=mean_tail_norm_detected,
        final_norm_mean_consensus=final_norm_mean_consensus,
        mean_tail_norm_mean_consensus=mean_tail_norm_mean_consensus,
        min_tail_state_consensus=min_tail_state_consensus,
    )

    role_update_diagnostics = system.get_role_update_diagnostic_rows() if role_update_diagnostics_enabled else None
    return record, top_follower_series, status_counts_series, norm_mean_consensus_series, role_update_diagnostics


def summarize_role_update_diagnostics(
    output_csv: Path,
    rows: Sequence[Dict[str, object]],
    mode: str,
    gamma: float,
    kappa: float,
    seed: int,
) -> None:
    if not rows:
        return

    tagged_rows = []
    for row in rows:
        tagged = dict(row)
        tagged["mode"] = mode
        tagged["gamma"] = float(gamma)
        tagged["kappa"] = float(kappa)
        tagged["seed"] = int(seed)
        tagged_rows.append(tagged)

    existing_rows: List[dict] = []
    if output_csv.exists():
        with output_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)

    existing_rows.extend(tagged_rows)
    write_csv(output_csv, existing_rows)


def plot_top_followers_curve(
    mode: str,
    gamma: float,
    aggregate_rows: Sequence[AggregateRecord],
    output_file: Path,
) -> None:
    rows = sorted(aggregate_rows, key=lambda r: r.kappa)
    kappas = np.array([r.kappa for r in rows], dtype=float)
    means = np.array([r.mean_final_top_followers for r in rows], dtype=float)
    cis = np.array([r.ci95_final_top_followers for r in rows], dtype=float)

    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(kappas, means, yerr=cis, fmt="-o", capsize=4, linewidth=1.8)
    plt.title(f"Final Top Followers vs Kappa ({mode}, gamma={gamma:g})", fontsize=12, fontweight="bold")
    plt.xlabel("kappa")
    plt.ylabel("Final top followers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def plot_status_curve(
    mode: str,
    gamma: float,
    aggregate_rows: Sequence[AggregateRecord],
    output_file: Path,
) -> None:
    rows = sorted(aggregate_rows, key=lambda r: r.kappa)
    kappas = np.array([r.kappa for r in rows], dtype=float)
    means = np.array([r.mean_mean_tail_status_count for r in rows], dtype=float)
    cis = np.array([r.ci95_mean_tail_status_count for r in rows], dtype=float)

    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(kappas, means, yerr=cis, fmt="-o", capsize=4, linewidth=1.8)
    plt.title(f"Mean Tail Status Count vs Kappa ({mode}, gamma={gamma:g})", fontsize=12, fontweight="bold")
    plt.xlabel("kappa")
    plt.ylabel("Mean tail status count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def plot_welfare_curve(
    mode: str,
    gamma: float,
    aggregate_rows: Sequence[AggregateRecord],
    output_file: Path,
) -> None:
    rows = sorted(aggregate_rows, key=lambda r: r.kappa)
    kappas = np.array([r.kappa for r in rows], dtype=float)
    means = np.array([r.mean_tail_welfare for r in rows], dtype=float)
    cis = np.array([r.ci95_tail_welfare for r in rows], dtype=float)

    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(kappas, means, yerr=cis, fmt="-o", capsize=4, linewidth=1.8)
    plt.title(f"Tail Welfare vs Kappa ({mode}, gamma={gamma:g})", fontsize=12, fontweight="bold")
    plt.xlabel("kappa")
    plt.ylabel("Tail welfare")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def plot_norm_curve(
    mode: str,
    gamma: float,
    aggregate_rows: Sequence[AggregateRecord],
    output_file: Path,
) -> None:
    rows = sorted(aggregate_rows, key=lambda r: r.kappa)
    kappas = np.array([r.kappa for r in rows], dtype=float)
    means = np.array([r.mean_mean_tail_norm_mean_consensus for r in rows], dtype=float)
    cis = np.array([r.ci95_mean_tail_norm_mean_consensus for r in rows], dtype=float)

    plt.figure(figsize=(7.5, 4.5))
    plt.errorbar(kappas, means, yerr=cis, fmt="-o", capsize=4, linewidth=1.8)
    plt.title(f"Tail Norm Consensus vs Kappa ({mode}, gamma={gamma:g})", fontsize=12, fontweight="bold")
    plt.xlabel("kappa")
    plt.ylabel("Mean tail norm consensus")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=140, bbox_inches="tight")
    plt.close()


def plot_progression_examples(
    mode: str,
    gamma: float,
    kappas: Sequence[float],
    top_series_by_kappa: Dict[float, List[np.ndarray]],
    status_series_by_kappa: Dict[float, List[np.ndarray]],
    norm_series_by_kappa: Dict[float, List[np.ndarray]],
    output_file: Path,
    sample_interval: int,
    role_update_times: Sequence[int],
) -> None:
    n = len(kappas)
    cols = 3 if n >= 3 else n
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 3.8 * rows), squeeze=False)

    for idx, kappa in enumerate(kappas):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        top_stack = np.stack(top_series_by_kappa[kappa], axis=0)
        top_mean = np.mean(top_stack, axis=0)

        if status_series_by_kappa[kappa]:
            min_len_status = min(len(s) for s in status_series_by_kappa[kappa])
            status_stack = np.stack([s[:min_len_status] for s in status_series_by_kappa[kappa]], axis=0)
            status_mean = np.mean(status_stack, axis=0)
        else:
            status_mean = np.zeros_like(top_mean)

        if norm_series_by_kappa[kappa]:
            min_len_norm = min(len(s) for s in norm_series_by_kappa[kappa])
            norm_stack = np.stack([s[:min_len_norm] for s in norm_series_by_kappa[kappa]], axis=0)
            norm_mean = np.mean(norm_stack, axis=0)
        else:
            norm_mean = np.zeros_like(top_mean)

        T = min(len(top_mean), len(status_mean), len(norm_mean))
        x = np.arange(1, T + 1)
        top_mean = top_mean[:T]
        status_mean = status_mean[:T]
        norm_mean = norm_mean[:T]

        if sample_interval > 1:
            sample_mask = (x % sample_interval) == 0
            if not np.any(sample_mask):
                sample_mask[-1] = True
            x = x[sample_mask]
            top_mean = top_mean[sample_mask]
            status_mean = status_mean[sample_mask]
            norm_mean = norm_mean[sample_mask]

        ax.plot(x, top_mean, linewidth=1.8, label="top followers")
        ax.plot(x, status_mean, linewidth=1.8, label="status count")
        ax.plot(x, norm_mean, linewidth=1.8, label="norm consensus")

        for idx_line, step in enumerate(role_update_times):
            ax.axvline(
                int(step),
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.18,
                label="Role update" if idx == 0 and idx_line == 0 else None,
            )

        ax.set_title(f"kappa={kappa:g}")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Count / consensus")
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        rr, cc = divmod(idx, cols)
        axes[rr][cc].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(f"Status Scaling Progression ({mode}, gamma={gamma:g})", fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    kappas = parse_kappas(args.kappas)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("#" * 72, flush=True)
    print("Status Scaling Run", flush=True)
    print(f"mode={args.mode}", flush=True)
    print(f"gamma={args.gamma}", flush=True)
    print(f"kappas={kappas}", flush=True)
    print(f"num_agents={args.num_agents}, num_states={args.num_states}, num_actions={args.num_actions}", flush=True)
    print(f"num_steps={args.num_steps}, seeds={len(seeds)} ({seeds[0]}..{seeds[-1]})", flush=True)
    print(
        f"role_update_base_interval={args.role_update_base_interval}, "
        f"fixed_role_update_interval={args.fixed_role_update_interval}",
        flush=True,
    )
    print(
        f"initial_rates=(actor={args.initial_actor_rate}, participant={args.initial_participant_rate})",
        flush=True,
    )

    if args.reward_model == "shared_base_gaussian":
        print(
            "reward_model=shared_base_gaussian "
            f"(base_mu={args.reward_base_mu}, base_sigma={args.reward_base_sigma}, "
            f"agent_sigma={args.reward_agent_sigma}, clip=[{args.reward_clip_min},{args.reward_clip_max}])",
            flush=True,
        )
    elif args.reward_model == "shared_good_bad_heterogeneous":
        print(
            "reward_model=shared_good_bad_heterogeneous "
            f"(good={args.reward_good_value}, bad={args.reward_bad_value}, "
            f"order_gap={args.reward_order_gap}, agent_sigma={args.reward_agent_sigma}, "
            f"clip=[{args.reward_clip_min},{args.reward_clip_max}])",
            flush=True,
        )
    elif args.reward_model == "consensus_welfare_gaussian":
        print(
            "reward_model=consensus_welfare_gaussian "
            f"(consensus_high={args.reward_consensus_high}, "
            f"consensus_low={args.reward_consensus_low}, "
            f"welfare_high={args.reward_welfare_high}, "
            f"welfare_low={args.reward_welfare_low}, "
            f"lambda_range=[{args.reward_lambda_min},{args.reward_lambda_max}], "
            f"agent_sigma={args.reward_agent_sigma}, "
            f"clip=[{args.reward_clip_min},{args.reward_clip_max}])",
            flush=True,
        )
    else:
        print("reward_model=simple_preferred_action", flush=True)

    print(f"plot_sample_interval={args.plot_sample_interval}", flush=True)
    print(f"tracking_mode={args.tracking_mode}, numpy_fast_path={args.numpy_fast_path}", flush=True)
    print(f"c_threshold={args.c_threshold}", flush=True)
    print(f"B_R={args.B_R}, B_F={args.B_F}", flush=True)

    if args.mode == "async":
        if args.async_role_update_prob is None:
            async_t_seq, async_s0, async_src = _build_async_interval_sequence(args)
            print(
                f"async_mode=independent_agent_clocks(source={async_src}, s0={async_s0}, "
                f"T_n={async_t_seq}, random_phase_in=[1,{int(async_t_seq[0])}], activity_coupled=False)",
                flush=True,
            )
        else:
            print(
                f"async_mode=bernoulli_per_agent(p={float(args.async_role_update_prob):.6f})",
                flush=True,
            )

    print("#" * 72, flush=True)

    all_records: List[RunRecord] = []
    top_series_by_kappa: Dict[float, List[np.ndarray]] = {k: [] for k in kappas}
    status_series_by_kappa: Dict[float, List[np.ndarray]] = {k: [] for k in kappas}
    norm_series_by_kappa: Dict[float, List[np.ndarray]] = {k: [] for k in kappas}

    role_update_times = build_static_role_update_times(args, horizon=int(args.num_steps))
    total_jobs = len(kappas) * len(seeds)
    job = 0

    runs_csv = output_dir / f"status_scaling_runs_{args.mode}_g{args.gamma:g}.csv"
    agg_csv = output_dir / f"status_scaling_aggregate_{args.mode}_g{args.gamma:g}.csv"
    role_diag_csv = output_dir / f"status_scaling_role_update_diagnostics_{args.mode}_g{args.gamma:g}.csv"

    top_png = output_dir / f"status_scaling_top_followers_{args.mode}_g{args.gamma:g}.png"
    status_png = output_dir / f"status_scaling_status_counts_{args.mode}_g{args.gamma:g}.png"
    welfare_png = output_dir / f"status_scaling_tail_welfare_{args.mode}_g{args.gamma:g}.png"
    norm_png = output_dir / f"status_scaling_norm_consensus_{args.mode}_g{args.gamma:g}.png"
    prog_png = output_dir / f"status_scaling_progression_{args.mode}_g{args.gamma:g}.png"

    for kappa in kappas:
        for seed in seeds:
            job += 1
            print(
                f"[{job:03d}/{total_jobs:03d}] mode={args.mode} gamma={args.gamma:g} "
                f"kappa={kappa:g} seed={seed}",
                flush=True,
            )

            rec, top_series, status_series, norm_series, role_diag_rows = run_single(
                args=args,
                mode=args.mode,
                gamma=args.gamma,
                kappa=kappa,
                seed=seed,
            )

            all_records.append(rec)
            top_series_by_kappa[kappa].append(top_series)
            status_series_by_kappa[kappa].append(status_series)
            norm_series_by_kappa[kappa].append(norm_series)

            print(
                "    "
                f"leader={rec.leader_id}, "
                f"final_top_followers={rec.final_top_followers}, "
                f"tail_welfare={rec.tail_welfare:.4f}, "
                f"final_status_count={rec.final_status_count}, "
                f"final_norm_detected={rec.final_norm_detected}, "
                f"final_norm_mean_consensus={rec.final_norm_mean_consensus:.4f}",
                flush=True,
            )

            if role_diag_rows:
                summarize_role_update_diagnostics(
                    output_csv=role_diag_csv,
                    rows=role_diag_rows,
                    mode=args.mode,
                    gamma=args.gamma,
                    kappa=kappa,
                    seed=seed,
                )

    run_rows = [asdict(r) for r in all_records]
    write_csv(runs_csv, run_rows)

    agg_rows = aggregate(all_records)
    agg_dict_rows = [asdict(r) for r in agg_rows]
    write_csv(agg_csv, agg_dict_rows)

    print("-" * 72, flush=True)
    print(f"Wrote run-level CSV to: {runs_csv}", flush=True)
    print(f"Wrote aggregate CSV to: {agg_csv}", flush=True)
    if role_diag_csv.exists():
        print(f"Wrote role-update diagnostics CSV to: {role_diag_csv}", flush=True)

    filtered_agg_rows = [
        r for r in agg_rows
        if r.mode == args.mode and abs(r.gamma - args.gamma) < 1e-12
    ]

    if filtered_agg_rows:
        plot_top_followers_curve(
            mode=args.mode,
            gamma=args.gamma,
            aggregate_rows=filtered_agg_rows,
            output_file=top_png,
        )
        print(f"Wrote top-followers plot to: {top_png}", flush=True)

        plot_status_curve(
            mode=args.mode,
            gamma=args.gamma,
            aggregate_rows=filtered_agg_rows,
            output_file=status_png,
        )
        print(f"Wrote status-count plot to: {status_png}", flush=True)

        plot_welfare_curve(
            mode=args.mode,
            gamma=args.gamma,
            aggregate_rows=filtered_agg_rows,
            output_file=welfare_png,
        )
        print(f"Wrote welfare plot to: {welfare_png}", flush=True)

        plot_norm_curve(
            mode=args.mode,
            gamma=args.gamma,
            aggregate_rows=filtered_agg_rows,
            output_file=norm_png,
        )
        print(f"Wrote norm-consensus plot to: {norm_png}", flush=True)

        plot_progression_examples(
            mode=args.mode,
            gamma=args.gamma,
            kappas=kappas,
            top_series_by_kappa=top_series_by_kappa,
            status_series_by_kappa=status_series_by_kappa,
            norm_series_by_kappa=norm_series_by_kappa,
            output_file=prog_png,
            sample_interval=max(1, int(args.plot_sample_interval)),
            role_update_times=role_update_times,
        )
        print(f"Wrote progression plot to: {prog_png}", flush=True)

    print("#" * 72, flush=True)
    print("Done.", flush=True)
    print("#" * 72, flush=True)


if __name__ == "__main__":
    main()
           