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
from typing import Dict, List, Optional, Sequence, Tuple

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
    consensus_step: int


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


DetailedTrace = Dict[str, object]
AsyncDebugArtifacts = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reputation scaling gamma sweep harness.")
    parser.add_argument("--mode", choices=["static", "async"], required=True)
    parser.add_argument("--gammas", type=str, default="0,1,1.25,1.5,1.75,2,3,5")
    parser.add_argument("--kappas", type=str, default="0,1,1.25,1.5,1.75,2,3,5")
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--num-states", type=int, default=3)
    parser.add_argument("--num-actions", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="First seed (inclusive).")
    parser.add_argument(
        "--selected-seeds",
        type=str,
        default="",
        help="Optional explicit comma-separated seed list (e.g. \"2,7,9\"). Overrides --seeds/--seed-start.",
    )
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument(
        "--actor-rate-driver-mode",
        choices=["standard", "status_if_followers_kappa0"],
        default="standard",
        help="Actor-rate driver mode for Eq. (13): paper-faithful standard or experimental status override at kappa=0.",
    )
    parser.add_argument(
        "--actor-rate-status-override-min-followers",
        type=int,
        default=10,
        help="Follower-count threshold for the experimental status-driven actor-rate override.",
    )
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
        default=1,
        help="Downsample plotted time-series to every N steps (1 records every timestep).",
    )
    parser.add_argument(
        "--role-update-epochs",
        type=str,
        default="",
        help="Optional direct role-update epochs s_n as comma-separated positive ints "
             "(e.g., \"2000,3000,6000\"). Used when role_update_T_seq is empty.",
    )
    parser.add_argument("--tracking-mode", choices=["full", "light"], default="light")
    parser.add_argument(
        "--trace-detailed-seeds",
        choices=["none", "first", "all"],
        default="first",
        help="When tracking-mode=full, export per-agent PU/reputation traces for none, the first seed, or all seeds.",
    )
    parser.add_argument(
        "--small-n-trace-export",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For small-N debug runs only, export dense per-timestep reputation matrices in long-form CSVs.",
    )
    parser.add_argument(
        "--force-all-active-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug override: force all agents to be active as both actors and participants every step.",
    )
    parser.add_argument("--initial-actor-rate", type=float, default=0.2)
    parser.add_argument("--initial-participant-rate", type=float, default=0.2)
    parser.add_argument(
        "--reward-model",
        choices=["simple_preferred_action", "shared_base_gaussian"],
        default="simple_preferred_action",
        help="Reward model for payoff generation.",
    )
    parser.add_argument("--reward-base-mu", type=float, default=0.5)
    parser.add_argument("--reward-base-sigma", type=float, default=0.08)
    parser.add_argument("--reward-agent-sigma", type=float, default=0.1)
    parser.add_argument("--reward-clip-min", type=float, default=0.01)
    parser.add_argument("--reward-clip-max", type=float, default=2.5)
    parser.add_argument(
        "--eq9-averaging-mode",
        choices=["participants_only", "all_agents"],
        default="participants_only",
        help="Eq. (9) observed-reputation averaging set.",
    )
    parser.add_argument(
        "--leader-update-mode",
        choices=["participants_only_post_eq9", "all_agents_post_eq9", "participants_only_pre_eq9"],
        default="participants_only_post_eq9",
        help="Section 6.4.4 timing/scope for updating L_i(t+1).",
    )
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
    parser.add_argument(
        "--async-decision-audit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In async mode, write scheduler/decision audit CSVs and compact per-agent debug traces.",
    )
    parser.add_argument(
        "--role-update-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write lightweight role-update-only diagnostics for convergence/fragmentation analysis.",
    )
    return parser.parse_args()


def parse_gammas(gamma_text: str) -> List[float]:
    parts = [p.strip() for p in gamma_text.split(",") if p.strip()]
    return [float(x) for x in parts]

def parse_kappas(kappa_text: str) -> List[float]:
    parts = [p.strip() for p in kappa_text.split(",") if p.strip()]
    return [float(x) for x in parts]


def parse_selected_seeds(seed_text: str) -> List[int]:
    if not seed_text.strip():
        return []
    parts = [p.strip() for p in seed_text.split(",") if p.strip()]
    seeds = [int(x) for x in parts]
    return sorted(set(seed for seed in seeds if seed >= 0))


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


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    selected = parse_selected_seeds(getattr(args, "selected_seeds", ""))
    if selected:
        return selected
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))


def make_config(args: argparse.Namespace, gamma: float, kappa: float, mode: str) -> SystemConfig:
    # Keep defaults aligned with experiments/experiments.py unless explicitly overridden.
    role_interval = args.role_update_base_interval
    role_s0 = int(args.role_update_s0)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)
    eq9_averaging_mode = getattr(args, "eq9_averaging_mode", "participants_only")
    leader_update_mode = getattr(args, "leader_update_mode", "participants_only_post_eq9")
    actor_rate_driver_mode = getattr(args, "actor_rate_driver_mode", "standard")
    actor_rate_status_override_min_followers = int(
        getattr(args, "actor_rate_status_override_min_followers", 10)
    )
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
        actor_rate_driver_mode=actor_rate_driver_mode,
        actor_rate_status_override_min_followers=actor_rate_status_override_min_followers,
        gamma=gamma,
        kappa=kappa,
        c_threshold=0.1,
        B_R=0.3,
        B_F=1_000_000.0,  # Disable hysteresis in Experiment B family; only Experiment D uses it.
        delta=args.delta,
        eq9_averaging_mode=eq9_averaging_mode,
        leader_update_mode=leader_update_mode,
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
        force_all_active_debug=getattr(args, "force_all_active_debug", False),
        initial_actor_interaction_rate=args.initial_actor_rate,
        initial_participant_interaction_rate=args.initial_participant_rate,
        reward_model=args.reward_model,
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
            leaders[t] = int(candidates[0])  # deterministic tie-break
    return leaders


def _leader_switches(leader_series: np.ndarray) -> int:
    non_null = [x for x in leader_series.tolist() if x >= 0]
    if len(non_null) <= 1:
        return 0
    return sum(1 for a, b in zip(non_null[:-1], non_null[1:]) if a != b)


def _serialize_int_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=int).reshape(-1)
    return "|".join(str(int(v)) for v in arr.tolist())


def _extract_trace_bundle(results: Dict[str, object]) -> Optional[DetailedTrace]:
    pu_history = np.asarray(results.get("estimated_reward_pu_history", []), dtype=float)
    rep_reward_history = np.asarray(results.get("estimated_reward_rep_history", []), dtype=float)
    status_reward_history = np.asarray(results.get("estimated_reward_status_history", []), dtype=float)
    actor_rate_history = np.asarray(results.get("actor_interaction_rate_history", []), dtype=float)
    rep_history = np.asarray(results.get("weighted_selected_reputation_history", []), dtype=float)
    raw_rep_history = np.asarray(results.get("selected_reputation_history", []), dtype=float)
    highest_rep_history = np.asarray(results.get("highest_rep_agent_history", []), dtype=int)
    following_history = np.asarray(results.get("following_history", []), dtype=int)
    role_label_history = np.asarray(results.get("role_label_history", []), dtype=object)
    dense_reputation_history = np.asarray(results.get("dense_reputation_history", []), dtype=float)
    dense_personal_benefit_history = np.asarray(results.get("dense_personal_benefit_history", []), dtype=float)
    true_reputation_history = np.asarray(results.get("true_reputation_history", []), dtype=float)
    true_reputation_rank_history = np.asarray(results.get("true_reputation_rank_history", []), dtype=int)
    true_reputation_theta_history = np.asarray(results.get("true_reputation_theta_history", []), dtype=float)
    true_reputation_sum_expected_history = np.asarray(
        results.get("true_reputation_sum_expected_history", []),
        dtype=float,
    )
    follower_count_history = np.asarray(results.get("follower_counts", []), dtype=int)
    active_actor_ids_history = list(results.get("active_actor_ids_history", []))
    active_participant_ids_history = list(results.get("active_participant_ids_history", []))
    observed_utility_matrix_history = [
        np.asarray(x, dtype=float) if x is not None else None
        for x in results.get("observed_utility_matrix_history", [])
    ]
    eta_v_history = [float(x) for x in results.get("eta_v_history", [])]
    gossip_target_ids_history = list(results.get("gossip_target_ids_history", []))
    averaging_agent_ids_history = list(results.get("averaging_agent_ids_history", []))
    avg_s_by_target_history = list(results.get("avg_s_by_target_history", []))
    delta_v_matrix_history = [
        np.asarray(x, dtype=float)
        for x in results.get("delta_v_matrix_history", [])
    ]
    if pu_history.size == 0 or rep_history.size == 0:
        return None
    return {
        "estimated_reward_pu_history": pu_history,
        "estimated_reward_rep_history": rep_reward_history,
        "estimated_reward_status_history": status_reward_history,
        "actor_interaction_rate_history": actor_rate_history,
        "weighted_selected_reputation_history": rep_history,
        "selected_reputation_history": raw_rep_history,
        "highest_rep_agent_history": highest_rep_history,
        "following_history": following_history,
        "role_label_history": role_label_history,
        "dense_reputation_history": dense_reputation_history,
        "dense_personal_benefit_history": dense_personal_benefit_history,
        "true_reputation_history": true_reputation_history,
        "true_reputation_rank_history": true_reputation_rank_history,
        "true_reputation_theta_history": true_reputation_theta_history,
        "true_reputation_sum_expected_history": true_reputation_sum_expected_history,
        "follower_count_history": follower_count_history,
        "role_update_times": np.asarray(results.get("role_update_times", []), dtype=int),
        "active_actor_ids_history": active_actor_ids_history,
        "active_participant_ids_history": active_participant_ids_history,
        "observed_utility_matrix_history": observed_utility_matrix_history,
        "eta_v_history": eta_v_history,
        "gossip_target_ids_history": gossip_target_ids_history,
        "averaging_agent_ids_history": averaging_agent_ids_history,
        "avg_s_by_target_history": avg_s_by_target_history,
        "delta_v_matrix_history": delta_v_matrix_history,
    }


def _time_to_threshold(series: np.ndarray, threshold: int) -> int:
    idx = np.where(series >= threshold)[0]
    return int(idx[0] + 1) if idx.size > 0 else -1


def _select_async_focus_agents(
    system: MultiAgentSystem,
    trace: DetailedTrace,
) -> List[Dict[str, object]]:
    n_agents = system.config.num_agents
    follower_counts = [len(a.state.followers) for a in system.agents]
    ranked = sorted(range(n_agents), key=lambda i: (follower_counts[i], -i), reverse=True)
    role_labels = np.asarray(trace["role_label_history"], dtype=object)
    highest_rep_history = np.asarray(trace["highest_rep_agent_history"], dtype=int)
    following_history = np.asarray(trace["following_history"], dtype=int)
    final_roles = role_labels[-1].tolist() if role_labels.size > 0 else [a.state.role.value for a in system.agents]
    final_highest = highest_rep_history[-1].tolist() if highest_rep_history.size > 0 else [-1] * n_agents
    final_following = following_history[-1].tolist() if following_history.size > 0 else [
        -1 if a.state.following is None else int(a.state.following) for a in system.agents
    ]

    focus_rows: List[Dict[str, object]] = []
    seen = set()

    def add(agent_id: Optional[int], reason: str):
        if agent_id is None:
            return
        agent_id = int(agent_id)
        if agent_id < 0 or agent_id >= n_agents or agent_id in seen:
            return
        seen.add(agent_id)
        focus_rows.append({"agent_id": agent_id, "focus_reason": reason})

    top_leader = ranked[0] if ranked and follower_counts[ranked[0]] > 0 else None
    second_leader = next((i for i in ranked[1:] if follower_counts[i] > 0), None)
    add(top_leader, "top_leader")
    add(second_leader, "second_leader")

    if top_leader is not None:
        for follower_id in sorted(system.agents[top_leader].state.followers)[:3]:
            add(int(follower_id), "top_leader_follower")

    for agent_id, role_label in enumerate(final_roles):
        if str(role_label) == "personal_utility":
            add(agent_id, "final_pu")
        if len(focus_rows) >= 8:
            break

    mismatch_agents = [
        i for i in range(n_agents)
        if int(final_following[i]) >= 0 and int(final_highest[i]) != int(final_following[i])
    ]
    for agent_id in mismatch_agents[:2]:
        add(agent_id, "highest_follow_mismatch")

    for agent_id in ranked:
        if len(focus_rows) >= 10:
            break
        add(agent_id, "filler")

    return focus_rows


def write_async_scheduler_csv(rows: Sequence[dict], output_file: Path) -> None:
    write_csv(output_file, rows)


def write_async_focus_trace_csv(
    trace: DetailedTrace,
    audit_rows: Sequence[dict],
    focus_agents: Sequence[dict],
    output_file: Path,
) -> None:
    pu_history = np.asarray(trace["estimated_reward_pu_history"], dtype=float)
    rep_history = np.asarray(trace["weighted_selected_reputation_history"], dtype=float)
    raw_rep_history = np.asarray(trace["selected_reputation_history"], dtype=float)
    highest_rep_history = np.asarray(trace["highest_rep_agent_history"], dtype=int)
    following_history = np.asarray(trace["following_history"], dtype=int)
    role_label_history = np.asarray(trace["role_label_history"], dtype=object)
    role_update_times = set(int(t) for t in np.asarray(trace["role_update_times"], dtype=int).tolist())

    audit_map = {
        (int(row["t"]), int(row["agent_id"])): row
        for row in audit_rows
    }

    n_steps, _ = pu_history.shape
    rows = []
    for t_idx in range(n_steps):
        step = t_idx + 1
        for focus in focus_agents:
            agent_id = int(focus["agent_id"])
            audit = audit_map.get((step, agent_id))
            rows.append(
                {
                    "t": step,
                    "role_update_step": int(step in role_update_times),
                    "agent_id": agent_id,
                    "focus_reason": str(focus["focus_reason"]),
                    "role": str(role_label_history[t_idx, agent_id]),
                    "following": int(following_history[t_idx, agent_id]),
                    "highest_rep_agent": int(highest_rep_history[t_idx, agent_id]),
                    "selected_rep_raw": float(raw_rep_history[t_idx, agent_id]),
                    "selected_rep_weighted": float(rep_history[t_idx, agent_id]),
                    "estimated_reward_pu": float(pu_history[t_idx, agent_id]),
                    "effective_threshold": "" if audit is None or audit["effective_threshold"] is None else float(audit["effective_threshold"]),
                    "decision_code": "NO_UPDATE" if audit is None else str(audit["decision_code"]),
                    "scheduled_for_update": 0 if audit is None else 1,
                    "opinion_leader_count": "" if audit is None else int(audit["opinion_leader_count"]),
                    "hysteresis_active": "" if audit is None else int(bool(audit["hysteresis_active"])),
                    "redirect_applied": "" if audit is None else int(bool(audit["redirect_applied"])),
                }
            )

    write_csv(output_file, rows)


def diagnose_async_run(
    *,
    args: argparse.Namespace,
    record: RunRecord,
    system: MultiAgentSystem,
    decision_audit_rows: Sequence[dict],
    scheduler_rows: Sequence[dict],
    update_counts: np.ndarray,
) -> Dict[str, object]:
    n_agents = int(args.num_agents)
    expected_updates = float(args.num_steps) / max(1.0, float(args.role_update_base_interval))
    mean_updates = float(np.mean(update_counts)) if update_counts.size > 0 else 0.0
    min_updates = int(np.min(update_counts)) if update_counts.size > 0 else 0
    max_updates = int(np.max(update_counts)) if update_counts.size > 0 else 0
    zero_update_agents = int(np.sum(update_counts == 0)) if update_counts.size > 0 else 0

    step1_rows = [row for row in decision_audit_rows if bool(row.get("in_C", False))]
    threshold_fail_rows = [row for row in step1_rows if row.get("decision_code") == "STAY_PU_REP_BELOW_THRESHOLD"]
    follow_rows = [row for row in step1_rows if str(row.get("decision_code", "")).startswith("FOLLOW_")]
    redirect_rows = [row for row in follow_rows if bool(row.get("redirect_applied", False))]
    root_lock_rows = [row for row in decision_audit_rows if (not bool(row.get("in_C", False))) and bool(row.get("has_followers", False))]
    hysteresis_violations = [
        row for row in decision_audit_rows
        if bool(row.get("hysteresis_active", False)) and int(row.get("opinion_leader_count", 0)) > 1
    ]
    better_candidate_rows = [
        row for row in step1_rows
        if int(row.get("following_before", -1)) >= 0
        and float(row.get("selected_reputation_raw", 0.0)) > float(row.get("current_followed_reputation_raw", 0.0))
    ]
    better_candidate_switch_rows = [
        row for row in better_candidate_rows
        if row.get("decision_code") in {"FOLLOW_DIRECT", "FOLLOW_REDIRECT"}
        and int(row.get("following_after", -1)) != int(row.get("following_before", -1))
    ]

    highest_targets = [int(row["highest_rep_agent_estimate"]) for row in step1_rows if int(row["highest_rep_agent_estimate"]) >= 0]
    if highest_targets:
        counts = np.bincount(np.asarray(highest_targets, dtype=int), minlength=n_agents)
        distinct_highest_targets = int(np.sum(counts > 0))
        max_highest_target_share = float(np.max(counts) / max(1, len(highest_targets)))
    else:
        distinct_highest_targets = 0
        max_highest_target_share = 0.0

    final_root_count = int(sum(1 for a in system.agents if len(a.state.followers) > 0))

    bucket = "STEP1-THRESHOLD"
    if zero_update_agents > 0 or mean_updates < 0.5 * expected_updates:
        bucket = "ASYNC-SCHEDULING"
    elif final_root_count > 1 and len(root_lock_rows) >= max(len(threshold_fail_rows), len(redirect_rows)):
        bucket = "ROOT-LOCKING"
    elif len(redirect_rows) > 0 and final_root_count > 1 and len(redirect_rows) >= len(threshold_fail_rows):
        bucket = "REDIRECT-STICKINESS"
    elif distinct_highest_targets > max(10, n_agents // 5) and max_highest_target_share < 0.2:
        bucket = "NOISY-ARGMAX"
    elif len(threshold_fail_rows) >= max(len(root_lock_rows), len(redirect_rows)):
        bucket = "STEP1-THRESHOLD"

    if bucket == "ASYNC-SCHEDULING":
        diagnosis_text = (
            f"main reason followers do not reach N-1 is ASYNC-SCHEDULING: "
            f"mean updates/agent={mean_updates:.2f} versus expected ≈{expected_updates:.2f}, "
            f"with {zero_update_agents} agents never reevaluating."
        )
    elif bucket == "ROOT-LOCKING":
        diagnosis_text = (
            f"main reason followers do not reach N-1 is ROOT-LOCKING: "
            f"{final_root_count} root leaders finish with followers, and {len(root_lock_rows)} scheduled updates "
            f"were on agents blocked from Step 1 because they already had followers."
        )
    elif bucket == "REDIRECT-STICKINESS":
        diagnosis_text = (
            f"main reason followers do not reach N-1 is REDIRECT-STICKINESS: "
            f"{len(redirect_rows)} Step-1 follow decisions redirected through existing follower chains, "
            f"which keeps multiple roots alive."
        )
    elif bucket == "NOISY-ARGMAX":
        diagnosis_text = (
            f"main reason followers do not reach N-1 is NOISY-ARGMAX: "
            f"highest-reputation choices were split across {distinct_highest_targets} targets and "
            f"the largest target share was only {max_highest_target_share:.3f}."
        )
    else:
        diagnosis_text = (
            f"main reason followers do not reach N-1 is STEP1-THRESHOLD: "
            f"{len(threshold_fail_rows)} candidate updates failed because gamma*rep did not beat max(B_i, J_pu), "
            f"while only {len(better_candidate_switch_rows)}/{len(better_candidate_rows)} better-candidate opportunities switched."
        )

    return {
        "mode": str(record.mode),
        "gamma": float(record.gamma),
        "seed": int(record.seed),
        "diagnosis_bucket": bucket,
        "diagnosis_text": diagnosis_text,
        "mean_updates_per_agent": mean_updates,
        "min_updates_per_agent": min_updates,
        "max_updates_per_agent": max_updates,
        "zero_update_agents": zero_update_agents,
        "expected_updates_per_agent": expected_updates,
        "step1_candidate_rows": int(len(step1_rows)),
        "threshold_fail_rows": int(len(threshold_fail_rows)),
        "follow_rows": int(len(follow_rows)),
        "redirect_rows": int(len(redirect_rows)),
        "root_lock_rows": int(len(root_lock_rows)),
        "better_candidate_rows": int(len(better_candidate_rows)),
        "better_candidate_switch_rows": int(len(better_candidate_switch_rows)),
        "hysteresis_violations": int(len(hysteresis_violations)),
        "distinct_highest_targets": int(distinct_highest_targets),
        "max_highest_target_share": float(max_highest_target_share),
        "final_root_count": int(final_root_count),
        "leader_switches": int(record.leader_switches),
        "final_top_followers": int(record.final_top_followers),
        "time_to_90pct_followers": int(record.time_to_90pct_followers),
    }


def write_async_diagnosis_markdown(
    *,
    output_file: Path,
    diagnosis: Dict[str, object],
    focus_agents: Sequence[dict],
) -> None:
    lines = [
        "# Async Debug Diagnosis",
        "",
        f"- gamma: {diagnosis['gamma']}",
        f"- seed: {diagnosis['seed']}",
        f"- bucket: {diagnosis['diagnosis_bucket']}",
        f"- summary: {diagnosis['diagnosis_text']}",
        "",
        "## Key checks",
        f"- updates per agent: mean={diagnosis['mean_updates_per_agent']:.2f}, min={diagnosis['min_updates_per_agent']}, max={diagnosis['max_updates_per_agent']}, zero={diagnosis['zero_update_agents']}",
        f"- threshold failures: {diagnosis['threshold_fail_rows']} / {diagnosis['step1_candidate_rows']} Step-1 candidate updates",
        f"- redirect follows: {diagnosis['redirect_rows']} / {diagnosis['follow_rows']} follow decisions",
        f"- root-lock rows: {diagnosis['root_lock_rows']}",
        f"- hysteresis violations with multiple leaders: {diagnosis['hysteresis_violations']}",
        f"- better-candidate switches: {diagnosis['better_candidate_switch_rows']} / {diagnosis['better_candidate_rows']}",
        f"- distinct highest-reputation targets: {diagnosis['distinct_highest_targets']} (max share {diagnosis['max_highest_target_share']:.3f})",
        f"- final root leaders with followers: {diagnosis['final_root_count']}",
        "",
        "## Focus agents",
    ]
    for row in focus_agents:
        lines.append(f"- agent {int(row['agent_id'])}: {row['focus_reason']}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines) + "\n")


def run_single(
    args: argparse.Namespace,
    mode: str,
    gamma: float,
    kappa: float,
    seed: int,
) -> Tuple[
    RunRecord,
    np.ndarray,
    np.ndarray,
    Optional[DetailedTrace],
    Optional[AsyncDebugArtifacts],
    Optional[List[dict]],
    Optional[Dict[str, List[dict]]],
]:
    np.random.seed(seed)
    config = make_config(args, gamma=gamma, kappa=kappa, mode=mode)
    system = MultiAgentSystem(config)
    async_debug_enabled = bool(mode == "async" and args.async_decision_audit)
    if async_debug_enabled:
        system.enable_async_decision_audit()
    small_n_trace_export_enabled = bool(
        getattr(args, "small_n_trace_export", False) and int(args.num_agents) <= 12
    )
    if small_n_trace_export_enabled:
        system.enable_small_n_trace_export()
    role_update_diagnostics_enabled = bool(mode == "static" and getattr(args, "role_update_diagnostics", False))
    if role_update_diagnostics_enabled:
        system.enable_role_update_diagnostics()

    scheduler_rows: List[Dict[str, object]] = []
    update_counts = np.zeros(args.num_agents, dtype=int)

    consensus_step = None

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
                    role_timers_before = role_timers.copy()
                    role_timers -= 1
                    update_ids = np.where(role_timers <= 0)[0]
                    if update_ids.size > 0:
                        update_list = update_ids.tolist()
                        system._update_roles_sequential(update_list)
                        system.refresh_last_tracked_state()
                        system.results.setdefault("role_update_times", []).append(int(system.time_step))
                        update_counts[update_ids] += 1

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
                    if async_debug_enabled:
                        scheduler_rows.append(
                            {
                                "t": int(system.time_step),
                                "update_ids": _serialize_int_array(update_ids),
                                "update_count": int(update_ids.size),
                                "role_timers_before_decrement": _serialize_int_array(role_timers_before),
                                "role_timers_after_reset": _serialize_int_array(role_timers),
                                "interval_indices": _serialize_int_array(interval_indices),
                            }
                        )
                else:
                    update_mask = np.random.random(args.num_agents) < async_update_prob
                    update_ids = np.where(update_mask)[0]
                    if update_ids.size > 0:
                        system._update_roles_sequential(update_ids.tolist())
                        system.refresh_last_tracked_state()
                        # Track async update events for diagnostics.
                        system.role_update_epoch += 1
                        system.results.setdefault("role_update_times", []).append(int(system.time_step))
                        update_counts[update_ids] += 1
                    if async_debug_enabled:
                        scheduler_rows.append(
                            {
                                "t": int(system.time_step),
                                "update_ids": _serialize_int_array(update_ids),
                                "update_count": int(update_ids.size),
                                "role_timers_before_decrement": "",
                                "role_timers_after_reset": "",
                                "interval_indices": "",
                            }
                        )
                # check if consensus reached
                followers_list = [len(a.state.followers) for a in system.agents]
                if not consensus_step and (len(system.agents) * 0.95 <= max(followers_list)):
                    consensus_step = _
            results = _finalize_results(system)
            results["consensus_step"] = consensus_step
    else:
        with redirect_stdout(io.StringIO()):
            for _ in range(args.num_steps):
                system.step()
                # check if consensus reached
                followers_list = [len(a.state.followers) for a in system.agents]
                if not consensus_step and (len(system.agents) * 0.95 <= max(followers_list)):
                    consensus_step = _
            results = _finalize_results(system)
            results["consensus_step"] = consensus_step

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
        consensus_step=results["consensus_step"]
    )
    detailed_trace: Optional[DetailedTrace] = None
    if str(args.tracking_mode).lower() == "full" or small_n_trace_export_enabled:
        detailed_trace = _extract_trace_bundle(results)
        if detailed_trace is not None:
            detailed_trace["B_R"] = float(system.config.B_R)
            detailed_trace["B_F"] = float(system.config.B_F)
            detailed_trace["delta"] = float(system.config.delta)

    async_debug: Optional[AsyncDebugArtifacts] = None
    if async_debug_enabled:
        trace_bundle = _extract_trace_bundle(results)
        decision_audit_rows = system.get_async_decision_audit_rows()
        focus_agents = _select_async_focus_agents(system, trace_bundle) if trace_bundle is not None else []
        diagnosis = diagnose_async_run(
            args=args,
            record=record,
            system=system,
            decision_audit_rows=decision_audit_rows,
            scheduler_rows=scheduler_rows,
            update_counts=update_counts,
        )
        async_debug = {
            "scheduler_rows": scheduler_rows,
            "decision_audit_rows": decision_audit_rows,
            "focus_agents": focus_agents,
            "diagnosis": diagnosis,
            "trace_bundle": trace_bundle,
        }
    role_update_diagnostics = system.get_role_update_diagnostic_rows() if role_update_diagnostics_enabled else None
    checkpoint_audit_rows: Optional[Dict[str, List[dict]]] = None
    if role_update_diagnostics_enabled:
        checkpoint_audit_rows = {
            "true_reputation_checkpoints": list(system.get_true_reputation_checkpoint_rows()),
            "estimate_consensus_checkpoints": list(system.get_estimate_consensus_checkpoint_rows()),
            "rate_audit_checkpoints": list(system.get_rate_audit_checkpoint_rows()),
        }
        final_bundle = system.build_expb_checkpoint_audit_bundle(
            checkpoint_kind="final",
            role_update_index=len(results.get("role_update_times", [])),
        )
        for key, final_rows in final_bundle.items():
            checkpoint_audit_rows.setdefault(key, []).extend(final_rows)

    return (
        record,
        top_follower_series,
        leader_series,
        detailed_trace,
        async_debug,
        role_update_diagnostics,
        checkpoint_audit_rows,
    )


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


def enrich_checkpoint_rows(
    *,
    mode: str,
    gamma: float,
    seed: int,
    rows: Sequence[dict],
) -> List[dict]:
    enriched: List[dict] = []
    for row in rows:
        out = dict(row)
        out["mode"] = str(mode)
        out["gamma"] = float(gamma)
        out["seed"] = int(seed)
        enriched.append(out)
    return enriched


def _mode_with_share(values: Sequence[int]) -> Tuple[int, float]:
    if not values:
        return -1, 0.0
    counts: Dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    mode_value = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    share = float(counts[mode_value] / max(1, len(values)))
    return int(mode_value), share


def summarize_rank_alignment_checkpoints(
    *,
    true_rows: Sequence[dict],
    estimate_rows: Sequence[dict],
) -> List[dict]:
    grouped_true: Dict[Tuple[str, float, int, int, str, int], List[dict]] = {}
    grouped_estimate: Dict[Tuple[str, float, int, int, str, int], List[dict]] = {}

    def _group_key(row: dict) -> Tuple[str, float, int, int, str, int]:
        return (
            str(row["mode"]),
            float(row["gamma"]),
            int(row["seed"]),
            int(row["t"]),
            str(row["checkpoint_kind"]),
            int(row["role_update_index"]),
        )

    for row in true_rows:
        grouped_true.setdefault(_group_key(row), []).append(row)
    for row in estimate_rows:
        grouped_estimate.setdefault(_group_key(row), []).append(row)

    out_rows: List[dict] = []
    for key in sorted(set(grouped_true.keys()) & set(grouped_estimate.keys())):
        true_group = grouped_true[key]
        estimate_group = grouped_estimate[key]
        mode, gamma, seed, t, checkpoint_kind, role_update_index = key

        unique_true_candidates = sorted(
            {
                int(row["unique_true_top_agent"])
                for row in true_group
                if int(row.get("true_top_unique", 0)) == 1 and int(row["unique_true_top_agent"]) >= 0
            }
        )
        unique_true_top_agent = unique_true_candidates[0] if len(unique_true_candidates) == 1 else -1
        top_estimate_agents = [int(row["top_estimate_agent"]) for row in estimate_group if int(row["top_estimate_agent"]) >= 0]
        selected_target_agents = [
            int(row["highest_rep_agent_estimate"])
            for row in estimate_group
            if int(row["highest_rep_agent_estimate"]) >= 0
        ]
        current_root_leaders = [
            int(row["current_root_leader"])
            for row in estimate_group
            if int(row["current_root_leader"]) >= 0
        ]
        top_estimate_mode_agent, top_estimate_mode_share = _mode_with_share(top_estimate_agents)
        selected_target_mode_agent, selected_target_mode_share = _mode_with_share(selected_target_agents)

        top_estimate_match_share = 0.0
        selected_match_share = 0.0
        if unique_true_top_agent >= 0 and estimate_group:
            top_estimate_match_share = float(
                np.mean(
                    np.asarray(
                        [int(row["top_estimate_agent"]) == unique_true_top_agent for row in estimate_group],
                        dtype=float,
                    )
                )
            )
            selected_match_share = float(
                np.mean(
                    np.asarray(
                        [int(row["highest_rep_agent_estimate"]) == unique_true_top_agent for row in estimate_group],
                        dtype=float,
                    )
                )
            )

        out_rows.append(
            {
                "mode": str(mode),
                "gamma": float(gamma),
                "seed": int(seed),
                "t": int(t),
                "checkpoint_kind": str(checkpoint_kind),
                "role_update_index": int(role_update_index),
                "eq9_averaging_mode": str(
                    estimate_group[0].get("eq9_averaging_mode", true_group[0].get("eq9_averaging_mode", "unknown"))
                ),
                "leader_update_mode": str(
                    estimate_group[0].get("leader_update_mode", true_group[0].get("leader_update_mode", "unknown"))
                ),
                "true_top_unique": int(unique_true_top_agent >= 0),
                "unique_true_top_agent": int(unique_true_top_agent),
                "top_estimate_mode_agent": int(top_estimate_mode_agent),
                "top_estimate_mode_share": float(top_estimate_mode_share),
                "selected_target_mode_agent": int(selected_target_mode_agent),
                "selected_target_mode_share": float(selected_target_mode_share),
                "top_estimate_matches_true_top_share": float(top_estimate_match_share),
                "selected_matches_true_top_share": float(selected_match_share),
                "candidate_count_mean": float(
                    np.mean([float(row["candidate_count_within_delta"]) for row in estimate_group])
                    if estimate_group
                    else 0.0
                ),
                "candidate_count_max": int(
                    max((int(row["candidate_count_within_delta"]) for row in estimate_group), default=0)
                ),
                "mean_gap_top2": float(
                    np.mean([float(row["gap_top2"]) for row in estimate_group]) if estimate_group else 0.0
                ),
                "distinct_top_estimate_agents": int(len(set(top_estimate_agents))),
                "distinct_selected_targets": int(len(set(selected_target_agents))),
                "distinct_root_count": int(len(set(current_root_leaders))),
            }
        )
    return out_rows


def build_paper_gossip_scope(highest_rep_agents: Sequence[int]) -> List[int]:
    return sorted({int(agent_id) for agent_id in highest_rep_agents if int(agent_id) >= 0})


def characterize_changed_gossip_columns(
    rep_before: np.ndarray,
    rep_after: np.ndarray,
    paper_scope: Sequence[int],
) -> dict:
    before = np.asarray(rep_before, dtype=float)
    after = np.asarray(rep_after, dtype=float)
    if before.shape != after.shape:
        raise ValueError("rep_before and rep_after must have the same shape.")

    changed_mask = np.any(~np.isclose(after, before, atol=1e-12, rtol=0.0), axis=0)
    changed_columns = np.where(changed_mask)[0].astype(int).tolist()
    paper_scope_set = set(int(col) for col in paper_scope)
    off_scope_columns = [int(col) for col in changed_columns if int(col) not in paper_scope_set]
    return {
        "paper_scope_columns": [int(col) for col in sorted(paper_scope_set)],
        "changed_columns": changed_columns,
        "off_scope_changed_columns": off_scope_columns,
        "implementation_updates_only_paper_scope": int(len(off_scope_columns) == 0),
    }


def enrich_role_update_diagnostic_rows(
    *,
    mode: str,
    gamma: float,
    seed: int,
    rows: Sequence[dict],
) -> List[dict]:
    enriched: List[dict] = []
    prev_leader = None
    prev_top_followers = None
    for row in rows:
        out = dict(row)
        out["mode"] = str(mode)
        out["gamma"] = float(gamma)
        out["seed"] = int(seed)
        if prev_leader is None:
            out["leader_changed_since_prev_update"] = 0
            out["delta_top_followers"] = int(out["top_followers"])
        else:
            out["leader_changed_since_prev_update"] = int(int(out["top_leader_id"]) != int(prev_leader))
            out["delta_top_followers"] = int(int(out["top_followers"]) - int(prev_top_followers))
        prev_leader = int(out["top_leader_id"])
        prev_top_followers = int(out["top_followers"])
        enriched.append(out)
    return enriched


def classify_seed_failure_bucket(
    *,
    num_agents: int,
    final_top_followers: int,
    final_share_gate_margin_positive: float,
    final_top_highest_rep_target_share: float,
    final_second_followers: int,
    leader_switches_role_update_only: int,
) -> str:
    threshold_90 = int(np.ceil(0.90 * (num_agents - 1)))
    if final_top_followers >= threshold_90:
        return "full_convergence"
    if final_share_gate_margin_positive < 0.5:
        return "weak_following"
    if (
        leader_switches_role_update_only > 0
        or final_top_highest_rep_target_share < 0.5
        or final_second_followers >= max(10, int(np.ceil(0.5 * max(1, final_top_followers))))
    ):
        return "fragmented_following"
    return "stable_partial_convergence"


def summarize_role_update_diagnostics(
    *,
    mode: str,
    gamma: float,
    seed: int,
    num_agents: int,
    record: RunRecord,
    top_follower_series: np.ndarray,
    rows: Sequence[dict],
) -> dict:
    if not rows:
        return {
            "mode": str(mode),
            "gamma": float(gamma),
            "seed": int(seed),
            "final_top_followers": int(record.final_top_followers),
            "max_top_followers": int(np.max(top_follower_series)) if top_follower_series.size else 0,
            "final_second_followers": 0,
            "time_to_50pct_followers": -1,
            "time_to_75pct_followers": -1,
            "time_to_90pct_followers": int(record.time_to_90pct_followers),
            "leader_switches_role_update_only": 0,
            "mean_share_gate_margin_positive": 0.0,
            "final_share_gate_margin_positive": 0.0,
            "mean_top_highest_rep_target_share": 0.0,
            "final_top_highest_rep_target_share": 0.0,
            "mean_top_follower_share": 0.0,
            "final_top_follower_share": 0.0,
            "failure_bucket": "weak_following",
        }

    threshold_50 = int(np.ceil(0.50 * (num_agents - 1)))
    threshold_75 = int(np.ceil(0.75 * (num_agents - 1)))
    top_leader_series = np.array([int(row["top_leader_id"]) for row in rows], dtype=int)
    summary = {
        "mode": str(mode),
        "gamma": float(gamma),
        "seed": int(seed),
        "final_top_followers": int(record.final_top_followers),
        "max_top_followers": int(np.max(top_follower_series)) if top_follower_series.size else 0,
        "final_second_followers": int(rows[-1]["second_followers"]),
        "time_to_50pct_followers": int(_time_to_threshold(top_follower_series, threshold_50)),
        "time_to_75pct_followers": int(_time_to_threshold(top_follower_series, threshold_75)),
        "time_to_90pct_followers": int(record.time_to_90pct_followers),
        "leader_switches_role_update_only": int(_leader_switches(top_leader_series)),
        "mean_share_gate_margin_positive": float(np.mean([float(r["share_gate_margin_positive"]) for r in rows])),
        "final_share_gate_margin_positive": float(rows[-1]["share_gate_margin_positive"]),
        "mean_top_highest_rep_target_share": float(np.mean([float(r["top_highest_rep_target_share"]) for r in rows])),
        "final_top_highest_rep_target_share": float(rows[-1]["top_highest_rep_target_share"]),
        "mean_top_follower_share": float(np.mean([float(r["top_follower_share"]) for r in rows])),
        "final_top_follower_share": float(rows[-1]["top_follower_share"]),
    }
    summary["failure_bucket"] = classify_seed_failure_bucket(
        num_agents=num_agents,
        final_top_followers=int(summary["final_top_followers"]),
        final_share_gate_margin_positive=float(summary["final_share_gate_margin_positive"]),
        final_top_highest_rep_target_share=float(summary["final_top_highest_rep_target_share"]),
        final_second_followers=int(summary["final_second_followers"]),
        leader_switches_role_update_only=int(summary["leader_switches_role_update_only"]),
    )
    return summary


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


def plot_progression(
    mode: str,
    gammas: Sequence[float],
    top_series_by_gamma: Dict[float, List[np.ndarray]],
    output_file: Path,
    sample_interval: int,
    role_update_times: Sequence[int],
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
        is_single_run = stack.shape[0] == 1
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
        if not is_single_run:
            ax.fill_between(x, mean - std, mean + std, alpha=0.25)
        for idx_line, step in enumerate(role_update_times):
            ax.axvline(
                int(step),
                color="gray",
                linestyle="--",
                linewidth=0.8,
                alpha=0.22,
                label="Role update" if idx == 0 and idx_line == 0 else None,
            )
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
    is_single_run = all(int(r.n_runs) == 1 for r in rows)

    plt.figure(figsize=(7.5, 4.5))
    if is_single_run:
        plt.plot(gammas, means, "-o", linewidth=1.8)
    else:
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


def write_agent_trace_csv(
    trace: DetailedTrace,
    output_file: Path,
) -> None:
    pu_history = np.asarray(trace["estimated_reward_pu_history"], dtype=float)
    rep_history = np.asarray(trace["weighted_selected_reputation_history"], dtype=float)
    raw_rep_history = np.asarray(trace["selected_reputation_history"], dtype=float)
    highest_rep_history = np.asarray(trace["highest_rep_agent_history"], dtype=int)
    following_history = np.asarray(trace["following_history"], dtype=int)
    role_update_times = set(int(t) for t in np.asarray(trace["role_update_times"], dtype=int).tolist())

    n_steps, n_agents = pu_history.shape
    rows = []
    for t in range(n_steps):
        row = {
            "t": t + 1,
            "role_update_step": int((t + 1) in role_update_times),
        }
        for agent_id in range(n_agents):
            row[f"agent_{agent_id}_pu"] = float(pu_history[t, agent_id])
            row[f"agent_{agent_id}_rep_weighted"] = float(rep_history[t, agent_id])
            row[f"agent_{agent_id}_rep_raw"] = float(raw_rep_history[t, agent_id])
            row[f"agent_{agent_id}_highest_rep_agent"] = int(highest_rep_history[t, agent_id])
            row[f"agent_{agent_id}_following"] = int(following_history[t, agent_id])
        rows.append(row)

    write_csv(output_file, rows)


def write_small_n_reputation_trace_long_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        dense_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        if dense_history.size == 0:
            continue
        n_steps, n_observers, n_targets = dense_history.shape
        for t in range(n_steps):
            for observer_id in range(n_observers):
                for target_id in range(n_targets):
                    rows.append(
                        {
                            "t": int(t + 1),
                            "seed": int(seed),
                            "gamma": float(gamma),
                            "observer_id": int(observer_id),
                            "target_id": int(target_id),
                            "reputation_estimate": float(dense_history[t, observer_id, target_id]),
                        }
                    )
    write_csv(output_file, rows)


def write_small_n_agent_state_trace_long_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        pu_history = np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float)
        rep_history = np.asarray(trace.get("estimated_reward_rep_history", []), dtype=float)
        status_history = np.asarray(trace.get("estimated_reward_status_history", []), dtype=float)
        actor_rate_history = np.asarray(trace.get("actor_interaction_rate_history", []), dtype=float)
        highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
        following_history = np.asarray(trace.get("following_history", []), dtype=int)
        role_label_history = np.asarray(trace.get("role_label_history", []), dtype=object)
        if pu_history.size == 0:
            continue
        n_steps, n_agents = pu_history.shape
        for t in range(n_steps):
            for agent_id in range(n_agents):
                rows.append(
                    {
                        "t": int(t + 1),
                        "seed": int(seed),
                        "gamma": float(gamma),
                        "agent_id": int(agent_id),
                        "estimated_reward_pu": float(pu_history[t, agent_id]),
                        "estimated_reward_rep": float(rep_history[t, agent_id]) if rep_history.size else 0.0,
                        "estimated_reward_status": float(status_history[t, agent_id]) if status_history.size else 0.0,
                        "highest_rep_agent_estimate": int(highest_rep_history[t, agent_id]) if highest_rep_history.size else -1,
                        "following": int(following_history[t, agent_id]) if following_history.size else -1,
                        "role": str(role_label_history[t, agent_id]) if role_label_history.size else "",
                        "actor_interaction_rate": float(actor_rate_history[t, agent_id]) if actor_rate_history.size else 0.0,
                    }
                )
    write_csv(output_file, rows)


def _rank_desc(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    ranked = sorted(range(arr.size), key=lambda idx: (-float(arr[idx]), int(idx)))
    out = np.zeros(arr.size, dtype=int)
    for pos, agent_id in enumerate(ranked, start=1):
        out[int(agent_id)] = int(pos)
    return out


def _top_agent_id(values: np.ndarray) -> int:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return -1
    return int(max(range(arr.size), key=lambda idx: (float(arr[idx]), -int(idx))))


def _mode_int(values: Sequence[int]) -> int:
    counts: Dict[int, int] = {}
    for value in values:
        counts[int(value)] = counts.get(int(value), 0) + 1
    if not counts:
        return -1
    return int(max(counts.items(), key=lambda kv: (int(kv[1]), -int(kv[0])))[0])


def _safe_pearson(values_a: np.ndarray, values_b: np.ndarray) -> float:
    a = np.asarray(values_a, dtype=float).reshape(-1)
    b = np.asarray(values_b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    a_centered = a - float(np.mean(a))
    b_centered = b - float(np.mean(b))
    denom = float(np.linalg.norm(a_centered) * np.linalg.norm(b_centered))
    if denom <= 1e-15:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denom)


def _dominant_alignment_target(
    *,
    corr_true_reputation: float,
    corr_sum_expected_utility: float,
    corr_theta_mu: float,
    corr_mean_incoming_v: float,
) -> str:
    candidates = {
        "true_reputation": float(corr_true_reputation),
        "sum_expected_utility": float(corr_sum_expected_utility),
        "theta_mu": float(corr_theta_mu),
        "mean_incoming_v": float(corr_mean_incoming_v),
    }
    filtered = {
        label: value
        for label, value in candidates.items()
        if np.isfinite(value)
    }
    if not filtered:
        return "other_or_none"
    best_label, best_value = max(filtered.items(), key=lambda kv: (float(kv[1]), kv[0]))
    if float(best_value) <= 0.0:
        return "other_or_none"
    return str(best_label)


def _classify_toy_failure_stage(
    *,
    true_top_agent: int,
    observed_top_agent: int,
    modal_highest_rep_agent_estimate: int,
    modal_selected_target: int,
    dominant_alignment_target: str,
    share_step1_margin_positive: float,
    top_followers: int,
) -> str:
    if int(true_top_agent) != int(observed_top_agent) or str(dominant_alignment_target) != "true_reputation":
        return "learning_target_mismatch"
    if int(modal_highest_rep_agent_estimate) != int(observed_top_agent) or int(modal_selected_target) != int(modal_highest_rep_agent_estimate):
        return "ranking_selection_mismatch"
    if float(share_step1_margin_positive) <= 0.0:
        return "step1_conversion_mismatch"
    if int(top_followers) <= 0:
        return "follower_assignment_mismatch"
    return "mixed_ranking_plus_scale"


def _candidate_count_from_row(row: np.ndarray, observer_id: int, delta: float) -> int:
    rep_row = np.asarray(row, dtype=float).copy()
    if rep_row.size == 0:
        return 0
    rep_row[int(observer_id)] = -np.inf
    max_rep = float(np.max(rep_row))
    if not np.isfinite(max_rep):
        return 0
    return int(np.sum(rep_row >= (max_rep - float(delta))))


def _effective_follow_threshold(role_label: str, follower_count: int, *, B_R: float, B_F: float) -> float:
    if str(role_label) == "reputation" and int(follower_count) == 0 and float(B_F) < float(B_R):
        return float(B_F)
    return float(B_R)


def _resolve_root_leaders_from_step(
    following_row: np.ndarray,
    follower_count_row: Optional[np.ndarray] = None,
) -> np.ndarray:
    following = np.asarray(following_row, dtype=int).reshape(-1)
    n_agents = int(following.size)
    follower_counts = (
        np.zeros(n_agents, dtype=int)
        if follower_count_row is None
        else np.asarray(follower_count_row, dtype=int).reshape(-1)
    )
    roots = np.full(n_agents, -1, dtype=int)

    for agent_id in range(n_agents):
        current = int(following[agent_id])
        if current < 0:
            if agent_id < follower_counts.size and int(follower_counts[agent_id]) > 0:
                roots[agent_id] = int(agent_id)
            continue

        seen = {int(agent_id)}
        while 0 <= current < n_agents and current not in seen:
            seen.add(current)
            next_target = int(following[current])
            if next_target < 0:
                roots[agent_id] = int(current)
                break
            current = next_target

    return roots


def write_small_n_true_rep_vs_estimate_trace_long_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        pu_history = np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float)
        rep_reward_history = np.asarray(trace.get("estimated_reward_rep_history", []), dtype=float)
        raw_rep_history = np.asarray(trace.get("selected_reputation_history", []), dtype=float)
        weighted_selected_rep_history = np.asarray(trace.get("weighted_selected_reputation_history", []), dtype=float)
        highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
        following_history = np.asarray(trace.get("following_history", []), dtype=int)
        role_label_history = np.asarray(trace.get("role_label_history", []), dtype=object)
        actor_rate_history = np.asarray(trace.get("actor_interaction_rate_history", []), dtype=float)
        follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
        dense_reputation_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        true_reputation_history = np.asarray(trace.get("true_reputation_history", []), dtype=float)
        true_rank_history = np.asarray(trace.get("true_reputation_rank_history", []), dtype=int)
        if pu_history.size == 0 or dense_reputation_history.size == 0 or true_reputation_history.size == 0:
            continue

        B_R = float(trace.get("B_R", 0.0))
        B_F = float(trace.get("B_F", 0.0))
        delta = float(trace.get("delta", 0.0))
        n_steps, n_agents = pu_history.shape
        for t in range(n_steps):
            mean_observed_reputation = np.mean(dense_reputation_history[t], axis=0)
            observed_rank = _rank_desc(mean_observed_reputation)
            for agent_id in range(n_agents):
                selected_raw = float(raw_rep_history[t, agent_id]) if raw_rep_history.size else 0.0
                gamma_times_selected_rep = (
                    float(weighted_selected_rep_history[t, agent_id])
                    if weighted_selected_rep_history.size
                    else float(gamma) * selected_raw
                )
                estimated_reward_pu = float(pu_history[t, agent_id])
                effective_threshold = _effective_follow_threshold(
                    str(role_label_history[t, agent_id]) if role_label_history.size else "",
                    int(follower_count_history[t, agent_id]) if follower_count_history.size else 0,
                    B_R=B_R,
                    B_F=B_F,
                )
                rows.append(
                    {
                        "t": int(t + 1),
                        "seed": int(seed),
                        "gamma": float(gamma),
                        "agent_id": int(agent_id),
                        "true_reputation": float(true_reputation_history[t, agent_id]),
                        "true_rank": int(true_rank_history[t, agent_id]) if true_rank_history.size else 0,
                        "mean_observed_reputation": float(mean_observed_reputation[agent_id]),
                        "observed_rank": int(observed_rank[agent_id]),
                        "highest_rep_agent_estimate": int(highest_rep_history[t, agent_id]) if highest_rep_history.size else -1,
                        "selected_candidate_count": int(
                            _candidate_count_from_row(
                                dense_reputation_history[t, agent_id],
                                agent_id,
                                delta,
                            )
                        ),
                        "estimated_reward_pu": estimated_reward_pu,
                        "estimated_reward_rep": float(rep_reward_history[t, agent_id]) if rep_reward_history.size else 0.0,
                        "gamma_times_estimated_reward_rep": (
                            float(gamma) * float(rep_reward_history[t, agent_id]) if rep_reward_history.size else 0.0
                        ),
                        "selected_reputation_raw": selected_raw,
                        "gamma_times_selected_reputation": float(gamma_times_selected_rep),
                        "effective_threshold": float(effective_threshold),
                        "step1_margin": float(gamma_times_selected_rep - estimated_reward_pu),
                        "gate_margin": float(gamma_times_selected_rep - max(effective_threshold, estimated_reward_pu)),
                        "role": str(role_label_history[t, agent_id]) if role_label_history.size else "",
                        "following": int(following_history[t, agent_id]) if following_history.size else -1,
                        "actor_interaction_rate": float(actor_rate_history[t, agent_id]) if actor_rate_history.size else 0.0,
                    }
                )
    write_csv(output_file, rows)


def write_small_n_true_reputation_decomposition_long_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        actor_rate_history = np.asarray(trace.get("actor_interaction_rate_history", []), dtype=float)
        dense_reputation_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        dense_personal_benefit_history = np.asarray(trace.get("dense_personal_benefit_history", []), dtype=float)
        true_reputation_history = np.asarray(trace.get("true_reputation_history", []), dtype=float)
        true_rank_history = np.asarray(trace.get("true_reputation_rank_history", []), dtype=int)
        true_theta_history = np.asarray(trace.get("true_reputation_theta_history", []), dtype=float)
        true_sum_expected_history = np.asarray(trace.get("true_reputation_sum_expected_history", []), dtype=float)
        if (
            actor_rate_history.size == 0
            or dense_reputation_history.size == 0
            or dense_personal_benefit_history.size == 0
            or true_reputation_history.size == 0
        ):
            continue

        n_steps, n_targets = true_reputation_history.shape
        for t in range(n_steps):
            mean_observed_reputation = np.mean(dense_reputation_history[t], axis=0)
            observed_rank = _rank_desc(mean_observed_reputation)
            mean_incoming_v = np.mean(dense_personal_benefit_history[t], axis=0)
            for target_id in range(n_targets):
                rows.append(
                    {
                        "t": int(t + 1),
                        "seed": int(seed),
                        "gamma": float(gamma),
                        "target_id": int(target_id),
                        "theta_mu": float(true_theta_history[t, target_id]) if true_theta_history.size else 0.0,
                        "actor_rate": float(actor_rate_history[t, target_id]),
                        "sum_expected_utility_others": float(true_sum_expected_history[t, target_id]) if true_sum_expected_history.size else 0.0,
                        "true_reputation": float(true_reputation_history[t, target_id]),
                        "mean_observed_reputation": float(mean_observed_reputation[target_id]),
                        "mean_incoming_v": float(mean_incoming_v[target_id]),
                        "true_rank": int(true_rank_history[t, target_id]) if true_rank_history.size else 0,
                        "observed_rank": int(observed_rank[target_id]),
                    }
                )
    write_csv(output_file, rows)


def write_small_n_toy_alignment_by_update_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
        dense_reputation_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        dense_personal_benefit_history = np.asarray(trace.get("dense_personal_benefit_history", []), dtype=float)
        true_reputation_history = np.asarray(trace.get("true_reputation_history", []), dtype=float)
        true_sum_expected_history = np.asarray(trace.get("true_reputation_sum_expected_history", []), dtype=float)
        true_theta_history = np.asarray(trace.get("true_reputation_theta_history", []), dtype=float)
        highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
        following_history = np.asarray(trace.get("following_history", []), dtype=int)
        follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
        if (
            dense_reputation_history.size == 0
            or dense_personal_benefit_history.size == 0
            or true_reputation_history.size == 0
            or highest_rep_history.size == 0
            or not role_update_times
        ):
            continue

        for step in role_update_times:
            idx = int(step) - 1
            if idx < 0 or idx >= dense_reputation_history.shape[0]:
                continue
            mean_observed_reputation = np.mean(dense_reputation_history[idx], axis=0)
            mean_incoming_v = np.mean(dense_personal_benefit_history[idx], axis=0)
            true_reputation = np.asarray(true_reputation_history[idx], dtype=float)
            sum_expected = np.asarray(true_sum_expected_history[idx], dtype=float)
            theta_mu = np.asarray(true_theta_history[idx], dtype=float)
            highest_ids = [int(x) for x in np.asarray(highest_rep_history[idx], dtype=int).tolist() if int(x) >= 0]
            # Step 1 uses the currently selected highest-reputation target L_i(t).
            current_selected_targets = [
                int(highest_rep_history[idx, agent_id])
                for agent_id in range(highest_rep_history.shape[1])
            ]
            modal_highest = _mode_int(highest_ids)
            modal_selected = _mode_int(current_selected_targets)
            true_top_agent = _top_agent_id(true_reputation)
            observed_top_agent = _top_agent_id(mean_observed_reputation)
            corr_true = _safe_pearson(mean_observed_reputation, true_reputation)
            corr_sum_expected = _safe_pearson(mean_observed_reputation, sum_expected)
            corr_theta = _safe_pearson(mean_observed_reputation, theta_mu)
            corr_v = _safe_pearson(mean_observed_reputation, mean_incoming_v)
            dominant_target = _dominant_alignment_target(
                corr_true_reputation=corr_true,
                corr_sum_expected_utility=corr_sum_expected,
                corr_theta_mu=corr_theta,
                corr_mean_incoming_v=corr_v,
            )
            top_followers = int(np.max(follower_count_history[idx])) if follower_count_history.size else 0
            share_step1_positive = float(
                np.mean(
                    float(gamma) * np.asarray(trace.get("selected_reputation_history", []), dtype=float)[idx]
                    > np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float)[idx]
                )
            ) if np.asarray(trace.get("selected_reputation_history", []), dtype=float).size and np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float).size else 0.0
            rows.append(
                {
                    "gamma": float(gamma),
                    "seed": int(seed),
                    "t": int(step),
                    "true_top_agent": int(true_top_agent),
                    "observed_top_agent": int(observed_top_agent),
                    "modal_highest_rep_agent_estimate": int(modal_highest),
                    "modal_selected_target": int(modal_selected),
                    "top_match_true_vs_observed": bool(int(true_top_agent) == int(observed_top_agent)),
                    "top_match_observed_vs_highest": bool(int(observed_top_agent) == int(modal_highest)),
                    "top_match_highest_vs_selected": bool(int(modal_highest) == int(modal_selected)),
                    "corr_observed_vs_true_reputation": float(corr_true),
                    "corr_observed_vs_sum_expected_utility": float(corr_sum_expected),
                    "corr_observed_vs_theta_mu": float(corr_theta),
                    "corr_observed_vs_mean_incoming_v": float(corr_v),
                    "dominant_alignment_target": str(dominant_target),
                    "failure_stage_label": _classify_toy_failure_stage(
                        true_top_agent=int(true_top_agent),
                        observed_top_agent=int(observed_top_agent),
                        modal_highest_rep_agent_estimate=int(modal_highest),
                        modal_selected_target=int(modal_selected),
                        dominant_alignment_target=str(dominant_target),
                        share_step1_margin_positive=float(share_step1_positive),
                        top_followers=int(top_followers),
                    ),
                }
            )
    write_csv(output_file, rows)


def write_small_n_toy_v_to_s_by_update_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
        dense_reputation_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        dense_personal_benefit_history = np.asarray(trace.get("dense_personal_benefit_history", []), dtype=float)
        true_reputation_history = np.asarray(trace.get("true_reputation_history", []), dtype=float)
        true_sum_expected_history = np.asarray(trace.get("true_reputation_sum_expected_history", []), dtype=float)
        true_theta_history = np.asarray(trace.get("true_reputation_theta_history", []), dtype=float)
        if (
            dense_reputation_history.size == 0
            or dense_personal_benefit_history.size == 0
            or true_reputation_history.size == 0
            or not role_update_times
        ):
            continue

        for step in role_update_times:
            idx = int(step) - 1
            if idx < 0 or idx >= dense_reputation_history.shape[0]:
                continue
            mean_observed_reputation = np.mean(dense_reputation_history[idx], axis=0)
            mean_incoming_v = np.mean(dense_personal_benefit_history[idx], axis=0)
            true_reputation = np.asarray(true_reputation_history[idx], dtype=float)
            sum_expected = np.asarray(true_sum_expected_history[idx], dtype=float)
            theta_mu = np.asarray(true_theta_history[idx], dtype=float)

            corr_v_true = _safe_pearson(mean_incoming_v, true_reputation)
            corr_v_sum = _safe_pearson(mean_incoming_v, sum_expected)
            corr_v_theta = _safe_pearson(mean_incoming_v, theta_mu)
            corr_s_v = _safe_pearson(mean_observed_reputation, mean_incoming_v)
            v_alignment_target = _dominant_alignment_target(
                corr_true_reputation=corr_v_true,
                corr_sum_expected_utility=corr_v_sum,
                corr_theta_mu=corr_v_theta,
                corr_mean_incoming_v=float("nan"),
            )

            rows.append(
                {
                    "gamma": float(gamma),
                    "seed": int(seed),
                    "t": int(step),
                    "true_top_agent": int(_top_agent_id(true_reputation)),
                    "sum_expected_top_agent": int(_top_agent_id(sum_expected)),
                    "mean_incoming_v_top_agent": int(_top_agent_id(mean_incoming_v)),
                    "observed_top_agent": int(_top_agent_id(mean_observed_reputation)),
                    "corr_mean_incoming_v_vs_true_reputation": float(corr_v_true),
                    "corr_mean_incoming_v_vs_sum_expected_utility": float(corr_v_sum),
                    "corr_mean_incoming_v_vs_theta_mu": float(corr_v_theta),
                    "corr_observed_vs_mean_incoming_v": float(corr_s_v),
                    "top_match_mean_incoming_v_vs_observed": bool(
                        int(_top_agent_id(mean_incoming_v)) == int(_top_agent_id(mean_observed_reputation))
                    ),
                    "top_match_mean_incoming_v_vs_true": bool(
                        int(_top_agent_id(mean_incoming_v)) == int(_top_agent_id(true_reputation))
                    ),
                    "dominant_v_alignment_target": str(v_alignment_target),
                }
            )
    write_csv(output_file, rows)


def write_small_n_toy_v_to_s_recurrence_audit_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
        dense_v_history = np.asarray(trace.get("dense_personal_benefit_history", []), dtype=float)
        dense_s_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        observed_utility_history = trace.get("observed_utility_matrix_history", [])
        eta_v_history = [float(x) for x in trace.get("eta_v_history", [])]
        active_actor_ids_history = trace.get("active_actor_ids_history", [])
        active_participant_ids_history = trace.get("active_participant_ids_history", [])
        gossip_target_ids_history = trace.get("gossip_target_ids_history", [])
        averaging_agent_ids_history = trace.get("averaging_agent_ids_history", [])
        avg_s_by_target_history = trace.get("avg_s_by_target_history", [])
        delta_v_matrix_history = trace.get("delta_v_matrix_history", [])
        if (
            dense_v_history.size == 0
            or dense_s_history.size == 0
            or not role_update_times
            or len(observed_utility_history) == 0
        ):
            continue

        num_steps = dense_v_history.shape[0]
        num_agents = dense_v_history.shape[1]
        for step in role_update_times:
            idx = int(step) - 1
            if idx < 0 or idx >= num_steps:
                continue
            prev_v = np.zeros((num_agents, num_agents), dtype=float) if idx == 0 else np.asarray(dense_v_history[idx - 1], dtype=float)
            new_v = np.asarray(dense_v_history[idx], dtype=float)
            prev_s = np.zeros((num_agents, num_agents), dtype=float) if idx == 0 else np.asarray(dense_s_history[idx - 1], dtype=float)
            new_s = np.asarray(dense_s_history[idx], dtype=float)
            observed_utility_matrix = np.asarray(observed_utility_history[idx], dtype=float)
            eta_v_t = float(eta_v_history[idx]) if idx < len(eta_v_history) else 0.0
            active_actor_ids = {int(x) for x in active_actor_ids_history[idx]} if idx < len(active_actor_ids_history) else set()
            active_participant_ids = {int(x) for x in active_participant_ids_history[idx]} if idx < len(active_participant_ids_history) else set()
            gossip_target_ids = {int(x) for x in gossip_target_ids_history[idx]} if idx < len(gossip_target_ids_history) else set()
            averaging_agent_ids = [int(x) for x in averaging_agent_ids_history[idx]] if idx < len(averaging_agent_ids_history) else []
            avg_s_by_target = {
                int(k): float(v) for k, v in (avg_s_by_target_history[idx] or {}).items()
            } if idx < len(avg_s_by_target_history) else {}
            delta_v_matrix = np.asarray(delta_v_matrix_history[idx], dtype=float) if idx < len(delta_v_matrix_history) else np.zeros_like(new_v)

            for observer_id in range(num_agents):
                for target_id in range(num_agents):
                    prev_v_val = float(prev_v[observer_id, target_id])
                    actual_v_new = float(new_v[observer_id, target_id])
                    observed_u = float(observed_utility_matrix[observer_id, target_id])
                    is_active_actor = bool(target_id in active_actor_ids)
                    is_active_participant = bool(observer_id in active_participant_ids)
                    is_gossip_target = bool(target_id in gossip_target_ids)
                    expected_v_new = (
                        prev_v_val + eta_v_t * (observed_u - prev_v_val)
                        if is_active_actor
                        else prev_v_val * (1.0 - eta_v_t)
                    )
                    prev_s_val = float(prev_s[observer_id, target_id])
                    actual_s_new = float(new_s[observer_id, target_id])
                    avg_s_target = float(avg_s_by_target.get(target_id, 0.0))
                    expected_s_new = (
                        avg_s_target + (expected_v_new - prev_v_val)
                        if is_active_participant and is_gossip_target
                        else prev_s_val
                    )
                    rows.append(
                        {
                            "gamma": float(gamma),
                            "seed": int(seed),
                            "t": int(step),
                            "observer_id": int(observer_id),
                            "target_id": int(target_id),
                            "is_active_actor": bool(is_active_actor),
                            "is_active_participant": bool(is_active_participant),
                            "is_gossip_target": bool(is_gossip_target),
                            "eta_v_t": float(eta_v_t),
                            "observed_utility": float(observed_u),
                            "prev_v": float(prev_v_val),
                            "expected_v_new_paper": float(expected_v_new),
                            "actual_v_new_code": float(actual_v_new),
                            "delta_v_from_phase4": float(delta_v_matrix[observer_id, target_id]),
                            "v_matches_paper": bool(np.isclose(actual_v_new, expected_v_new, atol=1e-12, rtol=0.0)),
                            "prev_s": float(prev_s_val),
                            "avg_s_target": float(avg_s_target),
                            "expected_s_new_paper": float(expected_s_new),
                            "actual_s_new_code": float(actual_s_new),
                            "s_matches_paper": bool(np.isclose(actual_s_new, expected_s_new, atol=1e-12, rtol=0.0)),
                            "averaging_agent_count": int(len(averaging_agent_ids)),
                        }
                    )
    write_csv(output_file, rows)


def write_small_n_toy_s_to_highest_by_update_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
        dense_reputation_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
        delta = float(trace.get("delta", 0.0))
        if dense_reputation_history.size == 0 or highest_rep_history.size == 0 or not role_update_times:
            continue

        for step in role_update_times:
            idx = int(step) - 1
            if idx < 0 or idx >= dense_reputation_history.shape[0]:
                continue

            row_argmax_targets: List[int] = []
            stored_targets: List[int] = []
            candidate_counts: List[int] = []
            stored_within_delta: List[bool] = []
            stored_equals_argmax: List[bool] = []

            for agent_id in range(dense_reputation_history.shape[1]):
                row = np.asarray(dense_reputation_history[idx, agent_id], dtype=float).copy()
                row[agent_id] = -np.inf
                argmax_target = int(np.argmax(row)) if np.any(np.isfinite(row)) else -1
                max_rep = float(np.max(row)) if np.any(np.isfinite(row)) else float("-inf")
                candidates = np.where(row >= max_rep - delta)[0] if np.isfinite(max_rep) else np.array([], dtype=int)
                stored_target = int(highest_rep_history[idx, agent_id])

                row_argmax_targets.append(argmax_target)
                stored_targets.append(stored_target)
                candidate_counts.append(int(candidates.size))
                stored_within_delta.append(bool(stored_target in candidates.tolist()))
                stored_equals_argmax.append(bool(stored_target == argmax_target))

            mean_observed_reputation = np.mean(dense_reputation_history[idx], axis=0)
            rows.append(
                {
                    "gamma": float(gamma),
                    "seed": int(seed),
                    "t": int(step),
                    "observed_top_agent": int(_top_agent_id(mean_observed_reputation)),
                    "modal_row_argmax_target": int(_mode_int(row_argmax_targets)),
                    "modal_highest_rep_agent_estimate": int(_mode_int(stored_targets)),
                    "share_highest_equals_row_argmax": float(np.mean(np.asarray(stored_equals_argmax, dtype=float))),
                    "share_highest_within_delta_set": float(np.mean(np.asarray(stored_within_delta, dtype=float))),
                    "mean_candidate_count_within_delta": float(np.mean(np.asarray(candidate_counts, dtype=float))),
                    "max_candidate_count_within_delta": int(np.max(np.asarray(candidate_counts, dtype=int))),
                }
            )
    write_csv(output_file, rows)


def write_small_n_toy_step1_by_update_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
        raw_rep_history = np.asarray(trace.get("selected_reputation_history", []), dtype=float)
        weighted_selected_rep_history = np.asarray(trace.get("weighted_selected_reputation_history", []), dtype=float)
        pu_history = np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float)
        rep_reward_history = np.asarray(trace.get("estimated_reward_rep_history", []), dtype=float)
        dense_reputation_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
        highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
        role_label_history = np.asarray(trace.get("role_label_history", []), dtype=object)
        follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
        if raw_rep_history.size == 0 or weighted_selected_rep_history.size == 0 or pu_history.size == 0 or not role_update_times:
            continue

        first_positive_step = None
        for step in role_update_times:
            idx = int(step) - 1
            if idx < 0 or idx >= raw_rep_history.shape[0]:
                continue
            if np.any(np.asarray(weighted_selected_rep_history[idx], dtype=float) > np.asarray(pu_history[idx], dtype=float)):
                first_positive_step = int(step)
                break

        B_R = float(trace.get("B_R", 0.0))
        B_F = float(trace.get("B_F", 0.0))
        for step in role_update_times:
            idx = int(step) - 1
            if idx < 0 or idx >= raw_rep_history.shape[0]:
                continue
            current_selected_targets = [
                int(highest_rep_history[idx, agent_id]) if highest_rep_history.size else -1
                for agent_id in range(raw_rep_history.shape[1])
            ]
            effective_thresholds = []
            selected_rep_matches = []
            weighted_rep_matches = []
            for agent_id in range(raw_rep_history.shape[1]):
                effective_thresholds.append(
                    _effective_follow_threshold(
                        str(role_label_history[idx, agent_id]) if role_label_history.size else "",
                        int(follower_count_history[idx, agent_id]) if follower_count_history.size else 0,
                        B_R=B_R,
                        B_F=B_F,
                    )
                )
                target_id = int(current_selected_targets[agent_id])
                if 0 <= target_id < raw_rep_history.shape[1] and dense_reputation_history.size:
                    row_value = float(dense_reputation_history[idx, agent_id, target_id])
                    selected_rep_matches.append(
                        bool(np.isclose(float(raw_rep_history[idx, agent_id]), row_value, atol=1e-12, rtol=0.0))
                    )
                else:
                    selected_rep_matches.append(bool(np.isclose(float(raw_rep_history[idx, agent_id]), 0.0, atol=1e-12, rtol=0.0)))
                weighted_rep_matches.append(
                    bool(
                        np.isclose(
                            float(weighted_selected_rep_history[idx, agent_id]),
                            float(gamma) * float(raw_rep_history[idx, agent_id]),
                            atol=1e-12,
                            rtol=0.0,
                        )
                    )
                )
            step1_margins = np.asarray(weighted_selected_rep_history[idx], dtype=float) - np.asarray(pu_history[idx], dtype=float)
            gate_margins = np.asarray(weighted_selected_rep_history[idx], dtype=float) - np.maximum(
                np.asarray(effective_thresholds, dtype=float),
                np.asarray(pu_history[idx], dtype=float),
            )
            rows.append(
                {
                    "gamma": float(gamma),
                    "seed": int(seed),
                    "t": int(step),
                    "modal_selected_target": int(_mode_int(current_selected_targets)),
                    "mean_selected_reputation_raw": float(np.mean(raw_rep_history[idx])),
                    "max_selected_reputation_raw": float(np.max(raw_rep_history[idx])),
                    "mean_gamma_times_selected_reputation": float(np.mean(weighted_selected_rep_history[idx])),
                    "max_gamma_times_selected_reputation": float(np.max(weighted_selected_rep_history[idx])),
                    "mean_estimated_reward_rep": float(np.mean(rep_reward_history[idx])) if rep_reward_history.size else 0.0,
                    "mean_gamma_times_estimated_reward_rep": float(np.mean(float(gamma) * rep_reward_history[idx])) if rep_reward_history.size else 0.0,
                    "mean_estimated_reward_pu": float(np.mean(pu_history[idx])),
                    "max_estimated_reward_pu": float(np.max(pu_history[idx])),
                    "effective_threshold": float(np.mean(np.asarray(effective_thresholds, dtype=float))),
                    "share_selected_reputation_matches_highest_row_value": float(np.mean(np.asarray(selected_rep_matches, dtype=float))),
                    "share_weighted_signal_matches_gamma_times_selected": float(np.mean(np.asarray(weighted_rep_matches, dtype=float))),
                    "mean_step1_margin": float(np.mean(step1_margins)),
                    "max_step1_margin": float(np.max(step1_margins)),
                    "share_step1_margin_positive": float(np.mean(step1_margins > 0.0)),
                    "mean_gate_margin": float(np.mean(gate_margins)),
                    "max_gate_margin": float(np.max(gate_margins)),
                    "share_gate_margin_positive": float(np.mean(gate_margins > 0.0)),
                    "first_positive_follow_signal_reached": bool(first_positive_step is not None and int(step) >= int(first_positive_step)),
                }
            )
    write_csv(output_file, rows)


def write_small_n_toy_choice_trace_long_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        raw_rep_history = np.asarray(trace.get("selected_reputation_history", []), dtype=float)
        weighted_selected_rep_history = np.asarray(trace.get("weighted_selected_reputation_history", []), dtype=float)
        pu_history = np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float)
        highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
        following_history = np.asarray(trace.get("following_history", []), dtype=int)
        role_label_history = np.asarray(trace.get("role_label_history", []), dtype=object)
        follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
        if raw_rep_history.size == 0 or weighted_selected_rep_history.size == 0 or pu_history.size == 0:
            continue

        B_R = float(trace.get("B_R", 0.0))
        B_F = float(trace.get("B_F", 0.0))
        role_update_times = set(int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist())
        n_steps, n_agents = raw_rep_history.shape
        for t in range(n_steps):
            follower_counts_t = (
                np.asarray(follower_count_history[t], dtype=int)
                if follower_count_history.size
                else np.zeros(n_agents, dtype=int)
            )
            following_t = (
                np.asarray(following_history[t], dtype=int)
                if following_history.size
                else np.full(n_agents, -1, dtype=int)
            )
            roots_t = _resolve_root_leaders_from_step(following_t, follower_counts_t)
            for agent_id in range(n_agents):
                role_label = str(role_label_history[t, agent_id]) if role_label_history.size else ""
                estimated_reward_pu = float(pu_history[t, agent_id])
                gamma_times_selected_rep = float(weighted_selected_rep_history[t, agent_id])
                effective_threshold = _effective_follow_threshold(
                    role_label,
                    int(follower_counts_t[agent_id]),
                    B_R=B_R,
                    B_F=B_F,
                )
                step1_margin = float(gamma_times_selected_rep - estimated_reward_pu)
                gate_margin = float(gamma_times_selected_rep - max(effective_threshold, estimated_reward_pu))
                rows.append(
                    {
                        "t": int(t + 1),
                        "seed": int(seed),
                        "gamma": float(gamma),
                        "role_update_step": int((t + 1) in role_update_times),
                        "agent_id": int(agent_id),
                        "role": role_label,
                        "highest_rep_agent_estimate": int(highest_rep_history[t, agent_id]) if highest_rep_history.size else -1,
                        "selected_reputation_raw": float(raw_rep_history[t, agent_id]),
                        "gamma_times_selected_reputation": float(gamma_times_selected_rep),
                        "estimated_reward_pu": estimated_reward_pu,
                        "step1_margin": step1_margin,
                        "gate_margin": gate_margin,
                        "following": int(following_t[agent_id]),
                        "root_leader": int(roots_t[agent_id]),
                        "rep_beats_pu": bool(step1_margin > 0.0),
                        "rep_beats_gate": bool(gate_margin > 0.0),
                    }
                )
    write_csv(output_file, rows)


def write_small_n_toy_consensus_by_step_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
        following_history = np.asarray(trace.get("following_history", []), dtype=int)
        follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
        raw_rep_history = np.asarray(trace.get("selected_reputation_history", []), dtype=float)
        weighted_selected_rep_history = np.asarray(trace.get("weighted_selected_reputation_history", []), dtype=float)
        pu_history = np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float)
        if highest_rep_history.size == 0 or raw_rep_history.size == 0 or weighted_selected_rep_history.size == 0 or pu_history.size == 0:
            continue

        B_R = float(trace.get("B_R", 0.0))
        B_F = float(trace.get("B_F", 0.0))
        role_label_history = np.asarray(trace.get("role_label_history", []), dtype=object)
        role_update_times = set(int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist())
        n_steps, n_agents = highest_rep_history.shape
        for t in range(n_steps):
            highest_t = np.asarray(highest_rep_history[t], dtype=int)
            valid_highest = [int(x) for x in highest_t.tolist() if int(x) >= 0]
            modal_highest = _mode_int(valid_highest)
            modal_count = sum(1 for x in highest_t.tolist() if int(x) == int(modal_highest)) if modal_highest >= 0 else 0
            distinct_highest = len(set(valid_highest))
            follower_counts_t = (
                np.asarray(follower_count_history[t], dtype=int)
                if follower_count_history.size
                else np.zeros(n_agents, dtype=int)
            )
            following_t = (
                np.asarray(following_history[t], dtype=int)
                if following_history.size
                else np.full(n_agents, -1, dtype=int)
            )
            roots_t = _resolve_root_leaders_from_step(following_t, follower_counts_t)
            valid_roots = [int(x) for x in roots_t.tolist() if int(x) >= 0]
            distinct_root_leaders = len(set(valid_roots))
            largest_root_size = 0 if not valid_roots else max(sum(1 for root in valid_roots if root == leader) for leader in set(valid_roots))
            effective_thresholds = np.array(
                [
                    _effective_follow_threshold(
                        str(role_label_history[t, agent_id]) if role_label_history.size else "",
                        int(follower_counts_t[agent_id]),
                        B_R=B_R,
                        B_F=B_F,
                    )
                    for agent_id in range(n_agents)
                ],
                dtype=float,
            )
            step1_positive = np.asarray(weighted_selected_rep_history[t], dtype=float) > np.asarray(pu_history[t], dtype=float)
            gate_positive = np.asarray(weighted_selected_rep_history[t], dtype=float) > np.maximum(
                effective_thresholds,
                np.asarray(pu_history[t], dtype=float),
            )
            rows.append(
                {
                    "t": int(t + 1),
                    "seed": int(seed),
                    "gamma": float(gamma),
                    "role_update_step": int((t + 1) in role_update_times),
                    "modal_highest_target": int(modal_highest),
                    "modal_highest_share": float(modal_count / max(1, n_agents)),
                    "distinct_highest_targets": int(distinct_highest),
                    "all_agents_agree_on_highest": bool(distinct_highest == 1 and len(valid_highest) == n_agents),
                    "distinct_root_leaders": int(distinct_root_leaders),
                    "largest_root_size": int(largest_root_size),
                    "share_step1_positive": float(np.mean(step1_positive)),
                    "share_gate_positive": float(np.mean(gate_positive)),
                }
            )
    write_csv(output_file, rows)


def write_small_n_toy_follow_relationships_long_csv(
    traces: Dict[Tuple[float, int], DetailedTrace],
    output_file: Path,
) -> None:
    rows: List[dict] = []
    for (gamma, seed), trace in sorted(traces.items()):
        following_history = np.asarray(trace.get("following_history", []), dtype=int)
        role_label_history = np.asarray(trace.get("role_label_history", []), dtype=object)
        follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
        if following_history.size == 0:
            continue

        role_update_times = set(int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist())
        n_steps, n_agents = following_history.shape
        for t in range(n_steps):
            follower_counts_t = (
                np.asarray(follower_count_history[t], dtype=int)
                if follower_count_history.size
                else np.zeros(n_agents, dtype=int)
            )
            following_t = np.asarray(following_history[t], dtype=int)
            roots_t = _resolve_root_leaders_from_step(following_t, follower_counts_t)
            for agent_id in range(n_agents):
                rows.append(
                    {
                        "t": int(t + 1),
                        "seed": int(seed),
                        "gamma": float(gamma),
                        "role_update_step": int((t + 1) in role_update_times),
                        "agent_id": int(agent_id),
                        "role": str(role_label_history[t, agent_id]) if role_label_history.size else "",
                        "following": int(following_t[agent_id]),
                        "root_leader": int(roots_t[agent_id]),
                        "has_followers": bool(int(follower_counts_t[agent_id]) > 0),
                    }
                )
    write_csv(output_file, rows)


def plot_toy_mean_gate_signals(
    trace: DetailedTrace,
    output_file: Path,
    *,
    gamma: float,
    title: str,
) -> None:
    pu_history = np.asarray(trace.get("estimated_reward_pu_history", []), dtype=float)
    weighted_selected_rep_history = np.asarray(trace.get("weighted_selected_reputation_history", []), dtype=float)
    rep_reward_history = np.asarray(trace.get("estimated_reward_rep_history", []), dtype=float)
    role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
    if pu_history.size == 0 or weighted_selected_rep_history.size == 0:
        return

    x = np.arange(1, pu_history.shape[0] + 1)
    mean_pu = np.mean(pu_history, axis=1)
    mean_selected_rep = np.mean(weighted_selected_rep_history, axis=1)
    mean_estimated_rep = float(gamma) * np.mean(rep_reward_history, axis=1) if rep_reward_history.size else None

    plt.figure(figsize=(8.2, 4.6))
    plt.plot(x, mean_pu, label="Mean PU estimate", linewidth=1.8, color="tab:blue")
    plt.plot(x, mean_selected_rep, label="Mean gamma * selected reputation", linewidth=1.8, color="tab:red")
    if mean_estimated_rep is not None:
        plt.plot(
            x,
            mean_estimated_rep,
            label="Mean gamma * estimated_reward_rep",
            linewidth=1.2,
            linestyle=":",
            color="tab:orange",
        )
    for idx_line, step in enumerate(role_update_times):
        plt.axvline(
            int(step),
            color="gray",
            linestyle="--",
            linewidth=0.7,
            alpha=0.22,
            label="Role update" if idx_line == 0 else None,
        )
    plt.title(title, fontsize=12, fontweight="bold")
    plt.xlabel("Timestep")
    plt.ylabel("Mean signal")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_toy_agent_timeline_matrix(
    matrix: np.ndarray,
    output_file: Path,
    *,
    title: str,
    colorbar_label: str,
) -> None:
    values = np.asarray(matrix, dtype=int)
    if values.size == 0:
        return
    n_agents, n_steps = values.shape
    plotted = values + 1  # -1 -> 0 ("none")
    base_colors = ["#f5f5f5"] + [matplotlib.cm.tab20(i % 20) for i in range(max(1, n_agents))]
    cmap = matplotlib.colors.ListedColormap(base_colors)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_agents + 1.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.0035 * n_steps), 3.8))
    im = ax.imshow(plotted, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Agent id")
    ax.set_yticks(np.arange(n_agents))
    ax.set_yticklabels([str(i) for i in range(n_agents)])
    x_ticks = np.arange(n_steps)
    if n_steps > 12:
        keep = np.linspace(0, n_steps - 1, num=12, dtype=int)
        x_ticks = np.unique(keep)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(idx + 1)) for idx in x_ticks], rotation=45, ha="right")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks(np.arange(n_agents + 1))
    cbar.set_ticklabels(["none"] + [str(i) for i in range(n_agents)])
    cbar.set_label(colorbar_label)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_toy_highest_target_timeline(
    trace: DetailedTrace,
    output_file: Path,
    *,
    title: str,
) -> None:
    highest_rep_history = np.asarray(trace.get("highest_rep_agent_history", []), dtype=int)
    if highest_rep_history.size == 0:
        return
    _plot_toy_agent_timeline_matrix(
        highest_rep_history.T,
        output_file,
        title=title,
        colorbar_label="Highest-reputation target",
    )


def plot_toy_root_leader_timeline(
    trace: DetailedTrace,
    output_file: Path,
    *,
    title: str,
) -> None:
    following_history = np.asarray(trace.get("following_history", []), dtype=int)
    follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
    if following_history.size == 0:
        return

    root_history = np.zeros_like(following_history, dtype=int)
    for t in range(following_history.shape[0]):
        follower_counts_t = (
            np.asarray(follower_count_history[t], dtype=int)
            if follower_count_history.size
            else np.zeros(following_history.shape[1], dtype=int)
        )
        root_history[t] = _resolve_root_leaders_from_step(
            np.asarray(following_history[t], dtype=int),
            follower_counts_t,
        )
    _plot_toy_agent_timeline_matrix(
        root_history.T,
        output_file,
        title=title,
        colorbar_label="Current root leader",
    )


def plot_toy_follow_graph_snapshots(
    trace: DetailedTrace,
    output_dir: Path,
    *,
    title_prefix: str,
) -> None:
    following_history = np.asarray(trace.get("following_history", []), dtype=int)
    role_label_history = np.asarray(trace.get("role_label_history", []), dtype=object)
    follower_count_history = np.asarray(trace.get("follower_count_history", []), dtype=int)
    role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
    if following_history.size == 0 or role_label_history.size == 0 or not role_update_times:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    n_agents = int(following_history.shape[1])
    theta = np.linspace(0.0, 2.0 * np.pi, n_agents, endpoint=False)
    radius = 1.0
    coords = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    role_colors = {
        "personal_utility": "#5b8ff9",
        "reputation": "#e8684a",
        "status": "#f6bd16",
    }

    for step in role_update_times:
        idx = int(step) - 1
        if idx < 0 or idx >= following_history.shape[0]:
            continue
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        current_following = following_history[idx]
        current_roles = role_label_history[idx]
        current_followers = follower_count_history[idx] if follower_count_history.size else np.zeros(n_agents, dtype=int)
        top_followers = int(np.max(current_followers)) if current_followers.size else 0
        top_leader = int(np.argmax(current_followers)) if top_followers > 0 else -1

        for agent_id in range(n_agents):
            target = int(current_following[agent_id])
            if target < 0 or target == agent_id:
                continue
            start = coords[agent_id]
            end = coords[target]
            delta = end - start
            dist = float(np.linalg.norm(delta))
            if dist <= 1e-9:
                continue
            pad = 0.12 * delta / dist
            ax.annotate(
                "",
                xy=(end[0] - pad[0], end[1] - pad[1]),
                xytext=(start[0] + pad[0], start[1] + pad[1]),
                arrowprops=dict(arrowstyle="->", color="#7a7a7a", linewidth=1.2, alpha=0.9),
            )

        for agent_id in range(n_agents):
            role_label = str(current_roles[agent_id])
            color = role_colors.get(role_label, "#999999")
            ax.scatter(
                coords[agent_id, 0],
                coords[agent_id, 1],
                s=520,
                color=color,
                edgecolor="black",
                linewidth=0.8,
                zorder=3,
            )
            ax.text(
                coords[agent_id, 0],
                coords[agent_id, 1],
                str(agent_id),
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
                zorder=4,
            )

        ax.scatter([], [], s=180, color=role_colors["personal_utility"], label="PU")
        ax.scatter([], [], s=180, color=role_colors["reputation"], label="Reputation")
        ax.scatter([], [], s=180, color=role_colors["status"], label="Status")
        ax.legend(loc="upper right", frameon=False)
        ax.set_title(
            f"{title_prefix} t={step} | top leader={top_leader} | top followers={top_followers}",
            fontsize=11,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"toy_follow_graph_t{int(step):04d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_toy_follow_timeline(
    trace: DetailedTrace,
    output_file: Path,
    *,
    title: str,
) -> None:
    following_history = np.asarray(trace.get("following_history", []), dtype=int)
    role_update_times = [int(t) for t in np.asarray(trace.get("role_update_times", []), dtype=int).tolist()]
    if following_history.size == 0 or not role_update_times:
        return

    valid_steps = [step for step in role_update_times if 1 <= int(step) <= int(following_history.shape[0])]
    if not valid_steps:
        return
    update_indices = [int(step) - 1 for step in valid_steps]
    matrix = following_history[update_indices].T + 1  # -1 -> 0 for "none"
    n_agents = following_history.shape[1]

    base_colors = ["#f5f5f5"] + [matplotlib.cm.tab20(i % 20) for i in range(n_agents)]
    cmap = matplotlib.colors.ListedColormap(base_colors)
    norm = matplotlib.colors.BoundaryNorm(np.arange(-0.5, n_agents + 1.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(max(7.0, 0.24 * len(valid_steps)), 3.8))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Role update timestep")
    ax.set_ylabel("Agent id")
    ax.set_yticks(np.arange(n_agents))
    ax.set_yticklabels([str(i) for i in range(n_agents)])
    x_ticks = np.arange(len(valid_steps))
    if len(valid_steps) > 12:
        keep = np.linspace(0, len(valid_steps) - 1, num=12, dtype=int)
        x_ticks = np.unique(keep)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(valid_steps[idx]) for idx in x_ticks], rotation=45, ha="right")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_ticks(np.arange(n_agents + 1))
    cbar.set_ticklabels(["none"] + [str(i) for i in range(n_agents)])
    cbar.set_label("Current following target")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_agent_estimate_trajectories(
    trace: DetailedTrace,
    output_file: Path,
    title: str,
    *,
    zoom_to_follow_window: bool,
) -> None:
    pu_history = np.asarray(trace["estimated_reward_pu_history"], dtype=float)
    rep_history = np.asarray(trace["weighted_selected_reputation_history"], dtype=float)
    role_update_times = [int(t) for t in np.asarray(trace["role_update_times"], dtype=int).tolist()]
    n_steps, n_agents = pu_history.shape
    x = np.arange(1, n_steps + 1)

    cols = 4 if n_agents >= 8 else 2
    rows = int(np.ceil(n_agents / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.8 * rows), squeeze=False)

    for agent_id in range(n_agents):
        ax = axes[agent_id // cols][agent_id % cols]
        pu = pu_history[:, agent_id]
        rep = rep_history[:, agent_id]
        follow_mask = rep > pu

        ax.plot(x, pu, label="PU estimate", linewidth=1.2, color="tab:blue")
        ax.plot(x, rep, label="gamma * reputation", linewidth=1.2, color="tab:red")
        if np.any(follow_mask):
            ax.fill_between(x, pu, rep, where=follow_mask, color="tab:red", alpha=0.16, step=None)

        for idx_line, step in enumerate(role_update_times):
            ax.axvline(
                int(step),
                color="gray",
                linestyle="--",
                linewidth=0.7,
                alpha=0.22,
                label="Role update" if agent_id == 0 and idx_line == 0 else None,
            )

        if zoom_to_follow_window and np.any(follow_mask):
            idx = np.where(follow_mask)[0]
            pad = max(10, int(0.05 * n_steps))
            start = max(1, int(idx[0] + 1 - pad))
            end = min(n_steps, int(idx[-1] + 1 + pad))
            ax.set_xlim(start, end)
        elif zoom_to_follow_window:
            ax.text(
                0.5,
                0.90,
                "No rep > PU window",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="dimgray",
            )

        ax.set_title(f"Agent {agent_id}", fontsize=10)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Estimate")
        ax.grid(True, alpha=0.3)

    for idx in range(n_agents, rows * cols):
        axes[idx // cols][idx % cols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()


def plot_paper_style_summary(
    mode: str,
    gammas: Sequence[float],
    top_series_by_gamma: Dict[float, List[np.ndarray]],
    leader_series_by_gamma: Dict[float, List[np.ndarray]],
    output_file: Path,
    num_agents: int,
    sample_interval: int,
    role_update_times: Sequence[int],
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
    for idx, step in enumerate(role_update_times):
        ax.axvline(
            int(step),
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.22,
            label="Role update" if idx == 0 else None,
        )
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
    kappas = parse_kappas(args.kappas)
    role_t_seq = parse_role_update_T_seq(args.role_update_T_seq)
    role_epochs = parse_role_update_epochs(args.role_update_epochs)
    seeds = resolve_seeds(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("#" * 72, flush=True)
    print("Reputation Scaling Run", flush=True)
    print(f"mode={args.mode}", flush=True)
    print(f"gammas={gammas}", flush=True)
    print(f"kappas={kappas}", flush=True)
    print(f"num_agents={args.num_agents}, num_states={args.num_states}, num_actions={args.num_actions}", flush=True)
    if seeds:
        if seeds == list(range(seeds[0], seeds[-1] + 1)):
            seed_label = f"{seeds[0]}..{seeds[-1]}"
        else:
            seed_label = ",".join(str(seed) for seed in seeds)
        print(f"num_steps={args.num_steps}, seeds={len(seeds)} ({seed_label})", flush=True)
    else:
        print(f"num_steps={args.num_steps}, seeds=0", flush=True)
    print(f"delta={args.delta}", flush=True)
    print(
        "actor_rate_driver_mode="
        f"{args.actor_rate_driver_mode} "
        f"(status_override_min_followers={args.actor_rate_status_override_min_followers})",
        flush=True,
    )
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
    if args.reward_model == "shared_base_gaussian":
        print(
            "reward_model=shared_base_gaussian "
            f"(base_mu={args.reward_base_mu}, base_sigma={args.reward_base_sigma}, "
            f"agent_sigma={args.reward_agent_sigma}, clip=[{args.reward_clip_min},{args.reward_clip_max}])",
            flush=True,
        )
    else:
        print("reward_model=simple_preferred_action", flush=True)
    print(f"plot_sample_interval={args.plot_sample_interval}", flush=True)
    print(f"tracking_mode={args.tracking_mode}, numpy_fast_path={args.numpy_fast_path}", flush=True)
    print(f"eq9_averaging_mode={getattr(args, 'eq9_averaging_mode', 'participants_only')}", flush=True)
    print(
        f"leader_update_mode={getattr(args, 'leader_update_mode', 'participants_only_post_eq9')}",
        flush=True,
    )
    if args.mode == "async":
        if args.async_role_update_prob is None:
            async_t_seq, async_s0, async_src = _build_async_interval_sequence(args)
            print(
                f"async_mode=independent_agent_clocks(source={async_src}, s0={async_s0}, T_n={async_t_seq}, random_phase_in=[1,{int(async_t_seq[0])}], activity_coupled=False)",
                flush=True,
            )
        else:
            print(f"async_mode=bernoulli_per_agent(p={float(args.async_role_update_prob):.6f})", flush=True)
        print(f"async_decision_audit={bool(args.async_decision_audit)}", flush=True)
    print("#" * 72, flush=True)

    all_records: List[RunRecord] = []
    top_series_by_gamma: Dict[float, List[np.ndarray]] = {g: [] for g in gammas}
    leader_series_by_gamma: Dict[float, List[np.ndarray]] = {g: [] for g in gammas}
    detailed_traces: Dict[Tuple[float, int], DetailedTrace] = {}
    async_diagnosis_rows: List[dict] = []
    role_update_diagnostic_rows: List[dict] = []
    role_update_diagnostic_summaries: List[dict] = []
    true_reputation_checkpoint_rows: List[dict] = []
    estimate_consensus_checkpoint_rows: List[dict] = []
    rate_audit_checkpoint_rows: List[dict] = []
    role_update_times = build_static_role_update_times(args, horizon=int(args.num_steps))
    total_jobs = len(gammas) * len(seeds)
    job = 0
    runs_csv = output_dir / f"reputation_scaling_runs_{args.mode}.csv"
    agg_csv = output_dir / f"reputation_scaling_aggregate_{args.mode}.csv"
    table_csv = output_dir / f"reputation_scaling_table_values_{args.mode}.csv"
    role_diag_csv = output_dir / f"reputation_scaling_role_update_diagnostics_{args.mode}.csv"
    role_summary_csv = output_dir / f"reputation_scaling_seed_diagnostic_summary_{args.mode}.csv"
    true_reputation_csv = output_dir / "expB_true_reputation_checkpoints.csv"
    estimate_consensus_csv = output_dir / "expB_estimate_consensus_checkpoints.csv"
    rate_audit_csv = output_dir / "expB_rate_audit_checkpoints.csv"
    rank_alignment_csv = output_dir / "expB_rank_alignment_checkpoints.csv"
    prog_png = output_dir / f"reputation_scaling_progression_{args.mode}.png"
    curve_png = output_dir / f"reputation_scaling_top_followers_{args.mode}.png"
    paper_png = output_dir / f"reputation_scaling_paper_style_{args.mode}.png"
    small_n_rep_long_csv = output_dir / "expB_reputation_trace_long.csv"
    small_n_agent_long_csv = output_dir / "expB_agent_state_trace_long.csv"
    small_n_true_rep_vs_estimate_csv = output_dir / "expB_true_rep_vs_estimate_trace_long.csv"
    small_n_true_rep_decomp_csv = output_dir / "expB_true_reputation_decomposition_long.csv"
    small_n_alignment_by_update_csv = output_dir / "expB_toy_alignment_by_update.csv"
    small_n_v_to_s_by_update_csv = output_dir / "expB_toy_v_to_s_by_update.csv"
    small_n_v_to_s_recurrence_audit_csv = output_dir / "expB_toy_v_to_s_recurrence_audit_long.csv"
    small_n_s_to_highest_by_update_csv = output_dir / "expB_toy_s_to_highest_by_update.csv"
    small_n_step1_by_update_csv = output_dir / "expB_toy_step1_by_update.csv"
    small_n_choice_trace_csv = output_dir / "expB_toy_choice_trace_long.csv"
    small_n_consensus_by_step_csv = output_dir / "expB_toy_consensus_by_step.csv"
    small_n_follow_relationships_csv = output_dir / "expB_toy_follow_relationships_long.csv"
    small_n_trace_requested = bool(getattr(args, "small_n_trace_export", False) and int(args.num_agents) <= 12)

    for gamma, kappa in zip(gammas, kappas):
            for seed in seeds:
                job += 1
                print(f"[{job:03d}/{total_jobs:03d}] mode={args.mode} gamma={gamma:g} kappa={kappa:g} seed={seed}", flush=True)
                (
                    rec,
                    top_series,
                    leader_series,
                    detailed_trace,
                    async_debug,
                    run_role_update_rows,
                    run_checkpoint_audit_rows,
                ) = run_single(
                    args=args,
                    mode=args.mode,
                    gamma=gamma,
                    kappa=kappa,
                    seed=seed,
                )
                all_records.append(rec)
                top_series_by_gamma[gamma].append(top_series)
                leader_series_by_gamma[gamma].append(leader_series)
                if detailed_trace is not None:
                    save_trace = (
                        small_n_trace_requested
                        or
                        args.trace_detailed_seeds == "all"
                        or (args.trace_detailed_seeds == "first" and seed == seeds[0])
                    )
                    if save_trace:
                        detailed_traces[(gamma, seed)] = detailed_trace
                if async_debug is not None:
                    gamma_tag = _format_gamma(gamma)
                    scheduler_csv = output_dir / f"reputation_scaling_async_scheduler_g{gamma_tag}_seed{seed}.csv"
                    audit_csv = output_dir / f"reputation_scaling_async_decision_audit_g{gamma_tag}_seed{seed}.csv"
                    focus_csv = output_dir / f"reputation_scaling_async_focus_trace_g{gamma_tag}_seed{seed}.csv"
                    diagnosis_md = output_dir / f"reputation_scaling_async_diagnosis_g{gamma_tag}_seed{seed}.md"

                    write_async_scheduler_csv(async_debug["scheduler_rows"], scheduler_csv)
                    write_csv(audit_csv, async_debug["decision_audit_rows"])
                    if async_debug["trace_bundle"] is not None:
                        write_async_focus_trace_csv(
                            async_debug["trace_bundle"],
                            async_debug["decision_audit_rows"],
                            async_debug["focus_agents"],
                            focus_csv,
                        )
                    write_async_diagnosis_markdown(
                        output_file=diagnosis_md,
                        diagnosis=async_debug["diagnosis"],
                        focus_agents=async_debug["focus_agents"],
                    )
                    async_diagnosis_rows.append(dict(async_debug["diagnosis"]))
                if run_role_update_rows is not None:
                    enriched_rows = enrich_role_update_diagnostic_rows(
                        mode=args.mode,
                        gamma=gamma,
                        seed=seed,
                        rows=run_role_update_rows,
                    )
                    role_update_diagnostic_rows.extend(enriched_rows)
                    role_update_diagnostic_summaries.append(
                        summarize_role_update_diagnostics(
                            mode=args.mode,
                            gamma=gamma,
                            seed=seed,
                            num_agents=args.num_agents,
                            record=rec,
                            top_follower_series=top_series,
                            rows=enriched_rows,
                        )
                    )
                if run_checkpoint_audit_rows is not None:
                    true_reputation_checkpoint_rows.extend(
                        enrich_checkpoint_rows(
                            mode=args.mode,
                            gamma=gamma,
                            seed=seed,
                            rows=run_checkpoint_audit_rows.get("true_reputation_checkpoints", []),
                        )
                    )
                    estimate_consensus_checkpoint_rows.extend(
                        enrich_checkpoint_rows(
                            mode=args.mode,
                            gamma=gamma,
                            seed=seed,
                            rows=run_checkpoint_audit_rows.get("estimate_consensus_checkpoints", []),
                        )
                    )
                    rate_audit_checkpoint_rows.extend(
                        enrich_checkpoint_rows(
                            mode=args.mode,
                            gamma=gamma,
                            seed=seed,
                            rows=run_checkpoint_audit_rows.get("rate_audit_checkpoints", []),
                        )
                    )
                # Incremental checkpoint for long sweeps.
                write_csv(runs_csv, [asdict(r) for r in all_records])
                if role_update_diagnostic_rows:
                    write_csv(role_diag_csv, role_update_diagnostic_rows)
                if role_update_diagnostic_summaries:
                    write_csv(role_summary_csv, role_update_diagnostic_summaries)
                if true_reputation_checkpoint_rows:
                    write_csv(true_reputation_csv, true_reputation_checkpoint_rows)
                if estimate_consensus_checkpoint_rows:
                    write_csv(estimate_consensus_csv, estimate_consensus_checkpoint_rows)
                if rate_audit_checkpoint_rows:
                    write_csv(rate_audit_csv, rate_audit_checkpoint_rows)
                rank_alignment_rows = summarize_rank_alignment_checkpoints(
                    true_rows=true_reputation_checkpoint_rows,
                    estimate_rows=estimate_consensus_checkpoint_rows,
                )
                if rank_alignment_rows:
                    write_csv(rank_alignment_csv, rank_alignment_rows)

    agg_records = aggregate(all_records)

    write_csv(runs_csv, [asdict(r) for r in all_records])
    write_csv(agg_csv, [asdict(r) for r in agg_records])
    write_table_values_csv(gammas=gammas, aggregate_rows=agg_records, output_file=table_csv)
    if async_diagnosis_rows:
        write_csv(
            output_dir / "reputation_scaling_async_diagnosis_summary.csv",
            async_diagnosis_rows,
        )
    if role_update_diagnostic_rows:
        write_csv(role_diag_csv, role_update_diagnostic_rows)
    if role_update_diagnostic_summaries:
        write_csv(role_summary_csv, role_update_diagnostic_summaries)
    if true_reputation_checkpoint_rows:
        write_csv(true_reputation_csv, true_reputation_checkpoint_rows)
    if estimate_consensus_checkpoint_rows:
        write_csv(estimate_consensus_csv, estimate_consensus_checkpoint_rows)
    if rate_audit_checkpoint_rows:
        write_csv(rate_audit_csv, rate_audit_checkpoint_rows)
    rank_alignment_rows = summarize_rank_alignment_checkpoints(
        true_rows=true_reputation_checkpoint_rows,
        estimate_rows=estimate_consensus_checkpoint_rows,
    )
    if rank_alignment_rows:
        write_csv(rank_alignment_csv, rank_alignment_rows)
    plot_progression(
        args.mode,
        gammas,
        top_series_by_gamma,
        prog_png,
        sample_interval=max(1, int(args.plot_sample_interval)),
        role_update_times=role_update_times,
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
        role_update_times=role_update_times,
    )

    for (gamma, seed), trace in detailed_traces.items():
        gamma_tag = _format_gamma(gamma)
        trace_csv = output_dir / f"reputation_scaling_agent_traces_g{gamma_tag}_seed{seed}_{args.mode}.csv"
        full_png = output_dir / f"reputation_scaling_agent_traces_full_g{gamma_tag}_seed{seed}_{args.mode}.png"
        zoom_png = output_dir / f"reputation_scaling_agent_traces_zoom_g{gamma_tag}_seed{seed}_{args.mode}.png"

        write_agent_trace_csv(trace, trace_csv)
        plot_agent_estimate_trajectories(
            trace,
            full_png,
            title=f"Experiment B traces: gamma={gamma:g} (full)",
            zoom_to_follow_window=False,
        )
        plot_agent_estimate_trajectories(
            trace,
            zoom_png,
            title=f"Experiment B traces: gamma={gamma:g} (zoomed to rep > PU)",
            zoom_to_follow_window=True,
        )

    has_dense_small_n_trace = any(
        np.asarray(trace.get("dense_reputation_history", []), dtype=float).size > 0
        for trace in detailed_traces.values()
    )
    if has_dense_small_n_trace:
        write_small_n_reputation_trace_long_csv(detailed_traces, small_n_rep_long_csv)
        write_small_n_agent_state_trace_long_csv(detailed_traces, small_n_agent_long_csv)
        write_small_n_true_rep_vs_estimate_trace_long_csv(detailed_traces, small_n_true_rep_vs_estimate_csv)
        write_small_n_true_reputation_decomposition_long_csv(detailed_traces, small_n_true_rep_decomp_csv)
        write_small_n_toy_alignment_by_update_csv(detailed_traces, small_n_alignment_by_update_csv)
        write_small_n_toy_v_to_s_by_update_csv(detailed_traces, small_n_v_to_s_by_update_csv)
        write_small_n_toy_v_to_s_recurrence_audit_csv(detailed_traces, small_n_v_to_s_recurrence_audit_csv)
        write_small_n_toy_s_to_highest_by_update_csv(detailed_traces, small_n_s_to_highest_by_update_csv)
        write_small_n_toy_step1_by_update_csv(detailed_traces, small_n_step1_by_update_csv)
        write_small_n_toy_choice_trace_long_csv(detailed_traces, small_n_choice_trace_csv)
        write_small_n_toy_consensus_by_step_csv(detailed_traces, small_n_consensus_by_step_csv)
        write_small_n_toy_follow_relationships_long_csv(detailed_traces, small_n_follow_relationships_csv)
        for (gamma, seed), trace in sorted(detailed_traces.items()):
            dense_history = np.asarray(trace.get("dense_reputation_history", []), dtype=float)
            if dense_history.size == 0:
                continue
            gamma_tag = _format_gamma(gamma)
            mean_signal_png = output_dir / f"toy_mean_gate_signals_g{gamma_tag}_seed{seed}_{args.mode}.png"
            timeline_png = output_dir / f"toy_follow_timeline_g{gamma_tag}_seed{seed}_{args.mode}.png"
            highest_timeline_png = output_dir / f"toy_highest_target_timeline_g{gamma_tag}_seed{seed}_{args.mode}.png"
            root_timeline_png = output_dir / f"toy_root_leader_timeline_g{gamma_tag}_seed{seed}_{args.mode}.png"
            graph_dir = output_dir / f"toy_follow_graphs_g{gamma_tag}_seed{seed}_{args.mode}"
            plot_toy_mean_gate_signals(
                trace,
                mean_signal_png,
                gamma=gamma,
                title=f"Toy mean gate signals: gamma={gamma:g}, seed={seed}",
            )
            plot_toy_follow_timeline(
                trace,
                timeline_png,
                title=f"Toy following timeline: gamma={gamma:g}, seed={seed}",
            )
            plot_toy_highest_target_timeline(
                trace,
                highest_timeline_png,
                title=f"Toy highest target timeline: gamma={gamma:g}, seed={seed}",
            )
            plot_toy_root_leader_timeline(
                trace,
                root_timeline_png,
                title=f"Toy root leader timeline: gamma={gamma:g}, seed={seed}",
            )
            plot_toy_follow_graph_snapshots(
                trace,
                graph_dir,
                title_prefix=f"Toy follow graph (gamma={gamma:g}, seed={seed})",
            )

    print("\nCompleted.", flush=True)
    print(f"Per-run CSV:     {runs_csv}", flush=True)
    print(f"Aggregate CSV:   {agg_csv}", flush=True)
    print(f"Table CSV:       {table_csv}", flush=True)
    if role_update_diagnostic_rows:
        print(f"Role diag CSV:   {role_diag_csv}", flush=True)
    if role_update_diagnostic_summaries:
        print(f"Role sum CSV:    {role_summary_csv}", flush=True)
    if true_reputation_checkpoint_rows:
        print(f"True rep CSV:    {true_reputation_csv}", flush=True)
    if estimate_consensus_checkpoint_rows:
        print(f"Estimate CSV:    {estimate_consensus_csv}", flush=True)
    if rate_audit_checkpoint_rows:
        print(f"Rate audit CSV:  {rate_audit_csv}", flush=True)
    if rank_alignment_rows:
        print(f"Rank align CSV:  {rank_alignment_csv}", flush=True)
    print(f"Progression PNG: {prog_png}", flush=True)
    print(f"Curve PNG:       {curve_png}", flush=True)
    print(f"Paper PNG:       {paper_png}", flush=True)
    if has_dense_small_n_trace:
        print(f"Small-N rep CSV: {small_n_rep_long_csv}", flush=True)
        print(f"Small-N agent CSV: {small_n_agent_long_csv}", flush=True)
        print(f"Small-N true/estimate CSV: {small_n_true_rep_vs_estimate_csv}", flush=True)
        print(f"Small-N true rep decomp CSV: {small_n_true_rep_decomp_csv}", flush=True)
        print(f"Toy alignment CSV: {small_n_alignment_by_update_csv}", flush=True)
        print(f"Toy v->s CSV: {small_n_v_to_s_by_update_csv}", flush=True)
        print(f"Toy v->s audit CSV: {small_n_v_to_s_recurrence_audit_csv}", flush=True)
        print(f"Toy s->highest CSV: {small_n_s_to_highest_by_update_csv}", flush=True)
        print(f"Toy Step-1 CSV: {small_n_step1_by_update_csv}", flush=True)
    if detailed_traces:
        print("Detailed trace artifacts written for:", flush=True)
        for gamma, seed in sorted(detailed_traces.keys()):
            gamma_tag = _format_gamma(gamma)
            print(
                f"  gamma={gamma:g} seed={seed}: "
                f"reputation_scaling_agent_traces_g{gamma_tag}_seed{seed}_{args.mode}.csv",
                flush=True,
            )
            if has_dense_small_n_trace:
                print(
                    f"  gamma={gamma:g} seed={seed}: "
                    f"toy_follow_graphs_g{gamma_tag}_seed{seed}_{args.mode}/",
                    flush=True,
                )


if __name__ == "__main__":
    main()
