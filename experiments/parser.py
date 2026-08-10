"""
All parsing functionality for experiments
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def load_data() -> dict:
    """
    Transform CLI input into dictionary of experiment data

    Parameters:
    data_string: String of data
    """

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI
    """

    parser = argparse.ArgumentParser(description="Reputation scaling gamma sweep harness.")
    parser.add_argument("--mode", choices=["static", "async"], required=True)
    parser.add_argument("--gammas", type=str, default="0")
    parser.add_argument("--kappas", type=str, default="0")
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
        help="Optional explicit comma-separated seed list (e.g. \"2,7,9\"). " \
        "Overrides --seeds/--seed-start.",
    )
    parser.add_argument("--delta", type=float, default=0.15)
    parser.add_argument(
        "--actor-rate-driver-mode",
        choices=["standard", "status_if_followers_kappa0"],
        default="standard",
        help="Actor-rate driver mode for Eq. (13): paper-faithful standard or experimental " \
        "status override at kappa=0.",
    )
    parser.add_argument(
        "--actor-rate-status-override-min-followers",
        type=int,
        default=10,
        help="Follower-count threshold for the experimental status-driven actor-rate override.",
    )
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent
                                                              / "outputs"))
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
        help="When tracking-mode=full, export per-agent PU/reputation traces for none, " \
        "the first seed, or all seeds.",
    )
    parser.add_argument(
        "--small-n-trace-export",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For small-N debug runs only, export dense per-timestep reputation " \
        "matrices in long-form CSVs.",
    )
    parser.add_argument(
        "--force-all-active-debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug override: force all agents to be active as both actors "
        "and participants every step.",
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
        help="In async mode, write scheduler/decision audit CSVs "
        "and compact per-agent debug traces.",
    )
    parser.add_argument(
        "--role-update-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write lightweight role-update-only diagnostics for " \
        "convergence/fragmentation analysis.",
    )
    return parser.parse_args()

def _parse_string_to_list(text: str, dtype: Callable, sort: bool = False,) -> List[int|float]:
    """
    Parse text into a list of numerics

    Parameters:
    text: the string of numerics
    dtype: the numeric type to cast elements
    """

    if not text.strip():
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    l = [dtype(x) for x in parts]
    return sorted(set(n for n in l if n >= 0)) if sort else l

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
    t_seq = _parse_string_to_list(text=args.role_update_T_seq, dtype=int, sort=True)
    if t_seq:
        return t_seq, s0, "T_sequence"

    epochs = _parse_string_to_list(args.role_update_epochs, dtype=int, sort=True)
    if epochs:
        from_epochs = _interval_seq_from_epochs(s0=s0, epochs=epochs)
        if from_epochs:
            return from_epochs, s0, "epochs"

    return [max(1, int(args.role_update_base_interval))], s0, "base_interval"


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    """
    Parse seeds and handle empty seed input
    """
    selected = _parse_string_to_list(getattr(args, "selected_seeds", ""), dtype=int, sort=True)
    if selected:
        return selected
    return list(range(int(args.seed_start), int(args.seed_start) + int(args.seeds)))
