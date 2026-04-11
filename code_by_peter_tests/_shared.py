from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "src" / "code_debugged.py"


def load_model_module():
    spec = importlib.util.spec_from_file_location("code_debugged_module", TARGET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_system(model_module, *, num_agents=3, extra_config=None):
    kwargs = dict(
        num_agents=num_agents,
        num_states=3,
        num_actions=2,
        num_time_steps=1,
        role_update_base_interval=10**9,
        gossip_rate=0.0,
    )
    if extra_config:
        kwargs.update(extra_config)
    config = model_module.SystemConfig(**kwargs)
    return model_module.MultiAgentSystem(config)


def estimate_activation_frequency(system, *, steps: int, seed: int):
    np.random.seed(seed)
    for _ in range(steps):
        system.step()
    actor_empirical = float(np.mean(system.results["actor_counts"]))
    participant_empirical = float(np.mean(system.results["participant_counts"]))
    return actor_empirical, participant_empirical


def gossip_sync_update(rep_snapshot, delta_v_by_i, num_agents):
    participants = list(rep_snapshot.keys())
    new_rep = {i: {} for i in participants}
    for i in participants:
        for k in range(num_agents):
            avg_est = sum(rep_snapshot[j].get(k, 0.0) for j in participants) / len(participants)
            dv = delta_v_by_i.get(i, {}).get(k, 0.0)
            new_rep[i][k] = avg_est + dv
    return new_rep


def gossip_inplace_update(rep, delta_v_by_i, num_agents, update_order=None):
    participants = list(rep.keys())
    if update_order is None:
        update_order = participants[:]
    for i in update_order:
        for k in range(num_agents):
            avg_est = sum(rep[j].get(k, 0.0) for j in participants) / len(participants)
            dv = delta_v_by_i.get(i, {}).get(k, 0.0)
            rep[i][k] = avg_est + dv
    return rep


def variance(xs):
    xs = np.array(xs, dtype=float)
    return float(np.var(xs))


def set_reputation_learning_state(
    system,
    *,
    personal_benefit_matrix,
    reputation_matrix,
    highest_rep_agent_estimates,
):
    system._set_reputation_learning_state_for_audit(
        personal_benefit_matrix=np.asarray(personal_benefit_matrix, dtype=float),
        reputation_matrix=np.asarray(reputation_matrix, dtype=float),
        highest_rep_agent_estimates=list(highest_rep_agent_estimates),
    )


def get_reputation_learning_snapshot(system):
    snap = system.get_reputation_learning_snapshot()
    return {
        "v_matrix": np.asarray(snap["v_matrix"], dtype=float),
        "s_matrix": np.asarray(snap["s_matrix"], dtype=float),
        "highest_rep_agent_estimates": np.asarray(snap["highest_rep_agent_estimates"], dtype=int),
        "actor_rates": np.asarray(snap["actor_rates"], dtype=float),
    }


def _deterministic_highest_rep_choice(row, agent_id, delta):
    row = np.asarray(row, dtype=float).copy()
    if row.size <= 1:
        return int(agent_id)
    row[int(agent_id)] = -np.inf
    max_rep = np.max(row)
    if not np.isfinite(max_rep):
        candidates = [idx for idx in range(row.size) if idx != int(agent_id)]
        return int(candidates[0]) if candidates else int(agent_id)
    candidates = [
        idx for idx, value in enumerate(row)
        if idx != int(agent_id) and value >= float(max_rep) - float(delta)
    ]
    if not candidates:
        candidates = [idx for idx in range(row.size) if idx != int(agent_id)]
    return int(min(candidates)) if candidates else int(agent_id)


def gossip_phase_oracle(
    *,
    v_before,
    s_before,
    highest_rep_before,
    observed_utility_matrix,
    active_actor_ids,
    active_participant_ids,
    eta_v_t,
    delta,
    eq9_averaging_mode="participants_only",
    leader_update_mode="participants_only_post_eq9",
    averaging_agent_ids=None,
):
    v_prev = np.asarray(v_before, dtype=float)
    s_prev = np.asarray(s_before, dtype=float)
    observed = np.asarray(observed_utility_matrix, dtype=float)
    num_agents = int(v_prev.shape[0])

    if averaging_agent_ids is None:
        if eq9_averaging_mode == "participants_only":
            averaging_agent_ids = [int(i) for i in active_participant_ids]
        elif eq9_averaging_mode == "all_agents":
            averaging_agent_ids = list(range(num_agents))
        else:
            raise ValueError(
                f"Unsupported eq9_averaging_mode='{eq9_averaging_mode}'."
            )
    else:
        averaging_agent_ids = [int(i) for i in averaging_agent_ids]

    highest_prev = np.asarray(highest_rep_before, dtype=int)

    v_after = v_prev * (1.0 - float(eta_v_t))
    if active_actor_ids:
        active_actor_ids = [int(i) for i in active_actor_ids]
        v_after[:, active_actor_ids] = (
            v_prev[:, active_actor_ids]
            + float(eta_v_t) * (observed[:, active_actor_ids] - v_prev[:, active_actor_ids])
        )
    delta_v = v_after - v_prev

    s_after = np.array(s_prev, dtype=float, copy=True)
    gossip_target_ids = sorted(
        {
            int(highest_prev[participant_id])
            for participant_id in [int(i) for i in active_participant_ids]
            if 0 <= int(highest_prev[participant_id]) < num_agents
            and int(highest_prev[participant_id]) != int(participant_id)
        }
    )
    avg_s_by_target = {}
    for target_id in gossip_target_ids:
        if averaging_agent_ids:
            avg_s = float(np.mean(s_prev[np.array(averaging_agent_ids, dtype=int), target_id]))
        else:
            avg_s = 0.0
        avg_s_by_target[int(target_id)] = float(avg_s)
        for participant_id in [int(i) for i in active_participant_ids]:
            s_after[participant_id, target_id] = avg_s + delta_v[participant_id, target_id]

    highest_after = np.array(highest_prev, dtype=int, copy=True)
    if leader_update_mode == "participants_only_post_eq9":
        leader_update_agent_ids = [int(i) for i in active_participant_ids]
        leader_source = s_after
    elif leader_update_mode == "all_agents_post_eq9":
        leader_update_agent_ids = list(range(num_agents))
        leader_source = s_after
    elif leader_update_mode == "participants_only_pre_eq9":
        leader_update_agent_ids = [int(i) for i in active_participant_ids]
        leader_source = s_prev
    else:
        raise ValueError(
            f"Unsupported leader_update_mode='{leader_update_mode}'."
        )
    for participant_id in leader_update_agent_ids:
        highest_after[participant_id] = _deterministic_highest_rep_choice(
            leader_source[participant_id],
            participant_id,
            delta,
        )

    return {
        "v_matrix": v_after,
        "s_matrix": s_after,
        "highest_rep_agent_estimates": highest_after,
        "gossip_target_ids": gossip_target_ids,
        "leader_update_agent_ids": leader_update_agent_ids,
        "avg_s_by_target": avg_s_by_target,
        "delta_v_matrix": delta_v,
        "eq9_averaging_mode": str(eq9_averaging_mode),
        "leader_update_mode": str(leader_update_mode),
    }
