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
