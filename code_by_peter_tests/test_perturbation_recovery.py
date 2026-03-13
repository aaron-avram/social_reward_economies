"""Tests for Experiment D perturbation/recovery harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from _shared import load_model_module, make_system


ROOT = Path(__file__).resolve().parents[1]
PERTURBATION_RUNNER_PATH = ROOT / "experiments" / "perturbation_recovery.py"


def load_perturbation_module():
    spec = importlib.util.spec_from_file_location("perturbation_recovery_module", PERTURBATION_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def model_module():
    return load_model_module()


@pytest.fixture(scope="module")
def perturbation_module():
    return load_perturbation_module()


def test_apply_low_payoff_perturbation_prefers_nonpreferred_action(model_module, perturbation_module):
    system = make_system(model_module, num_agents=1)
    agent = system.agents[0]

    perturbation_module.apply_low_payoff_perturbation(agent, strength=8.0)

    num_states = system.config.num_states
    num_actions = system.config.num_actions
    pref = int(agent.preferred_action % num_actions)
    anti = int((pref + 1) % num_actions)

    for state in range(num_states):
        pu_policy = agent.get_softmax_policy(state, agent.state.weights_pu)
        st_policy = agent.get_softmax_policy(state, agent.state.weights_status)

        assert pu_policy[anti] > 0.95
        assert pu_policy[pref] < 0.05
        assert st_policy[anti] > 0.95
        assert st_policy[pref] < 0.05


def test_apply_targeted_low_payoff_perturbation_chooses_follower_worst_action(
    model_module,
    perturbation_module,
):
    config = model_module.SystemConfig(
        num_agents=3,
        num_states=2,
        num_actions=2,
        reward_model="shared_base_gaussian",
        reward_base_mu=0.5,
        reward_base_sigma=0.01,
        reward_agent_sigma=0.01,
        use_numpy_fast_path=False,
    )
    system = model_module.MultiAgentSystem(config)
    leader = 0
    target_ids = [1, 2]

    system._reward_tables[1, 0, :] = [1.0, 0.1]
    system._reward_tables[2, 0, :] = [0.9, 0.2]
    system._reward_tables[1, 1, :] = [0.1, 1.0]
    system._reward_tables[2, 1, :] = [0.2, 0.9]

    perturbation_module.apply_targeted_low_payoff_perturbation(
        system,
        leader,
        strength=7.0,
        target_ids=target_ids,
    )

    policy_s0 = system.agents[leader].get_softmax_policy(0, system.agents[leader].state.weights_pu)
    policy_s1 = system.agents[leader].get_softmax_policy(1, system.agents[leader].state.weights_pu)
    assert policy_s0[1] > 0.95  # action 1 hurts followers most in state 0
    assert policy_s1[0] > 0.95  # action 0 hurts followers most in state 1


def test_apply_reputation_shock_scales_leader_column(model_module, perturbation_module):
    system = make_system(model_module, num_agents=3)
    leader_id = 1

    for a in system.agents:
        a.state.reputation_estimates[leader_id] = 2.0

    if getattr(system, "_s_matrix", None) is not None:
        system._s_matrix[:, leader_id] = 2.0

    perturbation_module.apply_reputation_shock(system, leader_id=leader_id, factor=0.25)

    for a in system.agents:
        assert a.state.reputation_estimates[leader_id] == pytest.approx(0.5, abs=1e-12)

    if getattr(system, "_s_matrix", None) is not None:
        assert np.allclose(system._s_matrix[:, leader_id], 0.5)


def test_collapse_leader_followership_clears_followers_and_demotes_to_pu(model_module, perturbation_module):
    system = make_system(model_module, num_agents=4)
    AgentRole = model_module.AgentRole

    leader = 0
    follower_ids = [1, 2]
    system.agents[leader].state.followers = set(follower_ids)

    for fid in follower_ids:
        agent = system.agents[fid]
        agent.state.role = AgentRole.REPUTATION
        agent.state.following = leader
        agent.state.was_following = True

    perturbation_module.collapse_leader_followership(system, leader)

    assert system.agents[leader].state.followers == set()
    for fid in follower_ids:
        agent = system.agents[fid]
        assert agent.state.role.value == AgentRole.PERSONAL_UTILITY.value
        assert agent.state.following is None
        assert agent.state.was_following is False


def test_detect_first_hold_requires_full_hold_window(perturbation_module):
    series = [4, 5, 5, 4, 5, 5, 5]
    idx = perturbation_module.detect_first_hold_index(series, threshold=5, hold_steps=3)
    assert idx == 6


def test_compute_normless_duration_finds_longest_contiguous_segment(perturbation_module):
    top_series = [7, 6, 2, 1, 3, 2, 1, 0, 5]
    duration = perturbation_module.compute_normless_duration(
        top_series,
        dominant_threshold=4,
        start_idx=2,
        end_idx=8,
    )
    assert duration == 6


def test_recovery_detector_respects_start_index_and_hold_window(perturbation_module):
    series = [1, 2, 9, 9, 4, 9, 9, 9]
    idx = perturbation_module.detect_first_hold_index(
        series,
        threshold=9,
        hold_steps=2,
        start_idx=4,
    )
    assert idx == 6


def test_perturbation_applies_to_detected_leader(monkeypatch, perturbation_module):
    calls = []
    original_apply = perturbation_module.apply_targeted_low_payoff_perturbation

    def spy_apply(system, leader_id, *, strength, target_ids):
        calls.append(leader_id)
        return original_apply(system, leader_id, strength=strength, target_ids=target_ids)

    monkeypatch.setattr(perturbation_module, "apply_targeted_low_payoff_perturbation", spy_apply)

    result = perturbation_module.run_experiment(
        mode="static",
        num_agents=8,
        num_states=3,
        num_actions=2,
        num_steps_max=300,
        gamma=2.0,
        kappa=2.0,
        seeds=1,
        seed_start=0,
        role_update_base_interval=20,
        fixed_role_update_interval=True,
        perturb_strength=7.0,
        perturb_duration=8,
        post_window=80,
        conv_threshold=1,   # intentionally low for deterministic, fast triggering
        conv_hold_steps=1,
        recovery_threshold=1,
        recovery_hold_steps=1,
        dominant_threshold=0.5,
        output_prefix="test_perturb_target",
    )

    rec = result["run_records"][0]
    if rec.t_perturb_start > 0:
        assert len(calls) > 0
        assert set(calls) == {rec.leader_pre}


def test_perturbation_recovery_integration_smoke(tmp_path, perturbation_module):
    result = perturbation_module.run_experiment(
        mode="static",
        num_agents=8,
        num_states=3,
        num_actions=2,
        num_steps_max=500,
        gamma=2.0,
        kappa=2.0,
        seeds=1,
        seed_start=0,
        role_update_base_interval=25,
        fixed_role_update_interval=True,
        perturb_strength=8.0,
        perturb_duration=20,
        post_window=120,
        conv_threshold=1,
        conv_hold_steps=1,
        recovery_threshold=1,
        recovery_hold_steps=1,
        dominant_threshold=0.5,
        output_dir=str(tmp_path),
        plot_sample_interval=25,
        tracking_mode="light",
        output_prefix="smoke_d",
    )

    runs_csv = Path(result["runs_csv"])
    agg_csv = Path(result["aggregate_csv"])
    plot_file = Path(next(iter(result["plot_files"].values())))

    assert runs_csv.exists()
    assert agg_csv.exists()
    assert plot_file.exists()

    rec = result["run_records"][0]
    assert np.isfinite(rec.final_top_followers)
    assert np.isfinite(rec.welfare_drop) or np.isnan(rec.welfare_drop)
    assert isinstance(rec.stable_recovery, bool)
    assert rec.stable_tail_window >= 1

    if rec.recovery_time > 0:
        assert rec.t_conv < rec.t_perturb_start <= rec.t_perturb_end < rec.recovery_time
