"""Tests for Experiment D perturbation/recovery harness."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from _shared import load_model_module, make_system


ROOT = Path(__file__).resolve().parents[1]
PERTURBATION_RUNNER_PATH = ROOT / "experiments" / "perturbation_recovery.py"
EXPERIMENTS_PATH = ROOT / "experiments" / "experiments.py"


def load_perturbation_module():
    spec = importlib.util.spec_from_file_location("perturbation_recovery_module", PERTURBATION_RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_experiments_module():
    spec = importlib.util.spec_from_file_location("experiments_module", EXPERIMENTS_PATH)
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


@pytest.fixture(scope="module")
def experiments_module():
    return load_experiments_module()


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


def test_apply_force_bad_action_perturbation_avoids_shared_good_action(model_module, perturbation_module):
    config = model_module.SystemConfig(
        num_agents=2,
        num_states=3,
        num_actions=2,
        reward_model="shared_good_bad_heterogeneous",
        reward_good_value=1.0,
        reward_bad_value=0.1,
        reward_agent_sigma=0.1,
        reward_order_gap=0.02,
        use_numpy_fast_path=False,
    )
    system = model_module.MultiAgentSystem(config)
    leader = 0

    perturbation_module.apply_force_bad_action_perturbation(system, leader, strength=9.0)

    for state, good_action in enumerate(system._shared_good_actions):
        policy = system.agents[leader].get_softmax_policy(state, system.agents[leader].state.weights_pu)
        assert policy[int(good_action)] < 0.05
        if system.config.num_actions == 2:
            bad_action = 1 - int(good_action)
            assert policy[bad_action] > 0.95


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


def test_derive_interval_scaled_windows_matches_1000_step_plan(perturbation_module):
    windows = perturbation_module.derive_interval_scaled_windows(1000)
    assert windows == {
        "perturb_duration": 3000,
        "conv_hold_steps": 1200,
        "recovery_hold_steps": 800,
        "stable_tail_window": 2000,
    }


def test_compute_alt_leader_stats_excludes_preleader(perturbation_module):
    alt_leader_id, alt_followers = perturbation_module.compute_alt_leader_stats([7, 3, 5, 2], ex_leader_id=0)
    assert alt_leader_id == 2
    assert alt_followers == 5


def test_summarize_positive_step1_margins_reports_share_and_mean(perturbation_module):
    share, mean_positive = perturbation_module.summarize_positive_step1_margins([-0.2, 0.0, 0.3, 0.6])
    assert share == pytest.approx(0.5, abs=1e-12)
    assert mean_positive == pytest.approx(0.45, abs=1e-12)


def test_step1_diagnostics_use_selected_target_not_self_inclusive_max(perturbation_module):
    config = perturbation_module.SystemConfig(
        num_agents=3,
        num_states=3,
        num_actions=2,
        num_time_steps=1,
        role_update_base_interval=10**9,
        gossip_rate=0.0,
        gamma=1.0,
        B_R=0.5,
        B_F=0.1,
        use_numpy_fast_path=False,
    )
    system = perturbation_module.MultiAgentSystem(config)

    agent = system.agents[1]
    agent.state.role = perturbation_module.AgentRole.PERSONAL_UTILITY
    agent.state.followers = set()
    agent.state.following = None
    agent.state.highest_rep_agent_estimate = 0
    agent.state.reputation_estimates = {0: 0.4, 1: 9.0, 2: 0.2}
    agent.state.estimated_reward_pu = 0.0

    terms = perturbation_module._compute_step1_diagnostic_terms(system, 1)

    assert terms["target_id"] == 0
    assert terms["selected_rep_weighted"] == pytest.approx(0.4, abs=1e-12)
    assert terms["step1_margin"] == pytest.approx(-0.1, abs=1e-12)


def test_step1_diagnostics_use_hysteresis_only_for_reputation_agents_in_C(perturbation_module):
    config = perturbation_module.SystemConfig(
        num_agents=3,
        num_states=3,
        num_actions=2,
        num_time_steps=1,
        role_update_base_interval=10**9,
        gossip_rate=0.0,
        gamma=1.0,
        B_R=0.8,
        B_F=0.6,
        use_numpy_fast_path=False,
    )
    system = perturbation_module.MultiAgentSystem(config)

    rep_agent = system.agents[0]
    rep_agent.state.role = perturbation_module.AgentRole.REPUTATION
    rep_agent.state.followers = set()
    rep_agent.state.following = 2
    rep_agent.state.highest_rep_agent_estimate = 2
    rep_agent.state.reputation_estimates = {0: 0.0, 1: 0.0, 2: 0.7}
    rep_agent.state.estimated_reward_pu = 0.0

    pu_agent = system.agents[1]
    pu_agent.state.role = perturbation_module.AgentRole.PERSONAL_UTILITY
    pu_agent.state.followers = set()
    pu_agent.state.following = None
    pu_agent.state.highest_rep_agent_estimate = 2
    pu_agent.state.reputation_estimates = {0: 0.0, 1: 0.0, 2: 0.7}
    pu_agent.state.estimated_reward_pu = 0.0

    rep_terms = perturbation_module._compute_step1_diagnostic_terms(system, 0)
    pu_terms = perturbation_module._compute_step1_diagnostic_terms(system, 1)

    assert rep_terms["hysteresis_active"] is True
    assert rep_terms["effective_threshold"] == pytest.approx(0.6, abs=1e-12)
    assert pu_terms["hysteresis_active"] is False
    assert pu_terms["effective_threshold"] == pytest.approx(0.8, abs=1e-12)


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


def test_parse_and_make_config_support_shared_good_bad_mode(monkeypatch, perturbation_module):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "perturbation_recovery.py",
            "--reward-model", "shared_good_bad_heterogeneous",
            "--perturb-policy-mode", "force_bad_action",
            "--reward-good-value", "1.2",
            "--reward-bad-value", "0.2",
            "--reward-order-gap", "0.05",
        ],
    )
    args = perturbation_module.parse_args()
    config = perturbation_module.make_config(args, mode="static")

    assert args.reward_model == "shared_good_bad_heterogeneous"
    assert args.perturb_policy_mode == "force_bad_action"
    assert config.reward_model == "shared_good_bad_heterogeneous"
    assert config.reward_good_value == pytest.approx(1.2, abs=1e-12)
    assert config.reward_bad_value == pytest.approx(0.2, abs=1e-12)
    assert config.reward_order_gap == pytest.approx(0.05, abs=1e-12)


def test_run_experiment_selected_seeds_override_contiguous_range(tmp_path, monkeypatch, perturbation_module):
    seen = []

    def fake_run_single(args, seed):
        seen.append(int(seed))
        record = perturbation_module.RunRecord(
            mode=str(args.mode),
            gamma=float(args.gamma),
            kappa=float(args.kappa),
            seed=int(seed),
            converged=False,
            t_conv=-1,
            leader_pre=-1,
            pre_followers=0,
            t_perturb_start=-1,
            t_perturb_end=-1,
            drop_min=float("nan"),
            drop_fraction=float("nan"),
            time_to_drop=-1,
            normless_duration=0,
            pu_share_peak_during_drop=float("nan"),
            recovery_time=-1,
            leader_post_recovery=-1,
            leader_changed=False,
            stable_recovery=False,
            stable_tail_window=int(args.stable_tail_window),
            welfare_pre=float("nan"),
            welfare_drop=float("nan"),
            welfare_recovered=float("nan"),
            final_leader=-1,
            final_leader_changed=False,
            final_top_followers=0,
            post_perturb_role_updates_available=0,
            max_alt_leader_followers_post=0,
            time_to_alt_leader_25pct=-1,
            time_to_alt_leader_50pct=-1,
            time_to_alt_leader_75pct=-1,
            final_share_positive_step1_margin=float("nan"),
            final_pu_share=1.0,
        )
        details = {
            "top_series": np.array([0.0]),
            "ex_leader_followers_series": np.array([np.nan]),
            "pu_share_series": np.array([1.0]),
            "welfare_series": np.array([0.0]),
            "exit_diagnostics": [],
        }
        return record, details

    def fake_plot_seed_trajectory(*, output_file, **kwargs):
        Path(output_file).touch()

    monkeypatch.setattr(perturbation_module, "run_single", fake_run_single)
    monkeypatch.setattr(perturbation_module, "_plot_seed_trajectory", fake_plot_seed_trajectory)

    result = perturbation_module.run_experiment(
        mode="static",
        num_agents=8,
        num_states=3,
        num_actions=2,
        num_steps_max=10,
        gamma=2.0,
        kappa=2.0,
        seeds=10,
        seed_start=20,
        selected_seeds=[0, 1, 3],
        output_dir=str(tmp_path),
        auto_run_subdir=False,
        output_prefix="selected_seed_test",
    )

    assert seen == [0, 1, 3]
    assert [record.seed for record in result["run_records"]] == [0, 1, 3]
    assert Path(result["runs_csv"]).exists()
    assert Path(result["aggregate_csv"]).exists()


def test_build_run_subdir_name_uses_explicit_seedset_suffix(perturbation_module):
    args = SimpleNamespace(
        run_label="",
        output_prefix="demo",
        mode="static",
        gamma=10.0,
        kappa=2.0,
        num_agents=100,
        num_states=10,
        num_steps_max=80000,
        seed_start=0,
        seeds=10,
        selected_seeds=[0, 1, 3, 4, 5, 8],
    )

    name = perturbation_module._build_run_subdir_name(args)

    assert "seedset_0_1_3_4_5_8" in name
    assert "_seed0to9_" not in name


def test_experiment_d_wrappers_keep_toy_and_scaled_configs_distinct(monkeypatch, experiments_module):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        seed = int(kwargs["seed_start"])
        record = SimpleNamespace(
            seed=seed,
            t_perturb_start=-1,
            leader_pre=-1,
            final_leader=-1,
            pre_followers=0,
            drop_fraction=float("nan"),
            normless_duration=0,
            stable_tail_window=1,
            stable_recovery=False,
            leader_changed=False,
            final_leader_changed=False,
            t_conv=-1,
            t_perturb_end=-1,
            recovery_time=-1,
            final_top_followers=0,
        )
        return {
            "run_records": [record],
            "details_by_seed": {seed: {"follower_rows": [[0] * int(kwargs["num_agents"])], "welfare_series": np.array([0.0])}},
            "runs_csv": "runs.csv",
            "aggregate_csv": "aggregate.csv",
            "plot_files": {seed: "plot.png"},
        }

    monkeypatch.setattr(experiments_module, "run_perturbation_recovery", fake_run)

    experiments_module.experiment_D()
    experiments_module.experiment_D_scaled_expB()
    experiments_module.experiment_D_scaled_expB_good_bad()
    experiments_module.experiment_D_scaled_expB_good_bad_recovery_debug()

    toy_call, scaled_call, good_bad_call, recovery_debug_call = calls
    assert toy_call["num_agents"] == 8
    assert toy_call["num_states"] == 3
    assert toy_call["seed_start"] == 25
    assert toy_call["run_label"] == "exp_D_perturbation_seed25"

    assert scaled_call["num_agents"] == 100
    assert scaled_call["num_states"] == 10
    assert scaled_call["seeds"] == 10
    assert scaled_call["role_update_base_interval"] == 3000
    assert scaled_call["run_label"] == "exp_D_perturbation_gamma5_expB_10seeds"

    assert good_bad_call["num_agents"] == 100
    assert good_bad_call["num_states"] == 10
    assert good_bad_call["reward_model"] == "shared_good_bad_heterogeneous"
    assert good_bad_call["perturb_policy_mode"] == "force_bad_action"
    assert good_bad_call["run_label"] == "exp_D_perturbation_gamma5_expB_good_bad_10seeds"

    assert recovery_debug_call["num_agents"] == 100
    assert recovery_debug_call["num_states"] == 10
    assert recovery_debug_call["seeds"] == 3
    assert recovery_debug_call["role_update_base_interval"] == 1000
    assert recovery_debug_call["perturb_duration"] == 3000
    assert recovery_debug_call["conv_hold_steps"] == 1200
    assert recovery_debug_call["recovery_hold_steps"] == 800
    assert recovery_debug_call["stable_tail_window"] == 2000
    assert recovery_debug_call["run_label"] == "exp_D_perturbation_gamma5_expB_good_bad_recovery_debug_1000_smoke3"


def test_experiment_d_gamma10_wrappers_forward_expected_parameters(monkeypatch, experiments_module):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        seed = int(kwargs["seed_start"])
        record = SimpleNamespace(
            seed=seed,
            t_perturb_start=-1,
            leader_pre=-1,
            final_leader=-1,
            pre_followers=0,
            drop_fraction=float("nan"),
            normless_duration=0,
            stable_tail_window=1,
            stable_recovery=False,
            leader_changed=False,
            final_leader_changed=False,
            t_conv=-1,
            t_perturb_end=-1,
            recovery_time=-1,
            final_top_followers=0,
        )
        return {
            "run_records": [record],
            "details_by_seed": {seed: {"follower_rows": [[0] * int(kwargs["num_agents"])], "welfare_series": np.array([0.0])}},
            "runs_csv": "runs.csv",
            "aggregate_csv": "aggregate.csv",
            "plot_files": {seed: "plot.png"},
        }

    monkeypatch.setattr(experiments_module, "run_perturbation_recovery", fake_run)

    experiments_module.experiment_D_scaled_expB_good_bad_gamma10_smoke()
    experiments_module.experiment_D_scaled_expB_good_bad_gamma10_gap_smoke()
    experiments_module.experiment_D_scaled_expB_good_bad_gamma10_h80000_selected_converged()

    gamma10_call, gamma10_gap_call, gamma10_h80000_call = calls
    assert gamma10_call["gamma"] == 10.0
    assert gamma10_call["reward_good_value"] == 1.0
    assert gamma10_call["reward_bad_value"] == 0.1
    assert gamma10_call["perturb_policy_mode"] == "force_bad_action"
    assert gamma10_call["collapse_followers_on_perturb"] is False
    assert gamma10_call["reputation_shock_factor"] == 1.0
    assert gamma10_call["run_label"] == "exp_D_perturbation_gamma10_expB_good_bad_smoke3"

    assert gamma10_gap_call["gamma"] == 10.0
    assert gamma10_gap_call["reward_good_value"] == 1.09
    assert gamma10_gap_call["reward_bad_value"] == 0.01
    assert gamma10_gap_call["perturb_policy_mode"] == "force_bad_action"
    assert gamma10_gap_call["collapse_followers_on_perturb"] is False
    assert gamma10_gap_call["reputation_shock_factor"] == 1.0
    assert gamma10_gap_call["run_label"] == "exp_D_perturbation_gamma10_expB_good_bad_gap_smoke3"

    assert gamma10_h80000_call["gamma"] == 10.0
    assert gamma10_h80000_call["selected_seeds"] == [0, 1, 3, 4, 5, 8]
    assert gamma10_h80000_call["num_steps_max"] == 80000
    assert gamma10_h80000_call["post_window"] == 80000
    assert gamma10_h80000_call["perturb_policy_mode"] == "force_bad_action"
    assert gamma10_h80000_call["collapse_followers_on_perturb"] is False
    assert gamma10_h80000_call["reputation_shock_factor"] == 1.0
    assert gamma10_h80000_call["run_label"] == "exp_D_perturbation_gamma10_expB_good_bad_h80000_converged6"
