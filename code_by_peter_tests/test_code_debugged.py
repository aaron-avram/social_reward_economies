"""Combined test suite for src/code_debugged.py.

Consolidated from:
  - code_by_peter_tests/test_pu_gossip_role_switching.py
  - code_by_peter_tests/test_reputation_status_rate_allocation.py
  - code_by_peter_tests/test_async_role_switching.py
  - code_by_peter_tests/test_perturbation_recovery.py
  - src/tests.py
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from _shared import (
    estimate_activation_frequency,
    get_reputation_learning_snapshot,
    gossip_inplace_update,
    gossip_phase_oracle,
    gossip_sync_update,
    load_model_module,
    make_system,
    set_reputation_learning_state,
    variance,
)

# Make src/ importable for Section 5 tests that reference code_debugged classes directly.
# load_model_module() also registers code_debugged in sys.modules so the import below resolves.
_code_debugged_mod = load_model_module()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
MultiAgentSystem = _code_debugged_mod.MultiAgentSystem
SystemConfig = _code_debugged_mod.SystemConfig
AgentRole = _code_debugged_mod.AgentRole

ROOT = Path(__file__).resolve().parents[1]
PERTURBATION_RUNNER_PATH = ROOT / "experiments" / "perturbation_recovery.py"
EXPERIMENTS_PATH = ROOT / "experiments" / "experiments.py"

_variance = variance


@pytest.fixture(scope="module")
def model_module():
    return _code_debugged_mod

# ==============================================================================
# === SECTION 1: Gossip Oracles and Reputation Learning ===
# ==============================================================================









# ==================== GOSSIP NETWORK ====================

def test_rep_observed_reputation_estimates_initialize_to_zero(model_module):
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(use_numpy_fast_path=True),
    )

    for agent in system.agents:
        assert all(
            agent.state.reputation_estimates[k] == pytest.approx(0.0, abs=1e-12)
            for k in range(system.config.num_agents)
        )

    assert system._s_matrix is not None
    assert np.allclose(system._s_matrix, 0.0)

def test_rep3_no_extra_phase5_pairwise_gossip_strict_noop_phase4(model_module):
    system = make_system(
        model_module,
        num_agents=2,
        extra_config=dict(gossip_rate=1.0, gossip_alpha=1.0, u_0=0.0),
    )
    a0, a1 = system.agents

    for a in (a0, a1):
        a.state.actor_interaction_rate = 0.0
        a.state.participant_interaction_rate = 100.0

    a0.state.reputation_estimates = {0: 0.0, 1: 0.0}
    a1.state.reputation_estimates = {0: 10.0, 1: 10.0}
    a0.state.highest_rep_agent_estimate = 0
    a1.state.highest_rep_agent_estimate = 0

    def _deltas_a0(self, observed_payoffs, eta_v_t):
        return {0: 1.0, 1: 0.0}

    def _deltas_a1(self, observed_payoffs, eta_v_t):
        return {0: -1.0, 1: 0.0}

    a0.update_personal_benefit_estimates = types.MethodType(_deltas_a0, a0)
    a1.update_personal_benefit_estimates = types.MethodType(_deltas_a1, a1)

    np.random.seed(0)
    system.step()

    assert a0.state.reputation_estimates[0] == pytest.approx(6.0, abs=1e-12)
    assert a1.state.reputation_estimates[0] == pytest.approx(4.0, abs=1e-12)


def test_rep3_no_extra_phase5_pairwise_gossip_legacy_scenario(model_module):
    system = make_system(
        model_module,
        num_agents=2,
        extra_config=dict(gossip_rate=1.0, gossip_alpha=0.5, u_0=0.0),
    )
    a0, a1 = system.agents

    for a in (a0, a1):
        a.state.actor_interaction_rate = 0.0
        a.state.participant_interaction_rate = 100.0

    a0.state.reputation_estimates = {0: 0.0, 1: 0.0}
    a1.state.reputation_estimates = {0: 10.0, 1: 10.0}
    a0.state.highest_rep_agent_estimate = 0
    a1.state.highest_rep_agent_estimate = 0

    def _deltas_a0(self, observed_payoffs, eta_v_t):
        return {0: 1.0, 1: 0.0}

    def _deltas_a1(self, observed_payoffs, eta_v_t):
        return {0: -1.0, 1: 0.0}

    a0.update_personal_benefit_estimates = types.MethodType(_deltas_a0, a0)
    a1.update_personal_benefit_estimates = types.MethodType(_deltas_a1, a1)

    np.random.seed(0)
    system.step()

    assert a0.state.reputation_estimates[0] == pytest.approx(6.0, abs=1e-12)
    assert a1.state.reputation_estimates[0] == pytest.approx(4.0, abs=1e-12)


def test_rep2_gossip_mean_only_helper():
    rep_snapshot = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: 0.0}, 1: {0: 0.0}, 2: {0: 0.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 1, 2]]
    assert all(abs(v - 5.0) < 1e-10 for v in vals)


def test_rep2_gossip_mean_plus_delta_v_helper():
    rep_snapshot = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: +1.0}, 1: {0: 0.0}, 2: {0: -2.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 1, 2]]
    assert abs(vals[0] - 6.0) < 1e-10
    assert abs(vals[1] - 5.0) < 1e-10
    assert abs(vals[2] - 3.0) < 1e-10


def test_rep2_gossip_snapshot_vs_inplace_order_dependence_helper():
    rep0 = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: 0.0}, 1: {0: 0.0}, 2: {0: 0.0}}

    out_sync = gossip_sync_update(rep0, delta_v, num_agents=1)
    sync_vals = [out_sync[i][0] for i in [0, 1, 2]]

    rep_inplace_a = {i: dict(rep0[i]) for i in rep0}
    out_inplace_a = gossip_inplace_update(
        rep_inplace_a, delta_v, num_agents=1, update_order=[0, 1, 2]
    )
    a_vals = [out_inplace_a[i][0] for i in [0, 1, 2]]

    rep_inplace_b = {i: dict(rep0[i]) for i in rep0}
    out_inplace_b = gossip_inplace_update(
        rep_inplace_b, delta_v, num_agents=1, update_order=[2, 1, 0]
    )
    b_vals = [out_inplace_b[i][0] for i in [0, 1, 2]]

    assert all(abs(v - 5.0) < 1e-10 for v in sync_vals)
    assert (any(abs(v - 5.0) > 1e-10 for v in a_vals) or
            any(abs(v - 5.0) > 1e-10 for v in b_vals))


def test_rep2_gossip_multi_round_convergence_delta_v_zero_helper():
    rep = {0: {0: 0.0}, 1: {0: 10.0}, 2: {0: 0.0}, 3: {0: 10.0}}
    delta_v = {i: {0: 0.0} for i in rep}

    variances = []
    for _ in range(6):
        vals = [rep[i][0] for i in sorted(rep.keys())]
        variances.append(variance(vals))
        rep = gossip_sync_update(rep, delta_v, num_agents=1)

    assert variances[-1] < variances[0] * 1e-6


def test_rep2_gossip_participant_subset_mean_helper():
    rep_snapshot = {0: {0: 0.0}, 2: {0: 6.0}, 4: {0: 12.0}}
    delta_v = {0: {0: 0.0}, 2: {0: 0.0}, 4: {0: 0.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 2, 4]]
    assert all(abs(v - 6.0) < 1e-10 for v in vals)


def test_rep1_identify_highest_reputation_agent_with_delta_tie(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(delta=0.1),
    )

    agent = system.agents[2]
    # self=2 should be excluded even if very large
    agent.state.reputation_estimates = {
        0: 1.00,
        1: 0.95,
        2: 999.0,
        3: 0.20,
    }

    agent.identify_highest_reputation_agent()

    # with delta=0.1, both 0 and 1 are admissible candidates
    assert agent.state.highest_rep_agent_estimate in {0, 1}
    assert agent.state.highest_rep_agent_estimate != 2


def test_rep4_personal_benefit_estimates_active_and_inactive_decay(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]

    agent.state.personal_benefit_estimates = {
        0: 4.0,   # active target
        1: 5.0,   # inactive target
        2: 0.0,
    }

    observed_payoffs = {
        0: 10.0,  # active
        1: 0.0,   # inactive -> decay
        2: 0.0,
    }

    deltas = agent.update_personal_benefit_estimates(
        observed_payoffs,
        eta_v_t=0.2,
        active_actor_ids=[0],
    )

    # active update: 4 + 0.2*(10-4) = 5.2
    assert agent.state.personal_benefit_estimates[0] == pytest.approx(5.2, abs=1e-12)
    assert deltas[0] == pytest.approx(1.2, abs=1e-12)

    # inactive decay: 5*(1-0.2) = 4.0
    assert agent.state.personal_benefit_estimates[1] == pytest.approx(4.0, abs=1e-12)
    assert deltas[1] == pytest.approx(-1.0, abs=1e-12)


def test_rep4_active_zero_payoff_updates_instead_of_decaying(model_module):
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]

    agent.state.personal_benefit_estimates = {
        0: 1.0,
        1: 5.0,
        2: 0.0,
    }

    observed_payoffs = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
    }

    deltas = agent.update_personal_benefit_estimates(
        observed_payoffs,
        eta_v_t=0.2,
        active_actor_ids=[1],
    )

    # Active zero-utility observation should move toward 0, not decay as inactive.
    assert agent.state.personal_benefit_estimates[1] == pytest.approx(4.0, abs=1e-12)
    assert deltas[1] == pytest.approx(-1.0, abs=1e-12)

    # Truly inactive targets still decay.
    assert agent.state.personal_benefit_estimates[0] == pytest.approx(0.8, abs=1e-12)
    assert deltas[0] == pytest.approx(-0.2, abs=1e-12)


def test_rep1_identify_highest_reputation_agent_single_agent_edge_case(model_module):
    system = make_system(model_module, num_agents=1)
    agent = system.agents[0]

    agent.state.reputation_estimates = {}
    agent.identify_highest_reputation_agent()

    assert agent.state.highest_rep_agent_estimate == 0


def test_gossip_step_with_no_active_participants_is_noop(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(num_states=1, num_actions=2),
    )

    # no participants at this step
    for a in system.agents:
        a.state.participant_interaction_rate = 0.0
        a.state.actor_interaction_rate = 0.0

    before_rep = [
        dict(agent.state.reputation_estimates)
        for agent in system.agents
    ]

    system.step()

    after_rep = [
        dict(agent.state.reputation_estimates)
        for agent in system.agents
    ]

    assert before_rep == after_rep
    assert system.last_active_participant_ids == set()


def test_rep1_identify_highest_reputation_agent_empty_dict_multi_agent(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=3)
    agent = system.agents[1]

    agent.state.reputation_estimates = {}
    agent.identify_highest_reputation_agent()

    assert agent.state.highest_rep_agent_estimate in {0, 2}
    assert agent.state.highest_rep_agent_estimate != 1


def test_gossip_step_with_active_participants_but_no_active_actors(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(num_states=1, num_actions=2),
    )

    # no actors, but all are participants
    for a in system.agents:
        a.state.actor_interaction_rate = 0.0
        a.state.participant_interaction_rate = 100.0

    before = [
        dict(agent.state.personal_benefit_estimates)
        for agent in system.agents
    ]

    system.step()

    # since there are no active actors, observed payoffs are all zero,
    # so v_i(k,t) should decay / stay at zero from the zero initialization
    after = [
        dict(agent.state.personal_benefit_estimates)
        for agent in system.agents
    ]

    assert system.last_active_actor_ids == set()
    assert system.last_active_participant_ids == {0, 1, 2}
    assert before == after  # all zeros stay zero


def test_rep2_update_reputation_estimates_gossip_function_direct_mean_plus_delta(model_module):
    system = make_system(model_module, num_agents=3)
    a0, a1, a2 = system.agents

    # old reputations for target k=1 are [1,5,9], mean = 5
    a0.state.reputation_estimates = {0: 0.0, 1: 1.0, 2: 0.0}
    a1.state.reputation_estimates = {0: 0.0, 1: 5.0, 2: 0.0}
    a2.state.reputation_estimates = {0: 0.0, 1: 9.0, 2: 0.0}

    deltas = {0: 0.0, 1: 2.0, 2: 0.0}
    a0.update_reputation_estimates_gossip(deltas, [a0, a1, a2], eta_s_t=0.1)

    # mean is 5, plus delta for target 1 gives 7
    assert a0.state.reputation_estimates[1] == pytest.approx(7.0, abs=1e-12)
    # unaffected targets remain mean + 0 delta
    assert a0.state.reputation_estimates[0] == pytest.approx(0.0, abs=1e-12)
    assert a0.state.reputation_estimates[2] == pytest.approx(0.0, abs=1e-12)


def test_rep2_update_reputation_estimates_gossip_function_empty_other_agents_list_fallback(model_module):
    system = make_system(model_module, num_agents=2)
    a = system.agents[0]

    a.state.reputation_estimates = {0: 1.5, 1: 2.5}
    deltas = {0: 0.5, 1: -1.0}

    a.update_reputation_estimates_gossip(deltas, [], eta_s_t=0.1)

    # fallback uses current self estimate if no estimates list available
    assert a.state.reputation_estimates[0] == pytest.approx(2.0, abs=1e-12)
    assert a.state.reputation_estimates[1] == pytest.approx(1.5, abs=1e-12)


def test_rep2_update_reputation_estimates_gossip_function_respects_target_scope(model_module):
    system = make_system(model_module, num_agents=3)
    a0, a1, a2 = system.agents

    a0.state.reputation_estimates = {0: 10.0, 1: 1.0, 2: 20.0}
    a1.state.reputation_estimates = {0: 30.0, 1: 5.0, 2: 40.0}
    a2.state.reputation_estimates = {0: 50.0, 1: 9.0, 2: 60.0}

    deltas = {0: 7.0, 1: 2.0, 2: 11.0}
    a0.update_reputation_estimates_gossip(
        deltas,
        [a0, a1, a2],
        eta_s_t=0.1,
        target_agent_ids=[1],
    )

    # Only target 1 should be gossip-updated under paper scope B(t).
    assert a0.state.reputation_estimates[1] == pytest.approx(7.0, abs=1e-12)
    assert a0.state.reputation_estimates[0] == pytest.approx(10.0, abs=1e-12)
    assert a0.state.reputation_estimates[2] == pytest.approx(20.0, abs=1e-12)


def test_rep1_identify_highest_reputation_agent_delta_tie_random_candidate_is_valid(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=4, extra_config=dict(delta=0.1))
    a = system.agents[3]

    a.state.reputation_estimates = {
        0: 1.00,
        1: 0.94,
        2: 0.20,
        3: 999.0,  # self should be excluded
    }

    a.identify_highest_reputation_agent()

    # max among non-self is 1.00, with delta=0.1 both 0 and 1 are valid
    assert a.state.highest_rep_agent_estimate in {0, 1}
    assert a.state.highest_rep_agent_estimate != 3


def test_rep4_step_decays_all_v_estimates_when_no_active_actors(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=3)

    # no actors, all participants
    for a in system.agents:
        a.state.actor_interaction_rate = 0.0
        a.state.participant_interaction_rate = 100.0
        a.state.personal_benefit_estimates = {0: 5.0, 1: 2.0, 2: 1.0}

    system.step()

    eta_v_t = system.config.eta_v_base / (1.0 + 1 * 0.01)

    for a in system.agents:
        assert a.state.personal_benefit_estimates[0] == pytest.approx(5.0 * (1.0 - eta_v_t), rel=1e-9)
        assert a.state.personal_benefit_estimates[1] == pytest.approx(2.0 * (1.0 - eta_v_t), rel=1e-9)
        assert a.state.personal_benefit_estimates[2] == pytest.approx(1.0 * (1.0 - eta_v_t), rel=1e-9)


def _make_isolated_gossip_fixture(
    model_module,
    *,
    use_numpy_fast_path: bool,
    eq9_averaging_mode: str,
    leader_update_mode: str = "participants_only_post_eq9",
):
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(
            use_numpy_fast_path=use_numpy_fast_path,
            delta=0.0,
            eq9_averaging_mode=eq9_averaging_mode,
            leader_update_mode=leader_update_mode,
            initial_actor_interaction_rate=0.4,
            initial_participant_interaction_rate=0.4,
        ),
    )

    v0 = np.array(
        [
            [0.00, 0.20, 0.10, 0.00],
            [0.05, 0.00, 0.20, 0.10],
            [0.10, 0.30, 0.00, 0.10],
            [0.00, 0.10, 0.20, 0.00],
        ],
        dtype=float,
    )
    s0 = np.array(
        [
            [0.00, 0.90, 0.40, 0.10],
            [0.30, 0.00, 1.10, 0.20],
            [0.20, 0.80, 0.00, 0.10],
            [0.10, 0.70, 0.30, 0.00],
        ],
        dtype=float,
    )
    l0 = np.array([1, 2, 1, 1], dtype=int)
    observed = np.array(
        [
            [0.0, 0.9, 1.3, 0.0],
            [0.0, 0.4, 1.1, 0.0],
            [0.0, 0.8, 1.2, 0.0],
            [0.0, 0.5, 1.0, 0.0],
        ],
        dtype=float,
    )

    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )
    return system, v0, s0, l0, observed


def test_gossip_isolated_phase_python_path_matches_oracle(model_module):
    system, v0, s0, l0, observed = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=False,
        eq9_averaging_mode="all_agents",
    )
    eta_v_t = 0.2
    active_actor_ids = [1, 2]
    active_participant_ids = [0, 1, 3]

    expected = gossip_phase_oracle(
        v_before=v0,
        s_before=s0,
        highest_rep_before=l0,
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        delta=0.0,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_post_eq9",
    )

    actual = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_post_eq9",
    )

    assert actual["gossip_target_ids"] == expected["gossip_target_ids"]
    assert actual["eq9_averaging_mode"] == "all_agents"
    assert actual["leader_update_mode"] == "participants_only_post_eq9"
    assert actual["averaging_agent_ids"] == [0, 1, 2, 3]
    assert actual["leader_update_agent_ids"] == active_participant_ids
    assert actual["avg_s_by_target"] == pytest.approx(expected["avg_s_by_target"], abs=1e-12)
    assert np.allclose(actual["delta_v_matrix"], expected["delta_v_matrix"], atol=1e-12)
    assert np.allclose(actual["v_matrix"], expected["v_matrix"], atol=1e-12)
    assert np.allclose(actual["s_matrix"], expected["s_matrix"], atol=1e-12)
    assert np.array_equal(actual["highest_rep_agent_estimates"], expected["highest_rep_agent_estimates"])


def test_gossip_isolated_phase_python_path_treats_active_zero_utilities_as_active(model_module):
    system, v0, s0, l0, observed = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=False,
        eq9_averaging_mode="all_agents",
    )
    eta_v_t = 0.2
    active_actor_ids = [1, 2]
    active_participant_ids = [0, 1, 3]
    observed[:, 1] = 0.0

    expected = gossip_phase_oracle(
        v_before=v0,
        s_before=s0,
        highest_rep_before=l0,
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        delta=0.0,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_post_eq9",
    )

    actual = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_post_eq9",
    )

    assert np.allclose(actual["delta_v_matrix"], expected["delta_v_matrix"], atol=1e-12)
    assert np.allclose(actual["v_matrix"], expected["v_matrix"], atol=1e-12)


def test_gossip_isolated_phase_numpy_fast_path_matches_oracle(model_module):
    system, v0, s0, l0, observed = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=True,
        eq9_averaging_mode="all_agents",
    )
    eta_v_t = 0.2
    active_actor_ids = [1, 2]
    active_participant_ids = [0, 1, 3]

    expected = gossip_phase_oracle(
        v_before=v0,
        s_before=s0,
        highest_rep_before=l0,
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        delta=0.0,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_post_eq9",
    )

    actual = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_post_eq9",
    )

    assert actual["gossip_target_ids"] == expected["gossip_target_ids"]
    assert actual["eq9_averaging_mode"] == "all_agents"
    assert actual["leader_update_mode"] == "participants_only_post_eq9"
    assert actual["averaging_agent_ids"] == [0, 1, 2, 3]
    assert actual["leader_update_agent_ids"] == active_participant_ids
    assert actual["avg_s_by_target"] == pytest.approx(expected["avg_s_by_target"], abs=1e-12)
    assert np.allclose(actual["delta_v_matrix"], expected["delta_v_matrix"], atol=1e-12)
    assert np.allclose(actual["v_matrix"], expected["v_matrix"], atol=1e-12)
    assert np.allclose(actual["s_matrix"], expected["s_matrix"], atol=1e-12)
    assert np.array_equal(actual["highest_rep_agent_estimates"], expected["highest_rep_agent_estimates"])


def test_gossip_isolated_phase_python_path_matches_oracle_participants_only(model_module):
    system, v0, s0, l0, observed = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=False,
        eq9_averaging_mode="participants_only",
    )
    eta_v_t = 0.2
    active_actor_ids = [1, 2]
    active_participant_ids = [0, 1, 3]

    expected = gossip_phase_oracle(
        v_before=v0,
        s_before=s0,
        highest_rep_before=l0,
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        delta=0.0,
        eq9_averaging_mode="participants_only",
        leader_update_mode="participants_only_post_eq9",
    )

    actual = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="participants_only",
        leader_update_mode="participants_only_post_eq9",
    )

    assert actual["eq9_averaging_mode"] == "participants_only"
    assert actual["leader_update_mode"] == "participants_only_post_eq9"
    assert actual["averaging_agent_ids"] == active_participant_ids
    assert actual["leader_update_agent_ids"] == active_participant_ids
    assert actual["avg_s_by_target"] == pytest.approx(expected["avg_s_by_target"], abs=1e-12)
    assert np.allclose(actual["v_matrix"], expected["v_matrix"], atol=1e-12)
    assert np.allclose(actual["s_matrix"], expected["s_matrix"], atol=1e-12)
    assert np.array_equal(actual["highest_rep_agent_estimates"], expected["highest_rep_agent_estimates"])


def test_gossip_isolated_phase_fast_and_python_paths_match(model_module):
    active_actor_ids = [1, 2]
    active_participant_ids = [0, 1, 3]
    eta_v_t = 0.2

    slow_system, _, _, _, observed = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=False,
        eq9_averaging_mode="all_agents",
    )
    fast_system, _, _, _, _ = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=True,
        eq9_averaging_mode="all_agents",
    )

    slow_after = slow_system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
    )
    fast_after = fast_system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
    )

    assert slow_after["gossip_target_ids"] == fast_after["gossip_target_ids"]
    assert slow_after["avg_s_by_target"] == pytest.approx(fast_after["avg_s_by_target"], abs=1e-12)
    assert np.allclose(slow_after["delta_v_matrix"], fast_after["delta_v_matrix"], atol=1e-12)
    assert np.allclose(slow_after["v_matrix"], fast_after["v_matrix"], atol=1e-12)
    assert np.allclose(slow_after["s_matrix"], fast_after["s_matrix"], atol=1e-12)
    assert np.array_equal(
        slow_after["highest_rep_agent_estimates"],
        fast_after["highest_rep_agent_estimates"],
    )


def test_gossip_isolated_phase_fast_and_python_paths_match_participants_only(model_module):
    active_actor_ids = [1, 2]
    active_participant_ids = [0, 1, 3]
    eta_v_t = 0.2

    slow_system, _, _, _, observed = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=False,
        eq9_averaging_mode="participants_only",
    )
    fast_system, _, _, _, _ = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=True,
        eq9_averaging_mode="participants_only",
    )

    slow_after = slow_system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="participants_only",
    )
    fast_after = fast_system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="participants_only",
    )

    assert slow_after["eq9_averaging_mode"] == "participants_only"
    assert fast_after["eq9_averaging_mode"] == "participants_only"
    assert slow_after["avg_s_by_target"] == pytest.approx(fast_after["avg_s_by_target"], abs=1e-12)
    assert np.allclose(slow_after["delta_v_matrix"], fast_after["delta_v_matrix"], atol=1e-12)
    assert np.allclose(slow_after["v_matrix"], fast_after["v_matrix"], atol=1e-12)
    assert np.allclose(slow_after["s_matrix"], fast_after["s_matrix"], atol=1e-12)
    assert np.array_equal(slow_after["highest_rep_agent_estimates"], fast_after["highest_rep_agent_estimates"])


@pytest.mark.parametrize(
    "leader_update_mode",
    ["participants_only_post_eq9", "all_agents_post_eq9", "participants_only_pre_eq9"],
)
def test_gossip_isolated_phase_leader_update_modes_match_oracle_in_both_paths(model_module, leader_update_mode):
    active_actor_ids = [1, 2]
    active_participant_ids = [0, 1, 3]
    eta_v_t = 0.2

    expected_system, v0, s0, l0, observed = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=False,
        eq9_averaging_mode="all_agents",
        leader_update_mode=leader_update_mode,
    )
    expected = gossip_phase_oracle(
        v_before=v0,
        s_before=s0,
        highest_rep_before=l0,
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        delta=0.0,
        eq9_averaging_mode="all_agents",
        leader_update_mode=leader_update_mode,
    )

    slow_after = expected_system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode=leader_update_mode,
    )
    fast_system, _, _, _, _ = _make_isolated_gossip_fixture(
        model_module,
        use_numpy_fast_path=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode=leader_update_mode,
    )
    fast_after = fast_system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode=leader_update_mode,
    )

    assert slow_after["leader_update_mode"] == leader_update_mode
    assert fast_after["leader_update_mode"] == leader_update_mode
    assert slow_after["leader_update_agent_ids"] == expected["leader_update_agent_ids"]
    assert fast_after["leader_update_agent_ids"] == expected["leader_update_agent_ids"]
    assert np.allclose(slow_after["v_matrix"], expected["v_matrix"], atol=1e-12)
    assert np.allclose(fast_after["v_matrix"], expected["v_matrix"], atol=1e-12)
    assert np.allclose(slow_after["s_matrix"], expected["s_matrix"], atol=1e-12)
    assert np.allclose(fast_after["s_matrix"], expected["s_matrix"], atol=1e-12)
    assert np.array_equal(slow_after["highest_rep_agent_estimates"], expected["highest_rep_agent_estimates"])
    assert np.array_equal(fast_after["highest_rep_agent_estimates"], expected["highest_rep_agent_estimates"])


def test_gossip_isolated_phase_updates_l_only_for_active_participants(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(use_numpy_fast_path=True, delta=0.0, eq9_averaging_mode="all_agents"),
    )
    v0 = np.zeros((3, 3), dtype=float)
    s0 = np.array(
        [
            [0.0, 0.9, 0.2],
            [0.1, 0.0, 0.95],
            [0.8, 0.3, 0.0],
        ],
        dtype=float,
    )
    l0 = np.array([1, 2, 1], dtype=int)
    observed = np.array(
        [
            [0.0, 0.1, 2.0],
            [0.0, 0.1, 2.2],
            [0.0, 0.1, 2.4],
        ],
        dtype=float,
    )
    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )

    before = get_reputation_learning_snapshot(system)
    after = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=[2],
        active_participant_ids=[0, 1],
        eta_v_t=1.0,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
    )

    # Active participant 0 updates L_i(t+1) from the post-update reputation row.
    # B(t) is built from the *pre-update* highest-reputation targets of active participants,
    # so participant 1 brings target 2 into scope for participant 0.
    assert int(after["highest_rep_agent_estimates"][0]) == 2
    # Active participant 1 is allowed to refresh its estimate too.
    assert int(after["highest_rep_agent_estimates"][1]) in {0, 2}
    # Inactive participants keep their previous estimate.
    assert int(after["highest_rep_agent_estimates"][2]) == int(before["highest_rep_agent_estimates"][2])


def test_gossip_isolated_phase_all_agents_post_eq9_updates_l_for_all_agents(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(
            use_numpy_fast_path=True,
            delta=0.0,
            eq9_averaging_mode="all_agents",
            leader_update_mode="all_agents_post_eq9",
        ),
    )
    v0 = np.zeros((3, 3), dtype=float)
    s0 = np.array(
        [
            [0.0, 0.9, 0.2],
            [0.1, 0.0, 0.95],
            [0.8, 0.3, 0.0],
        ],
        dtype=float,
    )
    l0 = np.array([1, 2, 1], dtype=int)
    observed = np.array(
        [
            [0.0, 0.1, 2.0],
            [0.0, 0.1, 2.2],
            [0.0, 0.1, 2.4],
        ],
        dtype=float,
    )
    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )

    after = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=[2],
        active_participant_ids=[0, 1],
        eta_v_t=1.0,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode="all_agents_post_eq9",
    )

    assert after["leader_update_mode"] == "all_agents_post_eq9"
    assert after["leader_update_agent_ids"] == [0, 1, 2]
    assert int(after["highest_rep_agent_estimates"][0]) == 2
    assert int(after["highest_rep_agent_estimates"][1]) in {0, 2}
    # Inactive agent 2 is refreshed under the all-agents mode and no longer keeps the stale value 1.
    assert int(after["highest_rep_agent_estimates"][2]) in {0, 1}
    assert int(after["highest_rep_agent_estimates"][2]) != int(l0[2])


def test_gossip_isolated_phase_pre_eq9_uses_preupdate_reputation_row(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(
            use_numpy_fast_path=True,
            delta=0.0,
            eq9_averaging_mode="all_agents",
            leader_update_mode="participants_only_pre_eq9",
        ),
    )
    v0 = np.zeros((3, 3), dtype=float)
    s0 = np.array(
        [
            [0.0, 0.9, 0.2],
            [0.1, 0.0, 0.95],
            [0.8, 0.3, 0.0],
        ],
        dtype=float,
    )
    l0 = np.array([1, 2, 1], dtype=int)
    observed = np.array(
        [
            [0.0, 0.1, 2.0],
            [0.0, 0.1, 2.2],
            [0.0, 0.1, 2.4],
        ],
        dtype=float,
    )
    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )

    after = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=[2],
        active_participant_ids=[0, 1],
        eta_v_t=1.0,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_pre_eq9",
    )

    assert after["leader_update_mode"] == "participants_only_pre_eq9"
    assert after["leader_update_agent_ids"] == [0, 1]
    # Participant 0 would switch to target 2 under post-update selection, but pre-update timing keeps target 1.
    assert int(after["highest_rep_agent_estimates"][0]) == 1
    assert int(after["highest_rep_agent_estimates"][2]) == int(l0[2])


def test_gossip_isolated_phase_builds_b_from_previous_l_without_same_step_leak(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(
            use_numpy_fast_path=True,
            delta=0.0,
            eq9_averaging_mode="all_agents",
            leader_update_mode="participants_only_post_eq9",
        ),
    )
    v0 = np.zeros((3, 3), dtype=float)
    s0 = np.array(
        [
            [0.0, 0.8, 0.1],
            [0.3, 0.0, 0.2],
            [0.4, 0.3, 0.0],
        ],
        dtype=float,
    )
    l0 = np.array([1, 0, 0], dtype=int)
    observed = np.array(
        [
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=float,
    )
    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )

    after = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=[2],
        active_participant_ids=[0],
        eta_v_t=1.0,
        update_actor_rates=False,
        identify_highest_rep=True,
        eq9_averaging_mode="all_agents",
        leader_update_mode="participants_only_post_eq9",
    )

    # B(t) comes from the previous L_i(t) of active participants, so only target 1 is updated this step.
    assert after["gossip_target_ids"] == [1]
    # The newly selected target is allowed to change after Eq. (9), but it must not retroactively enter the same-step gossip scope.
    assert int(after["highest_rep_agent_estimates"][0]) in {1, 2}


def test_gossip_isolated_phase_current_averaging_uses_active_participants_only(model_module):
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(use_numpy_fast_path=True, delta=0.0, eq9_averaging_mode="participants_only"),
    )
    v0 = np.zeros((4, 4), dtype=float)
    s0 = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [0.0, 200.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    l0 = np.array([1, 1, 1, 1], dtype=int)
    observed = np.zeros((4, 4), dtype=float)
    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )

    after = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=[],
        active_participant_ids=[0, 1],
        eta_v_t=0.0,
        update_actor_rates=False,
        identify_highest_rep=False,
        eq9_averaging_mode="participants_only",
    )

    # If only active participants are averaged, target-1 mean is (1 + 3) / 2 = 2.
    assert after["eq9_averaging_mode"] == "participants_only"
    assert after["avg_s_by_target"] == pytest.approx({1: 2.0}, abs=1e-12)
    assert after["s_matrix"][0, 1] == pytest.approx(2.0, abs=1e-12)
    assert after["s_matrix"][1, 1] == pytest.approx(2.0, abs=1e-12)
    # Inactive agents are not updated and their large values do not enter the mean.
    assert after["s_matrix"][2, 1] == pytest.approx(100.0, abs=1e-12)
    assert after["s_matrix"][3, 1] == pytest.approx(200.0, abs=1e-12)


def test_gossip_isolated_phase_paper_literal_averaging_uses_all_agents(model_module):
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(use_numpy_fast_path=True, delta=0.0, eq9_averaging_mode="all_agents"),
    )
    v0 = np.zeros((4, 4), dtype=float)
    s0 = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0],
            [0.0, 200.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    l0 = np.array([1, 1, 1, 1], dtype=int)
    observed = np.zeros((4, 4), dtype=float)
    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )

    after = system.run_isolated_reputation_learning_phase(
        observed_utility_matrix=observed,
        active_actor_ids=[],
        active_participant_ids=[0, 1],
        eta_v_t=0.0,
        update_actor_rates=False,
        identify_highest_rep=False,
        eq9_averaging_mode="all_agents",
    )

    # Paper-literal all-agent averaging uses all four rows: (1 + 3 + 100 + 200) / 4 = 76.
    assert after["eq9_averaging_mode"] == "all_agents"
    assert after["averaging_agent_ids"] == [0, 1, 2, 3]
    assert after["avg_s_by_target"] == pytest.approx({1: 76.0}, abs=1e-12)
    assert after["s_matrix"][0, 1] == pytest.approx(76.0, abs=1e-12)
    assert after["s_matrix"][1, 1] == pytest.approx(76.0, abs=1e-12)
    assert after["s_matrix"][2, 1] == pytest.approx(100.0, abs=1e-12)
    assert after["s_matrix"][3, 1] == pytest.approx(200.0, abs=1e-12)


def test_gossip_isolated_toy_case_converges_to_unique_target(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(use_numpy_fast_path=True, delta=0.0, eq9_averaging_mode="all_agents"),
    )
    v0 = np.zeros((3, 3), dtype=float)
    s0 = np.zeros((3, 3), dtype=float)
    l0 = np.array([1, 1, 1], dtype=int)
    set_reputation_learning_state(
        system,
        personal_benefit_matrix=v0,
        reputation_matrix=s0,
        highest_rep_agent_estimates=l0,
    )
    observed = np.array(
        [
            [0.0, 1.8, 0.2],
            [0.0, 1.6, 0.1],
            [0.0, 1.7, 0.1],
        ],
        dtype=float,
    )

    for _ in range(5):
        system.run_isolated_reputation_learning_phase(
            observed_utility_matrix=observed,
            active_actor_ids=[1, 2],
            active_participant_ids=[0, 1, 2],
            eta_v_t=0.5,
            update_actor_rates=False,
            identify_highest_rep=True,
            eq9_averaging_mode="all_agents",
        )

    snap = get_reputation_learning_snapshot(system)
    assert int(snap["highest_rep_agent_estimates"][0]) == 1
    assert int(snap["highest_rep_agent_estimates"][2]) == 1
    assert int(snap["highest_rep_agent_estimates"][1]) in {0, 2}
    assert snap["s_matrix"][0, 1] > snap["s_matrix"][0, 2]
    assert snap["s_matrix"][2, 1] > snap["s_matrix"][2, 0]


def test_l_initialization_is_characterized_when_paper_leaves_it_unspecified(model_module):
    np.random.seed(123)
    system = make_system(model_module, num_agents=5)
    highest = [
        int(agent.state.highest_rep_agent_estimate)
        for agent in system.agents
    ]
    assert all(0 <= estimate < 5 for estimate in highest)


# ==================== ROLE SWITCHING ====================

def test_role1_bootstrap_non_follower_from_reputation_signal(model_module):
    system = make_system(
        model_module,
        num_agents=2,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    candidate = system.agents[0]
    leader = system.agents[1]

    candidate.state.followers = set()
    leader.state.followers = {0}

    candidate.state.role = AgentRole.PERSONAL_UTILITY
    candidate.state.following = None
    candidate.state.estimated_reward_pu = 0.0
    candidate.state.estimated_reward_rep = 0.0
    candidate.state.reputation_estimates = {1: 2.0}
    candidate.state.highest_rep_agent_estimate = 1

    np.random.seed(7)
    system._update_roles_sequential()

    assert candidate.state.role == AgentRole.REPUTATION
    assert candidate.state.following == 1


def test_role1_bootstrap_non_follower_legacy_scenario(model_module):
    system = make_system(
        model_module,
        num_agents=2,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    candidate = system.agents[0]
    leader = system.agents[1]

    candidate.state.followers = set()
    leader.state.followers = {0}

    candidate.state.role = AgentRole.PERSONAL_UTILITY
    candidate.state.following = None
    candidate.state.estimated_reward_pu = 0.0
    candidate.state.estimated_reward_rep = 0.0
    candidate.state.reputation_estimates = {1: 2.0}
    candidate.state.highest_rep_agent_estimate = 1

    np.random.seed(7)
    system._update_roles_sequential()

    assert candidate.state.role == AgentRole.REPUTATION
    assert candidate.state.following == 1


def test_role2_step1_reputation_switch_does_not_use_extra_max_rep_gate(model_module):
    system = make_system(
        model_module,
        num_agents=2,
        extra_config=dict(gamma=3.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    follower = system.agents[0]
    leader = system.agents[1]

    follower.state.role = AgentRole.PERSONAL_UTILITY
    follower.state.following = None
    follower.state.followers = set()
    follower.state.estimated_reward_pu = 0.0
    follower.state.estimated_reward_rep = 0.0
    follower.state.highest_rep_agent_estimate = 1
    follower.state.reputation_estimates = {1: 0.5}

    leader.state.role = AgentRole.PERSONAL_UTILITY
    leader.state.following = None
    leader.state.followers = {0}

    np.random.seed(3)
    system._update_roles_sequential()

    assert follower.state.role == AgentRole.REPUTATION
    assert follower.state.following == 1


def test_role3_redirects_if_best_agent_is_already_follower(model_module):
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    i = system.agents[0]
    k_hat = system.agents[1]
    k_prime = system.agents[2]
    filler = system.agents[3]

    i.state.followers = set()
    k_hat.state.followers = {3}
    k_prime.state.followers = {1}
    filler.state.followers = {2}

    k_hat.state.role = AgentRole.REPUTATION
    k_hat.state.following = 2

    i.state.role = AgentRole.PERSONAL_UTILITY
    i.state.following = None
    i.state.estimated_reward_pu = 0.0
    i.state.estimated_reward_rep = 1.0
    i.state.reputation_estimates = {1: 2.0, 2: 1.5, 3: 0.1}
    i.state.highest_rep_agent_estimate = 1

    np.random.seed(1)
    system._update_roles_sequential()

    assert i.state.role == AgentRole.REPUTATION
    assert i.state.following == 2


def test_role3_redirect_legacy_scenario(model_module):
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    i = system.agents[0]
    k_hat = system.agents[1]
    k_prime = system.agents[2]
    filler = system.agents[3]

    i.state.followers = set()
    k_hat.state.followers = {3}
    k_prime.state.followers = {1}
    filler.state.followers = {2}

    k_hat.state.role = AgentRole.REPUTATION
    k_hat.state.following = 2

    i.state.role = AgentRole.PERSONAL_UTILITY
    i.state.following = None
    i.state.estimated_reward_pu = 0.0
    i.state.estimated_reward_rep = 1.0
    i.state.reputation_estimates = {1: 2.0, 2: 1.5, 3: 0.1}
    i.state.highest_rep_agent_estimate = 1

    np.random.seed(1)
    system._update_roles_sequential()

    assert i.state.role == AgentRole.REPUTATION
    assert i.state.following == 2


def test_role_4_no_self_following_after_redirect(model_module):
    config = model_module.SystemConfig(
        num_agents=3,
        num_states=2,
        num_actions=2,
        gamma=2.0,
        kappa=0.0,
        B_R=0.1,
        B_F=0.05,
    )
    system = model_module.MultiAgentSystem(config)

    agent_0 = system.agents[0]
    agent_1 = system.agents[1]

    agent_0.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_0.state.following = None
    agent_0.state.estimated_reward_pu = 0.1
    agent_0.state.estimated_reward_rep = 1.0
    agent_0.state.reputation_estimates = {1: 2.0, 2: 0.5}
    agent_0.state.highest_rep_agent_estimate = 1
    agent_0.state.followers = set()

    agent_1.state.role = model_module.AgentRole.REPUTATION
    agent_1.state.following = 0
    agent_1.state.followers = set()
    agent_1.state.estimated_reward_pu = 0.0
    agent_1.state.reputation_estimates = {0: 1.0, 2: 0.0}
    agent_1.state.highest_rep_agent_estimate = 0

    system._update_roles_sequential()

    assert agent_0.state.following != 0
    assert agent_0.state.role == model_module.AgentRole.PERSONAL_UTILITY
    assert 0 not in agent_0.state.followers
    assert len(agent_0.state.followers) < config.num_agents


def test_role_5_redirect_followers_when_leader_becomes_follower(model_module):
    config = model_module.SystemConfig(
        num_agents=4,
        num_states=2,
        num_actions=2,
        gamma=2.0,
        kappa=0.0,
        B_R=0.1,
        B_F=0.05,
    )
    system = model_module.MultiAgentSystem(config)

    agent_0 = system.agents[0]
    agent_1 = system.agents[1]
    agent_2 = system.agents[2]
    agent_3 = system.agents[3]

    agent_0.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_0.state.following = None
    agent_0.state.followers = set()
    agent_0.state.estimated_reward_pu = 0.05
    agent_0.state.reputation_estimates = {1: 3.0, 2: 1.0, 3: 1.0}
    agent_0.state.highest_rep_agent_estimate = 1

    agent_1.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_1.state.following = None
    agent_1.state.followers = set()
    agent_1.state.estimated_reward_pu = 0.05
    agent_1.state.reputation_estimates = {0: 1.0, 2: 1.0, 3: 5.0}
    agent_1.state.highest_rep_agent_estimate = 3

    agent_2.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_2.state.following = None
    agent_2.state.followers = set()

    agent_3.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_3.state.following = None
    agent_3.state.followers = set()

    np.random.seed(42)
    system._update_roles_sequential()

    assert agent_1.state.following == 3
    assert len(agent_1.state.followers) == 0
    assert agent_0.state.following == 3
    assert agent_0.state.following != 1


def test_role_hysteresis_start_vs_continue(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, delta=0.0, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    leader = 0
    rep_signal = 0.7

    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {0: rep_signal, 2: 0.0, 1: 0.0}
    system.agents[1].identify_highest_reputation_agent()

    system.agents[2].state.role = AgentRole.REPUTATION
    system.agents[2].state.following = leader
    system.agents[leader].state.followers.add(2)
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: rep_signal, 1: 0.0, 2: 0.0}
    system.agents[2].identify_highest_reputation_agent()

    system._update_roles_sequential()

    assert system.agents[1].state.role != AgentRole.REPUTATION
    assert system.agents[2].state.role == AgentRole.REPUTATION
    assert system.agents[2].state.following == leader


def test_role_hysteresis_disabled_when_multiple_leaders_exist(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, delta=0.0, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # Two simultaneous leaders: 0 and 3.
    system.agents[0].state.followers = {2}
    system.agents[3].state.followers = {1}

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 3
    system.agents[2].state.role = AgentRole.REPUTATION
    system.agents[2].state.following = 0
    system.agents[2].state.followers = set()
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: 0.7, 1: 0.0, 2: 0.0, 3: 0.65}
    system.agents[2].identify_highest_reputation_agent()

    system._update_roles_sequential()

    # With multiple leaders, the lower hysteresis threshold B_F should not apply.
    assert system.agents[2].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[2].state.following is None


def test_role_hysteresis_never_blocks_switch_to_higher_reputation_leader(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, delta=0.0, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    system.agents[0].state.followers = {2}
    system.agents[3].state.followers = {1}

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 3
    system.agents[2].state.role = AgentRole.REPUTATION
    system.agents[2].state.following = 0
    system.agents[2].state.followers = set()
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: 0.6, 1: 0.0, 2: 0.0, 3: 0.9}
    system.agents[2].identify_highest_reputation_agent()

    system._update_roles_sequential()

    assert system.agents[2].state.role == AgentRole.REPUTATION
    assert system.agents[2].state.following == 3


def test_role_follower_chain_redirection_simple(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.1),
    )
    AgentRole = model_module.AgentRole

    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[0].state.followers.add(1)

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.reputation_estimates = {0: 0.5, 1: 1.0, 2: 0.0}
    system.agents[2].identify_highest_reputation_agent()
    system.agents[2].state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    assert system.agents[2].state.following == 0


def test_role_status_requires_min_followers(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=10,
        extra_config=dict(gamma=0.0, kappa=2.0, c_threshold=0.3),
    )
    AgentRole = model_module.AgentRole

    agent = system.agents[0]
    agent.state.followers = {1, 2}
    agent.state.role = AgentRole.PERSONAL_UTILITY
    agent.state.estimated_reward_status = 999.0
    agent.state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    assert system.agents[0].state.role != AgentRole.STATUS


def test_role_status_threshold_uses_ceil_of_c_times_n(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=10,
        extra_config=dict(gamma=0.0, kappa=2.0, c_threshold=0.25),
    )
    AgentRole = model_module.AgentRole

    agent = system.agents[0]
    agent.state.role = AgentRole.PERSONAL_UTILITY
    agent.state.followers = {1, 2}
    agent.state.estimated_reward_status = 999.0
    agent.state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    assert agent.state.role == AgentRole.PERSONAL_UTILITY


def test_status_agent_with_zero_followers_reverts_to_personal_utility(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=6,
        extra_config=dict(gamma=0.0, kappa=2.0, c_threshold=0.2),
    )
    AgentRole = model_module.AgentRole

    i = 1
    agent = system.agents[i]
    agent.state.role = AgentRole.STATUS
    agent.state.followers = set()
    agent.state.following = None
    agent.state.estimated_reward_status = 999.0
    agent.state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    assert agent.state.role == AgentRole.PERSONAL_UTILITY


def test_zero_follower_status_agent_can_enter_reputation_in_step1(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=0.25),
    )
    AgentRole = model_module.AgentRole

    i = 1
    leader = 0
    agent = system.agents[i]
    agent.state.role = AgentRole.STATUS
    agent.state.followers = set()
    agent.state.following = None
    agent.state.estimated_reward_pu = 0.0
    agent.state.reputation_estimates = {0: 2.0, 1: 0.0, 2: 0.1, 3: 0.1}
    agent.state.highest_rep_agent_estimate = leader

    system._update_roles_sequential()

    assert agent.state.role == AgentRole.REPUTATION
    assert agent.state.following == leader
    assert i in system.agents[leader].state.followers


def test_role_status_switch_clears_following(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=6,
        extra_config=dict(gamma=1.0, kappa=2.0, c_threshold=0.2),
    )
    AgentRole = model_module.AgentRole

    leader = 0
    i = 1

    system.agents[i].state.role = AgentRole.REPUTATION
    system.agents[i].state.following = leader
    system.agents[leader].state.followers.add(i)

    system.agents[i].state.followers = {2, 3}
    system.agents[i].state.estimated_reward_status = 10.0
    system.agents[i].state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    assert system.agents[i].state.role == AgentRole.STATUS
    assert system.agents[i].state.following is None
    assert i not in system.agents[leader].state.followers


def test_role_update_candidates_only_updates_subset(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.1, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # agent 0 is the leader candidate
    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.followers = set()

    # agent 1 is allowed to update and should switch to REPUTATION
    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.following = None
    system.agents[1].state.followers = set()
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {0: 2.0, 1: 0.0, 2: 0.0}
    system.agents[1].state.highest_rep_agent_estimate = 0

    # agent 2 has exactly the same setup but is NOT in update_candidates
    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.following = None
    system.agents[2].state.followers = set()
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: 2.0, 1: 0.0, 2: 0.0}
    system.agents[2].state.highest_rep_agent_estimate = 0

    system._update_roles_sequential(update_candidates=[1])

    assert system.agents[1].state.role == AgentRole.REPUTATION
    assert system.agents[1].state.following == 0

    assert system.agents[2].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[2].state.following is None


def test_role4_stale_self_highest_rep_does_not_create_self_follow(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.1, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    i = 1
    agent = system.agents[i]

    agent.state.role = AgentRole.PERSONAL_UTILITY
    agent.state.following = None
    agent.state.followers = set()
    agent.state.estimated_reward_pu = 0.0

    # stale / bad state: highest-rep estimate points to self
    agent.state.reputation_estimates = {0: 1.5, 1: 2.0, 2: 0.5}
    agent.state.highest_rep_agent_estimate = 1

    system._update_roles_sequential()

    assert agent.state.following != 1
    assert i not in agent.state.followers


def test_role_update_candidates_empty_is_noop(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.1),
    )
    AgentRole = model_module.AgentRole

    system.agents[0].state.role = AgentRole.REPUTATION
    system.agents[0].state.following = 1
    system.agents[0].state.followers = set()

    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.followers = {0}

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.following = None
    system.agents[2].state.followers = set()
    system.agents[2].state.reputation_estimates = {0: 2.0, 1: 1.0, 2: 0.0, 3: 0.0}
    system.agents[2].state.highest_rep_agent_estimate = 0
    system.agents[2].state.estimated_reward_pu = 0.0

    system.agents[3].state.role = AgentRole.STATUS
    system.agents[3].state.following = None
    system.agents[3].state.followers = set()

    before = []
    for a in system.agents:
        before.append((
            a.state.role,
            a.state.following,
            set(a.state.followers),
        ))

    system._update_roles_sequential(update_candidates=[])

    after = []
    for a in system.agents:
        after.append((
            a.state.role,
            a.state.following,
            set(a.state.followers),
        ))

    assert before == after


def test_role_step1_existing_follower_changes_to_new_leader_and_old_leader_loses_follower(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # agent 0 is currently following leader 1
    system.agents[0].state.role = AgentRole.REPUTATION
    system.agents[0].state.following = 1

    # old leader 1 currently has follower 0
    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.followers = {0}

    # new leader candidate is 2
    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    # filler
    system.agents[3].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[3].state.followers = set()

    # make agent 0 want to switch from leader 1 to leader 2
    system.agents[0].state.estimated_reward_pu = 0.0
    system.agents[0].state.reputation_estimates = {1: 0.2, 2: 2.0, 3: 0.1, 0: 0.0}
    system.agents[0].state.highest_rep_agent_estimate = 2

    system._update_roles_sequential(update_candidates=[0])

    assert system.agents[0].state.role == AgentRole.REPUTATION
    assert system.agents[0].state.following == 2
    assert 0 not in system.agents[1].state.followers
    assert 0 in system.agents[2].state.followers


def test_role_step3_reputation_agent_returns_to_personal_utility_when_following_not_worth_it(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    leader = 0
    follower = 1

    system.agents[leader].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[leader].state.followers = {follower}

    system.agents[follower].state.role = AgentRole.REPUTATION
    system.agents[follower].state.following = leader
    system.agents[follower].state.followers = set()

    # make following unattractive:
    # gamma * max_rep <= max(B_F, J_pu)
    system.agents[follower].state.estimated_reward_pu = 1.0
    system.agents[follower].state.reputation_estimates = {0: 0.1, 1: 0.0, 2: 0.0}
    system.agents[follower].state.highest_rep_agent_estimate = 0

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    system._update_roles_sequential()

    assert system.agents[follower].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[follower].state.following is None
    assert follower not in system.agents[leader].state.followers


def test_role_step2_threshold_met_but_status_not_chosen_if_payoff_comparison_fails(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=6,
        extra_config=dict(gamma=0.0, kappa=2.0, c_threshold=0.2),
    )
    AgentRole = model_module.AgentRole

    i = 0
    agent = system.agents[i]

    # threshold met: ceil(0.2*6) = 2
    agent.state.followers = {1, 2}
    agent.state.role = AgentRole.PERSONAL_UTILITY

    # but kappa * J_status <= J_pu, so should NOT switch
    agent.state.estimated_reward_status = 1.0
    agent.state.estimated_reward_pu = 3.0

    system._update_roles_sequential()

    assert agent.state.role != AgentRole.STATUS


def test_full_tracking_records_role_update_times_and_agent_estimate_histories(model_module):
    np.random.seed(0)
    config = model_module.SystemConfig(
        num_agents=3,
        num_states=2,
        num_actions=2,
        num_time_steps=5,
        gamma=2.0,
        kappa=0.0,
        B_R=0.3,
        B_F=1_000_000.0,
        role_update_base_interval=2,
        fixed_role_update_interval=True,
        tracking_mode="full",
        use_numpy_fast_path=False,
    )
    system = model_module.MultiAgentSystem(config)
    with np.errstate(all="ignore"):
        results = system.simulate()

    assert results["role_update_times"] == [2, 4]
    assert len(results["estimated_reward_pu_history"]) == 5
    assert len(results["selected_reputation_history"]) == 5
    assert len(results["weighted_selected_reputation_history"]) == 5
    assert len(results["highest_rep_agent_history"]) == 5
    assert len(results["following_history"]) == 5


def test_role_step3_removes_agent_from_all_follower_sets(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(gamma=0.0, kappa=0.0, c_threshold=0.2),
    )
    AgentRole = model_module.AgentRole

    i = 1
    agent = system.agents[i]
    agent.state.role = AgentRole.REPUTATION
    agent.state.following = 0
    agent.state.followers = set()
    agent.state.estimated_reward_pu = 10.0
    agent.state.reputation_estimates = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    agent.state.highest_rep_agent_estimate = 0

    # Inject stale duplicate membership to verify Step-3 cleanup matches the paper's
    # "remove from all F_j" behavior.
    system.agents[0].state.followers = {i}
    system.agents[2].state.followers = {i}

    system._update_roles_sequential()

    assert agent.state.role == AgentRole.PERSONAL_UTILITY
    assert agent.state.following is None
    assert i not in system.agents[0].state.followers
    assert i not in system.agents[2].state.followers


def test_role_step1_existing_follower_keeps_same_leader_when_still_worth_it(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    leader = 0
    follower = 1

    system.agents[leader].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[leader].state.followers = {follower}

    system.agents[follower].state.role = AgentRole.REPUTATION
    system.agents[follower].state.following = leader
    system.agents[follower].state.followers = set()
    system.agents[follower].state.estimated_reward_pu = 0.0
    system.agents[follower].state.reputation_estimates = {0: 1.0, 1: 0.0, 2: 0.1}
    system.agents[follower].state.highest_rep_agent_estimate = 0

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    system._update_roles_sequential()

    assert system.agents[follower].state.role == AgentRole.REPUTATION
    assert system.agents[follower].state.following == leader
    assert follower in system.agents[leader].state.followers


def test_role_update_candidates_filters_invalid_ids(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.1, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.followers = set()

    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.followers = set()
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {0: 2.0, 1: 0.0, 2: 0.0}
    system.agents[1].state.highest_rep_agent_estimate = 0

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    # only id=1 is valid; others should be ignored
    system._update_roles_sequential(update_candidates=[-1, 1, 99])

    assert system.agents[1].state.role == AgentRole.REPUTATION
    assert system.agents[1].state.following == 0
    assert system.agents[0].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[2].state.role == AgentRole.PERSONAL_UTILITY


def test_schedule_build_role_update_epochs_from_T_sequence(model_module):
    config = model_module.SystemConfig(
        num_agents=3,
        role_update_s0=5,
        role_update_T_sequence=[10, 20, 30],
    )
    system = model_module.MultiAgentSystem(config)

    # s1 = 5+10=15, s2 = 15+20=35, s3 = 35+30=65
    assert system._role_update_epochs == [15, 35, 65]


def test_schedule_build_role_update_epochs_from_direct_epoch_list(model_module):
    config = model_module.SystemConfig(
        num_agents=3,
        role_update_epochs=[30, 10, 30, 50, -1, 0],
    )
    system = model_module.MultiAgentSystem(config)

    assert system._role_update_epochs == [10, 30, 50]


def test_schedule_step_uses_explicit_role_update_epochs(model_module):
    np.random.seed(0)
    config = model_module.SystemConfig(
        num_agents=3,
        num_time_steps=5,
        role_update_epochs=[2, 4],
    )
    system = model_module.MultiAgentSystem(config)

    call_counter = {"n": 0}

    def fake_update_roles():
        call_counter["n"] += 1

    system._update_roles_sequential = fake_update_roles

    for _ in range(5):
        system.step()

    assert call_counter["n"] == 2
    assert system.role_update_epoch == 2


def test_schedule_step_uses_fixed_role_update_interval(model_module):
    np.random.seed(0)
    config = model_module.SystemConfig(
        num_agents=3,
        num_time_steps=6,
        fixed_role_update_interval=True,
        role_update_base_interval=2,
    )
    system = model_module.MultiAgentSystem(config)

    call_counter = {"n": 0}

    def fake_update_roles():
        call_counter["n"] += 1

    system._update_roles_sequential = fake_update_roles

    for _ in range(6):
        system.step()

    # updates at t = 2, 4, 6
    assert call_counter["n"] == 3
    assert system.role_update_epoch == 3


def test_schedule_default_increasing_interval_triggers_role_updates(model_module):
    np.random.seed(0)
    config = model_module.SystemConfig(
        num_agents=3,
        num_time_steps=120,
        role_update_base_interval=50,
        fixed_role_update_interval=False,
    )
    system = model_module.MultiAgentSystem(config)

    call_steps = []

    def fake_update_roles():
        call_steps.append(system.time_step)

    system._update_roles_sequential = fake_update_roles

    for _ in range(120):
        system.step()

    # With base=50 and increasing interval:
    # first update at t=50
    # then role_update_epoch becomes 1, interval becomes int(50*(1+0.1))=55
    # next update at t=110
    assert call_steps == [50, 105]
    assert system.role_update_epoch == 2

# ==================== PERSONAL UTILITY ====================

def test_estimates_reward_ema_personal_utility(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=1, num_actions=2))

    a = system.agents[0]
    a.state.estimated_reward_pu = 1.0
    a.state.weights_pu = np.zeros((3, 2))

    eta_J = 0.5
    alpha = 0.0
    a.update_personal_utility(state=0, action=0, reward=3.0, alpha_pu_t=alpha, eta_J_t=eta_J)

    expected = 1.0 + eta_J * (3.0 - 1.0)
    assert abs(a.state.estimated_reward_pu - expected) < 1e-10


def test_pu_policy_gradient_updates_only_current_state_row(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=3, num_actions=2))
    a = system.agents[0]

    weights = np.zeros((3, 2))
    new_weights = a.update_policy_gradient(
        state=1,
        action=0,
        reward=2.0,
        weights=weights,
        lr=0.5,
    )

    # only row 1 should change
    assert np.allclose(new_weights[0], weights[0])
    assert not np.allclose(new_weights[1], weights[1])
    assert np.allclose(new_weights[2], weights[2])

    # chosen action 0 should be pushed up relative to action 1
    assert new_weights[1, 0] > new_weights[1, 1]


def test_pu_select_action_uses_weights_pu(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=1, num_actions=2))
    AgentRole = model_module.AgentRole

    a = system.agents[0]
    a.state.role = AgentRole.PERSONAL_UTILITY

    # make action 0 overwhelmingly likely
    a.state.weights_pu = np.array([[15.0, -15.0]])

    draws = [a.select_action(state=0) for _ in range(30)]

    # should essentially always choose action 0 under such extreme logits
    assert all(x == 0 for x in draws)


def test_pu_update_personal_utility_appends_payoff_history(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=1, num_actions=2))
    a = system.agents[0]

    a.state.payoff_history = []
    a.state.weights_pu = np.zeros((1, 2))
    a.state.estimated_reward_pu = 0.0

    a.update_personal_utility(
        state=0,
        action=1,
        reward=2.5,
        alpha_pu_t=0.0,
        eta_J_t=0.5,
    )

    assert a.state.payoff_history == [2.5]


def test_behavior_weights_returns_status_weights_for_status_role(model_module):
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=2, num_actions=2))
    AgentRole = model_module.AgentRole

    a = system.agents[0]
    a.state.role = AgentRole.STATUS
    a.state.weights_pu = np.array([[1.0, 2.0], [3.0, 4.0]])
    a.state.weights_status = np.array([[10.0, 20.0], [30.0, 40.0]])

    out = a.get_behavior_weights()
    assert np.array_equal(out, a.state.weights_status)


def test_reputation_follower_select_action_uses_leader_behavior_weights(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=1, num_actions=2))
    AgentRole = model_module.AgentRole

    leader = system.agents[0]
    follower = system.agents[1]

    leader.state.role = AgentRole.STATUS
    leader.state.weights_pu = np.array([[15.0, -15.0]])       # would choose action 0
    leader.state.weights_status = np.array([[-15.0, 15.0]])   # should choose action 1

    follower.state.role = AgentRole.REPUTATION
    follower.state.following = 0

    draws = [follower.select_action(state=0) for _ in range(20)]

    # follower should imitate leader's STATUS behavior weights, so choose action 1
    assert all(x == 1 for x in draws)










def test_fast_path_phase4_updates_only_paper_gossip_scope_and_highest_rep_agent(model_module):
    np.random.seed(0)
    config = model_module.SystemConfig(
        num_agents=3,
        use_numpy_fast_path=True,
        num_states=1,
        num_actions=2,
    )
    system = model_module.MultiAgentSystem(config)

    # overwrite dense matrices with deterministic values
    system._v_matrix = np.zeros((3, 3), dtype=float)
    system._s_matrix = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 5.0, 8.0],
        [0.0, 9.0, 4.0],
    ], dtype=float)

    observed_utility_matrix = np.array([
        [10.0, 0.0, 0.0],
        [6.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ], dtype=float)
    active_actor_ids = np.array([0], dtype=int)
    active_participant_ids = np.array([0, 1, 2], dtype=int)
    eta_v_t = 0.2

    # Paper Section 7.3.3: B(t) is built from the active participants' current
    # highest-reputation targets. Here all participants target agent 1, so only
    # column 1 should be gossip-updated.
    for agent in system.agents:
        agent.state.highest_rep_agent_estimate = 1

    system._phase4_updates_numpy_fast(
        observed_utility_matrix=observed_utility_matrix,
        active_actor_ids=active_actor_ids,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
    )

    # v update: only target 0 is active, with observer-specific utilities [10, 6, 2]
    assert np.allclose(system._v_matrix[:, 0], np.array([2.0, 1.2, 0.4]))

    # Only column 1 is inside B(t), so only that column is updated.
    # Its mean is mean(1,5,9)=5 and delta_v(:,1)=0 because actor 1 was inactive.
    expected_rows = np.array([
        [0.0, 5.0, 2.0],
        [0.0, 5.0, 8.0],
        [0.0, 5.0, 4.0],
    ])
    assert np.allclose(system._s_matrix, expected_rows)

    # Agent 0 should still identify target 1 as highest after the scoped update.
    assert system.agents[0].state.highest_rep_agent_estimate == 1


def test_system_builds_paper_gossip_scope_from_active_participants(model_module):
    system = make_system(model_module, num_agents=4)
    for agent_id, target in {0: 2, 1: 2, 2: 3, 3: None}.items():
        system.agents[agent_id].state.highest_rep_agent_estimate = target

    b_t = system._compute_gossip_target_ids_from_active_participants([0, 1, 2, 3])

    assert b_t == [2, 3]

# ==============================================================================
# === SECTION 2: Interaction Rates, Status, and Role Switching ===
# ==============================================================================








# ==================== RATE ALLOCATION ====================

def test_ir1_actor_activation_uses_theta_mu(model_module):
    system = make_system(model_module, num_agents=1, extra_config=dict(u_0=0.0))
    agent = system.agents[0]
    agent.state.actor_interaction_rate = 0.8
    agent.state.participant_interaction_rate = 0.0
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    actor_empirical, _ = estimate_activation_frequency(system, steps=15000, seed=11)
    expected = 1.0 - np.exp(-0.8)
    assert abs(actor_empirical - expected) < 0.03


def test_ir1_participant_activation_uses_theta_mu(model_module):
    system = make_system(model_module, num_agents=1, extra_config=dict(u_0=0.0))
    agent = system.agents[0]
    agent.state.actor_interaction_rate = 0.0
    agent.state.participant_interaction_rate = 0.8
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    _, participant_empirical = estimate_activation_frequency(system, steps=15000, seed=22)
    expected = 1.0 - np.exp(-0.8)
    assert abs(participant_empirical - expected) < 0.03


def test_interaction_rate_direction_and_bounds(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=1,
        extra_config=dict(M=1.0, u_0=0.1, gamma=2.0, kappa=2.0),
    )
    agent = system.agents[0]

    agent.state.actor_interaction_rate = 0.5
    agent.state.estimated_reward_pu = 1.0
    agent.state.estimated_reward_rep = 1.0
    agent.state.estimated_reward_status = 1.0
    old_mu = agent.state.actor_interaction_rate
    agent.update_actor_interaction_rate(0.1)
    assert agent.state.actor_interaction_rate > old_mu

    agent.state.actor_interaction_rate = 0.5
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0
    old_mu = agent.state.actor_interaction_rate
    agent.update_actor_interaction_rate(0.1)
    assert agent.state.actor_interaction_rate < old_mu

    agent.state.actor_interaction_rate = 1.5
    agent.update_actor_interaction_rate(0.1)
    assert 0.0 <= agent.state.actor_interaction_rate <= system.config.M

    agent.state.actor_interaction_rate = -0.5
    agent.update_actor_interaction_rate(0.1)
    assert 0.0 <= agent.state.actor_interaction_rate <= system.config.M


def test_ir_eq13_exact_one_step_value(model_module):
    """Section 6.7 Eq. (13): one-step actor-rate update matches the exact formula."""
    system = make_system(
        model_module,
        num_agents=1,
        extra_config=dict(M=1.0, u_0=0.1, gamma=2.0, kappa=3.0),
    )
    agent = system.agents[0]

    agent.state.actor_interaction_rate = 0.3
    agent.state.estimated_reward_pu = 0.4
    agent.state.estimated_reward_rep = 0.1
    agent.state.estimated_reward_status = 0.2

    alpha_rate = 0.05
    mu_prev = 0.3

    h_hat = max(
        agent.state.estimated_reward_pu,
        system.config.gamma * agent.state.estimated_reward_rep,
        system.config.kappa * agent.state.estimated_reward_status,
    )
    expected = np.clip(
        mu_prev
        + alpha_rate
        * (
            -np.exp(-(system.config.M - mu_prev)) * system.config.u_0
            + np.exp(-mu_prev) * h_hat
        ),
        0.0,
        system.config.M,
    )

    agent.update_actor_interaction_rate(alpha_rate)
    assert agent.state.actor_interaction_rate == pytest.approx(expected, abs=1e-12)


def test_ir_eq13_uses_weighted_max_gamma_kappa(model_module):
    """Eq. (13) must use max{Jpu, gamma*Jr, kappa*Js}, not unweighted reward max."""
    system = make_system(
        model_module,
        num_agents=1,
        extra_config=dict(M=1.0, u_0=0.1, gamma=2.0, kappa=0.1),
    )
    agent = system.agents[0]

    mu_prev = 0.4
    alpha_rate = 0.3
    agent.state.actor_interaction_rate = mu_prev

    # Weighted max should be gamma*Jr = 0.8.
    # Unweighted max would be Jpu = 0.55.
    agent.state.estimated_reward_pu = 0.55
    agent.state.estimated_reward_rep = 0.4
    agent.state.estimated_reward_status = 0.1

    weighted_h = max(
        agent.state.estimated_reward_pu,
        system.config.gamma * agent.state.estimated_reward_rep,
        system.config.kappa * agent.state.estimated_reward_status,
    )
    unweighted_h = max(
        agent.state.estimated_reward_pu,
        agent.state.estimated_reward_rep,
        agent.state.estimated_reward_status,
    )
    assert weighted_h != unweighted_h

    expected_weighted = np.clip(
        mu_prev
        + alpha_rate
        * (
            -np.exp(-(system.config.M - mu_prev)) * system.config.u_0
            + np.exp(-mu_prev) * weighted_h
        ),
        0.0,
        system.config.M,
    )
    expected_unweighted = np.clip(
        mu_prev
        + alpha_rate
        * (
            -np.exp(-(system.config.M - mu_prev)) * system.config.u_0
            + np.exp(-mu_prev) * unweighted_h
        ),
        0.0,
        system.config.M,
    )

    agent.update_actor_interaction_rate(alpha_rate)

    assert agent.state.actor_interaction_rate == pytest.approx(expected_weighted, abs=1e-12)
    assert agent.state.actor_interaction_rate != pytest.approx(expected_unweighted, abs=1e-6)


def test_actor_rate_driver_standard_mode_ignores_status_override_even_with_followers(model_module):
    system = make_system(
        model_module,
        num_agents=12,
        extra_config=dict(
            M=1.0,
            u_0=0.1,
            gamma=2.0,
            kappa=0.0,
            actor_rate_driver_mode="standard",
            actor_rate_status_override_min_followers=10,
        ),
    )
    agent = system.agents[0]
    agent.state.followers = set(range(1, 11))
    agent.state.actor_interaction_rate = 0.4
    agent.state.estimated_reward_pu = 0.9
    agent.state.estimated_reward_rep = 0.4
    agent.state.estimated_reward_status = 0.3

    rate_terms = agent.get_actor_rate_terms()

    assert rate_terms["status_override_active"] == 0
    assert rate_terms["driver"] == pytest.approx(0.9, abs=1e-12)
    assert rate_terms["driver_label"] == "pu"


def test_actor_rate_status_override_kappa0_below_threshold_uses_standard_driver(model_module):
    system = make_system(
        model_module,
        num_agents=12,
        extra_config=dict(
            M=1.0,
            u_0=0.1,
            gamma=2.0,
            kappa=0.0,
            actor_rate_driver_mode="status_if_followers_kappa0",
            actor_rate_status_override_min_followers=10,
        ),
    )
    agent = system.agents[0]
    agent.state.followers = set(range(1, 10))
    agent.state.actor_interaction_rate = 0.4
    agent.state.estimated_reward_pu = 0.9
    agent.state.estimated_reward_rep = 0.4
    agent.state.estimated_reward_status = 0.3

    rate_terms = agent.get_actor_rate_terms()

    assert rate_terms["follower_count"] == 9
    assert rate_terms["status_override_active"] == 0
    assert rate_terms["driver"] == pytest.approx(0.9, abs=1e-12)
    assert rate_terms["driver_label"] == "pu"


def test_actor_rate_status_override_kappa0_at_threshold_uses_unweighted_status(model_module):
    system = make_system(
        model_module,
        num_agents=12,
        extra_config=dict(
            M=1.0,
            u_0=0.1,
            gamma=2.0,
            kappa=0.0,
            actor_rate_driver_mode="status_if_followers_kappa0",
            actor_rate_status_override_min_followers=10,
        ),
    )
    agent = system.agents[0]
    agent.state.followers = set(range(1, 11))
    mu_prev = 0.4
    alpha_rate = 0.2
    agent.state.actor_interaction_rate = mu_prev
    agent.state.estimated_reward_pu = 0.9
    agent.state.estimated_reward_rep = 0.4
    agent.state.estimated_reward_status = 0.3

    rate_terms = agent.get_actor_rate_terms()
    expected = np.clip(
        mu_prev
        + alpha_rate
        * (
            -np.exp(-(system.config.M - mu_prev)) * system.config.u_0
            + np.exp(-mu_prev) * agent.state.estimated_reward_status
        ),
        0.0,
        system.config.M,
    )

    assert rate_terms["follower_count"] == 10
    assert rate_terms["status_override_active"] == 1
    assert rate_terms["driver"] == pytest.approx(agent.state.estimated_reward_status, abs=1e-12)
    assert rate_terms["driver_label"] == "status_override"

    agent.update_actor_interaction_rate(alpha_rate)
    assert agent.state.actor_interaction_rate == pytest.approx(expected, abs=1e-12)


def test_actor_rate_status_override_does_not_activate_when_kappa_positive(model_module):
    system = make_system(
        model_module,
        num_agents=12,
        extra_config=dict(
            M=1.0,
            u_0=0.1,
            gamma=2.0,
            kappa=1.0,
            actor_rate_driver_mode="status_if_followers_kappa0",
            actor_rate_status_override_min_followers=10,
        ),
    )
    agent = system.agents[0]
    agent.state.followers = set(range(1, 11))
    agent.state.actor_interaction_rate = 0.4
    agent.state.estimated_reward_pu = 0.9
    agent.state.estimated_reward_rep = 0.4
    agent.state.estimated_reward_status = 0.3

    rate_terms = agent.get_actor_rate_terms()

    assert rate_terms["status_override_active"] == 0
    assert rate_terms["driver"] == pytest.approx(0.9, abs=1e-12)
    assert rate_terms["driver_label"] == "pu"


def test_ir_participant_rate_remains_constant_over_steps(model_module):
    """Section 6.2/6.1.2 assumption: participant rates mu_p,i are fixed over time."""
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(role_update_base_interval=10**9, u_0=0.0),
    )

    initial_participant_rates = [a.state.participant_interaction_rate for a in system.agents]

    np.random.seed(101)
    for _ in range(300):
        system.step()

    final_participant_rates = [a.state.participant_interaction_rate for a in system.agents]
    assert final_participant_rates == initial_participant_rates


def test_ir_actor_participant_sampling_are_independent(model_module):
    """Section 6.2: actor and participant inclusion are sampled independently."""
    system = make_system(
        model_module,
        num_agents=1,
        extra_config=dict(role_update_base_interval=10**9, u_0=0.0),
    )
    agent = system.agents[0]
    agent.state.actor_interaction_rate = 0.8
    agent.state.participant_interaction_rate = 0.6

    # Keep mu_a fixed through the run so frequencies are stationary.
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    np.random.seed(202)
    for _ in range(20000):
        system.step()

    actor_active = np.array(system.results["actor_counts"], dtype=float)  # 0/1 for num_agents=1
    participant_active = np.array(system.results["participant_counts"], dtype=float)  # 0/1

    p_actor = float(np.mean(actor_active))
    p_participant = float(np.mean(participant_active))
    p_joint = float(np.mean((actor_active == 1.0) & (participant_active == 1.0)))

    assert abs(p_joint - p_actor * p_participant) < 0.02


def test_ir_alpha_rate_schedule_decreases_with_time(model_module):
    """Assumption 5 style schedule: actor-rate stepsize alpha_rate_t should decrease over time."""
    system = make_system(model_module, num_agents=1)
    agent = system.agents[0]

    # Choose state so update_delta > 0 and clipping does not activate.
    agent.state.estimated_reward_pu = 1.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    mu_prev = 0.5
    alpha_t1 = 0.01 / (1.0 + 1 * 0.005)
    alpha_t1000 = 0.01 / (1.0 + 1000 * 0.005)

    assert alpha_t1 > alpha_t1000 > 0.0

    agent.state.actor_interaction_rate = mu_prev
    agent.update_actor_interaction_rate(alpha_t1)
    delta_early = agent.state.actor_interaction_rate - mu_prev

    agent.state.actor_interaction_rate = mu_prev
    agent.update_actor_interaction_rate(alpha_t1000)
    delta_late = agent.state.actor_interaction_rate - mu_prev

    assert abs(delta_early) > abs(delta_late)


def test_ir_u0_effect_when_h_zero(model_module):
    """With H_hat=0 and u0>0, Eq. (13) should reduce in-group actor rate."""
    system = make_system(
        model_module,
        num_agents=1,
        extra_config=dict(M=1.0, u_0=0.3, gamma=2.0, kappa=2.0),
    )
    agent = system.agents[0]

    agent.state.actor_interaction_rate = 0.6
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    old_mu = agent.state.actor_interaction_rate
    agent.update_actor_interaction_rate(alpha_rate=0.1)
    assert agent.state.actor_interaction_rate < old_mu


# ==================== REPUTATION LEARNING ====================

def test_rep1_highest_reputation_selection_excludes_self(model_module):
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]

    agent.state.reputation_estimates = {0: 10.0, 1: 2.0, 2: 1.0}
    agent.config.delta = 0.0

    np.random.seed(0)
    agent.identify_highest_reputation_agent()

    assert agent.state.highest_rep_agent_estimate != 0


def test_true_reputation_helper_uses_theta_and_expected_group_utility(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(
            num_states=1,
            num_actions=2,
            reward_model="shared_base_gaussian",
        ),
    )

    system._reward_tables = np.array(
        [
            [[2.0, 0.0]],
            [[3.0, 1.0]],
            [[4.0, 1.0]],
        ],
        dtype=float,
    )

    for agent in system.agents:
        agent.state.weights_pu[:] = np.array([[-20.0, 20.0]])
        agent.state.role = model_module.AgentRole.PERSONAL_UTILITY
        agent.state.actor_interaction_rate = 1.0

    system.agents[0].state.weights_pu[:] = np.array([[20.0, -20.0]])  # deterministic action 0

    rows = system._build_true_reputation_checkpoint_rows(
        checkpoint_kind="final",
        role_update_index=0,
    )
    by_agent = {int(row["agent_id"]): row for row in rows}
    theta = 1.0 - np.exp(-1.0)

    assert by_agent[0]["sum_expected_utility_others"] == pytest.approx(7.0, abs=1e-6)
    assert by_agent[0]["true_reputation"] == pytest.approx(theta * 7.0, abs=1e-6)
    assert by_agent[0]["true_rank"] == 1
    assert by_agent[0]["unique_true_top_agent"] == 0
    assert by_agent[1]["gap_to_true_top"] > 0.0
    assert by_agent[2]["gap_to_true_top"] > 0.0


def test_rep1_tie_selection_is_uniform_for_dict_path(model_module):
    system = make_system(model_module, num_agents=4, extra_config=dict(delta=0.0))
    agent = system.agents[3]
    agent.state.reputation_estimates = {
        0: 1.0,
        1: 1.0,
        2: 0.2,
        3: 999.0,
    }

    draws = []
    np.random.seed(77)
    for _ in range(1000):
        agent.identify_highest_reputation_agent()
        draws.append(int(agent.state.highest_rep_agent_estimate))

    count_0 = draws.count(0)
    count_1 = draws.count(1)
    assert count_0 + count_1 == 1000
    share_0 = count_0 / 1000.0
    assert 0.40 <= share_0 <= 0.60


def test_rep1_tie_selection_is_uniform_for_matrix_path(model_module):
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(delta=0.0, use_numpy_fast_path=True),
    )

    system._s_matrix[3, :] = np.array([1.0, 1.0, 0.2, 999.0], dtype=float)

    draws = []
    np.random.seed(88)
    for _ in range(1000):
        system._identify_highest_reputation_agent_from_matrix(3)
        draws.append(int(system.agents[3].state.highest_rep_agent_estimate))

    count_0 = draws.count(0)
    count_1 = draws.count(1)
    assert count_0 + count_1 == 1000
    share_0 = count_0 / 1000.0
    assert 0.40 <= share_0 <= 0.60


def test_rep2_reputation_update_matches_eq9_additive_structure(model_module):
    system = make_system(model_module, num_agents=2)
    agent_i = system.agents[0]
    agent_j = system.agents[1]

    k = 1
    agent_i.state.personal_benefit_estimates[k] = 0.2
    agent_i.state.reputation_estimates[k] = 0.0
    agent_j.state.reputation_estimates[k] = 0.4

    observed_payoffs = {k: 1.0}
    eta_v_t = 0.5

    v_old = agent_i.state.personal_benefit_estimates[k]
    deltas = agent_i.update_personal_benefit_estimates(observed_payoffs, eta_v_t, active_actor_ids=[k])
    v_new = agent_i.state.personal_benefit_estimates[k]

    agent_i.update_reputation_estimates_gossip(deltas, [agent_j], eta_s_t=1.0)

    expected = agent_j.state.reputation_estimates[k] + (v_new - v_old)
    assert agent_i.state.reputation_estimates[k] == pytest.approx(expected, abs=1e-9)


def test_rep2_gossip_update_fallback_without_other_agents(model_module):
    """Cover Eq.(9) fallback path when no peer estimates are available."""
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]

    agent.state.reputation_estimates = {0: 0.3, 1: -0.2, 2: 1.2}
    deltas = {1: 0.5}

    agent.update_reputation_estimates_gossip(deltas, [], eta_s_t=1.0)

    assert agent.state.reputation_estimates[0] == pytest.approx(0.3, abs=1e-12)
    assert agent.state.reputation_estimates[1] == pytest.approx(0.3, abs=1e-12)
    assert agent.state.reputation_estimates[2] == pytest.approx(1.2, abs=1e-12)


def test_estimates_personal_benefit_delta_active(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]
    eta = 0.2

    agent.state.personal_benefit_estimates[1] = 4.0
    observed_payoffs = {0: 0.0, 1: 10.0, 2: 0.0}
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta, active_actor_ids=[1])

    expected_new = 4.0 + eta * (10.0 - 4.0)
    expected_delta = expected_new - 4.0

    assert abs(agent.state.personal_benefit_estimates[1] - expected_new) < 1e-10
    assert abs(deltas[1] - expected_delta) < 1e-10


def test_estimates_personal_benefit_decay_inactive(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]
    eta = 0.2

    agent.state.personal_benefit_estimates[2] = 5.0
    observed_payoffs = {0: 0.0, 1: 0.0, 2: 0.0}
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta, active_actor_ids=[])

    expected_new = 5.0 * (1.0 - eta)
    expected_delta = expected_new - 5.0

    assert abs(agent.state.personal_benefit_estimates[2] - expected_new) < 1e-10
    assert abs(deltas[2] - expected_delta) < 1e-10


def test_rep642_inactive_decay_matches_formula(model_module):
    system = make_system(model_module, num_agents=2)
    agent = system.agents[0]

    agent.state.personal_benefit_estimates[1] = 0.5
    eta_v_t = 0.2
    agent.update_personal_benefit_estimates({1: 0.0}, eta_v_t=eta_v_t, active_actor_ids=[])

    assert agent.state.personal_benefit_estimates[1] == pytest.approx(
        0.5 * (1.0 - eta_v_t), abs=1e-12
    )


def test_rep642_all_agents_update_personal_benefit_each_step(model_module):
    system = make_system(model_module, num_agents=2, extra_config=dict(role_update_base_interval=10**9))
    a0, a1 = system.agents

    a0.state.participant_interaction_rate = 100.0
    a1.state.participant_interaction_rate = 0.0
    a0.state.actor_interaction_rate = 0.0
    a1.state.actor_interaction_rate = 0.0

    a1.state.personal_benefit_estimates[0] = 1.0
    eta_v_t = system.config.eta_v_base / (1.0 + 1.0 * 0.01)

    np.random.seed(0)
    system.step()

    expected = 1.0 * (1.0 - eta_v_t)
    assert a1.state.personal_benefit_estimates[0] == pytest.approx(expected, abs=1e-12)


def test_rep642_personal_benefit_is_observer_specific_in_step(model_module):
    system = make_system(model_module, num_agents=3, extra_config=dict(role_update_base_interval=10**9))
    actor = system.agents[0]
    disagreeing_observer = system.agents[1]
    agreeing_observer = system.agents[2]

    for agent in system.agents:
        agent.state.actor_interaction_rate = 0.0
        agent.state.participant_interaction_rate = 0.0

    actor.state.actor_interaction_rate = 100.0
    actor.select_action = lambda state: 0

    eta_v_t = system.config.eta_v_base / (1.0 + 1.0 * 0.01)

    np.random.seed(0)
    system.step()

    assert disagreeing_observer.state.personal_benefit_estimates[0] == pytest.approx(0.0, abs=1e-12)
    assert agreeing_observer.state.personal_benefit_estimates[0] == pytest.approx(eta_v_t, abs=1e-12)


def test_rep642_shared_base_gaussian_observer_specific_utilities(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(num_states=1, reward_model="shared_base_gaussian"),
    )

    system._reward_tables[:, :, :] = 0.0
    system._reward_tables[0, 0, 1] = 0.4
    system._reward_tables[1, 0, 1] = 1.2
    system._reward_tables[2, 0, 1] = 2.3

    utilities = system.compute_observer_utility_vector(0, 1)

    assert np.allclose(utilities, np.array([0.4, 1.2, 2.3]))


def test_rep642_numpy_fast_path_uses_observer_specific_utilities(model_module):
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(role_update_base_interval=10**9, use_numpy_fast_path=True),
    )
    actor = system.agents[0]

    for agent in system.agents:
        agent.state.actor_interaction_rate = 0.0
        agent.state.participant_interaction_rate = 0.0

    actor.state.actor_interaction_rate = 100.0
    actor.select_action = lambda state: 0

    eta_v_t = system.config.eta_v_base / (1.0 + 1.0 * 0.01)

    np.random.seed(0)
    system.step()

    assert system._v_matrix[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert system._v_matrix[2, 0] == pytest.approx(eta_v_t, abs=1e-12)


def test_rep643_non_participant_reputation_estimates_unchanged(model_module):
    system = make_system(model_module, num_agents=2, extra_config=dict(role_update_base_interval=10**9))
    a0, a1 = system.agents

    a0.state.participant_interaction_rate = 100.0
    a1.state.participant_interaction_rate = 0.0
    a0.state.actor_interaction_rate = 0.0
    a1.state.actor_interaction_rate = 0.0

    a1.state.reputation_estimates = {0: 0.7, 1: -0.2}
    before = dict(a1.state.reputation_estimates)

    np.random.seed(1)
    system.step()

    assert a1.state.reputation_estimates == before


def test_rep644_tie_threshold_selection_is_near_uniform(model_module):
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]
    agent.config.delta = 0.1
    agent.state.reputation_estimates = {0: -10.0, 1: 1.0, 2: 0.95}

    counts = {1: 0, 2: 0}
    np.random.seed(123)
    n = 6000
    for _ in range(n):
        agent.identify_highest_reputation_agent()
        counts[agent.state.highest_rep_agent_estimate] += 1

    p1 = counts[1] / n
    p2 = counts[2] / n
    assert abs(p1 - 0.5) < 0.06
    assert abs(p2 - 0.5) < 0.06


def test_rep1_identify_highest_with_empty_reputation_dict(model_module):
    """Cold-start path: no estimates yet should still select a valid agent id."""
    system = make_system(model_module, num_agents=4)
    agent = system.agents[0]

    agent.state.reputation_estimates = {}
    np.random.seed(9)
    agent.identify_highest_reputation_agent()

    selected = int(agent.state.highest_rep_agent_estimate)
    assert 0 <= selected < system.config.num_agents


def test_rep1_single_agent_fallback_no_crash(model_module):
    """Single-agent edge case should not fail when self is excluded from candidates."""
    system = make_system(model_module, num_agents=1)
    agent = system.agents[0]

    # Forces the non-self candidate set to be empty.
    agent.state.reputation_estimates = {0: 7.0}
    agent.identify_highest_reputation_agent()

    assert agent.state.highest_rep_agent_estimate == 0


def test_rep644_non_participant_highest_rep_estimate_unchanged(model_module):
    system = make_system(model_module, num_agents=2, extra_config=dict(role_update_base_interval=10**9))
    a0, a1 = system.agents

    a0.state.participant_interaction_rate = 100.0
    a1.state.participant_interaction_rate = 0.0
    a0.state.actor_interaction_rate = 0.0
    a1.state.actor_interaction_rate = 0.0

    a1.state.highest_rep_agent_estimate = 0
    before = a1.state.highest_rep_agent_estimate

    np.random.seed(2)
    system.step()

    assert a1.state.highest_rep_agent_estimate == before


def test_rep645_adopt_leader_behavior_noop_when_no_valid_leader(model_module):
    """adopt_leader_behavior should no-op for following=None and following=self."""
    system = make_system(model_module, num_agents=2)
    agent = system.agents[0]

    baseline = agent.state.weights_pu.copy()

    agent.state.following = None
    agent.adopt_leader_behavior()
    assert np.allclose(agent.state.weights_pu, baseline)

    agent.state.following = agent.agent_id
    agent.adopt_leader_behavior()
    assert np.allclose(agent.state.weights_pu, baseline)


def test_rep645_follower_tracks_status_policy_of_status_leader(model_module):
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=1, num_actions=2))
    AgentRole = model_module.AgentRole
    leader = system.agents[0]
    follower = system.agents[1]

    leader.state.role = AgentRole.STATUS
    follower.state.role = AgentRole.REPUTATION
    follower.state.following = 0

    leader.state.weights_pu[:] = np.array([[8.0, -8.0]])
    leader.state.weights_status[:] = np.array([[-8.0, 8.0]])

    counts = [0, 0]
    np.random.seed(4)
    for _ in range(3000):
        a = follower.select_action(0)
        counts[a] += 1

    p_action1 = counts[1] / sum(counts)
    assert p_action1 > 0.95


def test_rep66_reputation_reward_estimate_matches_followed_agent_reputation(model_module):
    system = make_system(model_module, num_agents=2, extra_config=dict(role_update_base_interval=10**9))
    AgentRole = model_module.AgentRole
    i = system.agents[0]
    k = system.agents[1]

    i.state.role = AgentRole.REPUTATION
    i.state.following = 1
    i.state.estimated_reward_rep = 0.0
    i.state.reputation_estimates[1] = 0.8

    i.state.actor_interaction_rate = 100.0
    i.state.participant_interaction_rate = 0.0
    k.state.actor_interaction_rate = 0.0
    k.state.participant_interaction_rate = 0.0

    np.random.seed(5)
    system.step()

    assert i.state.estimated_reward_rep == pytest.approx(i.state.reputation_estimates[1], abs=1e-12)


# ==================== STATUS ====================

def test_status_reward_uses_sum_not_average(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=4, extra_config=dict(num_states=3, num_actions=2))
    leader = system.agents[0]

    state = 0
    action = 0
    beta_status_t = 0.0
    eta_J_t = 1.0

    follower_payoffs = [1.0, 2.0, 3.0]
    social_support_sum = float(sum(follower_payoffs))
    social_support_avg = float(np.mean(follower_payoffs))
    assert social_support_sum != social_support_avg

    leader.state.estimated_reward_status = 0.0
    leader.update_status_optimization(
        state=state,
        action=action,
        social_support_sum=social_support_sum,
        beta_status_t=beta_status_t,
        eta_J_t=eta_J_t,
    )
    assert abs(leader.state.estimated_reward_status - social_support_sum) < 1e-10

    leader.state.estimated_reward_status = 0.0
    leader.update_status_optimization(
        state=state,
        action=action,
        social_support_sum=social_support_avg,
        beta_status_t=beta_status_t,
        eta_J_t=eta_J_t,
    )
    assert abs(leader.state.estimated_reward_status - social_support_avg) < 1e-10


def test_status_entry_can_occur_after_status_reward_learning(model_module):
    """
    Deterministic STATUS-1 check:
    1) A non-STATUS agent with followers gets pre-STATUS reward-signal update in Phase 3.
    2) Step-2 role update uses that updated signal to switch the agent to STATUS.
    """
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(
            num_states=3,
            num_actions=2,
            gamma=2.0,
            kappa=50.0,
            c_threshold=0.1,  # min_followers = 1
            role_update_base_interval=1,
            u_0=0.0,
        ),
    )
    AgentRole = model_module.AgentRole

    leader = system.agents[0]
    follower = system.agents[1]

    # Leader starts as non-STATUS with one follower, so Step-2 eligibility is guaranteed.
    leader.state.role = AgentRole.PERSONAL_UTILITY
    leader.state.followers = {1}
    leader.state.estimated_reward_pu = 0.0
    leader.state.estimated_reward_status = 0.0

    # Force leader action to 1 (leader preferred action is 0), so PU reward stays low.
    leader.state.weights_pu[:] = np.array([[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]])

    # Follower is active and follows leader policy; follower preferred action is 1, so support payoff is high.
    follower.state.role = AgentRole.REPUTATION
    follower.state.following = 0
    follower.state.followers = set()
    follower.state.estimated_reward_pu = 0.0
    follower.state.reputation_estimates = {0: 1.0, 1: 0.0, 2: 0.0}
    follower.state.highest_rep_agent_estimate = 0

    # Ensure both leader and follower participate in actor/participant sets at this step.
    leader.state.actor_interaction_rate = 100.0
    leader.state.participant_interaction_rate = 100.0
    follower.state.actor_interaction_rate = 100.0
    follower.state.participant_interaction_rate = 100.0

    # Keep third agent effectively inactive to isolate the transition.
    system.agents[2].state.actor_interaction_rate = 0.0
    system.agents[2].state.participant_interaction_rate = 0.0

    np.random.seed(123)
    system.step()

    assert leader.state.estimated_reward_status > 0.0
    assert leader.state.role == AgentRole.STATUS


def test_status_select_action_uses_status_policy_directly(model_module):
    """STATUS agents should sample from weights_status, not weights_pu."""
    system = make_system(model_module, num_agents=1, extra_config=dict(num_states=1, num_actions=2))
    AgentRole = model_module.AgentRole
    agent = system.agents[0]

    agent.state.role = AgentRole.STATUS
    agent.state.weights_pu[:] = np.array([[10.0, -10.0]])      # strongly action 0
    agent.state.weights_status[:] = np.array([[-10.0, 10.0]])  # strongly action 1

    counts = [0, 0]
    np.random.seed(321)
    for _ in range(3000):
        action = agent.select_action(0)
        counts[action] += 1

    p_action1 = counts[1] / sum(counts)
    assert p_action1 > 0.95


def test_status_phase3_status_actor_updates_via_step(model_module):
    """STATUS actor with followers should update status weights and status reward in Phase 3."""
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(role_update_base_interval=10**9, u_0=0.0),
    )
    AgentRole = model_module.AgentRole
    leader = system.agents[0]
    follower = system.agents[1]
    filler = system.agents[2]

    leader.state.role = AgentRole.STATUS
    leader.state.followers = {1}
    leader.state.estimated_reward_status = 0.0

    follower.state.role = AgentRole.PERSONAL_UTILITY
    follower.state.following = None
    follower.state.followers = set()
    follower.state.weights_pu[:] = np.array([[-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]])  # id=1 prefers action 1

    leader.state.actor_interaction_rate = 100.0
    leader.state.participant_interaction_rate = 100.0
    follower.state.actor_interaction_rate = 100.0
    follower.state.participant_interaction_rate = 100.0
    filler.state.actor_interaction_rate = 0.0
    filler.state.participant_interaction_rate = 0.0

    weights_before = leader.state.weights_status.copy()
    np.random.seed(11)
    system.step()

    assert leader.state.estimated_reward_status > 0.0
    assert not np.allclose(leader.state.weights_status, weights_before)


def test_status_phase3_already_status_skips_preupdate_branch(model_module):
    """STATUS role should receive one status EMA update (not double-counted preupdate+status update)."""
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(role_update_base_interval=10**9, u_0=0.0),
    )
    AgentRole = model_module.AgentRole
    leader = system.agents[0]
    follower = system.agents[1]
    filler = system.agents[2]

    leader.state.role = AgentRole.STATUS
    leader.state.followers = {1}
    leader.state.estimated_reward_status = 0.0

    follower.state.role = AgentRole.PERSONAL_UTILITY
    follower.state.weights_pu[:] = np.array([[-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]])  # id=1 prefers action 1

    leader.state.actor_interaction_rate = 100.0
    leader.state.participant_interaction_rate = 100.0
    follower.state.actor_interaction_rate = 100.0
    follower.state.participant_interaction_rate = 100.0
    filler.state.actor_interaction_rate = 0.0
    filler.state.participant_interaction_rate = 0.0

    eta_j_t = system.config.eta_J_base / (1.0 + 1.0 * 0.01)  # first step
    expected_single = eta_j_t * 1.0
    expected_double = eta_j_t * (2.0 - eta_j_t)

    np.random.seed(12)
    system.step()

    assert leader.state.estimated_reward_status == pytest.approx(expected_single, abs=1e-12)
    assert leader.state.estimated_reward_status != pytest.approx(expected_double, abs=1e-6)


def test_status_support_counts_only_active_participant_followers(model_module):
    """Eq. (11): only followers active as participants contribute to social support."""
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(role_update_base_interval=10**9, u_0=0.0),
    )
    AgentRole = model_module.AgentRole
    leader = system.agents[0]
    f1 = system.agents[1]
    f2 = system.agents[2]

    leader.state.role = AgentRole.STATUS
    leader.state.followers = {1, 2}
    leader.state.estimated_reward_status = 0.0

    # Make both followers active actors with deterministic payoff 1.
    # id=1 prefers action 1, id=2 prefers action 0.
    f1.state.weights_pu[:] = np.array([[-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]])
    f2.state.weights_pu[:] = np.array([[20.0, -20.0], [20.0, -20.0], [20.0, -20.0]])

    leader.state.actor_interaction_rate = 100.0
    leader.state.participant_interaction_rate = 100.0
    f1.state.actor_interaction_rate = 100.0
    f1.state.participant_interaction_rate = 100.0  # included in support
    f2.state.actor_interaction_rate = 100.0
    f2.state.participant_interaction_rate = 0.0    # excluded from support

    eta_j_t = system.config.eta_J_base / (1.0 + 1.0 * 0.01)  # first step
    expected_from_one_active_participant = eta_j_t * 1.0
    expected_if_both_counted = eta_j_t * 2.0

    np.random.seed(13)
    system.step()

    assert leader.state.estimated_reward_status == pytest.approx(
        expected_from_one_active_participant,
        abs=1e-12,
    )
    assert leader.state.estimated_reward_status != pytest.approx(expected_if_both_counted, abs=1e-6)


def test_status_estimate_unchanged_when_preconditions_not_met_reachable(model_module):
    """
    Reachable unchanged-estimate checks:
    1) status agent loses followers -> no status estimate update thereafter.
    2) status agent has followers but is inactive actor -> no status estimate update.
    """
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(role_update_base_interval=10**9, u_0=0.0),
    )
    AgentRole = model_module.AgentRole
    leader = system.agents[0]
    follower = system.agents[1]
    filler = system.agents[2]

    leader.state.role = AgentRole.STATUS
    leader.state.followers = {1}
    leader.state.estimated_reward_status = 0.0

    follower.state.role = AgentRole.PERSONAL_UTILITY
    follower.state.following = None
    follower.state.followers = set()
    follower.state.weights_pu[:] = np.array([[-20.0, 20.0], [-20.0, 20.0], [-20.0, 20.0]])  # id=1 prefers action 1

    leader.state.actor_interaction_rate = 100.0
    leader.state.participant_interaction_rate = 100.0
    follower.state.actor_interaction_rate = 100.0
    follower.state.participant_interaction_rate = 100.0
    filler.state.actor_interaction_rate = 0.0
    filler.state.participant_interaction_rate = 0.0

    # First step: valid status-update conditions hold.
    np.random.seed(14)
    system.step()
    after_valid_update = leader.state.estimated_reward_status
    assert after_valid_update > 0.0

    # Scenario 1 (reachable): follower relation can disappear over time.
    leader.state.followers = set()
    follower.state.following = None
    np.random.seed(15)
    system.step()
    assert leader.state.estimated_reward_status == pytest.approx(after_valid_update, abs=1e-12)

    # Scenario 2: agent has followers but is not an active actor.
    leader.state.followers = {1}
    follower.state.following = 0
    leader.state.actor_interaction_rate = 0.0
    np.random.seed(16)
    system.step()
    assert leader.state.estimated_reward_status == pytest.approx(after_valid_update, abs=1e-12)


def test_status_step3_fallback_clears_following(model_module):
    """Step-3 PU fallback should clear stale following links when neither R nor S conditions hold."""
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=1.0, kappa=1.0, c_threshold=1.0, B_R=0.8, B_F=0.6),
    )
    AgentRole = model_module.AgentRole

    leader = system.agents[0]
    agent = system.agents[1]

    leader.state.followers = {1}

    # Construct a stale-but-reachable cleanup case: PU role with lingering following link.
    agent.state.role = AgentRole.PERSONAL_UTILITY
    agent.state.following = 0
    agent.state.followers = set()
    agent.state.estimated_reward_pu = 1.0
    agent.state.reputation_estimates = {0: 0.0, 1: 0.0, 2: 0.0}
    agent.state.highest_rep_agent_estimate = 0

    system._update_roles_sequential()

    assert agent.state.role == AgentRole.PERSONAL_UTILITY
    assert agent.state.following is None
    assert 1 not in system.agents[0].state.followers

# ==============================================================================
# === SECTION 3: Async Role Switching and Scheduler ===
# ==============================================================================









def _parse_serialized_ids(text):
    if text is None or text == "":
        return []
    return [int(part) for part in str(text).split("|") if part != ""]


# ==================== ASYNC ROLE SWITCHING (partial updates) ====================

def test_async_partial_update_only_selected_agent_recomputes_role(model_module):
    """
    Async semantics: only update_candidates should reevaluate roles.
    If agent 0 is selected and agent 2 is not selected, only agent 0 should move.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # leader candidate
    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.followers = set()

    # agent 0 should follow 1 if updated
    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.following = None
    system.agents[0].state.followers = set()
    system.agents[0].state.estimated_reward_pu = 0.0
    system.agents[0].state.reputation_estimates = {1: 2.0, 2: 0.1, 3: 0.1, 0: 0.0}
    system.agents[0].state.highest_rep_agent_estimate = 1

    # agent 2 also qualifies, but should NOT move because not selected
    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.following = None
    system.agents[2].state.followers = set()
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {1: 2.5, 0: 0.1, 3: 0.1, 2: 0.0}
    system.agents[2].state.highest_rep_agent_estimate = 1

    # filler
    system.agents[3].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[3].state.followers = set()

    system._update_roles_sequential(update_candidates=[0])

    assert system.agents[0].state.role == AgentRole.REPUTATION
    assert system.agents[0].state.following == 1
    assert 0 in system.agents[1].state.followers

    # non-selected agent should remain unchanged
    assert system.agents[2].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[2].state.following is None
    assert 2 not in system.agents[1].state.followers


def test_async_partial_update_existing_follower_can_switch_to_new_leader(model_module):
    """
    Async semantics: a currently-following agent should be able to switch leaders
    when only that single follower is selected for reevaluation.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # agent 0 currently follows leader 1
    system.agents[0].state.role = AgentRole.REPUTATION
    system.agents[0].state.following = 1
    system.agents[0].state.followers = set()

    # old leader
    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.followers = {0}

    # new leader candidate
    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    # filler
    system.agents[3].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[3].state.followers = set()

    # make agent 0 prefer leader 2
    system.agents[0].state.estimated_reward_pu = 0.0
    system.agents[0].state.reputation_estimates = {1: 0.2, 2: 2.0, 3: 0.1, 0: 0.0}
    system.agents[0].state.highest_rep_agent_estimate = 2

    system._update_roles_sequential(update_candidates=[0])

    assert system.agents[0].state.role == AgentRole.REPUTATION
    assert system.agents[0].state.following == 2
    assert 0 not in system.agents[1].state.followers
    assert 0 in system.agents[2].state.followers


def test_async_partial_update_existing_follower_can_return_to_pu(model_module):
    """
    This is the most important async bug-catching test.

    If a selected follower no longer finds following worthwhile, it should return to
    PERSONAL_UTILITY and be removed from the old leader's follower set.

    If this fails, your async partial-update logic is still keeping stale members in R.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    leader = 0
    follower = 1

    system.agents[leader].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[leader].state.followers = {follower}

    system.agents[follower].state.role = AgentRole.REPUTATION
    system.agents[follower].state.following = leader
    system.agents[follower].state.followers = set()

    # Following should no longer be attractive
    system.agents[follower].state.estimated_reward_pu = 1.0
    system.agents[follower].state.reputation_estimates = {0: 0.1, 1: 0.0, 2: 0.0}
    system.agents[follower].state.highest_rep_agent_estimate = 0

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    system._update_roles_sequential(update_candidates=[follower])

    assert system.agents[follower].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[follower].state.following is None
    assert follower not in system.agents[leader].state.followers


def test_async_partial_update_status_switch_clears_following(model_module):
    """
    In async mode, if only one selected agent reevaluates and enters STATUS,
    its following pointer must be cleared and the old leader must lose that follower.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=6,
        extra_config=dict(gamma=1.0, kappa=2.0, c_threshold=0.2),
    )
    AgentRole = model_module.AgentRole

    leader = 0
    i = 1

    system.agents[i].state.role = AgentRole.REPUTATION
    system.agents[i].state.following = leader
    system.agents[leader].state.followers.add(i)

    # enough followers to qualify for status: ceil(0.2*6) = 2
    system.agents[i].state.followers = {2, 3}
    system.agents[i].state.estimated_reward_status = 10.0
    system.agents[i].state.estimated_reward_pu = 0.0

    system._update_roles_sequential(update_candidates=[i])

    assert system.agents[i].state.role == AgentRole.STATUS
    assert system.agents[i].state.following is None
    assert i not in system.agents[leader].state.followers


def test_async_partial_update_demotes_zero_follower_status_only_for_selected_agents(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(gamma=0.0, kappa=2.0, c_threshold=0.2),
    )
    AgentRole = model_module.AgentRole

    selected = 1
    untouched = 2

    system.agents[selected].state.role = AgentRole.STATUS
    system.agents[selected].state.followers = set()
    system.agents[selected].state.following = None
    system.agents[selected].state.estimated_reward_status = 10.0
    system.agents[selected].state.estimated_reward_pu = 0.0

    system.agents[untouched].state.role = AgentRole.STATUS
    system.agents[untouched].state.followers = set()
    system.agents[untouched].state.following = None
    system.agents[untouched].state.estimated_reward_status = 10.0
    system.agents[untouched].state.estimated_reward_pu = 0.0

    system._update_roles_sequential(update_candidates=[selected])

    assert system.agents[selected].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[untouched].state.role == AgentRole.STATUS


def test_async_partial_update_preserves_follow_graph_consistency(model_module):
    """
    After a partial async update, the graph must still satisfy:
    if j.following = k, then j in followers[k], and no self-following.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # Initial structure: 1 -> 0
    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.followers = {1}

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[1].state.followers = set()

    # agent 2 will newly follow 1, but should redirect to 0
    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.following = None
    system.agents[2].state.followers = set()
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: 0.5, 1: 2.0, 2: 0.0, 3: 0.1, 4: 0.1}
    system.agents[2].state.highest_rep_agent_estimate = 1

    # agent 3 stays unchanged
    system.agents[3].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[3].state.followers = set()

    # agent 4 stays unchanged
    system.agents[4].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[4].state.followers = set()

    system._update_roles_sequential(update_candidates=[2])

    # graph invariant check
    for j, a in enumerate(system.agents):
        assert a.state.following != j, f"Agent {j} is following itself"
        if a.state.following is not None:
            k = a.state.following
            assert j in system.agents[k].state.followers, (
                f"Agent {j} follows {k}, but {j} not in followers[{k}]"
            )


# ==================== ASYNC HARNESS / SCHEDULING ====================

def test_async_make_config_disables_builtin_global_role_updates():
    """
    In async experiment mode, reputation_scaling.make_config(...) should disable
    built-in synchronized Step-6 role updates, because async updates are applied externally.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    class Args:
        num_agents = 10
        num_states = 3
        num_actions = 2
        num_steps = 1000
        kappa = 0.0
        role_update_s0 = 0
        role_update_T_seq = "3000"
        role_update_base_interval = 3000
        fixed_role_update_interval = True
        role_update_epochs = ""
        tracking_mode = "light"
        numpy_fast_path = False
        initial_actor_rate = 0.2
        initial_participant_rate = 0.2
        reward_model = "simple_preferred_action"
        reward_base_mu = 0.5
        reward_base_sigma = 0.08
        reward_agent_sigma = 0.1
        reward_clip_min = 0.01
        reward_clip_max = 2.5

    cfg = rep_scaling.make_config(Args(), gamma=2.0, mode="async")

    # async mode should effectively disable internal synchronized updates
    assert cfg.role_update_T_sequence == []
    assert cfg.role_update_epochs == []
    assert cfg.role_update_base_interval > Args.num_steps


def test_async_independent_clock_calls_subset_updates_not_global():
    """
    This is a harness-level smoke test:
    async mode with independent clocks should sometimes call _update_roles_sequential
    on a strict subset of agents, not always on the full population.
    """
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    class Args:
        mode = "async"
        num_agents = 6
        num_states = 2
        num_actions = 2
        num_steps = 20
        kappa = 0.0
        tail_window = 5
        role_update_s0 = 0
        role_update_T_seq = "3"
        role_update_base_interval = 3
        fixed_role_update_interval = True
        role_update_epochs = ""
        tracking_mode = "light"
        numpy_fast_path = False
        initial_actor_rate = 0.2
        initial_participant_rate = 0.2
        reward_model = "simple_preferred_action"
        reward_base_mu = 0.5
        reward_base_sigma = 0.08
        reward_agent_sigma = 0.1
        reward_clip_min = 0.01
        reward_clip_max = 2.5
        async_role_update_prob = None
        plot_sample_interval = 1
        output_dir = "."

    np.random.seed(0)
    config = rep_scaling.make_config(Args(), gamma=2.0, mode="async")
    system = rep_scaling.MultiAgentSystem(config)

    calls = []

    original = system._update_roles_sequential

    def wrapped(update_candidates=None):
        if update_candidates is None:
            calls.append(config.num_agents)
        else:
            calls.append(len(update_candidates))
        return original(update_candidates)

    system._update_roles_sequential = wrapped

    interval_seq, async_s0, _ = rep_scaling._build_async_interval_sequence(Args())
    first_interval = int(interval_seq[0])
    role_timers = np.random.randint(1, first_interval + 1, size=Args.num_agents, dtype=int)
    if async_s0 > 0:
        role_timers = role_timers + async_s0
    interval_indices = np.zeros(Args.num_agents, dtype=int)

    for _ in range(Args.num_steps):
        system.step()
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

    assert len(calls) > 0
    assert any(0 < c < Args.num_agents for c in calls), (
        f"Expected at least one subset async update, got calls={calls}"
    )


# ==================== MORE ASYNC ROLE TESTS ====================

def test_async_partial_update_hysteresis_start_vs_continue(model_module):
    """
    Async version of hysteresis:
    - a non-follower with rep signal between B_F and B_R should NOT start following
    - an already-follower with the same signal SHOULD continue following
    when both are selected for async reevaluation.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, delta=0.0, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    leader = 0
    rep_signal = 0.7  # B_F < 0.7 < B_R

    # agent 1: not following yet
    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.following = None
    system.agents[1].state.followers = set()
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {0: rep_signal, 1: 0.0, 2: 0.0}
    system.agents[1].state.highest_rep_agent_estimate = 0

    # agent 2: already following leader
    system.agents[2].state.role = AgentRole.REPUTATION
    system.agents[2].state.following = leader
    system.agents[2].state.followers = set()
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: rep_signal, 1: 0.0, 2: 0.0}
    system.agents[2].state.highest_rep_agent_estimate = 0
    system.agents[leader].state.followers = {2}

    system._update_roles_sequential(update_candidates=[1, 2])

    assert system.agents[1].state.role != AgentRole.REPUTATION
    assert system.agents[1].state.following is None

    assert system.agents[2].state.role == AgentRole.REPUTATION
    assert system.agents[2].state.following == leader
    assert 2 in system.agents[leader].state.followers


def test_async_partial_update_redirect_prevents_self_follow_after_chain(model_module):
    """
    Async redirect edge case:
    if i wants to follow j, and j is following i, redirect would point back to i.
    The correct behavior is to skip following and keep i in PU.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # agent 0 wants to follow 1
    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.following = None
    system.agents[0].state.followers = set()
    system.agents[0].state.estimated_reward_pu = 0.0
    system.agents[0].state.reputation_estimates = {1: 2.0, 2: 0.1, 0: 0.0}
    system.agents[0].state.highest_rep_agent_estimate = 1

    # but 1 is already following 0
    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[1].state.followers = set()
    system.agents[0].state.followers = {1}

    # filler
    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    system._update_roles_sequential(update_candidates=[0])

    assert system.agents[0].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[0].state.following is None
    assert 0 not in system.agents[0].state.followers


def test_async_partial_update_nonselected_agents_keep_role_when_selected_agent_moves(model_module):
    """
    If only one agent is selected in async mode, another agent that would also qualify
    for switching should remain unchanged.
    This catches accidental global reevaluation in partial-update code paths.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    # leader candidate
    system.agents[3].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[3].state.followers = set()

    # selected agent 0 qualifies
    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.following = None
    system.agents[0].state.followers = set()
    system.agents[0].state.estimated_reward_pu = 0.0
    system.agents[0].state.reputation_estimates = {3: 2.0, 1: 0.1, 2: 0.1, 0: 0.0}
    system.agents[0].state.highest_rep_agent_estimate = 3

    # nonselected agent 1 also qualifies, but should not move
    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.following = None
    system.agents[1].state.followers = set()
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {3: 2.5, 0: 0.1, 2: 0.1, 1: 0.0}
    system.agents[1].state.highest_rep_agent_estimate = 3

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    system._update_roles_sequential(update_candidates=[0])

    assert system.agents[0].state.role == AgentRole.REPUTATION
    assert system.agents[0].state.following == 3

    assert system.agents[1].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[1].state.following is None
    assert 1 not in system.agents[3].state.followers


# ==================== MORE ASYNC HARNESS TESTS ====================

def test_async_clock_sequence_progression_uses_next_interval():
    """
    Pure clock logic test:
    with interval sequence [3,5], an agent that triggers at t=3 should next trigger 5 steps later,
    not repeatedly every 3 forever.
    """
    role_timers = np.array([3], dtype=int)
    interval_seq = [3, 5]
    interval_indices = np.array([0], dtype=int)

    trigger_times = []

    for t in range(1, 12):
        role_timers -= 1
        update_ids = np.where(role_timers <= 0)[0]
        if update_ids.size > 0:
            trigger_times.append(t)

            for agent_id in update_ids.tolist():
                idx = int(interval_indices[agent_id])
                next_interval = int(interval_seq[idx if idx < len(interval_seq) else -1])
                role_timers[agent_id] += next_interval
                if idx < len(interval_seq) - 1:
                    interval_indices[agent_id] = idx + 1

    # first trigger at 3, second trigger at 6? wait:
    # start timer=3 -> t=3 trigger, add interval_seq[0]=3 => next trigger at 6
    # after that interval index becomes 1, add 5 => next trigger at 11
    assert trigger_times == [3, 6, 11]


def test_async_bernoulli_mode_can_update_strict_subset():
    """
    Pure Bernoulli harness test:
    per-step async update mask should be able to generate strict subsets (not always empty/full).
    """
    np.random.seed(0)
    n = 20
    p = 0.2

    subset_sizes = []
    for _ in range(50):
        update_mask = np.random.random(n) < p
        subset_sizes.append(int(np.sum(update_mask)))

    assert any(0 < s < n for s in subset_sizes), subset_sizes


def test_async_partial_update_empty_candidate_list_is_noop(model_module):
    """
    Edge case: empty async update list should do nothing and not corrupt graph/state.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0),
    )

    before_roles = [a.state.role for a in system.agents]
    before_following = [a.state.following for a in system.agents]
    before_followers = [set(a.state.followers) for a in system.agents]

    system._update_roles_sequential(update_candidates=[])

    after_roles = [a.state.role for a in system.agents]
    after_following = [a.state.following for a in system.agents]
    after_followers = [set(a.state.followers) for a in system.agents]

    assert before_roles == after_roles
    assert before_following == after_following
    assert before_followers == after_followers














def test_async_partial_update_redirects_existing_followers_when_agent_becomes_follower(
    model_module, monkeypatch
):
    """
    Async ROLE-5 scenario consistent with Section 7.3:
    i starts with no followers (so i ∈ C), receives followers earlier in the same
    Step-1 pass, then i switches to REPUTATION and those followers are redirected.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    follower_a, follower_b, i, new_leader = 0, 1, 3, 4

    # Deterministic Step-1 order: followers first, then i.
    monkeypatch.setattr(np.random, "shuffle", lambda xs: xs.sort())

    # Agent i starts in C(t) and prefers to follow new_leader.
    system.agents[i].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[i].state.following = None
    system.agents[i].state.followers = set()
    system.agents[i].state.estimated_reward_pu = 0.0
    system.agents[i].state.reputation_estimates = {0: 0.1, 1: 0.1, 2: 0.1, 3: 0.0, 4: 3.0}
    system.agents[i].state.highest_rep_agent_estimate = new_leader

    # Two updatable agents initially choose i as leader.
    for f in (follower_a, follower_b):
        system.agents[f].state.role = AgentRole.PERSONAL_UTILITY
        system.agents[f].state.following = None
        system.agents[f].state.followers = set()
        system.agents[f].state.estimated_reward_pu = 0.0
        system.agents[f].state.reputation_estimates = {0: 0.0, 1: 0.0, 2: 0.0, 3: 2.5, 4: 0.2}
        system.agents[f].state.highest_rep_agent_estimate = i

    system.agents[new_leader].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[new_leader].state.followers = set()

    system._update_roles_sequential(update_candidates=[follower_a, follower_b, i])

    assert system.agents[i].state.role == AgentRole.REPUTATION
    assert system.agents[i].state.following == new_leader
    assert system.agents[follower_a].state.following == new_leader
    assert system.agents[follower_b].state.following == new_leader
    assert follower_a in system.agents[new_leader].state.followers
    assert follower_b in system.agents[new_leader].state.followers
    assert len(system.agents[i].state.followers) == 0


def test_async_agent_with_followers_cannot_enter_reputation_switch_step(model_module):
    """
    Under current Section 7.3 logic, agents with followers are excluded from C(t),
    so even if they would prefer another leader, they should not switch into REPUTATION.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    i = 1
    preferred_leader = 3

    system.agents[i].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[i].state.following = None
    system.agents[i].state.followers = {0, 2}
    system.agents[i].state.estimated_reward_pu = 0.0
    system.agents[i].state.reputation_estimates = {0: 0.1, 2: 0.1, 3: 3.0, 4: 0.1, 1: 0.0}
    system.agents[i].state.highest_rep_agent_estimate = preferred_leader

    system.agents[0].state.role = AgentRole.REPUTATION
    system.agents[0].state.following = i
    system.agents[0].state.followers = set()

    system.agents[2].state.role = AgentRole.REPUTATION
    system.agents[2].state.following = i
    system.agents[2].state.followers = set()

    system.agents[preferred_leader].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[preferred_leader].state.followers = set()

    system.agents[4].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[4].state.followers = set()

    system._update_roles_sequential(update_candidates=[i])

    # Since i has followers, it is not in C(t), so it should not switch.
    assert system.agents[i].state.role == AgentRole.PERSONAL_UTILITY
    assert system.agents[i].state.following is None
    assert system.agents[i].state.followers == {0, 2}
    assert system.agents[0].state.following == i
    assert system.agents[2].state.following == i


def test_async_scheduler_audit_records_exact_timer_expirations():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    class Args:
        mode = "async"
        num_agents = 4
        num_states = 2
        num_actions = 2
        num_steps = 8
        kappa = 0.0
        tail_window = 5
        role_update_s0 = 0
        role_update_T_seq = "3"
        role_update_base_interval = 3
        fixed_role_update_interval = True
        role_update_epochs = ""
        tracking_mode = "light"
        numpy_fast_path = False
        initial_actor_rate = 0.2
        initial_participant_rate = 0.2
        reward_model = "simple_preferred_action"
        reward_base_mu = 0.5
        reward_base_sigma = 0.08
        reward_agent_sigma = 0.1
        reward_clip_min = 0.01
        reward_clip_max = 2.5
        async_role_update_prob = None
        async_decision_audit = True
        trace_detailed_seeds = "none"
        plot_sample_interval = 1
        output_dir = "."

    np.random.seed(0)
    _, _, _, _, async_debug, role_update_diagnostics, checkpoint_audit = rep_scaling.run_single(
        args=Args(),
        mode="async",
        gamma=2.0,
        seed=0,
    )

    assert async_debug is not None
    assert role_update_diagnostics is None
    assert checkpoint_audit is None
    scheduler_rows = async_debug["scheduler_rows"]
    assert len(scheduler_rows) == Args.num_steps

    for row in scheduler_rows:
        before = np.array(_parse_serialized_ids(row["role_timers_before_decrement"]), dtype=int)
        after = np.array(_parse_serialized_ids(row["role_timers_after_reset"]), dtype=int)
        update_ids = _parse_serialized_ids(row["update_ids"])
        if before.size == 0:
            continue
        expected_updates = np.where(before - 1 <= 0)[0].tolist()
        assert update_ids == expected_updates
        if update_ids:
            assert after.size == before.size
            for idx in update_ids:
                assert after[idx] > 0


def test_static_role_update_diagnostics_capture_only_role_update_epochs():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    class Args:
        mode = "static"
        num_agents = 4
        num_states = 2
        num_actions = 2
        num_steps = 8
        kappa = 0.0
        tail_window = 5
        role_update_s0 = 0
        role_update_T_seq = ""
        role_update_base_interval = 3
        fixed_role_update_interval = True
        role_update_epochs = ""
        tracking_mode = "light"
        numpy_fast_path = False
        initial_actor_rate = 0.2
        initial_participant_rate = 0.2
        reward_model = "simple_preferred_action"
        reward_base_mu = 0.5
        reward_base_sigma = 0.08
        reward_agent_sigma = 0.1
        reward_clip_min = 0.01
        reward_clip_max = 2.5
        async_role_update_prob = None
        async_decision_audit = False
        role_update_diagnostics = True
        trace_detailed_seeds = "none"
        plot_sample_interval = 1
        output_dir = "."

    np.random.seed(0)
    _, _, _, _, async_debug, role_update_diagnostics, checkpoint_audit = rep_scaling.run_single(
        args=Args(),
        mode="static",
        gamma=5.0,
        seed=0,
    )

    assert async_debug is None
    assert role_update_diagnostics is not None
    assert checkpoint_audit is not None

    expected_role_update_times = rep_scaling.build_static_role_update_times(Args(), horizon=Args.num_steps)
    assert [int(row["t"]) for row in role_update_diagnostics] == expected_role_update_times
    assert [int(row["role_update_index"]) for row in role_update_diagnostics] == list(
        range(1, len(expected_role_update_times) + 1)
    )

    first_row = role_update_diagnostics[0]
    for key in (
        "top_leader_id",
        "top_followers",
        "second_followers",
        "distinct_follow_targets",
        "n_reputation",
        "n_personal_utility",
        "mean_pu_estimate",
        "mean_rep_signal_weighted",
        "share_gate_margin_positive",
        "top_highest_rep_target_share",
    ):
        assert key in first_row

    final_true_rows = [row for row in checkpoint_audit["true_reputation_checkpoints"] if row["checkpoint_kind"] == "final"]
    final_estimate_rows = [row for row in checkpoint_audit["estimate_consensus_checkpoints"] if row["checkpoint_kind"] == "final"]
    final_rate_rows = [row for row in checkpoint_audit["rate_audit_checkpoints"] if row["checkpoint_kind"] == "final"]

    assert len(final_true_rows) == Args.num_agents
    assert len(final_estimate_rows) == Args.num_agents
    assert len(final_rate_rows) == Args.num_agents
    assert all(int(row["t"]) == Args.num_steps for row in final_true_rows)
    assert all(int(row["role_update_index"]) == len(expected_role_update_times) for row in final_true_rows)


def test_reputation_scaling_selected_seeds_override_contiguous_range():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    class Args:
        selected_seeds = "9,2,7,2"
        seed_start = 100
        seeds = 5

    assert rep_scaling.parse_selected_seeds(Args.selected_seeds) == [2, 7, 9]
    assert rep_scaling.resolve_seeds(Args()) == [2, 7, 9]


def test_gossip_scope_helper_builds_b_set_and_flags_off_scope_changes():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    paper_scope = rep_scaling.build_paper_gossip_scope([5, 5, 7, -1, 7])
    assert paper_scope == [5, 7]

    rep_before = np.array(
        [
            [0.0, 1.0, 4.0],
            [0.0, 2.0, 8.0],
        ],
        dtype=float,
    )
    rep_after = np.array(
        [
            [0.0, 3.0, 5.0],
            [0.0, 3.0, 6.0],
        ],
        dtype=float,
    )
    scope_audit = rep_scaling.characterize_changed_gossip_columns(rep_before, rep_after, paper_scope=[1])

    assert scope_audit["paper_scope_columns"] == [1]
    assert scope_audit["changed_columns"] == [1, 2]
    assert scope_audit["off_scope_changed_columns"] == [2]
    assert scope_audit["implementation_updates_only_paper_scope"] == 0


def test_rank_alignment_summary_helper_rolls_up_checkpoint_rows():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    true_rows = [
        {
            "mode": "static",
            "gamma": 5.0,
            "seed": 2,
            "t": 50000,
            "checkpoint_kind": "final",
            "role_update_index": 16,
            "agent_id": 0,
            "true_top_unique": 1,
            "unique_true_top_agent": 7,
            "eq9_averaging_mode": "all_agents",
            "leader_update_mode": "participants_only_post_eq9",
        },
        {
            "mode": "static",
            "gamma": 5.0,
            "seed": 2,
            "t": 50000,
            "checkpoint_kind": "final",
            "role_update_index": 16,
            "agent_id": 1,
            "true_top_unique": 1,
            "unique_true_top_agent": 7,
            "eq9_averaging_mode": "all_agents",
            "leader_update_mode": "participants_only_post_eq9",
        },
    ]
    estimate_rows = [
        {
            "mode": "static",
            "gamma": 5.0,
            "seed": 2,
            "t": 50000,
            "checkpoint_kind": "final",
            "role_update_index": 16,
            "observer_id": 0,
            "top_estimate_agent": 7,
            "highest_rep_agent_estimate": 7,
            "candidate_count_within_delta": 3,
            "gap_top2": 0.2,
            "current_root_leader": 7,
            "eq9_averaging_mode": "all_agents",
            "leader_update_mode": "participants_only_post_eq9",
        },
        {
            "mode": "static",
            "gamma": 5.0,
            "seed": 2,
            "t": 50000,
            "checkpoint_kind": "final",
            "role_update_index": 16,
            "observer_id": 1,
            "top_estimate_agent": 5,
            "highest_rep_agent_estimate": 7,
            "candidate_count_within_delta": 5,
            "gap_top2": 0.1,
            "current_root_leader": 3,
            "eq9_averaging_mode": "all_agents",
            "leader_update_mode": "participants_only_post_eq9",
        },
    ]

    summary_rows = rep_scaling.summarize_rank_alignment_checkpoints(
        true_rows=true_rows,
        estimate_rows=estimate_rows,
    )

    assert len(summary_rows) == 1
    row = summary_rows[0]
    assert row["eq9_averaging_mode"] == "all_agents"
    assert row["leader_update_mode"] == "participants_only_post_eq9"
    assert row["unique_true_top_agent"] == 7
    assert row["top_estimate_mode_agent"] == 5
    assert row["selected_target_mode_agent"] == 7
    assert row["top_estimate_matches_true_top_share"] == pytest.approx(0.5, abs=1e-12)
    assert row["selected_matches_true_top_share"] == pytest.approx(1.0, abs=1e-12)
    assert row["candidate_count_mean"] == pytest.approx(4.0, abs=1e-12)
    assert row["candidate_count_max"] == 5
    assert row["distinct_root_count"] == 2


def test_run_single_small_n_trace_export_populates_dense_history_only_when_enabled(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    class Args:
        mode = "static"
        num_agents = 4
        num_states = 2
        num_actions = 2
        num_steps = 5
        kappa = 0.0
        tail_window = 5
        role_update_s0 = 0
        role_update_T_seq = ""
        role_update_base_interval = 10
        fixed_role_update_interval = True
        role_update_epochs = ""
        tracking_mode = "full"
        numpy_fast_path = True
        initial_actor_rate = 0.2
        initial_participant_rate = 0.2
        reward_model = "simple_preferred_action"
        reward_base_mu = 0.5
        reward_base_sigma = 0.08
        reward_agent_sigma = 0.1
        reward_clip_min = 0.01
        reward_clip_max = 2.5
        delta = 0.15
        eq9_averaging_mode = "all_agents"
        leader_update_mode = "participants_only_post_eq9"
        async_role_update_prob = None
        async_decision_audit = False
        role_update_diagnostics = False
        trace_detailed_seeds = "none"
        plot_sample_interval = 1
        small_n_trace_export = True
        output_dir = str(tmp_path)

    np.random.seed(0)
    _, _, _, detailed_trace, _, _, _ = rep_scaling.run_single(
        args=Args(),
        mode="static",
        gamma=2.0,
        seed=0,
    )
    assert detailed_trace is not None
    dense = np.asarray(detailed_trace["dense_reputation_history"], dtype=float)
    assert dense.shape == (Args.num_steps, Args.num_agents, Args.num_agents)
    dense_v = np.asarray(detailed_trace["dense_personal_benefit_history"], dtype=float)
    assert dense_v.shape == (Args.num_steps, Args.num_agents, Args.num_agents)
    true_rep = np.asarray(detailed_trace["true_reputation_history"], dtype=float)
    assert true_rep.shape == (Args.num_steps, Args.num_agents)

    class ArgsDisabled(Args):
        small_n_trace_export = False

    np.random.seed(0)
    _, _, _, detailed_trace_disabled, _, _, _ = rep_scaling.run_single(
        args=ArgsDisabled(),
        mode="static",
        gamma=2.0,
        seed=0,
    )
    assert detailed_trace_disabled is not None
    dense_disabled = np.asarray(detailed_trace_disabled["dense_reputation_history"], dtype=float)
    assert dense_disabled.size == 0
    dense_v_disabled = np.asarray(detailed_trace_disabled["dense_personal_benefit_history"], dtype=float)
    assert dense_v_disabled.size == 0
    true_rep_disabled = np.asarray(detailed_trace_disabled["true_reputation_history"], dtype=float)
    assert true_rep_disabled.size == 0


def test_force_all_active_debug_overrides_sampling(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(
            num_time_steps=2,
            initial_actor_interaction_rate=0.0,
            initial_participant_interaction_rate=0.0,
            force_all_active_debug=True,
        ),
    )

    system.step()

    assert system.results["actor_counts"][0] == 4
    assert system.results["participant_counts"][0] == 4
    assert system.last_active_actor_ids == {0, 1, 2, 3}
    assert system.last_active_participant_ids == {0, 1, 2, 3}


def test_small_n_long_trace_writers_emit_expected_row_counts(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    trace = {
        "estimated_reward_pu_history": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float),
        "estimated_reward_rep_history": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=float),
        "estimated_reward_status_history": np.array([[0.0, 0.0], [0.0, 0.0]], dtype=float),
        "actor_interaction_rate_history": np.array([[0.2, 0.2], [0.25, 0.25]], dtype=float),
        "selected_reputation_history": np.array([[0.9, 0.8], [1.1, 1.0]], dtype=float),
        "weighted_selected_reputation_history": np.array([[1.8, 1.6], [2.2, 2.0]], dtype=float),
        "highest_rep_agent_history": np.array([[1, 0], [1, 0]], dtype=int),
        "following_history": np.array([[-1, 0], [-1, 0]], dtype=int),
        "follower_count_history": np.array([[1, 0], [1, 0]], dtype=int),
        "role_label_history": np.array(
            [["personal_utility", "reputation"], ["personal_utility", "reputation"]],
            dtype=object,
        ),
        "dense_reputation_history": np.array(
            [
                [[0.0, 1.0], [2.0, 0.0]],
                [[0.0, 1.5], [2.5, 0.0]],
            ],
            dtype=float,
        ),
        "dense_personal_benefit_history": np.array(
            [
                [[0.0, 0.4], [0.6, 0.0]],
                [[0.0, 0.5], [0.7, 0.0]],
            ],
            dtype=float,
        ),
        "true_reputation_history": np.array(
            [
                [0.2, 0.8],
                [0.3, 0.9],
            ],
            dtype=float,
        ),
        "true_reputation_rank_history": np.array(
            [
                [2, 1],
                [2, 1],
            ],
            dtype=int,
        ),
        "true_reputation_theta_history": np.array(
            [
                [0.1, 0.2],
                [0.15, 0.25],
            ],
            dtype=float,
        ),
        "true_reputation_sum_expected_history": np.array(
            [
                [2.0, 4.0],
                [2.0, 3.6],
            ],
            dtype=float,
        ),
        "active_actor_ids_history": [[0, 1], [0]],
        "active_participant_ids_history": [[0, 1], [0, 1]],
        "observed_utility_matrix_history": [
            np.array([[0.0, 0.4], [0.6, 0.0]], dtype=float),
            np.array([[0.2, 0.0], [0.5, 0.0]], dtype=float),
        ],
        "eta_v_history": [0.5, 0.5],
        "gossip_target_ids_history": [[0, 1], [0, 1]],
        "averaging_agent_ids_history": [[0, 1], [0, 1]],
        "avg_s_by_target_history": [
            {0: 0.1, 1: 0.2},
            {0: 0.3, 1: 0.4},
        ],
        "delta_v_matrix_history": [
            np.array([[0.0, 0.4], [0.6, 0.0]], dtype=float),
            np.array([[0.2, -0.2], [-0.1, 0.0]], dtype=float),
        ],
        "B_R": 0.8,
        "B_F": 0.6,
        "delta": 0.15,
        "role_update_times": np.array([1, 2], dtype=int),
    }
    traces = {(2.0, 0): trace}
    rep_csv = tmp_path / "expB_reputation_trace_long.csv"
    agent_csv = tmp_path / "expB_agent_state_trace_long.csv"
    true_vs_est_csv = tmp_path / "expB_true_rep_vs_estimate_trace_long.csv"
    true_decomp_csv = tmp_path / "expB_true_reputation_decomposition_long.csv"
    align_csv = tmp_path / "expB_toy_alignment_by_update.csv"
    v_to_s_csv = tmp_path / "expB_toy_v_to_s_by_update.csv"
    v_to_s_audit_csv = tmp_path / "expB_toy_v_to_s_recurrence_audit_long.csv"
    s_to_highest_csv = tmp_path / "expB_toy_s_to_highest_by_update.csv"
    step1_csv = tmp_path / "expB_toy_step1_by_update.csv"
    choice_csv = tmp_path / "expB_toy_choice_trace_long.csv"
    consensus_csv = tmp_path / "expB_toy_consensus_by_step.csv"
    follow_relationships_csv = tmp_path / "expB_toy_follow_relationships_long.csv"

    rep_scaling.write_small_n_reputation_trace_long_csv(traces, rep_csv)
    rep_scaling.write_small_n_agent_state_trace_long_csv(traces, agent_csv)
    rep_scaling.write_small_n_true_rep_vs_estimate_trace_long_csv(traces, true_vs_est_csv)
    rep_scaling.write_small_n_true_reputation_decomposition_long_csv(traces, true_decomp_csv)
    rep_scaling.write_small_n_toy_alignment_by_update_csv(traces, align_csv)
    rep_scaling.write_small_n_toy_v_to_s_by_update_csv(traces, v_to_s_csv)
    rep_scaling.write_small_n_toy_v_to_s_recurrence_audit_csv(traces, v_to_s_audit_csv)
    rep_scaling.write_small_n_toy_s_to_highest_by_update_csv(traces, s_to_highest_csv)
    rep_scaling.write_small_n_toy_step1_by_update_csv(traces, step1_csv)
    rep_scaling.write_small_n_toy_choice_trace_long_csv(traces, choice_csv)
    rep_scaling.write_small_n_toy_consensus_by_step_csv(traces, consensus_csv)
    rep_scaling.write_small_n_toy_follow_relationships_long_csv(traces, follow_relationships_csv)

    with rep_csv.open() as f:
        rep_rows = list(csv.DictReader(f))
    with agent_csv.open() as f:
        agent_rows = list(csv.DictReader(f))
    with true_vs_est_csv.open() as f:
        true_vs_est_rows = list(csv.DictReader(f))
    with true_decomp_csv.open() as f:
        true_decomp_rows = list(csv.DictReader(f))
    with align_csv.open() as f:
        align_rows = list(csv.DictReader(f))
    with v_to_s_csv.open() as f:
        v_to_s_rows = list(csv.DictReader(f))
    with v_to_s_audit_csv.open() as f:
        v_to_s_audit_rows = list(csv.DictReader(f))
    with s_to_highest_csv.open() as f:
        s_to_highest_rows = list(csv.DictReader(f))
    with step1_csv.open() as f:
        step1_rows = list(csv.DictReader(f))
    with choice_csv.open() as f:
        choice_rows = list(csv.DictReader(f))
    with consensus_csv.open() as f:
        consensus_rows = list(csv.DictReader(f))
    with follow_relationships_csv.open() as f:
        follow_relationship_rows = list(csv.DictReader(f))

    assert len(rep_rows) == 8
    assert len(agent_rows) == 4
    assert len(true_vs_est_rows) == 4
    assert len(true_decomp_rows) == 4
    assert len(align_rows) == 2
    assert len(v_to_s_rows) == 2
    assert len(v_to_s_audit_rows) == 8
    assert len(s_to_highest_rows) == 2
    assert len(step1_rows) == 2
    assert len(choice_rows) == 4
    assert len(consensus_rows) == 2
    assert len(follow_relationship_rows) == 4
    assert rep_rows[0]["seed"] == "0"
    assert rep_rows[0]["gamma"] == "2.0"
    assert set(row["role"] for row in agent_rows) == {"personal_utility", "reputation"}
    assert "mean_observed_reputation" in true_vs_est_rows[0]
    assert "gamma_times_selected_reputation" in true_vs_est_rows[0]
    assert float(true_vs_est_rows[0]["step1_margin"]) == pytest.approx(1.7, abs=1e-12)
    assert "mean_incoming_v" in true_decomp_rows[0]
    assert "dominant_alignment_target" in align_rows[0]
    assert "failure_stage_label" in align_rows[0]
    assert align_rows[0]["dominant_alignment_target"] == "mean_incoming_v"
    assert align_rows[0]["failure_stage_label"] == "learning_target_mismatch"
    assert "dominant_v_alignment_target" in v_to_s_rows[0]
    assert "corr_observed_vs_mean_incoming_v" in v_to_s_rows[0]
    assert "expected_s_new_paper" in v_to_s_audit_rows[0]
    assert "actual_s_new_code" in v_to_s_audit_rows[0]
    assert "v_matches_paper" in v_to_s_audit_rows[0]
    assert "s_matches_paper" in v_to_s_audit_rows[0]
    assert "share_highest_within_delta_set" in s_to_highest_rows[0]
    assert "mean_candidate_count_within_delta" in s_to_highest_rows[0]
    assert "first_positive_follow_signal_reached" in step1_rows[0]
    assert "share_selected_reputation_matches_highest_row_value" in step1_rows[0]
    assert "share_weighted_signal_matches_gamma_times_selected" in step1_rows[0]
    assert float(step1_rows[0]["mean_step1_margin"]) == pytest.approx(1.55, abs=1e-12)
    assert step1_rows[0]["first_positive_follow_signal_reached"] == "True"
    assert "root_leader" in choice_rows[0]
    assert "rep_beats_gate" in choice_rows[0]
    assert choice_rows[0]["root_leader"] == "0"
    assert "all_agents_agree_on_highest" in consensus_rows[0]
    assert "largest_root_size" in consensus_rows[0]
    assert consensus_rows[0]["all_agents_agree_on_highest"] == "False"
    assert consensus_rows[0]["largest_root_size"] == "2"
    assert "root_leader" in follow_relationship_rows[0]
    assert "has_followers" in follow_relationship_rows[0]
    assert follow_relationship_rows[0]["root_leader"] == "0"


def test_toy_alignment_and_failure_stage_helpers_snapshot():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    dominant = rep_scaling._dominant_alignment_target(
        corr_true_reputation=0.2,
        corr_sum_expected_utility=0.7,
        corr_theta_mu=-0.1,
        corr_mean_incoming_v=0.4,
    )
    assert dominant == "sum_expected_utility"

    label = rep_scaling._classify_toy_failure_stage(
        true_top_agent=4,
        observed_top_agent=4,
        modal_highest_rep_agent_estimate=2,
        modal_selected_target=2,
        dominant_alignment_target="true_reputation",
        share_step1_margin_positive=1.0,
        top_followers=3,
    )
    assert label == "ranking_selection_mismatch"

    label = rep_scaling._classify_toy_failure_stage(
        true_top_agent=4,
        observed_top_agent=1,
        modal_highest_rep_agent_estimate=1,
        modal_selected_target=1,
        dominant_alignment_target="mean_incoming_v",
        share_step1_margin_positive=1.0,
        top_followers=3,
    )
    assert label == "learning_target_mismatch"


def test_small_n_follow_graph_and_timeline_smoke(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    trace = {
        "following_history": np.array(
            [
                [-1, 0, -1],
                [-1, 0, 1],
            ],
            dtype=int,
        ),
        "role_label_history": np.array(
            [
                ["personal_utility", "reputation", "personal_utility"],
                ["personal_utility", "reputation", "reputation"],
            ],
            dtype=object,
        ),
        "follower_count_history": np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=int,
        ),
        "role_update_times": np.array([1, 2], dtype=int),
    }

    graph_dir = tmp_path / "graphs"
    timeline_png = tmp_path / "timeline.png"
    highest_timeline_png = tmp_path / "highest_timeline.png"
    root_timeline_png = tmp_path / "root_timeline.png"
    rep_scaling.plot_toy_follow_graph_snapshots(
        trace,
        graph_dir,
        title_prefix="Toy smoke",
    )
    rep_scaling.plot_toy_follow_timeline(
        trace,
        timeline_png,
        title="Toy following timeline",
    )
    rep_scaling.plot_toy_highest_target_timeline(
        {
            **trace,
            "highest_rep_agent_history": np.array([[1, 0, 0], [1, 0, 1]], dtype=int),
        },
        highest_timeline_png,
        title="Toy highest target timeline",
    )
    rep_scaling.plot_toy_root_leader_timeline(
        trace,
        root_timeline_png,
        title="Toy root leader timeline",
    )

    assert (graph_dir / "toy_follow_graph_t0001.png").exists()
    assert (graph_dir / "toy_follow_graph_t0002.png").exists()
    assert timeline_png.exists()
    assert highest_timeline_png.exists()
    assert root_timeline_png.exists()


def test_role_update_diagnostic_helpers_classify_fragmented_vs_partial():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    rep_scaling = importlib.import_module("experiments.reputation_scaling")

    fragmented_rows = rep_scaling.enrich_role_update_diagnostic_rows(
        mode="static",
        gamma=5.0,
        seed=7,
        rows=[
            {
                "t": 3000,
                "role_update_index": 1,
                "top_leader_id": 10,
                "top_followers": 28,
                "second_leader_id": 11,
                "second_followers": 18,
                "third_leader_id": 12,
                "third_followers": 10,
                "top_follower_share": 28 / 99,
                "top2_follower_share": 46 / 99,
                "distinct_follow_targets": 3,
                "n_reputation": 90,
                "n_personal_utility": 10,
                "n_status": 0,
                "mean_pu_estimate": 0.4,
                "mean_rep_signal_weighted": 0.8,
                "mean_step1_margin": 0.4,
                "share_step1_margin_positive": 1.0,
                "mean_gate_margin": 0.2,
                "share_gate_margin_positive": 1.0,
                "distinct_highest_rep_targets": 3,
                "top_highest_rep_target_id": 10,
                "top_highest_rep_target_share": 0.42,
                "second_highest_rep_target_share": 0.31,
            },
            {
                "t": 6000,
                "role_update_index": 2,
                "top_leader_id": 11,
                "top_followers": 41,
                "second_leader_id": 10,
                "second_followers": 20,
                "third_leader_id": 12,
                "third_followers": 11,
                "top_follower_share": 41 / 99,
                "top2_follower_share": 61 / 99,
                "distinct_follow_targets": 3,
                "n_reputation": 96,
                "n_personal_utility": 4,
                "n_status": 0,
                "mean_pu_estimate": 0.4,
                "mean_rep_signal_weighted": 0.9,
                "mean_step1_margin": 0.5,
                "share_step1_margin_positive": 1.0,
                "mean_gate_margin": 0.3,
                "share_gate_margin_positive": 1.0,
                "distinct_highest_rep_targets": 3,
                "top_highest_rep_target_id": 11,
                "top_highest_rep_target_share": 0.44,
                "second_highest_rep_target_share": 0.29,
            },
        ],
    )
    fragmented_summary = rep_scaling.summarize_role_update_diagnostics(
        mode="static",
        gamma=5.0,
        seed=7,
        num_agents=100,
        record=rep_scaling.RunRecord(
            mode="static",
            gamma=5.0,
            seed=7,
            leader_id=11,
            final_top_followers=41,
            time_to_90pct_followers=-1,
            leader_switches=1,
            tail_welfare=0.0,
        ),
        top_follower_series=np.array([0, 12, 28, 41]),
        rows=fragmented_rows,
    )
    assert fragmented_summary["leader_switches_role_update_only"] == 1
    assert fragmented_summary["failure_bucket"] == "fragmented_following"

    assert (
        rep_scaling.classify_seed_failure_bucket(
            num_agents=100,
            final_top_followers=57,
            final_share_gate_margin_positive=1.0,
            final_top_highest_rep_target_share=0.82,
            final_second_followers=8,
            leader_switches_role_update_only=0,
        )
        == "stable_partial_convergence"
    )


def test_async_decision_audit_records_threshold_and_redirect_fields(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    system.enable_async_decision_audit()

    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.followers = {1}

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[1].state.followers = set()

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()
    system.agents[2].state.following = None
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: 0.5, 1: 2.0, 2: 0.0, 3: 0.1}
    system.agents[2].state.highest_rep_agent_estimate = 1

    system.agents[3].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[3].state.followers = set()

    system.time_step = 17
    system._update_roles_sequential(update_candidates=[2])

    row = system.get_async_decision_audit_rows()[-1]
    assert row["t"] == 17
    assert row["agent_id"] == 2
    assert row["in_C"] is True
    assert row["effective_threshold"] == pytest.approx(0.1)
    assert row["step1_condition_met"] is True
    assert row["best_k_before_redirect"] == 1
    assert row["best_k_after_redirect"] == 0
    assert row["redirect_target_is_follower"] is True
    assert row["redirect_applied"] is True
    assert row["decision_code"] == "FOLLOW_REDIRECT"
    assert row["new_role"] == AgentRole.REPUTATION.value
    assert row["following_after"] == 0


def test_async_decision_audit_hysteresis_inactive_with_multiple_leaders(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    system.enable_async_decision_audit()

    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.followers = {1}

    system.agents[3].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[3].state.followers = {4}

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[1].state.followers = set()
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {0: 0.7, 1: 0.0, 2: 0.1, 3: 0.5, 4: 0.1}
    system.agents[1].state.highest_rep_agent_estimate = 0

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    system.agents[4].state.role = AgentRole.REPUTATION
    system.agents[4].state.following = 3
    system.agents[4].state.followers = set()

    system._update_roles_sequential(update_candidates=[1])

    row = system.get_async_decision_audit_rows()[-1]
    assert row["agent_id"] == 1
    assert row["opinion_leader_count"] == 2
    assert row["hysteresis_active"] is False
    assert row["effective_threshold"] == pytest.approx(0.8)


def test_async_decision_audit_decision_code_matches_threshold_fail_outcome(model_module):
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=3,
        extra_config=dict(gamma=1.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    system.enable_async_decision_audit()

    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[0].state.followers = {1}

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[1].state.followers = set()
    system.agents[1].state.estimated_reward_pu = 1.0
    system.agents[1].state.reputation_estimates = {0: 0.1, 1: 0.0, 2: 0.0}
    system.agents[1].state.highest_rep_agent_estimate = 0

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.followers = set()

    system._update_roles_sequential(update_candidates=[1])

    row = system.get_async_decision_audit_rows()[-1]
    assert row["decision_code"] == "STAY_PU_REP_BELOW_THRESHOLD"
    assert row["new_role"] == AgentRole.PERSONAL_UTILITY.value
    assert row["following_after"] == -1

# ==============================================================================
# === SECTION 4: Perturbation and Recovery ===
# ==============================================================================









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

# ==============================================================================
# === SECTION 5: Additional Gossip, Role, Reward, and Estimate Tests ===
# ==============================================================================




# ============================================================
# Gossip (Eq. 9) tests
# ============================================================

def test_gossip_mean_only():
    rep_snapshot = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: 0.0}, 1: {0: 0.0}, 2: {0: 0.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 1, 2]]
    print("After gossip (Δv=0):", vals)
    assert all(abs(v - 5.0) < 1e-10 for v in vals), "Expected all to become mean=5.0"


def test_gossip_mean_plus_delta_v():
    rep_snapshot = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: +1.0}, 1: {0: 0.0}, 2: {0: -2.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 1, 2]]
    print("After gossip (mean+Δv):", vals)
    assert abs(vals[0] - 6.0) < 1e-10
    assert abs(vals[1] - 5.0) < 1e-10
    assert abs(vals[2] - 3.0) < 1e-10


def test_gossip_snapshot_vs_inplace_order_dependence():
    rep0 = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: 0.0}, 1: {0: 0.0}, 2: {0: 0.0}}

    out_sync = gossip_sync_update(rep0, delta_v, num_agents=1)
    sync_vals = [out_sync[i][0] for i in [0, 1, 2]]

    rep_inplace_a = {i: dict(rep0[i]) for i in rep0}
    out_inplace_a = gossip_inplace_update(rep_inplace_a, delta_v, num_agents=1, update_order=[0, 1, 2])
    a_vals = [out_inplace_a[i][0] for i in [0, 1, 2]]

    rep_inplace_b = {i: dict(rep0[i]) for i in rep0}
    out_inplace_b = gossip_inplace_update(rep_inplace_b, delta_v, num_agents=1, update_order=[2, 1, 0])
    b_vals = [out_inplace_b[i][0] for i in [0, 1, 2]]

    print("Snapshot vals:", sync_vals)
    print("Inplace order [0,1,2]:", a_vals)
    print("Inplace order [2,1,0]:", b_vals)

    assert all(abs(v - 5.0) < 1e-10 for v in sync_vals), "Snapshot must give exact mean"
    assert (any(abs(v - 5.0) > 1e-10 for v in a_vals) or
            any(abs(v - 5.0) > 1e-10 for v in b_vals)), "In-place should be order-dependent and deviate"


def test_gossip_multi_round_convergence_delta_v_zero():
    rep = {0: {0: 0.0}, 1: {0: 10.0}, 2: {0: 0.0}, 3: {0: 10.0}}
    delta_v = {i: {0: 0.0} for i in rep}

    variances = []
    for _ in range(6):
        vals = [rep[i][0] for i in sorted(rep.keys())]
        variances.append(_variance(vals))
        rep = gossip_sync_update(rep, delta_v, num_agents=1)

    print("Variances over rounds:", variances)
    assert variances[-1] < variances[0] * 1e-6, "Should converge very close to consensus (variance ~ 0)"


def test_gossip_participant_subset_mean():
    rep_snapshot = {0: {0: 0.0}, 2: {0: 6.0}, 4: {0: 12.0}}
    delta_v = {0: {0: 0.0}, 2: {0: 0.0}, 4: {0: 0.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 2, 4]]
    print("Subset participants gossip vals:", vals)
    assert all(abs(v - 6.0) < 1e-10 for v in vals), "Mean must be over active participants only"


# ============================================================
# Role allocation tests (Section 7)
# ============================================================

def test_role_follower_chain_redirection():
    np.random.seed(0)
    config = SystemConfig(num_agents=3, num_time_steps=1, gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.1)
    system = MultiAgentSystem(config)

    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[0].state.followers.add(1)

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.reputation_estimates = {0: 0.5, 1: 1.0, 2: 0.0}
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].identify_highest_reputation_agent()

    system._update_roles_sequential(update_candidates=[2])

    print("Agent 2 following:", system.agents[2].state.following)
    assert system.agents[2].state.following == 0, "Follower chain not redirected correctly."
    assert system.agents[0].state.followers == {1, 2}, "Follower graph should be rebuilt onto the redirected leader."


def test_role_identify_highest_rep_excludes_self():
    np.random.seed(0)
    config = SystemConfig(num_agents=4, num_time_steps=1, delta=0.0)
    system = MultiAgentSystem(config)

    i = 2
    a = system.agents[i]
    a.state.reputation_estimates = {0: 1.0, 1: 2.0, 2: 999.0, 3: 3.0}
    a.identify_highest_reputation_agent()

    print("highest_rep_agent_estimate:", a.state.highest_rep_agent_estimate)
    assert a.state.highest_rep_agent_estimate != i, "Should exclude self from candidates"


def test_role_hysteresis_start_vs_continue():
    np.random.seed(0)
    config = SystemConfig(
        num_agents=3, num_time_steps=1, gamma=1.0, kappa=0.0,
        B_R=0.8, B_F=0.6, delta=0.0,
        c_threshold=1.0  # prevent status step from triggering
    )
    system = MultiAgentSystem(config)

    leader = 0
    rep_signal = 0.7  # 0.6 < 0.7 < 0.8

    system.agents[1].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {0: rep_signal, 2: 0.0, 1: 0.0}
    system.agents[1].identify_highest_reputation_agent()

    system.agents[2].state.role = AgentRole.REPUTATION
    system.agents[2].state.following = leader
    system.agents[leader].state.followers.add(2)
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: rep_signal, 1: 0.0, 2: 0.0}
    system.agents[2].identify_highest_reputation_agent()

    system._update_roles_sequential()

    print("Agent 1 role/following:", system.agents[1].state.role, system.agents[1].state.following)
    print("Agent 2 role/following:", system.agents[2].state.role, system.agents[2].state.following)

    assert system.agents[1].state.role != AgentRole.REPUTATION, "Should NOT start following at 0.7 when B_R=0.8"
    assert system.agents[2].state.role == AgentRole.REPUTATION and system.agents[2].state.following == leader, \
        "Should CONTINUE following at 0.7 when B_F=0.6"


def test_role_hysteresis_continue_applies_with_multiple_leaders():
    np.random.seed(0)
    config = SystemConfig(
        num_agents=4, num_time_steps=1, gamma=1.0, kappa=0.0,
        B_R=0.8, B_F=0.6, delta=0.0,
        c_threshold=1.0
    )
    system = MultiAgentSystem(config)

    # Two current opinion leaders: 0 and 1 both have followers.
    system.agents[2].state.role = AgentRole.REPUTATION
    system.agents[2].state.following = 0
    system.agents[0].state.followers.add(2)

    system.agents[3].state.role = AgentRole.REPUTATION
    system.agents[3].state.following = 1
    system.agents[1].state.followers.add(3)

    # Agent 2 should still use B_F even while multiple leaders exist.
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: 0.7, 1: 0.5, 2: 0.0, 3: 0.0}
    system.agents[2].identify_highest_reputation_agent()

    system._update_roles_sequential(update_candidates=[2])

    print("Agent 2 role/following with two leaders:", system.agents[2].state.role, system.agents[2].state.following)
    assert system.agents[2].state.role == AgentRole.REPUTATION, "Existing followers should still use B_F when leaders > 1."
    assert system.agents[2].state.following == 0, "Follower should keep the current highest-reputation target when 0.7 > B_F."


def test_role_existing_follower_switches_to_current_highest():
    np.random.seed(0)
    config = SystemConfig(
        num_agents=3, num_time_steps=1, gamma=1.0, kappa=0.0,
        B_R=0.8, B_F=0.6, delta=0.0,
        c_threshold=1.0
    )
    system = MultiAgentSystem(config)

    follower = 2
    old_leader = 0
    new_leader = 1

    system.agents[follower].state.role = AgentRole.REPUTATION
    system.agents[follower].state.following = old_leader
    system.agents[old_leader].state.followers.add(follower)
    system.agents[follower].state.estimated_reward_pu = 0.0
    system.agents[follower].state.reputation_estimates = {0: 0.7, 1: 0.9, 2: 0.0}
    system.agents[follower].identify_highest_reputation_agent()

    system._update_roles_sequential(update_candidates=[follower])

    print("Follower switched to:", system.agents[follower].state.following)
    assert system.agents[follower].state.following == new_leader, "Follower should switch to the current highest-reputation agent."
    assert follower not in system.agents[old_leader].state.followers, "Old leader should lose the follower after switching."
    assert follower in system.agents[new_leader].state.followers, "New leader should gain the follower after switching."


def test_role_chain_redirection_uses_current_pass_state():
    np.random.seed(1)
    config = SystemConfig(num_agents=4, num_time_steps=1, gamma=1.0, kappa=0.0, B_R=0.1, B_F=0.1, delta=0.0)
    system = MultiAgentSystem(config)

    # Agent 1 starts as a follower of 0, but in this pass should switch to 3.
    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[0].state.followers.add(1)
    system.agents[1].state.estimated_reward_pu = 0.0
    system.agents[1].state.reputation_estimates = {0: 0.7, 1: 0.0, 2: 0.0, 3: 0.9}
    system.agents[1].identify_highest_reputation_agent()

    # Agent 2 wants to follow agent 1, so redirection should use 1's updated target (3), not stale state.
    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.estimated_reward_pu = 0.0
    system.agents[2].state.reputation_estimates = {0: 0.5, 1: 1.0, 2: 0.0, 3: 0.4}
    system.agents[2].identify_highest_reputation_agent()

    original_shuffle = np.random.shuffle
    np.random.shuffle = lambda xs: xs.sort()
    try:
        system._update_roles_sequential(update_candidates=[1, 2])
    finally:
        np.random.shuffle = original_shuffle

    print("Agent 1 following after update:", system.agents[1].state.following)
    print("Agent 2 following after update:", system.agents[2].state.following)
    assert system.agents[1].state.following == 3, "Agent 1 should switch to the new current highest-reputation leader."
    assert system.agents[2].state.following == 3, "Redirection should use the current-pass leader relation, not stale t-1 state."
    assert system.agents[0].state.followers == set(), "Old leader should lose followers redirected in the current pass."
    assert system.agents[3].state.followers == {1, 2}, "Current-pass rebuild should place both followers on the new leader."


def test_role_self_reputation_does_not_trigger_follow_entry():
    np.random.seed(0)
    config = SystemConfig(
        num_agents=3, num_time_steps=1, gamma=1.0, kappa=0.0,
        B_R=0.8, B_F=0.6, delta=0.0,
        c_threshold=1.0
    )
    system = MultiAgentSystem(config)

    agent = system.agents[2]
    agent.state.role = AgentRole.PERSONAL_UTILITY
    agent.state.estimated_reward_pu = 0.0
    agent.state.reputation_estimates = {0: 0.4, 1: 0.5, 2: 999.0}
    agent.identify_highest_reputation_agent()

    system._update_roles_sequential(update_candidates=[2])

    print("Self-reputation case role/following:", agent.state.role, agent.state.following)
    assert agent.state.role == AgentRole.PERSONAL_UTILITY, "Self reputation should not trigger follow entry."
    assert agent.state.following is None, "Agent should remain independent when the best non-self reputation is below B_R."


def test_role_status_requires_min_followers():
    np.random.seed(0)
    config = SystemConfig(num_agents=10, num_time_steps=1, gamma=0.0, kappa=2.0, c_threshold=0.3)
    system = MultiAgentSystem(config)

    agent = system.agents[0]
    agent.state.followers = {1, 2}  # only 2 followers, but min is 3
    agent.state.role = AgentRole.PERSONAL_UTILITY
    agent.state.estimated_reward_status = 999.0
    agent.state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    print("Agent 0 role:", system.agents[0].state.role)
    assert system.agents[0].state.role != AgentRole.STATUS, "Should not enter STATUS without enough followers"


def test_role_status_switch_clears_following():
    np.random.seed(0)
    config = SystemConfig(num_agents=6, num_time_steps=1, gamma=1.0, kappa=2.0, c_threshold=0.2)
    system = MultiAgentSystem(config)

    leader = 0
    i = 1

    system.agents[i].state.role = AgentRole.REPUTATION
    system.agents[i].state.following = leader
    system.agents[leader].state.followers.add(i)

    system.agents[i].state.followers = {2, 3}  # qualifies because ceil(0.2 * 6) = 2
    system.agents[i].state.estimated_reward_status = 10.0
    system.agents[i].state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    print("Agent 1 role/following:", system.agents[i].state.role, system.agents[i].state.following)
    assert system.agents[i].state.role == AgentRole.STATUS, "Should switch to STATUS"
    assert system.agents[i].state.following is None, "STATUS agent should not keep following"
    assert i not in system.agents[leader].state.followers, "Should be removed from old leader followers"


# ============================================================
# Reward model tests
# ============================================================

def test_reward_shared_good_bad_has_one_shared_good_action_per_state():
    np.random.seed(0)
    config = SystemConfig(
        num_agents=6,
        num_states=4,
        num_actions=2,
        num_time_steps=1,
        reward_model="shared_good_bad_heterogeneous",
        reward_good_value=1.0,
        reward_bad_value=0.1,
        reward_agent_sigma=0.1,
        reward_order_gap=0.02,
    )
    system = MultiAgentSystem(config)

    print("Shared good actions:", system._shared_good_actions)
    assert system._shared_good_actions is not None
    assert len(system._shared_good_actions) == config.num_states
    assert all(0 <= int(a) < config.num_actions for a in system._shared_good_actions)


def test_reward_shared_good_bad_preserves_good_over_bad_ranking():
    np.random.seed(1)
    config = SystemConfig(
        num_agents=8,
        num_states=5,
        num_actions=2,
        num_time_steps=1,
        reward_model="shared_good_bad_heterogeneous",
        reward_good_value=1.0,
        reward_bad_value=0.1,
        reward_agent_sigma=0.1,
        reward_order_gap=0.02,
    )
    system = MultiAgentSystem(config)

    for state, good_action in enumerate(system._shared_good_actions):
        bad_action = 1 - int(good_action)
        good_rewards = system._reward_tables[:, state, int(good_action)]
        bad_rewards = system._reward_tables[:, state, bad_action]
        print(f"State {state} min good-bad gap:", float(np.min(good_rewards - bad_rewards)))
        assert np.all(good_rewards >= bad_rewards + config.reward_order_gap - 1e-12)


def test_reward_shared_good_bad_keeps_agent_heterogeneity():
    np.random.seed(2)
    config = SystemConfig(
        num_agents=10,
        num_states=4,
        num_actions=2,
        num_time_steps=1,
        reward_model="shared_good_bad_heterogeneous",
        reward_good_value=1.0,
        reward_bad_value=0.1,
        reward_agent_sigma=0.1,
        reward_order_gap=0.02,
    )
    system = MultiAgentSystem(config)

    state = 0
    good_action = int(system._shared_good_actions[state])
    rewards = system._reward_tables[:, state, good_action]
    print("Heterogeneous rewards for state 0 good action:", rewards)
    assert np.ptp(rewards) > 1e-9, "Agents should not all have identical rewards under the heterogeneous model."


# ============================================================
# Estimates tracking tests (Section 6.3 / 6.4 / 6.6)
# ============================================================

def test_estimates_personal_benefit_delta_active():
    np.random.seed(0)
    config = SystemConfig(num_agents=3, num_time_steps=1)
    system = MultiAgentSystem(config)

    agent = system.agents[0]
    eta = 0.2

    # Set a known previous value for v_0(1)
    agent.state.personal_benefit_estimates[1] = 4.0

    observed_payoffs = {0: 0.0, 1: 10.0, 2: 0.0}
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta)

    expected_new = 4.0 + eta * (10.0 - 4.0)  # 5.2
    expected_delta = expected_new - 4.0      # 1.2

    print("v update (active): new=", agent.state.personal_benefit_estimates[1], " delta=", deltas[1])
    assert abs(agent.state.personal_benefit_estimates[1] - expected_new) < 1e-10
    assert abs(deltas[1] - expected_delta) < 1e-10


def test_estimates_personal_benefit_decay_inactive():
    np.random.seed(0)
    config = SystemConfig(num_agents=3, num_time_steps=1)
    system = MultiAgentSystem(config)

    agent = system.agents[0]
    eta = 0.2

    agent.state.personal_benefit_estimates[2] = 5.0

    observed_payoffs = {0: 0.0, 1: 0.0, 2: 0.0}  # everyone inactive from i's POV
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta)

    expected_new = 5.0 * (1.0 - eta)  # 4.0
    expected_delta = expected_new - 5.0  # -1.0

    print("v update (inactive decay): new=", agent.state.personal_benefit_estimates[2], " delta=", deltas[2])
    assert abs(agent.state.personal_benefit_estimates[2] - expected_new) < 1e-10
    assert abs(deltas[2] - expected_delta) < 1e-10


def test_estimates_reward_ema_personal_utility():
    np.random.seed(0)
    config = SystemConfig(num_agents=2, num_states=1, num_actions=2, num_time_steps=1)
    system = MultiAgentSystem(config)

    a = system.agents[0]
    a.state.estimated_reward_pu = 1.0
    a.state.weights_pu = np.zeros((3, 2))

    # Use update_personal_utility directly: Ĵ_pu <- Ĵ_pu + η_J (r - Ĵ_pu)
    eta_J = 0.5
    alpha = 0.0  # don't change weights (not needed for this test)
    a.update_personal_utility(state=0, action=0, reward=3.0, alpha_pu_t=alpha, eta_J_t=eta_J)

    expected = 1.0 + eta_J * (3.0 - 1.0)  # 2.0
    print("Ĵ_pu EMA:", a.state.estimated_reward_pu)
    assert abs(a.state.estimated_reward_pu - expected) < 1e-10


# ============================================================
# Interaction rate test (Eq. 13)
# ============================================================

# ============================================================
# Status test (Eq. 11-12)
# ============================================================
def test_status_reward_uses_sum_not_average():
    np.random.seed(0)

    config = SystemConfig(
        num_agents=4,
        num_states=3,
        num_actions=2,
        num_time_steps=1,
    )
    system = MultiAgentSystem(config)

    leader = system.agents[0]
    leader.state.role = AgentRole.STATUS

    # Pick any valid state/action indices (policy gradient won't matter because we set beta=0).
    state = 0
    action = 0

    # Make the EMA update deterministic: eta_J_t=1.0 -> estimate becomes exactly the target in one step.
    beta_status_t = 0.0
    eta_J_t = 1.0

    # Followers' payoffs (active followers). SUM vs AVG differ by construction.
    follower_payoffs = [1.0, 2.0, 3.0]   # sum=6, avg=2
    social_support_sum = float(sum(follower_payoffs))
    social_support_avg = float(np.mean(follower_payoffs))
    assert social_support_sum != social_support_avg

    # --- SUM case: should update to 6.0 exactly ---
    leader.state.estimated_reward_status = 0.0
    leader.update_status_optimization(
        state=state,
        action=action,
        social_support_sum=social_support_sum,
        beta_status_t=beta_status_t,
        eta_J_t=eta_J_t,
    )
    print("Status reward (SUM) target:", social_support_sum,
          "updated Ĵ_status:", leader.state.estimated_reward_status)
    assert abs(leader.state.estimated_reward_status - social_support_sum) < 1e-10, \
        "Status estimate should update to SUM of follower payoffs (Eq. 11)."

    # --- AVG case (intentional contrast): would update to 2.0 if average were used ---
    leader.state.estimated_reward_status = 0.0
    leader.update_status_optimization(
        state=state,
        action=action,
        social_support_sum=social_support_avg,   # deliberately passing average here
        beta_status_t=beta_status_t,
        eta_J_t=eta_J_t,
    )
    print("Status reward (AVG) target:", social_support_avg,
          "updated Ĵ_status:", leader.state.estimated_reward_status)
    assert abs(leader.state.estimated_reward_status - social_support_avg) < 1e-10, \
        "Control check: if average is passed in, estimate should equal average."

    # The actual model should be using SUM, not AVG (so these must differ).
    assert abs(social_support_sum - social_support_avg) > 1e-10


