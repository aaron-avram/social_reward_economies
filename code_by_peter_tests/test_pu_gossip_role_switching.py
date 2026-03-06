"""Tests for personal utility, gossip network mechanics, and role switching."""

from __future__ import annotations

import types

import numpy as np
import pytest

from _shared import (
    gossip_inplace_update,
    gossip_sync_update,
    load_model_module,
    make_system,
    variance,
)


@pytest.fixture(scope="module")
def model_module():
    return load_model_module()


# ==================== GOSSIP NETWORK ====================

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

    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=0.2)

    # active update: 4 + 0.2*(10-4) = 5.2
    assert agent.state.personal_benefit_estimates[0] == pytest.approx(5.2, abs=1e-12)
    assert deltas[0] == pytest.approx(1.2, abs=1e-12)

    # inactive decay: 5*(1-0.2) = 4.0
    assert agent.state.personal_benefit_estimates[1] == pytest.approx(4.0, abs=1e-12)
    assert deltas[1] == pytest.approx(-1.0, abs=1e-12)


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

    system.agents[i].state.followers = {2}
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

    # threshold met: min followers = max(1, int(0.2*6)) = 1
    agent.state.followers = {1}
    agent.state.role = AgentRole.PERSONAL_UTILITY

    # but kappa * J_status <= J_pu, so should NOT switch
    agent.state.estimated_reward_status = 1.0
    agent.state.estimated_reward_pu = 3.0

    system._update_roles_sequential()

    assert agent.state.role != AgentRole.STATUS


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










def test_fast_path_phase4_updates_reputation_and_highest_rep_agent(model_module):
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

    observed_payoff_vector = np.array([10.0, 0.0, 0.0], dtype=float)
    active_participant_ids = np.array([0, 1, 2], dtype=int)
    eta_v_t = 0.2

    system._phase4_updates_numpy_fast(
        observed_payoff_vector=observed_payoff_vector,
        active_participant_ids=active_participant_ids,
        eta_v_t=eta_v_t,
    )

    # v update: only target 0 is active, so all observers get delta_v(:,0)=+2.0
    assert np.allclose(system._v_matrix[:, 0], 2.0)

    # average old s for each target:
    # k=0: mean(0,0,0)=0
    # k=1: mean(1,5,9)=5
    # k=2: mean(2,8,4)=14/3
    #
    # then add delta_v. since only k=0 has delta +2, rows for active participants become:
    # [2, 5, 14/3]
    expected_row = np.array([2.0, 5.0, 14.0 / 3.0])
    for i in range(3):
        assert np.allclose(system._s_matrix[i], expected_row)

    # highest reputation agent should be target 1 for agent 0 (self excluded)
    # because among non-self entries for row [2,5,14/3]:
    # agent 0 excludes k=0, so compares 5 vs 14/3 -> highest is 1
    assert system.agents[0].state.highest_rep_agent_estimate == 1