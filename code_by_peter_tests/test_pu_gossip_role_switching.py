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
