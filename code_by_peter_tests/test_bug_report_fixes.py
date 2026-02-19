"""Regression tests for bug-report fixes in doc/code_by_peter.py.

Covers all findings listed in baseline_debug_reputation_interaction_2026-02-19/BUG_REPORT.md.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "doc" / "code_by_peter.py"


@pytest.fixture(scope="module")
def model_module():
    spec = importlib.util.spec_from_file_location("code_by_peter_module", TARGET_PATH)
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


def _estimate_activation_frequency(system, *, steps: int, seed: int):
    np.random.seed(seed)
    for _ in range(steps):
        system.step()

    actor_empirical = float(np.mean(system.results["actor_counts"]))
    participant_empirical = float(np.mean(system.results["participant_counts"]))
    return actor_empirical, participant_empirical


def test_actor_activation_uses_theta_mu(model_module):
    system = make_system(model_module, num_agents=1, extra_config=dict(u_0=0.0))
    agent = system.agents[0]
    agent.state.actor_interaction_rate = 0.8
    agent.state.participant_interaction_rate = 0.0
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    actor_empirical, _ = _estimate_activation_frequency(system, steps=15000, seed=11)
    expected = 1.0 - np.exp(-0.8)
    assert abs(actor_empirical - expected) < 0.03


def test_participant_activation_uses_theta_mu(model_module):
    system = make_system(model_module, num_agents=1, extra_config=dict(u_0=0.0))
    agent = system.agents[0]
    agent.state.actor_interaction_rate = 0.0
    agent.state.participant_interaction_rate = 0.8
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    _, participant_empirical = _estimate_activation_frequency(system, steps=15000, seed=22)
    expected = 1.0 - np.exp(-0.8)
    assert abs(participant_empirical - expected) < 0.03


def test_highest_reputation_selection_excludes_self(model_module):
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]

    agent.state.reputation_estimates = {0: 10.0, 1: 2.0, 2: 1.0}
    agent.config.delta = 0.0

    np.random.seed(0)
    agent.identify_highest_reputation_agent()

    assert agent.state.highest_rep_agent_estimate != 0


def test_reputation_update_matches_eq9_additive_structure(model_module):
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
    deltas = agent_i.update_personal_benefit_estimates(observed_payoffs, eta_v_t)
    v_new = agent_i.state.personal_benefit_estimates[k]

    agent_i.update_reputation_estimates_gossip(deltas, [agent_j], eta_s_t=1.0)

    expected = agent_j.state.reputation_estimates[k] + (v_new - v_old)
    assert agent_i.state.reputation_estimates[k] == pytest.approx(expected, abs=1e-9)


def test_step1_reputation_switch_does_not_use_extra_max_rep_gate(model_module):
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
    # max_rep below B_R, but gamma*max_rep > B_R should still pass Section 7.3 condition.
    follower.state.reputation_estimates = {1: 0.5}

    leader.state.role = AgentRole.PERSONAL_UTILITY
    leader.state.following = None
    leader.state.followers = {0}  # Keep C={0} to isolate update-order effects.

    np.random.seed(3)
    system._update_roles_sequential()

    assert follower.state.role == AgentRole.REPUTATION
    assert follower.state.following == 1


def test_bug1_bootstrap_non_follower_from_reputation_signal(model_module):
    system = make_system(
        model_module,
        num_agents=2,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    candidate = system.agents[0]
    leader = system.agents[1]

    candidate.state.followers = set()
    leader.state.followers = {0}  # Keep C={0}

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


def test_bug5_redirects_if_best_agent_is_already_follower(model_module):
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


def test_bug4_no_extra_phase5_pairwise_gossip(model_module):
    system = make_system(
        model_module,
        num_agents=2,
        extra_config=dict(gossip_rate=1.0, gossip_alpha=1.0, u_0=0.0),
    )
    a0, a1 = system.agents

    # Force no actors and near-certain participant activity under theta(mu).
    for a in (a0, a1):
        a.state.actor_interaction_rate = 0.0
        a.state.participant_interaction_rate = 100.0

    a0.state.reputation_estimates = {0: 0.0, 1: 0.0}
    a1.state.reputation_estimates = {0: 10.0, 1: 10.0}

    def _noop_personal_benefit(self, observed_payoffs, eta_v_t):
        return {k: 0.0 for k in observed_payoffs}

    def _noop_gossip(self, personal_benefit_deltas, other_agents_list, eta_s_t):
        return None

    # Disable Phase-4 estimate changes so any mutation would indicate extra gossip pass.
    a0.update_personal_benefit_estimates = types.MethodType(_noop_personal_benefit, a0)
    a1.update_personal_benefit_estimates = types.MethodType(_noop_personal_benefit, a1)
    a0.update_reputation_estimates_gossip = types.MethodType(_noop_gossip, a0)
    a1.update_reputation_estimates_gossip = types.MethodType(_noop_gossip, a1)

    np.random.seed(0)
    system.step()

    assert a0.state.reputation_estimates[0] == pytest.approx(0.0, abs=1e-12)
    assert a1.state.reputation_estimates[0] == pytest.approx(10.0, abs=1e-12)
