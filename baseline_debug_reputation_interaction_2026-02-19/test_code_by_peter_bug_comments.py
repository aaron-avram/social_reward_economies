"""Targeted checks for BUG 1, BUG 4, BUG 5 against doc/code_by_peter.py."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "doc" / "code_by_peter.py"


@pytest.fixture(scope="module")
def module_under_test():
    spec = importlib.util.spec_from_file_location("code_by_peter_module", TARGET_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_system(module_under_test, *, num_agents, extra_config=None):
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

    config = module_under_test.SystemConfig(**kwargs)
    return module_under_test.MultiAgentSystem(config)


def test_bug1_non_followers_do_not_switch_even_with_high_reputation_signal(module_under_test):
    system = make_system(
        module_under_test,
        num_agents=2,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = module_under_test.AgentRole

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

    # Should switch if BUG 1 is fixed.
    assert candidate.state.role == AgentRole.REPUTATION
    assert candidate.state.following == 1


def test_bug4_step_contains_extra_pairwise_gossip_after_participant_updates(module_under_test):
    system = make_system(
        module_under_test,
        num_agents=2,
        extra_config=dict(gossip_rate=1.0, gossip_alpha=1.0, u_0=0.0),
    )

    a0, a1 = system.agents

    for a in (a0, a1):
        a.state.actor_interaction_rate = 0.0
        a.state.participant_interaction_rate = 1.0

    a0.state.reputation_estimates = {0: 0.0, 1: 0.0}
    a1.state.reputation_estimates = {0: 10.0, 1: 10.0}

    def _noop(self, observed_payoffs, other_agents_list, eta_s_t):
        return None

    a0.update_reputation_estimates_gossip = types.MethodType(_noop, a0)
    a1.update_reputation_estimates_gossip = types.MethodType(_noop, a1)

    np.random.seed(0)
    system.step()

    # If BUG 4 is fixed, these should remain unchanged.
    assert a0.state.reputation_estimates[0] == pytest.approx(0.0, abs=1e-12)
    assert a1.state.reputation_estimates[0] == pytest.approx(10.0, abs=1e-12)


def test_bug5_indirect_following_redirect_not_applied(module_under_test):
    system = make_system(
        module_under_test,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = module_under_test.AgentRole

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

    # Should redirect to k_prime if BUG 5 is fixed.
    assert i.state.role == AgentRole.REPUTATION
    assert i.state.following == 2
