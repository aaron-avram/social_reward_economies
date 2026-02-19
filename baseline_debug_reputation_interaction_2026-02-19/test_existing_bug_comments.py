"""Targeted checks for BUG 1, BUG 4, BUG 5 comments in doc/code_old.py.

These tests are designed to verify whether the commented bug behaviors are
indeed present in the current baseline implementation.
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "doc" / "code_old.py"


@pytest.fixture(scope="module")
def baseline_module():
    spec = importlib.util.spec_from_file_location("baseline_code_old", BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_system(baseline_module, *, num_agents, extra_config=None):
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

    config = baseline_module.SystemConfig(**kwargs)
    return baseline_module.MultiAgentSystem(config)


def test_bug1_non_followers_do_not_switch_even_with_high_reputation_signal(baseline_module):
    """BUG 1 check: Step-1 uses estimated_reward_rep (stuck at 0 for non-followers)."""
    system = make_system(
        baseline_module,
        num_agents=2,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = baseline_module.AgentRole

    candidate = system.agents[0]
    leader = system.agents[1]

    # Keep C={0} to remove update-order effects.
    candidate.state.followers = set()
    leader.state.followers = {0}

    candidate.state.role = AgentRole.PERSONAL_UTILITY
    candidate.state.following = None
    candidate.state.estimated_reward_pu = 0.0
    candidate.state.estimated_reward_rep = 0.0

    # Reputation evidence is strong and above B_R.
    candidate.state.reputation_estimates = {1: 2.0}
    candidate.state.highest_rep_agent_estimate = 1

    np.random.seed(7)
    system._update_roles_sequential()

    # If reputation reward were derived from observed reputation in Step-1,
    # this would switch to REPUTATION. Current code stays PERSONAL_UTILITY.
    assert candidate.state.role == AgentRole.REPUTATION
    assert candidate.state.following == 1


def test_bug4_step_contains_extra_pairwise_gossip_after_participant_updates(
    baseline_module,
):
    """BUG 4 check: even with participant gossip update disabled, Phase 5 still mutates estimates."""
    system = make_system(
        baseline_module,
        num_agents=2,
        extra_config=dict(gossip_rate=1.0, gossip_alpha=1.0, u_0=0.0),
    )

    a0, a1 = system.agents

    # Force participant activity and no actors, to isolate gossip phases.
    for a in (a0, a1):
        a.state.actor_interaction_rate = 0.0
        a.state.participant_interaction_rate = 1.0

    # Distinct estimates so pairwise gossip creates a visible change.
    a0.state.reputation_estimates = {0: 0.0, 1: 0.0}
    a1.state.reputation_estimates = {0: 10.0, 1: 10.0}

    # Disable Phase-4 reputation-update method to isolate Phase 5 effect.
    def _noop(self, observed_payoffs, other_agents_list, eta_s_t):
        return None

    a0.update_reputation_estimates_gossip = types.MethodType(_noop, a0)
    a1.update_reputation_estimates_gossip = types.MethodType(_noop, a1)

    np.random.seed(0)
    system.step()

    # Without Phase 5, these would remain unchanged.
    assert a0.state.reputation_estimates[0] == pytest.approx(0.0, abs=1e-12)
    assert a1.state.reputation_estimates[0] == pytest.approx(10.0, abs=1e-12)


def test_bug5_indirect_following_redirect_not_applied(baseline_module):
    """BUG 5 check: when best_k is already a follower, follower should redirect to best_k's leader."""
    system = make_system(
        baseline_module,
        num_agents=4,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = baseline_module.AgentRole

    i = system.agents[0]  # candidate to become follower
    k_hat = system.agents[1]  # chosen best_k, but already follower of k_prime
    k_prime = system.agents[2]  # should be followed instead
    filler = system.agents[3]

    # Ensure only agent 0 is in C (agents without followers).
    i.state.followers = set()
    k_hat.state.followers = {3}
    k_prime.state.followers = {1}
    filler.state.followers = {2}

    # Relationship: k_hat follows k_prime.
    k_hat.state.role = AgentRole.REPUTATION
    k_hat.state.following = 2

    i.state.role = AgentRole.PERSONAL_UTILITY
    i.state.following = None
    i.state.estimated_reward_pu = 0.0
    i.state.estimated_reward_rep = 1.0  # gamma*J_r = 2.0 > B_R
    i.state.reputation_estimates = {1: 2.0, 2: 1.5, 3: 0.1}
    i.state.highest_rep_agent_estimate = 1

    np.random.seed(1)
    system._update_roles_sequential()

    assert i.state.role == AgentRole.REPUTATION
    assert i.state.following == 2
