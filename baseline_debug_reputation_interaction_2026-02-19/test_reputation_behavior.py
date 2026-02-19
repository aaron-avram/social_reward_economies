"""Reputation-focused regression tests for doc/code_old.py baseline.

These tests are derived from Sections 6.4 and 7.3 in
learning_paper_newest_ver_transcription.md and inspired by the style in
single-reward-economy/reputation_tests/.
"""

from __future__ import annotations

import importlib.util
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


def make_system(baseline_module, *, num_agents=3, extra_config=None):
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
    system = baseline_module.MultiAgentSystem(config)
    return system


def test_identify_highest_reputation_excludes_self(baseline_module):
    """Section 6.4.4 says candidate set is C\\{i}; self should never be selected."""
    system = make_system(baseline_module, num_agents=3)
    agent = system.agents[0]

    # Self is the unique maximum estimate.
    agent.state.reputation_estimates = {0: 10.0, 1: 2.0, 2: 1.0}
    agent.config.delta = 0.0

    np.random.seed(0)
    agent.identify_highest_reputation_agent()

    assert (
        agent.state.highest_rep_agent_estimate != 0
    ), "highest_rep_agent_estimate should exclude self (agent 0)"


def test_reputation_update_matches_eq9_additive_delta_structure(baseline_module):
    """Section 6.4.3 Eq. (9): s_i(k,t+1) = avg_j s_j(k,t) + v_i(k,t+1)-v_i(k,t)."""
    system = make_system(baseline_module, num_agents=2)
    agent_i = system.agents[0]
    agent_j = system.agents[1]

    k = 1
    agent_i.state.personal_benefit_estimates[k] = 0.2
    agent_i.state.reputation_estimates[k] = 0.0
    agent_j.state.reputation_estimates[k] = 0.4

    observed_payoffs = {k: 1.0}
    eta_v_t = 0.5
    eta_s_t = 1.0

    v_old = agent_i.state.personal_benefit_estimates[k]
    agent_i.update_personal_benefit_estimates(observed_payoffs, eta_v_t)
    v_new = agent_i.state.personal_benefit_estimates[k]

    # Provide one external estimate source to make avg_j term unambiguous.
    agent_i.update_reputation_estimates_gossip(observed_payoffs, [agent_j], eta_s_t)

    expected_avg = agent_j.state.reputation_estimates[k]
    expected = expected_avg + (v_new - v_old)
    actual = agent_i.state.reputation_estimates[k]

    assert actual == pytest.approx(
        expected, abs=1e-9
    ), f"Eq.(9) expected {expected:.6f}, got {actual:.6f}"


def test_step1_reputation_switch_uses_role_criterion_without_extra_rep_gate(baseline_module):
    """Section 7.3 criterion should depend on gamma*J_r and max(B_i, J_pu) only."""
    system = make_system(
        baseline_module,
        num_agents=2,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.8, B_F=0.6, c_threshold=1.0),
    )
    AgentRole = baseline_module.AgentRole

    follower = system.agents[0]
    leader = system.agents[1]

    follower.state.role = AgentRole.PERSONAL_UTILITY
    follower.state.following = None
    follower.state.followers = set()
    follower.state.estimated_reward_pu = 0.0
    follower.state.estimated_reward_rep = 1.0  # gamma*J_r = 2.0
    follower.state.highest_rep_agent_estimate = 1
    follower.state.reputation_estimates = {1: 0.1}  # < B_R, but this should not block switch per Section 7.3

    leader.state.role = AgentRole.PERSONAL_UTILITY
    leader.state.following = None
    leader.state.followers = set()

    np.random.seed(3)
    system._update_roles_sequential()

    assert follower.state.role == AgentRole.REPUTATION
    assert follower.state.following == 1


def test_personal_benefit_decay_for_inactive_agent_matches_section_642(baseline_module):
    """Sanity check: inactive actors should decay v_i(k,t) by (1-eta_v)."""
    system = make_system(baseline_module, num_agents=2)
    agent = system.agents[0]

    agent.state.personal_benefit_estimates[1] = 0.5
    agent.update_personal_benefit_estimates({1: 0.0}, eta_v_t=0.2)

    assert agent.state.personal_benefit_estimates[1] == pytest.approx(0.4, abs=1e-12)
