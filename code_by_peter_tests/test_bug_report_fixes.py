"""Regression tests for bug fixes in src/code_debugged.py.

Organized by canonical bug IDs:
- IR-1  : activation sampling must use theta(mu)=1-exp(-mu)
- REP-1 : highest-reputation selection excludes self
- REP-2 : reputation update matches Eq. (9): avg + delta_v
- REP-3 : no extra pairwise gossip pass in same timestep
- ROLE-1: reputation-role entry can bootstrap from observed signal
- ROLE-2: no extra max_rep >= B_i gate
- ROLE-3: indirect-follow target redirects to leader
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TARGET_PATH = ROOT / "src" / "code_debugged.py"


@pytest.fixture(scope="module")
def model_module():
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


def _estimate_activation_frequency(system, *, steps: int, seed: int):
    np.random.seed(seed)
    for _ in range(steps):
        system.step()

    actor_empirical = float(np.mean(system.results["actor_counts"]))
    participant_empirical = float(np.mean(system.results["participant_counts"]))
    return actor_empirical, participant_empirical


# ==================== IR-1 ====================

def test_ir1_actor_activation_uses_theta_mu(model_module):
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


def test_ir1_participant_activation_uses_theta_mu(model_module):
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


# ==================== REP-1 ====================

def test_rep1_highest_reputation_selection_excludes_self(model_module):
    system = make_system(model_module, num_agents=3)
    agent = system.agents[0]

    agent.state.reputation_estimates = {0: 10.0, 1: 2.0, 2: 1.0}
    agent.config.delta = 0.0

    np.random.seed(0)
    agent.identify_highest_reputation_agent()

    assert agent.state.highest_rep_agent_estimate != 0


# ==================== REP-2 ====================

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
    deltas = agent_i.update_personal_benefit_estimates(observed_payoffs, eta_v_t)
    v_new = agent_i.state.personal_benefit_estimates[k]

    agent_i.update_reputation_estimates_gossip(deltas, [agent_j], eta_s_t=1.0)

    expected = agent_j.state.reputation_estimates[k] + (v_new - v_old)
    assert agent_i.state.reputation_estimates[k] == pytest.approx(expected, abs=1e-9)


# ==================== REP-3 ====================

def test_rep3_no_extra_phase5_pairwise_gossip_strict_noop_phase4(model_module):
    """Primary REP-3 check under current code path: only one reputation-averaging pass."""
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

    # Force distinct per-agent deltas so a second gossip pass would be visible.
    def _deltas_a0(self, observed_payoffs, eta_v_t):
        return {0: 1.0, 1: 0.0}

    def _deltas_a1(self, observed_payoffs, eta_v_t):
        return {0: -1.0, 1: 0.0}

    a0.update_personal_benefit_estimates = types.MethodType(_deltas_a0, a0)
    a1.update_personal_benefit_estimates = types.MethodType(_deltas_a1, a1)

    np.random.seed(0)
    system.step()

    # Single pass expectation:
    # avg_s[0] = (0 + 10)/2 = 5 -> a0: 5 + 1 = 6, a1: 5 - 1 = 4
    # With an extra pairwise gossip pass (alpha=1), both would collapse to 5.
    assert a0.state.reputation_estimates[0] == pytest.approx(6.0, abs=1e-12)
    assert a1.state.reputation_estimates[0] == pytest.approx(4.0, abs=1e-12)


def test_rep3_no_extra_phase5_pairwise_gossip_legacy_scenario(model_module):
    """Secondary REP-3 check with partial gossip alpha; still should reflect single-pass values."""
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

    # Without a second pass: 6 and 4.
    # With an extra pass at alpha=0.5: 5.5 and 4.5.
    assert a0.state.reputation_estimates[0] == pytest.approx(6.0, abs=1e-12)
    assert a1.state.reputation_estimates[0] == pytest.approx(4.0, abs=1e-12)


# ==================== ROLE-1 ====================

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


def test_role1_bootstrap_non_follower_legacy_scenario(model_module):
    """Legacy scenario from old bug-comment test, now under canonical ROLE-1 grouping."""
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


# ==================== ROLE-2 ====================

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
    # max_rep below B_R, but gamma*max_rep > B_R should still pass Section 7.3 condition.
    follower.state.reputation_estimates = {1: 0.5}

    leader.state.role = AgentRole.PERSONAL_UTILITY
    leader.state.following = None
    leader.state.followers = {0}  # Keep C={0} to isolate update-order effects.

    np.random.seed(3)
    system._update_roles_sequential()

    assert follower.state.role == AgentRole.REPUTATION
    assert follower.state.following == 1


# ==================== ROLE-3 ====================

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
    """Legacy scenario from old bug-comment test, now under canonical ROLE-3 grouping."""
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


# ───────────────────────────────────────────────────────────────────────────
# ROLE-4: Prevent self-following after redirect
# ───────────────────────────────────────────────────────────────────────────

def test_role_4_no_self_following_after_redirect(model_module):
    """
    [ROLE-4] Ensure agent cannot follow itself after redirect chain.

    Scenario: Agent 0 selects Agent 1 as highest reputation.
              Agent 1 is a follower, following Agent 0.
              Redirect would set Agent 0 to follow itself.
              Fix: Agent 0 should stay in PERSONAL_UTILITY instead.
    """
    config = model_module.SystemConfig(
        num_agents=3,
        num_states=2,
        num_actions=2,
        gamma=2.0,
        kappa=0.0,
        B_R=0.1,
        B_F=0.05
    )
    system = model_module.MultiAgentSystem(config)

    # Setup: Agent 0 selects Agent 1 (highest rep)
    # Agent 1 is following Agent 0 (would create self-loop via redirect)
    agent_0 = system.agents[0]
    agent_1 = system.agents[1]

    # Agent 0 starts in PU role, sees Agent 1 as highest reputation
    agent_0.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_0.state.following = None
    agent_0.state.estimated_reward_pu = 0.1
    agent_0.state.estimated_reward_rep = 1.0  # High enough for role switch
    agent_0.state.reputation_estimates = {1: 2.0, 2: 0.5}
    agent_0.state.highest_rep_agent_estimate = 1
    agent_0.state.followers = set()

    # Agent 1 is already following Agent 0
    agent_1.state.role = model_module.AgentRole.REPUTATION
    agent_1.state.following = 0
    agent_1.state.followers = set()

    # Simulate role update
    system._update_roles_sequential()

    # ROLE-4 fix: Agent 0 should NOT follow itself
    # After redirect: best_k = 1's leader = 0 (self), so skip following
    assert agent_0.state.following != 0, "Agent 0 should not follow itself"
    assert agent_0.state.role == model_module.AgentRole.PERSONAL_UTILITY, "Agent 0 should stay in PU role"
    assert 0 not in agent_0.state.followers, "Agent 0 should not be in its own followers"
    assert len(agent_0.state.followers) < config.num_agents, "Follower count should be < num_agents"


def test_role_5_redirect_followers_when_leader_becomes_follower(model_module):
    """
    [ROLE-5] When an agent becomes a follower, redirect its existing followers.

    Scenario: Agent 1 has NO followers initially, decides to follow Agent 3.
              During the same update, Agent 0 (processed earlier) decides to follow Agent 1.
              Without ROLE-5 fix: Agent 0 follows Agent 1, Agent 1 follows Agent 3 (multi-level).
              With ROLE-5 fix: Agent 0 gets redirected to Agent 3 when Agent 1 becomes follower.
    """
    config = model_module.SystemConfig(
        num_agents=4,
        num_states=2,
        num_actions=2,
        gamma=2.0,
        kappa=0.0,
        B_R=0.1,
        B_F=0.05
    )
    system = model_module.MultiAgentSystem(config)

    agent_0 = system.agents[0]
    agent_1 = system.agents[1]
    agent_2 = system.agents[2]
    agent_3 = system.agents[3]

    # Setup: All agents start with no followers (qualify for STEP 1)
    # Agent 0 will select Agent 1 as highest rep
    # Agent 1 will select Agent 3 as highest rep
    # Processing order matters: if 0 processed before 1, we get multi-level chain

    agent_0.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_0.state.following = None
    agent_0.state.followers = set()
    agent_0.state.estimated_reward_pu = 0.05
    agent_0.state.reputation_estimates = {1: 3.0, 2: 1.0, 3: 1.0}
    agent_0.state.highest_rep_agent_estimate = 1  # Wants to follow Agent 1

    agent_1.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_1.state.following = None
    agent_1.state.followers = set()  # NO followers initially
    agent_1.state.estimated_reward_pu = 0.05
    agent_1.state.reputation_estimates = {0: 1.0, 2: 1.0, 3: 5.0}
    agent_1.state.highest_rep_agent_estimate = 3  # Wants to follow Agent 3

    agent_2.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_2.state.following = None
    agent_2.state.followers = set()

    agent_3.state.role = model_module.AgentRole.PERSONAL_UTILITY
    agent_3.state.following = None
    agent_3.state.followers = set()

    # Force processing order: Agent 0, then Agent 1 (simulates worst case)
    np.random.seed(42)  # Seed for deterministic shuffle

    # Run role update
    system._update_roles_sequential()

    # ROLE-5 fix: When Agent 1 becomes follower, Agent 0 should be redirected
    # Expected final state: Agent 0 → Agent 3, Agent 1 → Agent 3 (NO multi-level chain)
    assert agent_1.state.following == 3, "Agent 1 should follow Agent 3"
    assert len(agent_1.state.followers) == 0, "Agent 1 should have no followers"

    # Agent 0 should follow Agent 3 (redirected from Agent 1)
    assert agent_0.state.following == 3, "Agent 0 should be redirected to Agent 3 (not Agent 1)"
    assert agent_0.state.following != 1, "Agent 0 should NOT follow Agent 1 (would create multi-level chain)"


# ==================== STATUS ENTRY ====================

def test_status_entry_can_occur_after_status_reward_learning(model_module):
    """
    Status-role entry regression:
    estimated_reward_status must be learnable before STATUS entry so Step-2 can fire.
    """
    system = make_system(
        model_module,
        num_agents=8,
        extra_config=dict(
            gamma=2.0,
            kappa=2.0,
            c_threshold=0.1,
            role_update_base_interval=1,
            u_0=0.0,
        ),
    )
    AgentRole = model_module.AgentRole

    np.random.seed(123)
    for _ in range(2000):
        system.step()

    min_followers = max(1, int(system.config.c_threshold * system.config.num_agents))
    max_follower_seen = max(max(step_counts) for step_counts in system.results["follower_counts"])
    max_status_seen = max(
        sum(1 for role in roles if role == AgentRole.STATUS)
        for roles in system.results["roles_history"]
    )

    # Ensure Step-2 eligibility is reached and STATUS becomes reachable in practice.
    assert max_follower_seen >= min_followers
    assert max_status_seen >= 1
