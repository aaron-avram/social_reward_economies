"""Tests for reputation learning, status learning, and interaction-rate allocation."""

from __future__ import annotations

import numpy as np
import pytest

from _shared import (
    estimate_activation_frequency,
    load_model_module,
    make_system,
)


@pytest.fixture(scope="module")
def model_module():
    return load_model_module()


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
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta)

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
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta)

    expected_new = 5.0 * (1.0 - eta)
    expected_delta = expected_new - 5.0

    assert abs(agent.state.personal_benefit_estimates[2] - expected_new) < 1e-10
    assert abs(deltas[2] - expected_delta) < 1e-10


def test_rep642_inactive_decay_matches_formula(model_module):
    system = make_system(model_module, num_agents=2)
    agent = system.agents[0]

    agent.state.personal_benefit_estimates[1] = 0.5
    eta_v_t = 0.2
    agent.update_personal_benefit_estimates({1: 0.0}, eta_v_t=eta_v_t)

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
