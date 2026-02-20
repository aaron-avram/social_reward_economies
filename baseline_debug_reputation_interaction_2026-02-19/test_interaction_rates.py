"""Interaction-rate-focused tests for src/code_old.py baseline.

Covers Section 6.2 activation probabilities and Section 6.7 Eq. (13).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "src" / "code_old.py"


@pytest.fixture(scope="module")
def baseline_module():
    spec = importlib.util.spec_from_file_location("baseline_code_old", BASELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_sampling_system(baseline_module, *, actor_mu: float, participant_mu: float):
    config = baseline_module.SystemConfig(
        num_agents=1,
        num_states=3,
        num_actions=2,
        num_time_steps=1,
        role_update_base_interval=10**9,
        gossip_rate=0.0,
        u_0=0.0,
        M=1.0,
    )
    system = baseline_module.MultiAgentSystem(config)
    agent = system.agents[0]
    agent.state.actor_interaction_rate = actor_mu
    agent.state.participant_interaction_rate = participant_mu
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0
    return system


def _estimate_activation_frequency(system, *, steps: int, seed: int):
    np.random.seed(seed)
    for _ in range(steps):
        system.step()

    actor_empirical = float(np.mean(system.results["actor_counts"]))
    participant_empirical = float(np.mean(system.results["participant_counts"]))
    return actor_empirical, participant_empirical


def test_actor_activation_probability_uses_theta_of_mu(baseline_module):
    """Section 6.2: actor activation probability should be theta(mu)=1-exp(-mu)."""
    mu = 0.8
    system = _make_sampling_system(baseline_module, actor_mu=mu, participant_mu=0.0)
    actor_empirical, _ = _estimate_activation_frequency(system, steps=20000, seed=11)

    expected = 1.0 - np.exp(-mu)
    assert abs(actor_empirical - expected) < 0.03, (
        f"Actor activation frequency should be near theta({mu})={expected:.4f}, "
        f"got {actor_empirical:.4f}"
    )


def test_participant_activation_probability_uses_theta_of_mu(baseline_module):
    """Section 6.2: participant activation probability should be theta(mu)=1-exp(-mu)."""
    mu = 0.8
    system = _make_sampling_system(baseline_module, actor_mu=0.0, participant_mu=mu)
    _, participant_empirical = _estimate_activation_frequency(system, steps=20000, seed=22)

    expected = 1.0 - np.exp(-mu)
    assert abs(participant_empirical - expected) < 0.03, (
        f"Participant activation frequency should be near theta({mu})={expected:.4f}, "
        f"got {participant_empirical:.4f}"
    )


def test_actor_rate_update_matches_eq13_formula(baseline_module):
    """Section 6.7 Eq. (13) one-step value should match exact formula."""
    config = baseline_module.SystemConfig(num_agents=1, M=1.0, u_0=0.1, gamma=2.0, kappa=3.0)
    system = baseline_module.MultiAgentSystem(config)
    agent = system.agents[0]

    agent.state.actor_interaction_rate = 0.3
    agent.state.estimated_reward_pu = 0.4
    agent.state.estimated_reward_rep = 0.1
    agent.state.estimated_reward_status = 0.2

    alpha_rate = 0.05

    H_hat = max(
        agent.state.estimated_reward_pu,
        config.gamma * agent.state.estimated_reward_rep,
        config.kappa * agent.state.estimated_reward_status,
    )
    mu_prev = 0.3
    expected = np.clip(
        mu_prev
        + alpha_rate
        * (
            -np.exp(-(config.M - mu_prev)) * config.u_0
            + np.exp(-mu_prev) * H_hat
        ),
        0.0,
        config.M,
    )

    agent.update_actor_interaction_rate(alpha_rate)

    assert agent.state.actor_interaction_rate == pytest.approx(expected, abs=1e-12)


def test_actor_rate_is_clipped_to_bounds(baseline_module):
    """Section 6.7: [x]_0^M clipping should enforce 0 <= mu <= M."""
    config = baseline_module.SystemConfig(num_agents=1, M=1.0, u_0=0.1, gamma=2.0, kappa=2.0)
    system = baseline_module.MultiAgentSystem(config)
    agent = system.agents[0]

    # Push above upper bound.
    agent.state.actor_interaction_rate = 0.99
    agent.state.estimated_reward_pu = 100.0
    agent.state.estimated_reward_rep = 100.0
    agent.state.estimated_reward_status = 100.0
    agent.update_actor_interaction_rate(alpha_rate=5.0)
    assert 0.0 <= agent.state.actor_interaction_rate <= config.M

    # Push below lower bound.
    agent.state.actor_interaction_rate = 0.01
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0
    config.u_0 = 50.0
    agent.update_actor_interaction_rate(alpha_rate=5.0)
    assert 0.0 <= agent.state.actor_interaction_rate <= config.M
