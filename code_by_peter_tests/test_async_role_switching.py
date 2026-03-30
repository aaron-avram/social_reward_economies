"""Async-specific tests for partial role updates and async experiment harness."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from _shared import load_model_module, make_system


@pytest.fixture(scope="module")
def model_module():
    return load_model_module()


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
    _, _, _, _, async_debug, role_update_diagnostics = rep_scaling.run_single(
        args=Args(),
        mode="async",
        gamma=2.0,
        seed=0,
    )

    assert async_debug is not None
    assert role_update_diagnostics is None
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
    _, _, _, _, async_debug, role_update_diagnostics = rep_scaling.run_single(
        args=Args(),
        mode="static",
        gamma=5.0,
        seed=0,
    )

    assert async_debug is None
    assert role_update_diagnostics is not None

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
