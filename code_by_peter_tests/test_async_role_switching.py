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

    # enough followers to qualify for status
    system.agents[i].state.followers = {2}
    system.agents[i].state.estimated_reward_status = 10.0
    system.agents[i].state.estimated_reward_pu = 0.0

    system._update_roles_sequential(update_candidates=[i])

    assert system.agents[i].state.role == AgentRole.STATUS
    assert system.agents[i].state.following is None
    assert i not in system.agents[leader].state.followers


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














# Not a valid invariant under the current Section 7.3 implementation:
# agents with followers are excluded from Step-1 reputation switching.
def test_async_partial_update_redirects_existing_followers_when_agent_becomes_follower(model_module):
    """
    Async version of ROLE-5:
    if selected agent i currently has followers and async reevaluation makes i follow best_k,
    then i's followers should be redirected to best_k.
    """
    np.random.seed(0)
    system = make_system(
        model_module,
        num_agents=5,
        extra_config=dict(gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.05, c_threshold=1.0),
    )
    AgentRole = model_module.AgentRole

    i = 1
    new_leader = 3
    old_follower_a = 0
    old_follower_b = 2

    # agent i currently a PU leader with followers {0,2}
    system.agents[i].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[i].state.following = None
    system.agents[i].state.followers = {old_follower_a, old_follower_b}
    system.agents[i].state.estimated_reward_pu = 0.0
    system.agents[i].state.reputation_estimates = {0: 0.1, 2: 0.1, 3: 3.0, 4: 0.1, 1: 0.0}
    system.agents[i].state.highest_rep_agent_estimate = new_leader

    # those followers currently follow i
    system.agents[old_follower_a].state.role = AgentRole.REPUTATION
    system.agents[old_follower_a].state.following = i
    system.agents[old_follower_a].state.followers = set()

    system.agents[old_follower_b].state.role = AgentRole.REPUTATION
    system.agents[old_follower_b].state.following = i
    system.agents[old_follower_b].state.followers = set()

    # new leader
    system.agents[new_leader].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[new_leader].state.followers = set()

    # filler
    system.agents[4].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[4].state.followers = set()

    system._update_roles_sequential(update_candidates=[i])

    assert system.agents[i].state.role == AgentRole.REPUTATION
    assert system.agents[i].state.following == new_leader

    assert system.agents[old_follower_a].state.following == new_leader
    assert system.agents[old_follower_b].state.following == new_leader

    assert old_follower_a in system.agents[new_leader].state.followers
    assert old_follower_b in system.agents[new_leader].state.followers
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