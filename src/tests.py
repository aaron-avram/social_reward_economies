import numpy as np
from copy import deepcopy

from code_debugged import MultiAgentSystem, SystemConfig, AgentRole


# ============================================================
# Gossip (Eq. 9) tests
# ============================================================

def gossip_sync_update(rep_snapshot, delta_v_by_i, num_agents):
    """
    Synchronous snapshot gossip (Eq. 9 structure):
      s_i(k,t+1) = avg_{j in A_p(t)} s_j(k,t) + Δv_i(k,t)
    """
    participants = list(rep_snapshot.keys())
    new_rep = {i: {} for i in participants}
    for i in participants:
        for k in range(num_agents):
            avg_est = sum(rep_snapshot[j].get(k, 0.0) for j in participants) / len(participants)
            dv = delta_v_by_i.get(i, {}).get(k, 0.0)
            new_rep[i][k] = avg_est + dv
    return new_rep


def gossip_inplace_update(rep, delta_v_by_i, num_agents, update_order=None):
    """
    INTENTIONALLY WRONG: in-place update (order-dependent).
    This mimics the bug if you don't snapshot before writing.
    """
    participants = list(rep.keys())
    if update_order is None:
        update_order = participants[:]
    for i in update_order:
        for k in range(num_agents):
            avg_est = sum(rep[j].get(k, 0.0) for j in participants) / len(participants)
            dv = delta_v_by_i.get(i, {}).get(k, 0.0)
            rep[i][k] = avg_est + dv
    return rep


def _variance(xs):
    xs = np.array(xs, dtype=float)
    return float(np.var(xs))


def test_gossip_mean_only():
    rep_snapshot = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: 0.0}, 1: {0: 0.0}, 2: {0: 0.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 1, 2]]
    print("After gossip (Δv=0):", vals)
    assert all(abs(v - 5.0) < 1e-10 for v in vals), "Expected all to become mean=5.0"


def test_gossip_mean_plus_delta_v():
    rep_snapshot = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: +1.0}, 1: {0: 0.0}, 2: {0: -2.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 1, 2]]
    print("After gossip (mean+Δv):", vals)
    assert abs(vals[0] - 6.0) < 1e-10
    assert abs(vals[1] - 5.0) < 1e-10
    assert abs(vals[2] - 3.0) < 1e-10


def test_gossip_snapshot_vs_inplace_order_dependence():
    rep0 = {0: {0: 1.0}, 1: {0: 5.0}, 2: {0: 9.0}}
    delta_v = {0: {0: 0.0}, 1: {0: 0.0}, 2: {0: 0.0}}

    out_sync = gossip_sync_update(rep0, delta_v, num_agents=1)
    sync_vals = [out_sync[i][0] for i in [0, 1, 2]]

    rep_inplace_a = {i: dict(rep0[i]) for i in rep0}
    out_inplace_a = gossip_inplace_update(rep_inplace_a, delta_v, num_agents=1, update_order=[0, 1, 2])
    a_vals = [out_inplace_a[i][0] for i in [0, 1, 2]]

    rep_inplace_b = {i: dict(rep0[i]) for i in rep0}
    out_inplace_b = gossip_inplace_update(rep_inplace_b, delta_v, num_agents=1, update_order=[2, 1, 0])
    b_vals = [out_inplace_b[i][0] for i in [0, 1, 2]]

    print("Snapshot vals:", sync_vals)
    print("Inplace order [0,1,2]:", a_vals)
    print("Inplace order [2,1,0]:", b_vals)

    assert all(abs(v - 5.0) < 1e-10 for v in sync_vals), "Snapshot must give exact mean"
    assert (any(abs(v - 5.0) > 1e-10 for v in a_vals) or
            any(abs(v - 5.0) > 1e-10 for v in b_vals)), "In-place should be order-dependent and deviate"


def test_gossip_multi_round_convergence_delta_v_zero():
    rep = {0: {0: 0.0}, 1: {0: 10.0}, 2: {0: 0.0}, 3: {0: 10.0}}
    delta_v = {i: {0: 0.0} for i in rep}

    variances = []
    for _ in range(6):
        vals = [rep[i][0] for i in sorted(rep.keys())]
        variances.append(_variance(vals))
        rep = gossip_sync_update(rep, delta_v, num_agents=1)

    print("Variances over rounds:", variances)
    assert variances[-1] < variances[0] * 1e-6, "Should converge very close to consensus (variance ~ 0)"


def test_gossip_participant_subset_mean():
    rep_snapshot = {0: {0: 0.0}, 2: {0: 6.0}, 4: {0: 12.0}}
    delta_v = {0: {0: 0.0}, 2: {0: 0.0}, 4: {0: 0.0}}
    out = gossip_sync_update(rep_snapshot, delta_v, num_agents=1)
    vals = [out[i][0] for i in [0, 2, 4]]
    print("Subset participants gossip vals:", vals)
    assert all(abs(v - 6.0) < 1e-10 for v in vals), "Mean must be over active participants only"


# ============================================================
# Role allocation tests (Section 7)
# ============================================================

def test_role_follower_chain_redirection():
    np.random.seed(0)
    config = SystemConfig(num_agents=3, num_time_steps=1, gamma=2.0, kappa=0.0, B_R=0.1, B_F=0.1)
    system = MultiAgentSystem(config)

    system.agents[0].state.role = AgentRole.PERSONAL_UTILITY

    system.agents[1].state.role = AgentRole.REPUTATION
    system.agents[1].state.following = 0
    system.agents[0].state.followers.add(1)

    system.agents[2].state.role = AgentRole.PERSONAL_UTILITY
    system.agents[2].state.reputation_estimates = {0: 0.5, 1: 1.0, 2: 0.0}
    system.agents[2].state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    print("Agent 2 following:", system.agents[2].state.following)
    assert system.agents[2].state.following == 0, "Follower chain not redirected correctly."


def test_role_identify_highest_rep_excludes_self():
    np.random.seed(0)
    config = SystemConfig(num_agents=4, num_time_steps=1, delta=0.0)
    system = MultiAgentSystem(config)

    i = 2
    a = system.agents[i]
    a.state.reputation_estimates = {0: 1.0, 1: 2.0, 2: 999.0, 3: 3.0}
    a.identify_highest_reputation_agent()

    print("highest_rep_agent_estimate:", a.state.highest_rep_agent_estimate)
    assert a.state.highest_rep_agent_estimate != i, "Should exclude self from candidates"


def test_role_hysteresis_start_vs_continue():
    np.random.seed(0)
    config = SystemConfig(
        num_agents=3, num_time_steps=1, gamma=1.0, kappa=0.0,
        B_R=0.8, B_F=0.6, delta=0.0,
        c_threshold=1.0  # prevent status step from triggering
    )
    system = MultiAgentSystem(config)

    leader = 0
    rep_signal = 0.7  # 0.6 < 0.7 < 0.8

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

    print("Agent 1 role/following:", system.agents[1].state.role, system.agents[1].state.following)
    print("Agent 2 role/following:", system.agents[2].state.role, system.agents[2].state.following)

    assert system.agents[1].state.role != AgentRole.REPUTATION, "Should NOT start following at 0.7 when B_R=0.8"
    assert system.agents[2].state.role == AgentRole.REPUTATION and system.agents[2].state.following == leader, \
        "Should CONTINUE following at 0.7 when B_F=0.6"


def test_role_status_requires_min_followers():
    np.random.seed(0)
    config = SystemConfig(num_agents=10, num_time_steps=1, gamma=0.0, kappa=2.0, c_threshold=0.3)
    system = MultiAgentSystem(config)

    agent = system.agents[0]
    agent.state.followers = {1, 2}  # only 2 followers, but min is 3
    agent.state.role = AgentRole.PERSONAL_UTILITY
    agent.state.estimated_reward_status = 999.0
    agent.state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    print("Agent 0 role:", system.agents[0].state.role)
    assert system.agents[0].state.role != AgentRole.STATUS, "Should not enter STATUS without enough followers"


def test_role_status_switch_clears_following():
    np.random.seed(0)
    config = SystemConfig(num_agents=6, num_time_steps=1, gamma=1.0, kappa=2.0, c_threshold=0.2)
    system = MultiAgentSystem(config)

    leader = 0
    i = 1

    system.agents[i].state.role = AgentRole.REPUTATION
    system.agents[i].state.following = leader
    system.agents[leader].state.followers.add(i)

    system.agents[i].state.followers = {2}  # qualifies (min_followers = 1)
    system.agents[i].state.estimated_reward_status = 10.0
    system.agents[i].state.estimated_reward_pu = 0.0

    system._update_roles_sequential()

    print("Agent 1 role/following:", system.agents[i].state.role, system.agents[i].state.following)
    assert system.agents[i].state.role == AgentRole.STATUS, "Should switch to STATUS"
    assert system.agents[i].state.following is None, "STATUS agent should not keep following"
    assert i not in system.agents[leader].state.followers, "Should be removed from old leader followers"


# ============================================================
# Estimates tracking tests (Section 6.3 / 6.4 / 6.6)
# ============================================================

def test_estimates_personal_benefit_delta_active():
    np.random.seed(0)
    config = SystemConfig(num_agents=3, num_time_steps=1)
    system = MultiAgentSystem(config)

    agent = system.agents[0]
    eta = 0.2

    # Set a known previous value for v_0(1)
    agent.state.personal_benefit_estimates[1] = 4.0

    observed_payoffs = {0: 0.0, 1: 10.0, 2: 0.0}
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta)

    expected_new = 4.0 + eta * (10.0 - 4.0)  # 5.2
    expected_delta = expected_new - 4.0      # 1.2

    print("v update (active): new=", agent.state.personal_benefit_estimates[1], " delta=", deltas[1])
    assert abs(agent.state.personal_benefit_estimates[1] - expected_new) < 1e-10
    assert abs(deltas[1] - expected_delta) < 1e-10


def test_estimates_personal_benefit_decay_inactive():
    np.random.seed(0)
    config = SystemConfig(num_agents=3, num_time_steps=1)
    system = MultiAgentSystem(config)

    agent = system.agents[0]
    eta = 0.2

    agent.state.personal_benefit_estimates[2] = 5.0

    observed_payoffs = {0: 0.0, 1: 0.0, 2: 0.0}  # everyone inactive from i's POV
    deltas = agent.update_personal_benefit_estimates(observed_payoffs, eta_v_t=eta)

    expected_new = 5.0 * (1.0 - eta)  # 4.0
    expected_delta = expected_new - 5.0  # -1.0

    print("v update (inactive decay): new=", agent.state.personal_benefit_estimates[2], " delta=", deltas[2])
    assert abs(agent.state.personal_benefit_estimates[2] - expected_new) < 1e-10
    assert abs(deltas[2] - expected_delta) < 1e-10


def test_estimates_reward_ema_personal_utility():
    np.random.seed(0)
    config = SystemConfig(num_agents=2, num_states=1, num_actions=2, num_time_steps=1)
    system = MultiAgentSystem(config)

    a = system.agents[0]
    a.state.estimated_reward_pu = 1.0
    a.state.weights_pu = np.zeros((3, 2))

    # Use update_personal_utility directly: Ĵ_pu <- Ĵ_pu + η_J (r - Ĵ_pu)
    eta_J = 0.5
    alpha = 0.0  # don't change weights (not needed for this test)
    a.update_personal_utility(state=0, action=0, reward=3.0, alpha_pu_t=alpha, eta_J_t=eta_J)

    expected = 1.0 + eta_J * (3.0 - 1.0)  # 2.0
    print("Ĵ_pu EMA:", a.state.estimated_reward_pu)
    assert abs(a.state.estimated_reward_pu - expected) < 1e-10


# ============================================================
# Interaction rate test (Eq. 13)
# ============================================================

def test_interaction_rate_direction_and_bounds():
    np.random.seed(0)

    config = SystemConfig(
        num_agents=1,
        num_time_steps=1,
        M=1.0,
        u_0=0.1,
        gamma=2.0,
        kappa=2.0,
    )

    system = MultiAgentSystem(config)
    agent = system.agents[0]

    # ---- Case 1: High incentive -> μ should increase ----
    agent.state.actor_interaction_rate = 0.5
    agent.state.estimated_reward_pu = 1.0
    agent.state.estimated_reward_rep = 1.0
    agent.state.estimated_reward_status = 1.0

    old_mu = agent.state.actor_interaction_rate
    agent.update_actor_interaction_rate(0.1)
    new_mu = agent.state.actor_interaction_rate

    print("μ high-incentive:", old_mu, "->", new_mu)
    assert new_mu > old_mu, "Interaction rate should increase when incentive is high"

    # ---- Case 2: Low incentive -> μ should decrease ----
    agent.state.actor_interaction_rate = 0.5
    agent.state.estimated_reward_pu = 0.0
    agent.state.estimated_reward_rep = 0.0
    agent.state.estimated_reward_status = 0.0

    old_mu = agent.state.actor_interaction_rate
    agent.update_actor_interaction_rate(0.1)
    new_mu = agent.state.actor_interaction_rate

    print("μ low-incentive:", old_mu, "->", new_mu)
    assert new_mu < old_mu, "Interaction rate should decrease when incentive is low"

    # ---- Case 3: Clip bounds ----
    agent.state.actor_interaction_rate = 1.5  # above M
    agent.update_actor_interaction_rate(0.1)
    assert 0.0 <= agent.state.actor_interaction_rate <= config.M

    agent.state.actor_interaction_rate = -0.5  # below 0
    agent.update_actor_interaction_rate(0.1)
    assert 0.0 <= agent.state.actor_interaction_rate <= config.M


# ============================================================
# Status test (Eq. 11-12)
# ============================================================
def test_status_reward_uses_sum_not_average():
    np.random.seed(0)

    config = SystemConfig(
        num_agents=4,
        num_states=3,
        num_actions=2,
        num_time_steps=1,
    )
    system = MultiAgentSystem(config)

    leader = system.agents[0]
    leader.state.role = AgentRole.STATUS

    # Pick any valid state/action indices (policy gradient won't matter because we set beta=0).
    state = 0
    action = 0

    # Make the EMA update deterministic: eta_J_t=1.0 -> estimate becomes exactly the target in one step.
    beta_status_t = 0.0
    eta_J_t = 1.0

    # Followers' payoffs (active followers). SUM vs AVG differ by construction.
    follower_payoffs = [1.0, 2.0, 3.0]   # sum=6, avg=2
    social_support_sum = float(sum(follower_payoffs))
    social_support_avg = float(np.mean(follower_payoffs))
    assert social_support_sum != social_support_avg

    # --- SUM case: should update to 6.0 exactly ---
    leader.state.estimated_reward_status = 0.0
    leader.update_status_optimization(
        state=state,
        action=action,
        social_support_sum=social_support_sum,
        beta_status_t=beta_status_t,
        eta_J_t=eta_J_t,
    )
    print("Status reward (SUM) target:", social_support_sum,
          "updated Ĵ_status:", leader.state.estimated_reward_status)
    assert abs(leader.state.estimated_reward_status - social_support_sum) < 1e-10, \
        "Status estimate should update to SUM of follower payoffs (Eq. 11)."

    # --- AVG case (intentional contrast): would update to 2.0 if average were used ---
    leader.state.estimated_reward_status = 0.0
    leader.update_status_optimization(
        state=state,
        action=action,
        social_support_sum=social_support_avg,   # deliberately passing average here
        beta_status_t=beta_status_t,
        eta_J_t=eta_J_t,
    )
    print("Status reward (AVG) target:", social_support_avg,
          "updated Ĵ_status:", leader.state.estimated_reward_status)
    assert abs(leader.state.estimated_reward_status - social_support_avg) < 1e-10, \
        "Control check: if average is passed in, estimate should equal average."

    # The actual model should be using SUM, not AVG (so these must differ).
    assert abs(social_support_sum - social_support_avg) > 1e-10


def run_all_tests():
    # Gossip
    test_gossip_mean_only()
    test_gossip_mean_plus_delta_v()
    test_gossip_snapshot_vs_inplace_order_dependence()
    test_gossip_multi_round_convergence_delta_v_zero()
    test_gossip_participant_subset_mean()

    # Roles
    test_role_follower_chain_redirection()
    test_role_identify_highest_rep_excludes_self()
    test_role_hysteresis_start_vs_continue()
    test_role_status_requires_min_followers()
    test_role_status_switch_clears_following()

    # Estimates
    test_estimates_personal_benefit_delta_active()
    test_estimates_personal_benefit_decay_inactive()
    test_estimates_reward_ema_personal_utility()

    test_interaction_rate_direction_and_bounds()

    test_status_reward_uses_sum_not_average()

    print("\nAll tests passed.")


if __name__ == "__main__":
    run_all_tests()