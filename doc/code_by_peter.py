"""

ALGORITHMIC FLOW:
- Agents maintain THREE reward estimates (personal utility, reputation, status)
- Reputation learning: agents track personal benefits v_i(k,t) from each other agent
- Gossip: participants average reputation estimates s_i(k,t) with tie tolerance Δ
- Role selection: sequential 3-step procedure (reputation → status → personal utility)
- Actor rates: learned via Eq. (13) with proper gamma/kappa weighting
- Time-scales: reputation learning (fast), behavior learning (slower), role updates (slowest)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Set, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class AgentRole(Enum):
    PERSONAL_UTILITY = "personal_utility"
    REPUTATION = "reputation"
    STATUS = "status"


@dataclass
class SystemConfig:
    """Section 6–7 configuration with all required parameters"""
    
    # Basic setup
    num_agents: int = 6
    num_states: int = 3
    num_actions: int = 2
    num_time_steps: int = 2000
    
    # --- Section 6.7: Actor Interaction Rates (Eq. 13) ---
    M: float = 1.0  # Total interaction rate budget
    u_0: float = 0.1  # Utility from outside interactions
    
    # --- Section 7.1: Role Incentives ---
    gamma: float = 2.0  # Reputation weight
    kappa: float = 2.0  # Status weight
    
    # --- Section 7.1.2: Follower Threshold for Status ---
    c_threshold: float = 0.1  # Minimum fraction of followers needed for status (c·N)
    
    # --- Section 7.1.3: Hysteresis Thresholds ---
    B_R: float = 0.8  # Reputation threshold to START following
    B_F: float = 0.6  # Reputation threshold to CONTINUE following (B_F < B_R)
    
    # --- Section 6.4.4: Tie Threshold for Reputation Selection ---
    delta: float = 0.1  # Tolerance for near-ties in reputation
    
    # --- Section 8: Time-Scale Aware Stepsizes (Assumptions 5–8) ---
    # Base stepsizes (will be decayed as 1/t)
    alpha_pu_base: float = 0.05      # Policy gradient for personal utility
    beta_status_base: float = 0.05   # Policy gradient for status
    eta_v_base: float = 0.1          # Personal benefit estimates v_i(k,t)
    eta_s_base: float = 0.1          # Reputation estimates s_i(k,t)
    eta_J_base: float = 0.05         # Reward estimate updates
    
    # --- Section 7.1.4: Update Intervals (increasing T_n) ---
    role_update_base_interval: int = 50  # Base interval, increases over time
    
    # --- Gossip (Section 6.4) ---
    gossip_rate: float = 0.5  # Probability of gossip at each step
    gossip_alpha: float = 0.5  # Averaging parameter in gossip


@dataclass
class AgentState:
    """Complete agent state for Sections 6–7"""
    
    # --- Policy weights ---
    weights_pu: np.ndarray = field(default_factory=lambda: np.random.randn(3, 2) * 0.1)
    weights_status: np.ndarray = field(default_factory=lambda: np.random.randn(3, 2) * 0.1)
    
    # --- Role and follower relationships ---
    role: AgentRole = AgentRole.PERSONAL_UTILITY
    following: int = None  # Which agent this agent follows (for REPUTATION role)
    followers: Set[int] = field(default_factory=set)  # Who follows this agent
    
    # --- Section 6.4: Reputation Learning ---
    # v_i(k,t): Personal benefit estimates (how much benefit agent i gets from agent k's behavior)
    personal_benefit_estimates: Dict[int, float] = field(default_factory=dict)
    
    # s_i(k,t): Observed reputation estimates (what agent i believes agent k's reputation is)
    reputation_estimates: Dict[int, float] = field(default_factory=dict)
    
    # L_i(t): Estimate of which agent has highest reputation
    highest_rep_agent_estimate: int = None
    
    # --- Section 6.6: Reward Estimates (three separate) ---
    estimated_reward_pu: float = 0.0      # Ĵ^{pu}_i(t)
    estimated_reward_rep: float = 0.0     # Ĵ^r_i(t)
    estimated_reward_status: float = 0.0  # Ĵ^s_i(t)
    
    # --- Section 6.7: Actor Interaction Rates ---
    actor_interaction_rate: float = 0.7   # μ_{a,i}(t) - learned
    participant_interaction_rate: float = 0.7  # μ_{p,i} - fixed
    
    # Tracking for history
    payoff_history: List[float] = field(default_factory=list)
    
    # Flag to track if agent was following before (for hysteresis)
    was_following: bool = False


class Agent:
    """Agent implementing Sections 6–7 algorithms"""
    
    def __init__(self, agent_id: int, config: SystemConfig, system):
        self.agent_id = agent_id
        self.config = config
        self.system = system
        self.state = AgentState()
        
        # Random preference for personal utility (base payoff)
        self.preferred_action = agent_id % config.num_actions
        
        # Initialize reputation and personal benefit estimates for all agents
        for other_id in range(config.num_agents):
            self.state.personal_benefit_estimates[other_id] = 0.0
            self.state.reputation_estimates[other_id] = np.random.randn() * 0.1
        
        # Initial highest rep agent estimate
        self.state.highest_rep_agent_estimate = np.random.randint(config.num_agents)
        
        # Track last action for gradient updates
        self.last_action = None
        self.last_state = None
    
    # ==================== Section 6.3: Personal Utility ====================
    
    def get_softmax_policy(self, state: int, weights: np.ndarray) -> np.ndarray:
        """Convert policy weights to softmax probabilities"""
        logits = weights[state, :]
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / (np.sum(exp_logits) + 1e-8)
    
    def select_action(self, state: int) -> int:
        """Select action based on current role"""
        self.last_state = state
        
        if self.state.role == AgentRole.REPUTATION and self.state.following is not None:
            # Follow leader's policy (Section 6.4.5)
            leader_weights = self.system.agents[self.state.following].state.weights_pu
            policy = self.get_softmax_policy(state, leader_weights)
        elif self.state.role == AgentRole.STATUS:
            # Use status-optimized policy
            policy = self.get_softmax_policy(state, self.state.weights_status)
        else:
            # Personal utility role
            policy = self.get_softmax_policy(state, self.state.weights_pu)
        
        action = np.random.choice(len(policy), p=policy)
        self.last_action = action
        return action
    
    def update_policy_gradient(self, state: int, action: int, reward: float, 
                               weights: np.ndarray, lr: float) -> np.ndarray:
        """
        Policy gradient update (Eq. 8 for PU, Eq. 12 for status)
        w(t+1) = w(t) + α(t) · u_i(s,x) · ∇_w log π(w; s, x)
        where ∇_w log π = a - π(w; s, ·)
        """
        policy = self.get_softmax_policy(state, weights)
        action_one_hot = np.zeros(self.config.num_actions)
        action_one_hot[action] = 1
        gradient = action_one_hot - policy
        
        weights_new = weights.copy()
        weights_new[state, :] += lr * gradient * reward
        return weights_new
    
    def update_personal_utility(self, state: int, action: int, reward: float, 
                                alpha_pu_t: float, eta_J_t: float):
        """
        Section 6.3: Personal utility optimization
        - Update policy weights via gradient (Eq. 8)
        - Update reward estimate Ĵ^{pu}_i(t)
        """
        self.state.weights_pu = self.update_policy_gradient(
            state, action, reward, self.state.weights_pu, alpha_pu_t
        )
        
        # Update personal utility reward estimate (Eq. in Section 6.6)
        self.state.estimated_reward_pu += eta_J_t * (reward - self.state.estimated_reward_pu)
        self.state.payoff_history.append(reward)
    
    # ==================== Section 6.4: Reputation Learning ====================
    
    def update_personal_benefit_estimates(self, observed_payoffs: Dict[int, float], 
                                          eta_v_t: float) -> Dict[int, float]:
        """
        Section 6.4.2: Learn personal benefit from each agent's actions
        v_i(k, t+1) = v_i(k, t) + η_v(t) [u_i(s(t), x_k(t)) - v_i(k, t)]
        
        observed_payoffs[k] is u_i(s(t), x_k(t)) if k is active, else 0

        Returns:
            Dictionary of deltas Δv_i(k,t) = v_i(k,t+1) - v_i(k,t), used by Eq. (9).
        """
        personal_benefit_deltas = {}
        for agent_k, payoff in observed_payoffs.items():
            prev_val = self.state.personal_benefit_estimates.get(agent_k, 0.0)

            # Update if active; otherwise decay for inactive agents.
            if payoff != 0.0:
                new_val = prev_val + eta_v_t * (payoff - prev_val)
            else:
                new_val = prev_val * (1.0 - eta_v_t)

            self.state.personal_benefit_estimates[agent_k] = new_val
            personal_benefit_deltas[agent_k] = new_val - prev_val

        return personal_benefit_deltas
    
    def update_reputation_estimates_gossip(self, personal_benefit_deltas: Dict[int, float],
                                          other_agents_list: List['Agent'],
                                          eta_s_t: float):
        """
        Section 6.4.3: Update reputation estimates via gossip
        s_i(k, t+1) = (Σ_j s_j(k, t)) / |A_p(t)| + v_i(k, t+1) - v_i(k, t)
        """
        for agent_k in range(self.config.num_agents):
            estimates = [
                other_agent.state.reputation_estimates.get(agent_k, 0.0)
                for other_agent in other_agents_list
            ]

            if estimates:
                avg_estimate = float(np.mean(estimates))
            else:
                avg_estimate = self.state.reputation_estimates.get(agent_k, 0.0)

            delta_v = personal_benefit_deltas.get(agent_k, 0.0)
            self.state.reputation_estimates[agent_k] = avg_estimate + delta_v
    
    def identify_highest_reputation_agent(self):
        """
        Section 6.4.4: Identify agent with highest reputation using tie threshold Δ
        
        K_i(t) = {k ∈ C\\{i} : s_i(k, t) ≥ max_k' s_i(k', t) - Δ}
        Select uniformly at random from K_i(t)
        """
        if not self.state.reputation_estimates:
            self.state.highest_rep_agent_estimate = np.random.randint(self.config.num_agents)
            return
        
        # Section 6.4.4 requires selecting from C\{i}; never select self.
        non_self_estimates = {
            k: rep
            for k, rep in self.state.reputation_estimates.items()
            if k != self.agent_id
        }
        if not non_self_estimates:
            all_other_ids = [i for i in range(self.config.num_agents) if i != self.agent_id]
            if all_other_ids:
                self.state.highest_rep_agent_estimate = np.random.choice(all_other_ids)
            else:
                self.state.highest_rep_agent_estimate = self.agent_id
            return

        max_rep = max(non_self_estimates.values())
        
        # Find all agents within delta of the maximum
        candidates = [
            k for k, rep in non_self_estimates.items()
            if rep >= max_rep - self.config.delta
        ]
        
        if candidates:
            self.state.highest_rep_agent_estimate = np.random.choice(candidates)
        else:
            all_other_ids = [i for i in range(self.config.num_agents) if i != self.agent_id]
            if all_other_ids:
                self.state.highest_rep_agent_estimate = np.random.choice(all_other_ids)
            else:
                self.state.highest_rep_agent_estimate = self.agent_id
    
    # ==================== Section 6.5: Status Optimization ====================
    
    def update_status_optimization(self, state: int, action: int, 
                                   social_support_sum: float, beta_status_t: float,
                                   eta_J_t: float):
        """
        Section 6.5: Status optimization
        - Update policy weights via gradient (Eq. 12)
        - Update reward estimate Ĵ^s_i(t)
        
        NOTE: social_support_sum is the SUM of follower payoffs (Eq. 11),
        not the average. This is critical for convergence.
        """
        self.state.weights_status = self.update_policy_gradient(
            state, action, social_support_sum, self.state.weights_status, beta_status_t
        )
        
        # Update status reward estimate
        self.state.estimated_reward_status += eta_J_t * \
            (social_support_sum - self.state.estimated_reward_status)
    
    # ==================== Section 6.4.5: Reputation Optimization ====================
    
    def adopt_leader_behavior(self):
        """
        Section 6.4.5: Agent following a leader directly copies the leader's behavior
        """
        if self.state.following is not None and self.state.following != self.agent_id:
            leader = self.system.agents[self.state.following]
            self.state.weights_pu = np.copy(leader.state.weights_pu)
    
    def update_reputation_reward_estimate(self, leader_payoff: float, eta_J_t: float):
        """
        Update estimated reward from reputation optimization
        """
        self.state.estimated_reward_rep += eta_J_t * (leader_payoff - self.state.estimated_reward_rep)
    
    # ==================== Section 6.7: Actor Interaction Rates ====================
    
    def update_actor_interaction_rate(self, alpha_rate: float):
        """
        Section 6.7: Learn actor interaction rates via Eq. (13)
        
        μ_{a,i}(t) = [μ_{a,i}(t-1) + α(t) · (-θ'(M - μ_{a,i}) · u_0 + θ'(μ_{a,i}) · H_i)]^M_0
        
        where:
        - θ(μ) = 1 - exp(-μ), so θ'(μ) = exp(-μ)
        - H_i = max{Ĵ^{pu}_i, γ Ĵ^r_i, κ Ĵ^s_i}  <-- WEIGHTED MAX
        - u_0 is utility from outside interactions
        - M is total interaction budget
        - [x]^M_0 = clip(x, 0, M)
        """
        # Compute H_i with PROPER gamma and kappa weighting
        max_reward = max(
            self.state.estimated_reward_pu,
            self.config.gamma * self.state.estimated_reward_rep,
            self.config.kappa * self.state.estimated_reward_status
        )
        
        # Compute derivatives of θ
        mu_prev = self.state.actor_interaction_rate
        theta_prime_mu = np.exp(-mu_prev)
        theta_prime_M_minus_mu = np.exp(-(self.config.M - mu_prev))
        
        # Update per Eq. (13)
        update_delta = -theta_prime_M_minus_mu * self.config.u_0 + theta_prime_mu * max_reward
        mu_new = mu_prev + alpha_rate * update_delta
        
        # Clip to [0, M]
        self.state.actor_interaction_rate = np.clip(mu_new, 0.0, self.config.M)


class MultiAgentSystem:
    """Multi-agent system implementing Sections 6–7"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.agents = [Agent(i, config, self) for i in range(config.num_agents)]
        self.time_step = 0
        self.role_update_epoch = 0  # Track which role update epoch we're in
        
        # Results tracking
        self.results = {
            'norm_consensus': [],
            'expected_utilities': [],
            'follower_counts': [],
            'actor_counts': [],
            'participant_counts': [],
            'actor_rates': [],
            'roles_history': [],
            'actual_payoffs': [],
            'social_welfare': [],
        }
    
    def compute_actual_payoff(self, agent_id: int, state: int, action: int) -> float:
        """
        Compute actual payoff for agent taking action in state
        Includes base preference bonus and follower bonus for status agents
        """
        agent = self.agents[agent_id]
        preference_bonus = 1.0 if action == agent.preferred_action else 0.0
        follower_bonus = 0.0  # Followers get social support, not direct bonus
        return preference_bonus + follower_bonus
    
    def step(self):
        """
        Execute one time step following Sections 6–7
        """
        self.time_step += 1
        
        # --- Time-scale aware stepsizes (Section 8, Assumption 5) ---
        # Decay as 1/t to satisfy summability conditions
        t = max(1, self.time_step)
        alpha_pu_t = self.config.alpha_pu_base / (1.0 + t * 0.01)
        beta_status_t = self.config.beta_status_base / (1.0 + t * 0.01)
        eta_v_t = self.config.eta_v_base / (1.0 + t * 0.01)
        eta_s_t = self.config.eta_s_base / (1.0 + t * 0.01)
        eta_J_t = self.config.eta_J_base / (1.0 + t * 0.01)
        alpha_rate_t = 0.01 / (1.0 + t * 0.005)  # Actor rate update
        
        # === PHASE 1: Sample Active Actors and Participants (Section 6.2) ===
        
        # A_a(t): Active actors (Section 6.2) sampled using θ(μ)=1-exp(-μ)
        active_actors = set()
        for agent in self.agents:
            actor_prob = 1.0 - np.exp(-agent.state.actor_interaction_rate)
            if np.random.random() < actor_prob:
                active_actors.add(agent.agent_id)
        
        # A_p(t): Active participants (Section 6.2) sampled using θ(μ)=1-exp(-μ)
        active_participants = []
        for agent in self.agents:
            participant_prob = 1.0 - np.exp(-agent.state.participant_interaction_rate)
            if np.random.random() < participant_prob:
                active_participants.append(agent)
        
        active_participant_ids = {a.agent_id for a in active_participants}
        
        # === PHASE 2: Actors Take Actions (Section 6, Step 1) ===
        
        actions = {}
        this_step_payoffs = {}
        for agent_id in active_actors:
            agent = self.agents[agent_id]
            state = np.random.randint(self.config.num_states)
            action = agent.select_action(state)
            actions[agent_id] = (state, action)
            payoff = self.compute_actual_payoff(agent_id, state, action)
            this_step_payoffs[agent_id] = payoff
            agent.state.payoff_history.append(payoff)
        
        # Observed payoffs: fresh for active actors, 0 for others
        observed_payoffs = {i: this_step_payoffs.get(i, 0.0) for i in range(self.config.num_agents)}
        
        # === PHASE 3: Role-Based Updates for Active Actors (Section 6) ===
        
        for agent_id in active_actors:
            agent = self.agents[agent_id]
            state, action = actions[agent_id]
            payoff = this_step_payoffs[agent_id]
            
            if agent.state.role == AgentRole.PERSONAL_UTILITY:
                # Section 6.3: Update personal utility
                agent.update_personal_utility(state, action, payoff, alpha_pu_t, eta_J_t)
            
            elif agent.state.role == AgentRole.REPUTATION:
                # Section 6.4.5: Reputation agent gets social support from leader
                if agent.state.following is not None:
                    leader_payoff = this_step_payoffs.get(agent.state.following, 0.0)
                    agent.update_reputation_reward_estimate(leader_payoff, eta_J_t)
            
            elif agent.state.role == AgentRole.STATUS:
                # Section 6.5: Status agent receives social support from followers
                if len(agent.state.followers) > 0:
                    # Sum of follower payoffs (NOT average!) - Eq. (11)
                    followers_payoffs = [
                        this_step_payoffs.get(f, 0.0) for f in agent.state.followers
                        if f in active_participant_ids  # Only active participants provide support
                    ]
                    social_support_sum = sum(followers_payoffs)
                    agent.update_status_optimization(
                        state, action, social_support_sum, beta_status_t, eta_J_t
                    )
        
        # === PHASE 4: Updates for Active Participants (Section 6.4) ===
        
        for agent in active_participants:
            # Section 6.4.2: Update personal benefit estimates v_i(k,t)
            personal_benefit_deltas = agent.update_personal_benefit_estimates(
                observed_payoffs, eta_v_t
            )
            
            # Section 6.4.3: Update reputation estimates s_i(k,t) via gossip
            agent.update_reputation_estimates_gossip(
                personal_benefit_deltas, active_participants, eta_s_t
            )
            
            # Section 6.4.4: Identify highest reputation agent
            agent.identify_highest_reputation_agent()
            
            # Section 6.7: Update actor interaction rates
            agent.update_actor_interaction_rate(alpha_rate_t)
        
        # === PHASE 5: Adoption of Leader Behavior (Section 6.4.5) ===
        
        for agent in self.agents:
            if agent.state.role == AgentRole.REPUTATION:
                agent.adopt_leader_behavior()
        
        # === PHASE 6: Periodic Role Updates (Section 7) ===
        
        # Increasing update interval: T_n → ∞ as per Assumption 6
        current_interval = max(
            self.config.role_update_base_interval,
            int(self.config.role_update_base_interval * (1.0 + self.role_update_epoch * 0.1))
        )
        
        if self.time_step % current_interval == 0:
            self._update_roles_sequential()  # Sequential procedure from Section 7
            self.role_update_epoch += 1
        
        # === PHASE 7: Tracking ===
        
        self._track_results(this_step_payoffs, len(active_actors), len(active_participants))
    
    def _update_roles_sequential(self):
        """
        Section 7: Sequential 3-step role update procedure
        
        This is critical: updates must occur in order to properly handle
        indirect follower relationships (Section 7.2)
        """
        
        # Initialize: copy current state
        P = set(i for i, a in enumerate(self.agents) if a.state.role == AgentRole.PERSONAL_UTILITY)
        R = set(i for i, a in enumerate(self.agents) if a.state.role == AgentRole.REPUTATION)
        S = set(i for i, a in enumerate(self.agents) if a.state.role == AgentRole.STATUS)
        
        # Maintain follower relationships during update
        followers = {i: set(self.agents[i].state.followers) for i in range(self.config.num_agents)}
        
        # === STEP 1: Reputation Optimization (Section 7.3) ===
        # Agents without followers decide if they want to follow someone
        
        # C(t) = agents without followers
        C = set(i for i in range(self.config.num_agents) if len(followers[i]) == 0)
        
        # Partition C into agents already following (C_r) and not following (C_pu)
        C_r = C & R  # Already following
        C_pu = C & P  # Not yet following
        
        # Process in random order to avoid bias
        update_order = list(C)
        np.random.shuffle(update_order)
        
        for i in update_order:
            agent = self.agents[i]
            
            # Determine effective threshold with hysteresis (Section 7.1.3)
            if i in C_r:
                B_i = self.config.B_F  # Continue following with lower threshold
            else:
                B_i = self.config.B_R  # Start following with higher threshold
            
            # Get estimated rewards (weighted by gamma for reputation)
            # Bug 1 fix: Ĵ^r_i = s_i(L_i, t) = max reputation estimate (Section 6.6)
            # estimated_reward_rep is only non-zero for current followers; for non-followers
            # the follow decision must use max_rep (the best candidate's s_i value) directly.
            max_rep = max(agent.state.reputation_estimates.values()) if agent.state.reputation_estimates else 0.0
            est_rep_weighted = self.config.gamma * max_rep  # γ·s_i(L_i,t) per Section 6.6 + 7.3
            est_pu = agent.state.estimated_reward_pu

            # Decision: should optimize reputation? (Section 7.3, line 1)
            # Bug 2 fix: paper condition is only γĴ^r_i > max{B_i, Ĵ^pu_i}; the extra
            # "and max_rep >= B_i" gate is not in the paper and blocks all following.
            if est_rep_weighted > max(B_i, est_pu):
                # Follow highest reputation agent
                best_k = agent.state.highest_rep_agent_estimate
                
                # Check if best_k is already a follower (Section 7.2)
                # Bug 5 fix: check if best_k is in R (already following), then follow
                # best_k's leader instead to prevent indirect follower chains.
                if best_k in R and self.agents[best_k].state.following is not None:
                    best_k = self.agents[best_k].state.following
                
                # Update role and follower relationships
                if i in R:
                    # Was already following, change to new leader
                    old_leader = agent.state.following
                    if old_leader is not None:
                        followers[old_leader].discard(i)
                
                agent.state.role = AgentRole.REPUTATION
                agent.state.following = best_k
                followers[best_k].add(i)
                agent.state.was_following = True
                
                R.add(i)
                P.discard(i)
        
        # === STEP 2: Status Optimization (Section 7.4) ===
        # Agents with sufficient followers decide if they want to optimize status
        
        min_followers = max(1, int(self.config.c_threshold * self.config.num_agents))
        
        for i in range(self.config.num_agents):
            if len(followers[i]) >= min_followers:
                agent = self.agents[i]
                
                # Decision: should optimize status?
                if self.config.kappa * agent.state.estimated_reward_status > agent.state.estimated_reward_pu:
                    agent.state.role = AgentRole.STATUS
                    if agent.state.following is not None:
                        followers[agent.state.following].discard(i)
                        agent.state.following = None
                    
                    S.add(i)
                    P.discard(i)
                    R.discard(i)
        
        # === STEP 3: Personal Utility (Section 7.5) ===
        # All remaining agents optimize personal utility
        
        for i in range(self.config.num_agents):
            agent = self.agents[i]
            
            if i not in R and i not in S:
                agent.state.role = AgentRole.PERSONAL_UTILITY
                if agent.state.following is not None:
                    followers[agent.state.following].discard(i)
                    agent.state.following = None
                
                P.add(i)
        
        # === Apply updated follower relationships ===
        for i in range(self.config.num_agents):
            self.agents[i].state.followers = followers[i]
    
    def _track_results(self, this_step_payoffs: Dict, num_actors: int, num_participants: int):
        """Track simulation results for analysis"""
        
        # Norm consensus: variance of policy weights across agents
        all_weights = np.array([a.state.weights_pu.flatten() for a in self.agents])
        norm_variance = np.mean(np.var(all_weights, axis=0))
        self.results['norm_consensus'].append(norm_variance)
        
        # Expected utilities (based on payoff history)
        utils = {
            i: np.mean(a.state.payoff_history) if a.state.payoff_history else 0.0
            for i, a in enumerate(self.agents)
        }
        self.results['expected_utilities'].append(utils)
        
        # Follower counts
        followers = [len(a.state.followers) for a in self.agents]
        self.results['follower_counts'].append(followers)
        
        # Actor and participant counts
        self.results['actor_counts'].append(num_actors)
        self.results['participant_counts'].append(num_participants)
        
        # Learned actor rates
        self.results['actor_rates'].append([a.state.actor_interaction_rate for a in self.agents])
        
        # Roles
        roles = [a.state.role for a in self.agents]
        self.results['roles_history'].append(roles)
        
        # Payoffs
        self.results['actual_payoffs'].append(this_step_payoffs)
        
        # Social welfare
        total_welfare = sum(this_step_payoffs.values())
        self.results['social_welfare'].append(total_welfare)
    
    def simulate(self) -> Dict:
        """Run the full simulation"""
        print(f"Running Sections 6–7 simulation for {self.config.num_time_steps} timesteps...")
        
        for t in range(self.config.num_time_steps):
            self.step()
            if (t + 1) % 500 == 0:
                print(f"  Step {t+1}/{self.config.num_time_steps}")
        
        # Print final results
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)
        
        final_roles = [a.state.role for a in self.agents]
        final_followers = [len(a.state.followers) for a in self.agents]
        opinion_leader = np.argmax(final_followers) if max(final_followers) > 0 else -1
        
        print("\nFinal Roles:")
        for i, role in enumerate(final_roles):
            print(f"  Agent {i}: {role.value:20s} (followers: {final_followers[i]})")
        
        if opinion_leader >= 0:
            print(f"\nOpinion Leader: Agent {opinion_leader} with {final_followers[opinion_leader]} followers")
        
        print(f"\nExpected Utilities (mean payoff over trajectory):")
        for i, a in enumerate(self.agents):
            exp_util = np.mean(a.state.payoff_history) if a.state.payoff_history else 0.0
            print(f"  Agent {i}: {exp_util:.4f}")
        
        print(f"\nFinal Actor Interaction Rates (learned):")
        for i, a in enumerate(self.agents):
            print(f"  Agent {i}: {a.state.actor_interaction_rate:.4f}")
        
        self.results['final_roles'] = final_roles
        self.results['final_followers'] = final_followers
        self.results['opinion_leader'] = opinion_leader
        
        return self.results
    
    def plot_results(self, filename: str = "sections_6_7_corrected.png"):
        """Plot comprehensive simulation results"""
        
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 3, figure=fig)
        
        # 1. Norm Consensus
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(self.results['norm_consensus'], linewidth=2, color='darkblue')
        ax1.set_title('Norm Convergence (Policy Weight Variance)', fontsize=11, fontweight='bold')
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Variance (log scale)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Expected Utilities
        ax2 = fig.add_subplot(gs[0, 1])
        utils_array = np.zeros((len(self.results['expected_utilities']), self.config.num_agents))
        for t, utils in enumerate(self.results['expected_utilities']):
            for agent_id, utility in utils.items():
                utils_array[t, agent_id] = utility
        colors = plt.cm.tab10(np.linspace(0, 1, self.config.num_agents))
        for i in range(self.config.num_agents):
            ax2.plot(utils_array[:, i], label=f'Agent {i}', color=colors[i], alpha=0.8, linewidth=1.5)
        ax2.set_title('Utility Learning (Section 6.3, 6.5)', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Expected Utility')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # 3. Follower Emergence
        ax3 = fig.add_subplot(gs[0, 2])
        followers_array = np.array(self.results['follower_counts'])
        for i in range(self.config.num_agents):
            ax3.plot(followers_array[:, i], label=f'Agent {i}', linewidth=2, color=colors[i])
        ax3.set_title('Opinion Leader Emergence (Section 7)', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Timestep')
        ax3.set_ylabel('Follower Count')
        ax3.legend(fontsize=8, loc='best')
        ax3.grid(True, alpha=0.3)
        
        # 4. Role Evolution
        ax4 = fig.add_subplot(gs[1, :2])
        roles_num = np.array([[1 if r == AgentRole.PERSONAL_UTILITY else 
                              2 if r == AgentRole.REPUTATION else 3 
                              for r in roles] 
                             for roles in self.results['roles_history']])
        im = ax4.imshow(roles_num.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax4.set_title('Role Evolution (Section 7)', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Timestep')
        ax4.set_ylabel('Agent ID')
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('Role: 1=PU, 2=Rep, 3=Status')
        
        # 5. Active Sets
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(self.results['actor_counts'], label='Actors |A_a(t)|', color='teal', linewidth=2)
        ax5.plot(self.results['participant_counts'], label='Participants |A_p(t)|', 
                color='orange', linewidth=2)
        ax5.set_title('Active Actor/Participant Sets (Section 6)', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Timestep')
        ax5.set_ylabel('Count')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        
        # 6. Learned Actor Rates
        ax6 = fig.add_subplot(gs[2, 0])
        rates_array = np.array(self.results['actor_rates'])
        for i in range(self.config.num_agents):
            ax6.plot(rates_array[:, i], label=f'Agent {i}', color=colors[i], linewidth=1.5)
        ax6.set_title('Learned Actor Rates μ_{a,i}(t) (Eq. 13)', fontsize=11, fontweight='bold')
        ax6.set_xlabel('Timestep')
        ax6.set_ylabel('Actor Rate')
        ax6.set_ylim([0, self.config.M * 1.1])
        ax6.legend(fontsize=8, loc='best')
        ax6.grid(True, alpha=0.3)
        
        # 7. Social Welfare
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.plot(self.results['social_welfare'], linewidth=2, color='darkgreen')
        ax7.set_title('Social Welfare Over Time', fontsize=11, fontweight='bold')
        ax7.set_xlabel('Timestep')
        ax7.set_ylabel('Total Payoff')
        ax7.grid(True, alpha=0.3)
        
        # 8. Final Role Distribution
        ax8 = fig.add_subplot(gs[2, 2])
        final_roles = self.results['final_roles']
        role_counts = {
            'Personal Utility': sum(1 for r in final_roles if r == AgentRole.PERSONAL_UTILITY),
            'Reputation': sum(1 for r in final_roles if r == AgentRole.REPUTATION),
            'Status': sum(1 for r in final_roles if r == AgentRole.STATUS)
        }
        ax8.bar(role_counts.keys(), role_counts.values(), 
               color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='black', linewidth=2)
        ax8.set_title('Final Role Distribution', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Count')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Opinion Leader Composition
        ax9 = fig.add_subplot(gs[3, :])
        final_followers = self.results['final_followers']
        colors_bar = ['#ff6666' if i == self.results['opinion_leader'] else '#6666ff' 
                     for i in range(len(final_followers))]
        bars = ax9.bar(range(len(final_followers)), final_followers, color=colors_bar, 
                       edgecolor='black', linewidth=2)
        ax9.set_title(f"Final Follower Distribution (Opinion Leader: Agent {self.results['opinion_leader']})",
                     fontsize=11, fontweight='bold')
        ax9.set_xlabel('Agent ID')
        ax9.set_ylabel('Follower Count')
        ax9.set_xticks(range(len(final_followers)))
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(
            'Sections 6–7: Corrected Learning Algorithms\n' +
            'Personal Utility + Reputation Learning + Status Optimization + Actor Rates + Sequential Role Updates',
            fontsize=13, fontweight='bold', y=0.995
        )
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to {filename}")


if __name__ == "__main__":
    print("="*80)
    print("CORRECTED IMPLEMENTATION: Sections 6–7 Learning Algorithms")
    print("Learning Common Norms in Multi-Agent Systems (Sharma & Marbach, 2026)")
    print("="*80)
    
    config = SystemConfig(
        num_agents=6,
        num_states=3,
        num_actions=2,
        num_time_steps=3000,
        M=1.0,
        u_0=0.1,
        gamma=2.0,
        kappa=2.0,
        c_threshold=0.15,
        B_R=0.8,
        B_F=0.6,
        delta=0.15,
        alpha_pu_base=0.05,
        beta_status_base=0.05,
        eta_v_base=0.1,
        eta_s_base=0.1,
        eta_J_base=0.05,
        role_update_base_interval=100,
    )
    
    print("\n" + "="*80)
    print("CONFIGURATION")
    print("="*80)
    print(f"\nBasic Setup:")
    print(f"  Agents: {config.num_agents}")
    print(f"  States: {config.num_states}, Actions: {config.num_actions}")
    print(f"  Time Steps: {config.num_time_steps}")
    
    print(f"\nInteraction Budget (Section 6.7, Eq. 13):")
    print(f"  M (total budget): {config.M}")
    print(f"  u_0 (outside utility): {config.u_0}")
    
    print(f"\nRole Incentives (Section 7):")
    print(f"  γ (reputation weight): {config.gamma}")
    print(f"  κ (status weight): {config.kappa}")
    print(f"  c (follower threshold): {config.c_threshold:.2f} → {int(config.c_threshold * config.num_agents)} followers")
    
    print(f"\nHysteresis Thresholds (Section 7.1.3):")
    print(f"  B_R (start following): {config.B_R}")
    print(f"  B_F (continue following): {config.B_F}")
    
    print(f"\nReputation Tie Threshold (Section 6.4.4):")
    print(f"  Δ (delta): {config.delta}")
    
    print(f"\nTime-Scale Aware Stepsizes (Section 8):")
    print(f"  α_pu base: {config.alpha_pu_base}")
    print(f"  β_status base: {config.beta_status_base}")
    print(f"  η_v base: {config.eta_v_base}")
    print(f"  η_s base: {config.eta_s_base}")
    print(f"  η_J base: {config.eta_J_base}")
    print(f"  (All decay as 1/t to satisfy Assumption 5)")
    
    print(f"\nRole Update Intervals (Section 7.1.4, Assumption 6):")
    print(f"  Base interval: {config.role_update_base_interval}")
    print(f"  (Increases over time: T_n → ∞)")
    
    print("\n" + "="*80)
    print("IMPLEMENTING ALGORITHMS")
    print("="*80)
    print("\nPhase sequence per timestep:")
    print("  1. Sample A_a(t) [active actors] and A_p(t) [active participants]")
    print("  2. Actors take actions and receive payoffs")
    print("  3. Role-based updates for active actors:")
    print("     - PU agents: update policy via gradient (Eq. 8)")
    print("     - Reputation agents: track leader payoff")
    print("     - Status agents: receive follower social support (Eq. 11)")
    print("  4. Participants update estimates:")
    print("     - v_i(k,t): personal benefit estimates (Section 6.4.2)")
    print("     - s_i(k,t): reputation estimates via gossip (Section 6.4.3)")
    print("     - Identify highest reputation agent with tie threshold Δ (Section 6.4.4)")
    print("     - Update actor rates via Eq. (13) with γ/κ weighting (Section 6.7)")
    print("  5. Gossip: pairwise averaging of reputation estimates")
    print("  6. Followers adopt leader behavior (Section 6.4.5)")
    print("  7. Periodic sequential role updates (Section 7, 3-step procedure)")
    
    print("\n" + "="*80)
    print("Running simulation...")
    print("="*80 + "\n")
    
    system = MultiAgentSystem(config)
    results = system.simulate()
    system.plot_results("sections_6_7_corrected.png")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey Correctness Improvements:")
    print("  ✓ Section 6.3: Policy gradient personal utility (Eq. 8)")
    print("  ✓ Section 6.4: Full reputation learning with v_i(k,t), s_i(k,t), Δ threshold")
    print("  ✓ Section 6.5: Status optimization with SUM of social support (Eq. 11-12)")
    print("  ✓ Section 6.6: Three separate reward estimates")
    print("  ✓ Section 6.7: Actor rates Eq. (13) with γ/κ weighting")
    print("  ✓ Section 7: Sequential 3-step role update procedure")
    print("  ✓ Section 7: Follower count threshold c·N and hysteresis B_R/B_F")
    print("  ✓ Section 7: Tie threshold Δ for reputation selection")
    print("  ✓ Section 8: Time-scale separation with decreasing stepsizes")
    print("  ✓ Section 8: Increasing role update intervals T_n")
    print("\n" + "="*80)
