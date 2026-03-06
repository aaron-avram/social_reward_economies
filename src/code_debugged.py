"""

ALGORITHMIC FLOW:
- Agents maintain THREE reward estimates (personal utility, reputation, status)
- Reputation learning: agents track personal benefits v_i(k,t) from each other agent
- Gossip: participants average reputation estimates s_i(k,t) with tie tolerance Δ
- Role selection: sequential 3-step procedure (reputation → status → personal utility)
- Actor rates: learned via Eq. (13) with proper gamma/kappa weighting
- Time-scales: reputation learning (fast), behavior learning (slower), role updates (slowest)

BUG FIX ID MAP (grouped):
- IR-1: Active actor/participant sampling must use θ(μ)=1-exp(-μ)
- REP-1: Highest-reputation selection must exclude self (C\{i})
- REP-2: Reputation update must follow Eq. (9): avg reputation + Δv
- REP-3: Remove extra per-step pairwise gossip pass
- REP-4: Personal-benefit estimates v_i(k,t) must update for all agents each step
- REP-5: Reputation followers must emulate leader's active-role policy (PU vs STATUS)
- REP-6: Reputation reward estimate must match followed-agent reputation s_i(k,t)
- ROLE-1: Non-followers bootstrap reputation-role entry via γ*s_i(L_i,t)
- ROLE-2: Remove extra gate `max_rep >= B_i` in Step-1 follow decision
- ROLE-3: Redirect if selected leader is itself a follower
- ROLE-4: Prevent self-following when redirect chain points back to agent
- ROLE-5: Redirect existing followers when an agent becomes a follower
- STATUS-1: Update status reward signal before Step-2 gate so STATUS entry is reachable
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
    
    # --- Section 7.1.4: Update Epochs ---
    # Paper notation: s_0 and interval sequence {T_n}, with s_n = s_{n-1} + T_n.
    role_update_s0: int = 0
    role_update_T_sequence: List[int] = field(default_factory=list)
    role_update_base_interval: int = 50  # Base interval for constant/increasing schedules
    fixed_role_update_interval: bool = False  # If True, use constant spacing T_n = const
    role_update_epochs: List[int] = field(default_factory=list)  # Optional direct s_n list (alternative input)
    
    # --- Gossip (Section 6.4) ---
    gossip_rate: float = 0.5  # Probability of gossip at each step
    gossip_alpha: float = 0.5  # Averaging parameter in gossip

    # --- Runtime/Simulation controls ---
    tracking_mode: str = "full"  # "full" keeps all diagnostics, "light" keeps core metrics only
    use_numpy_fast_path: bool = False  # Enable vectorized reputation updates for large-N sweeps
    initial_actor_interaction_rate: float = 0.7
    initial_participant_interaction_rate: float = 0.7
    reward_model: str = "simple_preferred_action"  # "simple_preferred_action" or "shared_base_gaussian"
    reward_base_mu: float = 0.5
    reward_base_sigma: float = 0.08
    reward_agent_sigma: float = 0.1
    reward_clip_min: float = 0.01
    reward_clip_max: float = 2.5


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

        # Ensure policy parameter shapes follow runtime config (num_states, num_actions).
        # AgentState defaults are placeholders and may not match experiment overrides.
        self.state.weights_pu = np.random.randn(config.num_states, config.num_actions) * 0.1
        self.state.weights_status = np.random.randn(config.num_states, config.num_actions) * 0.1
        
        # Random preference for personal utility (base payoff)
        self.preferred_action = agent_id % config.num_actions
        
        # Initialize reputation and personal benefit estimates for all agents
        for other_id in range(config.num_agents):
            self.state.personal_benefit_estimates[other_id] = 0.0
            self.state.reputation_estimates[other_id] = np.random.randn() * 0.1

        # Runtime tuning: initial rates are configurable for faster large-scale experiments.
        self.state.actor_interaction_rate = float(config.initial_actor_interaction_rate)
        self.state.participant_interaction_rate = float(config.initial_participant_interaction_rate)
        
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

    def get_behavior_weights(self) -> np.ndarray:
        """Return role-consistent behavior weights w_i(t) used for imitation."""
        if self.state.role == AgentRole.STATUS:
            return self.state.weights_status
        return self.state.weights_pu
    
    def select_action(self, state: int) -> int:
        """Select action based on current role"""
        self.last_state = state
        
        if self.state.role == AgentRole.REPUTATION and self.state.following is not None:
            # [REP-5] Followers emulate the leader's active-role policy w_k(t):
            # status weights for STATUS leaders, PU weights otherwise.
            leader_weights = self.system.agents[self.state.following].get_behavior_weights()
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

        # [REP-2] Eq. (9) needs the fresh change term v_i(k,t+1)-v_i(k,t).
        # We return these per-agent deltas so reputation can be updated as:
        # avg_j s_j(k,t) + delta_v, rather than a second EMA toward raw payoff.
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
        # [REP-2] Implement Eq. (9) directly:
        # s_i(k,t+1) = average active estimate for k + delta_v_i(k,t).
        # This replaces the old two-step EMA logic (toward avg, then toward payoff).
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
        
        # [REP-1] Section 6.4.4 defines candidates over C\{i}.
        # Excluding self prevents invalid self-target selection in follow decisions.
        non_self_estimates = {
            k: rep
            for k, rep in self.state.reputation_estimates.items()
            if k != self.agent_id
        }
        if not non_self_estimates:
            # Edge-case guard for single-agent configs.
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
            # Edge-case guard for single-agent configs.
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
            # [REP-5] Copy role-consistent leader behavior w_k(t), not always w_k^pu.
            self.state.weights_pu = np.copy(leader.get_behavior_weights())
    
    def update_reputation_reward_estimate(self, followed_rep_estimate: float, eta_J_t: float = None):
        """
        [REP-6] Section 6.6:
        For active reputation optimizers, J^r_i(t) is the current reputation
        estimate of followed agent k, i.e., s_i(k,t), not an EMA of leader payoff.
        """
        self.state.estimated_reward_rep = followed_rep_estimate
    
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
        if self.config.reward_model not in {"simple_preferred_action", "shared_base_gaussian"}:
            raise ValueError(
                f"Unsupported reward_model='{self.config.reward_model}'. "
                "Use 'simple_preferred_action' or 'shared_base_gaussian'."
            )
        self.agents = [Agent(i, config, self) for i in range(config.num_agents)]
        self.time_step = 0
        self.role_update_epoch = 0  # Track which role update epoch we're in
        self.last_active_actor_ids: Set[int] = set()
        self.last_active_participant_ids: Set[int] = set()
        self._role_update_epochs = self._build_role_update_epochs()
        self._next_role_update_epoch_idx = 0
        self._next_role_update_time = max(1, int(self.config.role_update_base_interval))
        
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

        # Optional reward table r_i(s,a) for richer state/action-dependent experiments.
        self._reward_tables = None
        if self.config.reward_model == "shared_base_gaussian":
            self._initialize_shared_base_gaussian_rewards()

        # Optional vectorized caches used by the large-scale experiment harness.
        self._v_matrix = None
        self._s_matrix = None
        if self.config.use_numpy_fast_path:
            self._initialize_numpy_fast_state()

    def _initialize_numpy_fast_state(self):
        """Initialize dense NxN buffers for v_i(k,t) and s_i(k,t)."""
        num_agents = self.config.num_agents
        self._v_matrix = np.zeros((num_agents, num_agents), dtype=float)
        self._s_matrix = np.zeros((num_agents, num_agents), dtype=float)
        for i, agent in enumerate(self.agents):
            for k in range(num_agents):
                self._v_matrix[i, k] = float(agent.state.personal_benefit_estimates.get(k, 0.0))
                self._s_matrix[i, k] = float(agent.state.reputation_estimates.get(k, 0.0))

    def _initialize_shared_base_gaussian_rewards(self):
        """
        Build reward table r_i(s,a) with shared base means:
        - Draw base means m(s,a) once.
        - For each agent i, draw r_i(s,a) ~ Normal(m(s,a), sigma_agent).
        - Clip to a positive range to avoid sign-related artifacts in PU baselines.
        """
        base = np.random.normal(
            loc=self.config.reward_base_mu,
            scale=self.config.reward_base_sigma,
            size=(self.config.num_states, self.config.num_actions),
        )
        base = np.clip(base, self.config.reward_clip_min, self.config.reward_clip_max)

        tables = np.random.normal(
            loc=base[np.newaxis, :, :],
            scale=self.config.reward_agent_sigma,
            size=(self.config.num_agents, self.config.num_states, self.config.num_actions),
        )
        self._reward_tables = np.clip(tables, self.config.reward_clip_min, self.config.reward_clip_max)

    def _build_role_update_epochs(self) -> List[int]:
        """
        Build explicit update epochs s_n for Step-6 role updates.

        Priority:
        1) Paper notation schedule from (s_0, T_n sequence):
           s_n = s_{n-1} + T_n.
        2) Direct epoch list s_n from config.role_update_epochs.
        3) Empty list -> fall back to interval-based schedules.
        """
        t_seq = [int(t) for t in self.config.role_update_T_sequence if int(t) > 0]
        if t_seq:
            s_prev = max(0, int(self.config.role_update_s0))
            epochs = []
            for t_n in t_seq:
                s_prev += t_n
                if s_prev > 0:
                    epochs.append(int(s_prev))
            return sorted(set(epochs))

        return sorted(set(int(t) for t in self.config.role_update_epochs if int(t) > 0))

    def _sync_s_matrix_to_state_dicts(self, agent_ids=None):
        """Materialize dense s_i(k,t) rows into per-agent dicts when needed."""
        if self._s_matrix is None:
            return
        num_agents = self.config.num_agents
        if agent_ids is None:
            agent_ids = range(num_agents)
        for i in agent_ids:
            row = self._s_matrix[i]
            self.agents[i].state.reputation_estimates = {
                k: float(row[k]) for k in range(num_agents)
            }

    def _identify_highest_reputation_agent_from_matrix(self, agent_id: int):
        """
        Fast REP-1 selection on dense s_i(k,t): choose from C\\{i} with tie tolerance delta.
        """
        num_agents = self.config.num_agents
        agent = self.agents[agent_id]
        if num_agents <= 1:
            agent.state.highest_rep_agent_estimate = agent_id
            return

        row = self._s_matrix[agent_id].copy()
        row[agent_id] = -np.inf  # REP-1: exclude self
        max_rep = np.max(row)
        if not np.isfinite(max_rep):
            others = [k for k in range(num_agents) if k != agent_id]
            agent.state.highest_rep_agent_estimate = int(np.random.choice(others))
            return

        candidates = np.where(row >= max_rep - self.config.delta)[0]
        candidates = candidates[candidates != agent_id]
        if candidates.size == 0:
            others = [k for k in range(num_agents) if k != agent_id]
            agent.state.highest_rep_agent_estimate = int(np.random.choice(others))
            return

        agent.state.highest_rep_agent_estimate = int(np.random.choice(candidates))

    def _phase4_updates_numpy_fast(
        self,
        observed_payoff_vector: np.ndarray,
        active_participant_ids: np.ndarray,
        eta_v_t: float,
    ):
        """
        Vectorized Phase-4 updates:
        - REP-4: update v_i(k,t) for all agents i and all k each step.
        - REP-2: update s_i(k,t+1)=avg_j s_j(k,t)+delta_v_i(k,t) for active participants.
        """
        prev_v = self._v_matrix

        # Broadcast active/inactive actor payoffs across all observer agents i.
        active_mask = observed_payoff_vector != 0.0
        new_v = np.where(
            active_mask[np.newaxis, :],
            prev_v + eta_v_t * (observed_payoff_vector[np.newaxis, :] - prev_v),
            prev_v * (1.0 - eta_v_t),
        )
        delta_v = new_v - prev_v
        self._v_matrix = new_v

        if active_participant_ids.size > 0:
            avg_s = np.mean(self._s_matrix[active_participant_ids, :], axis=0)
            self._s_matrix[active_participant_ids, :] = avg_s[np.newaxis, :] + delta_v[active_participant_ids, :]

            for agent_id in active_participant_ids:
                self._identify_highest_reputation_agent_from_matrix(int(agent_id))
    
    def compute_actual_payoff(self, agent_id: int, state: int, action: int) -> float:
        """
        Compute actual payoff for agent taking action in state
        Includes base preference bonus and follower bonus for status agents
        """
        if self.config.reward_model == "shared_base_gaussian":
            return float(self._reward_tables[agent_id, state, action])

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
        
        # [IR-1] Activation probability must be θ(μ)=1-exp(-μ).
        # The old implementation used raw μ directly, which over-activated agents.
        # A_a(t): Active actors (Section 6.2) sampled using θ(μ)=1-exp(-μ)
        active_actors = set()
        for agent in self.agents:
            actor_prob = 1.0 - np.exp(-agent.state.actor_interaction_rate)
            if np.random.random() < actor_prob:
                active_actors.add(agent.agent_id)
        
        # [IR-1] Activation probability must be θ(μ)=1-exp(-μ).
        # Keep participant sampling consistent with Section 6.2.
        # A_p(t): Active participants (Section 6.2) sampled using θ(μ)=1-exp(-μ)
        active_participants = []
        for agent in self.agents:
            participant_prob = 1.0 - np.exp(-agent.state.participant_interaction_rate)
            if np.random.random() < participant_prob:
                active_participants.append(agent)
        
        active_participant_ids = {a.agent_id for a in active_participants}
        self.last_active_actor_ids = set(active_actors)
        self.last_active_participant_ids = set(active_participant_ids)
        
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

            # [STATUS-1] Keep status-reward estimates current for any agent that currently has
            # followers. Step-2 role selection compares kappa * J^s_i against J^pu_i,
            # so J^s_i must be learned even before the agent enters STATUS role.
            social_support_sum = 0.0
            if len(agent.state.followers) > 0:
                followers_payoffs = [
                    this_step_payoffs.get(f, 0.0) for f in agent.state.followers
                    if f in active_participant_ids  # Only active participants provide support
                ]
                social_support_sum = sum(followers_payoffs)
                if agent.state.role != AgentRole.STATUS:
                    agent.state.estimated_reward_status += eta_J_t * (
                        social_support_sum - agent.state.estimated_reward_status
                    )
            
            if agent.state.role == AgentRole.PERSONAL_UTILITY:
                # Section 6.3: Update personal utility
                agent.update_personal_utility(state, action, payoff, alpha_pu_t, eta_J_t)
            
            elif agent.state.role == AgentRole.REPUTATION:
                # Section 6.4.5: Reputation agent gets social support from leader
                if agent.state.following is not None:
                    # [REP-6] Use current followed-agent reputation estimate s_i(k,t).
                    if self.config.use_numpy_fast_path and self._s_matrix is not None:
                        followed_rep_estimate = float(self._s_matrix[agent_id, agent.state.following])
                    else:
                        followed_rep_estimate = agent.state.reputation_estimates.get(
                            agent.state.following,
                            0.0,
                        )
                    agent.update_reputation_reward_estimate(followed_rep_estimate, eta_J_t)
            
            elif agent.state.role == AgentRole.STATUS:
                # Section 6.5: Status agent receives social support from followers
                if len(agent.state.followers) > 0:
                    agent.update_status_optimization(
                        state, action, social_support_sum, beta_status_t, eta_J_t
                    )
        
        # === PHASE 4: Updates for Active Participants (Section 6.4) ===

        if self.config.use_numpy_fast_path and self._v_matrix is not None and self._s_matrix is not None:
            observed_payoff_vector = np.array(
                [observed_payoffs[i] for i in range(self.config.num_agents)],
                dtype=float,
            )
            active_participant_array = np.array(
                [a.agent_id for a in active_participants],
                dtype=int,
            )
            self._phase4_updates_numpy_fast(observed_payoff_vector, active_participant_array, eta_v_t)

            for agent in active_participants:
                agent.update_actor_interaction_rate(alpha_rate_t)
        else:
            # (4.1)
            # [REP-4] Section 6.4.2 updates v_i(k,t) for every agent i each step
            # (active k use observed payoff, inactive k decay via zero payoff).
            delta_v_by_agent = {}
            for agent in self.agents:
                delta_v_by_agent[agent.agent_id] = agent.update_personal_benefit_estimates(
                    observed_payoffs, eta_v_t
                )

            # (4.2) snapshot
            snapshot_s = {}
            for k in range(self.config.num_agents):
                snapshot_s[k] = [
                    p.state.reputation_estimates.get(k, 0.0)
                    for p in active_participants
                ]
            avg_s = {k: float(np.mean(vals)) if len(vals) > 0 else 0.0 for k, vals in snapshot_s.items()}

            # (4.3) s_i(k,t+1) = avg_s[k] + delta_v_i(k)
            for agent in active_participants:
                deltas = delta_v_by_agent[agent.agent_id]
                for k in range(self.config.num_agents):
                    agent.state.reputation_estimates[k] = avg_s[k] + deltas.get(k, 0.0)

            for agent in active_participants:
                agent.identify_highest_reputation_agent()
                agent.update_actor_interaction_rate(alpha_rate_t)


        # [REP-3] No second gossip pass here.
        # Reputation averaging already happens in Phase 4; a second pass per step
        # would double-count gossip and distort convergence behavior.
        # === PHASE 5: Adoption of Leader Behavior (Section 6.4.5) ===
        
        for agent in self.agents:
            if agent.state.role == AgentRole.REPUTATION:
                agent.adopt_leader_behavior()
        
        # === PHASE 6: Periodic Role Updates (Section 7) ===

        if self._role_update_epochs:
            # Explicit epoch sequence s_n supplied by caller (e.g., 2000,3000,6000,...).
            while (
                self._next_role_update_epoch_idx < len(self._role_update_epochs)
                and self.time_step >= self._role_update_epochs[self._next_role_update_epoch_idx]
            ):
                self._update_roles_sequential()
                self.role_update_epoch += 1
                self._next_role_update_epoch_idx += 1
        else:
            if self.config.fixed_role_update_interval:
                # Fixed global epoch spacing (static schedule): s_n = n * T.
                current_interval = max(1, int(self.config.role_update_base_interval))
            else:
                # Increasing update interval: T_n → ∞ as per Assumption 6
                current_interval = max(
                    self.config.role_update_base_interval,
                    int(self.config.role_update_base_interval * (1.0 + self.role_update_epoch * 0.1))
                )

            if self.time_step >= self._next_role_update_time:
                self._update_roles_sequential()
                self.role_update_epoch += 1

                if self.config.fixed_role_update_interval:
                    next_interval = max(1, int(self.config.role_update_base_interval))
                else:
                    next_interval = max(
                        self.config.role_update_base_interval,
                        int(self.config.role_update_base_interval * (1.0 + self.role_update_epoch * 0.1))
                    )

                self._next_role_update_time += next_interval

            # if self.time_step % current_interval == 0:
            #     self._update_roles_sequential()  # Sequential procedure from Section 7
            #     self.role_update_epoch += 1
        
        # === PHASE 7: Tracking ===
        
        self._track_results(this_step_payoffs, len(active_actors), len(active_participants))
    
    def _update_roles_sequential(self, update_candidates=None):
        """
        Section 7: Sequential 3-step role update procedure
        
        This is critical: updates must occur in order to properly handle
        indirect follower relationships (Section 7.2).

        Args:
            update_candidates: Optional iterable of agent ids allowed to reevaluate
                roles in this call. If None, all agents reevaluate (synchronous mode).
                Used by asynchronous experiments where agents update on independent clocks.
        """
        if self.config.use_numpy_fast_path and self._s_matrix is not None:
            # Step-1 uses agent.state.reputation_estimates; sync dense cache before role logic.
            self._sync_s_matrix_to_state_dicts()
        
        # Initialize: copy current state
        P = set(i for i, a in enumerate(self.agents) if a.state.role == AgentRole.PERSONAL_UTILITY)
        R = set(i for i, a in enumerate(self.agents) if a.state.role == AgentRole.REPUTATION)
        S = set(i for i, a in enumerate(self.agents) if a.state.role == AgentRole.STATUS)
        
        # Maintain follower relationships during update
        followers = {i: set(self.agents[i].state.followers) for i in range(self.config.num_agents)}
        
        if update_candidates is None:
            updatable = set(range(self.config.num_agents))
        else:
            updatable = {int(i) for i in update_candidates if 0 <= int(i) < self.config.num_agents}
            if not updatable:
                return

        # === STEP 1: Reputation Optimization (Section 7.3) ===
        # Agents without followers decide if they want to follow someone
        
        # C(t) = agents without followers
        C = set(i for i in range(self.config.num_agents) if len(followers[i]) == 0)
        
        # Partition C into agents already following (C_r) and not following (C_pu)
        C_r = C & R  # Already following
        C_pu = C & P  # Not yet following
        
        # Process in random order to avoid bias
        update_order = list(C & updatable)
        np.random.shuffle(update_order)
        
        for i in update_order:
            agent = self.agents[i]
            
            # Determine effective threshold with hysteresis (Section 7.1.3)
            if i in C_r:
                B_i = self.config.B_F  # Continue following with lower threshold
            else:
                B_i = self.config.B_R  # Start following with higher threshold
            
            # Get estimated rewards (weighted by gamma for reputation)
            # [ROLE-1] For Step-1 role switching, use observed reputation signal
            # Ĵ^r_i ≈ s_i(L_i,t) = max reputation estimate.
            # estimated_reward_rep is only non-zero for current followers; for non-followers
            # the follow decision must use max_rep (the best candidate's s_i value) directly.
            max_rep = max(agent.state.reputation_estimates.values()) if agent.state.reputation_estimates else 0.0
            est_rep_weighted = self.config.gamma * max_rep  # γ·s_i(L_i,t) per Section 6.6 + 7.3
            est_pu = agent.state.estimated_reward_pu

            # Decision: should optimize reputation? (Section 7.3, line 1)
            # [ROLE-2] Section 7.3 uses a single condition:
            # γĴ^r_i > max{B_i, Ĵ^pu_i}. We intentionally do NOT add an extra
            # "and max_rep >= B_i" gate is not in the paper and blocks all following.
            if est_rep_weighted > max(B_i, est_pu):
                # Follow highest reputation agent
                best_k = agent.state.highest_rep_agent_estimate
                
                # Check if best_k is already a follower (Section 7.2)
                # [ROLE-3] If the selected target best_k is itself a follower,
                # redirect to best_k's leader so we avoid indirect follower chains.
                # Check if best_k is in R (already following), then follow
                # best_k's leader instead to prevent indirect follower chains.
                if best_k in R and self.agents[best_k].state.following is not None:
                    best_k = self.agents[best_k].state.following

                # [ROLE-4] After redirect, ensure we don't follow ourselves
                # (can happen if redirect chain points back to i)
                if best_k == i:
                    # Skip following - stay in personal utility instead
                    continue

                # Update role and follower relationships
                if i in R:
                    # Was already following, change to new leader
                    old_leader = agent.state.following
                    if old_leader is not None:
                        followers[old_leader].discard(i)

                # [ROLE-5] When agent i becomes a follower, redirect i's own followers
                # to i's new leader (best_k) to prevent multi-level chains.
                # This handles the case where i was previously a leader (e.g., from STATUS
                # role) and now switches to REPUTATION role following best_k.
                if len(followers[i]) > 0:
                    for follower_id in list(followers[i]):
                        self.agents[follower_id].state.following = best_k
                        followers[best_k].add(follower_id)
                    followers[i].clear()

                agent.state.role = AgentRole.REPUTATION
                agent.state.following = best_k
                followers[best_k].add(i)
                agent.state.was_following = True
                
                R.add(i)
                P.discard(i)
        
            else:
                # Existing follower no longer wants to keep following:
                # remove from R so Step 3 can send it back to PU
                if i in C_r:
                    old_leader = agent.state.following
                    if old_leader is not None:
                        followers[old_leader].discard(i)
                    agent.state.following = None
                    R.discard(i)

        # === STEP 2: Status Optimization (Section 7.4) ===
        # Agents with sufficient followers decide if they want to optimize status
        
        min_followers = max(1, int(self.config.c_threshold * self.config.num_agents))
        
        for i in updatable:
            if len(followers[i]) >= min_followers:
                agent = self.agents[i]
                
                # Decision: should optimize status?
                # [STATUS-1] This gate relies on estimated_reward_status being updated
                # before role assignment (handled in Phase 3 for follower-holding actors).
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
        
        for i in updatable:
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
        mode = str(self.config.tracking_mode).lower()
        if mode not in {"full", "light"}:
            raise ValueError(f"Unsupported tracking_mode='{self.config.tracking_mode}'. Use 'full' or 'light'.")

        # Core metrics used by experiment sweeps.
        followers = [len(a.state.followers) for a in self.agents]
        self.results['follower_counts'].append(followers)
        self.results['actor_counts'].append(num_actors)
        self.results['participant_counts'].append(num_participants)
        self.results['social_welfare'].append(sum(this_step_payoffs.values()))

        if mode == "light":
            return

        # Full diagnostics (more expensive for large-N long-horizon runs).
        all_weights = np.array([a.state.weights_pu.flatten() for a in self.agents])
        norm_variance = np.mean(np.var(all_weights, axis=0))
        self.results['norm_consensus'].append(norm_variance)

        utils = {
            i: np.mean(a.state.payoff_history) if a.state.payoff_history else 0.0
            for i, a in enumerate(self.agents)
        }
        self.results['expected_utilities'].append(utils)
        self.results['actor_rates'].append([a.state.actor_interaction_rate for a in self.agents])
        self.results['roles_history'].append([a.state.role for a in self.agents])
        self.results['actual_payoffs'].append(this_step_payoffs)
    
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


def test_pu_get_softmax_policy_sums_to_one_and_prefers_large_logit(model_module):
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=2, num_actions=3))
    a = system.agents[0]

    weights = np.array([
        [10.0, 0.0, -10.0],
        [0.0, 0.0, 0.0],
    ])

    probs = a.get_softmax_policy(state=0, weights=weights)

    assert probs.shape == (3,)
    assert np.sum(probs) == pytest.approx(1.0, abs=1e-12)
    assert probs[0] > probs[1] > probs[2]


def test_pu_select_action_uses_correct_state_row(model_module):
    np.random.seed(0)
    system = make_system(model_module, num_agents=2, extra_config=dict(num_states=2, num_actions=2))
    AgentRole = model_module.AgentRole

    a = system.agents[0]
    a.state.role = AgentRole.PERSONAL_UTILITY

    # state 0 strongly prefers action 0, state 1 strongly prefers action 1
    a.state.weights_pu = np.array([
        [15.0, -15.0],
        [-15.0, 15.0],
    ])

    draws_state0 = [a.select_action(state=0) for _ in range(20)]
    draws_state1 = [a.select_action(state=1) for _ in range(20)]

    assert all(x == 0 for x in draws_state0)
    assert all(x == 1 for x in draws_state1)