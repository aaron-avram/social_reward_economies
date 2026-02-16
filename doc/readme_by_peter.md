
### Example 1: Run with Default Configuration
```python
from corrected_sections_6_7 import SystemConfig, MultiAgentSystem

config = SystemConfig()
system = MultiAgentSystem(config)
results = system.simulate()
```

### Example 2: Custom Parameters
```python
config = SystemConfig(
    num_agents=8,
    num_time_steps=5000,
    M=1.5,
    u_0=0.2,
    gamma=3.0,      # Stronger reputation incentive
    kappa=2.5,      # Stronger status incentive
    c_threshold=0.2,  # 20% follower threshold
    delta=0.2,      # More tolerance for tie-breaking
)

system = MultiAgentSystem(config)
results = system.simulate()
system.plot_results("custom_results.png")
```

### Example 3: Access Results
```python
# Final state
print(f"Opinion leader: Agent {results['opinion_leader']}")
print(f"Final followers: {results['final_followers']}")
print(f"Final roles: {results['final_roles']}")

# Dynamics
import numpy as np
print(f"Mean norm consensus: {np.mean(results['norm_consensus'])}")
print(f"Mean social welfare: {np.mean(results['social_welfare'])}")
```

---

## 📊 Output Structure

The `results` dictionary contains:
```python
{
    'norm_consensus': [...],        # Policy weight variance over time
    'expected_utilities': [...],    # Per-agent expected payoffs
    'follower_counts': [...],       # Follower counts by timestep
    'actor_counts': [...],          # |A_a(t)| over time
    'participant_counts': [...],    # |A_p(t)| over time
    'actor_rates': [...],           # Learned μ_{a,i}(t)
    'roles_history': [...],         # Role sequence for each agent
    'actual_payoffs': [...],        # Step-by-step payoffs
    'social_welfare': [...],        # Total system reward
    'final_roles': [...],           # Final role assignment
    'final_followers': [...],       # Final follower counts
    'opinion_leader': int,          # Agent with most followers
}
```
