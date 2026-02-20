# code_by_peter Test Suite

This suite validates bug-report fixes implemented in:

- `/Users/xia/social_reward_economies/src/code_debugged.py`

## Coverage

- Active-set sampling via `theta(mu)=1-exp(-mu)` for actors and participants
- Reputation leader selection excludes self
- Reputation update follows Eq. (9) additive form (`avg + delta_v`)
- Step-1 role-switch criterion (no extra `max_rep >= B_i` gate)
- Bug 1 bootstrap behavior (non-followers can switch based on observed reputation)
- Bug 5 redirect behavior (avoid indirect follower chains)
- Bug 4 removal (no extra Phase-5 pairwise gossip pass)

## Run

```bash
cd /Users/xia/social_reward_economies
pytest -q code_by_peter_tests
```
