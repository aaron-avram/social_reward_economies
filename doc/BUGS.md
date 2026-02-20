# Bug Catalog (Canonical IDs)

This catalog uses grouped, stable IDs for the bugs originally identified in `src/code_old.py`.

## Canonical IDs

- `IR-1`: Active-set sampling must use `theta(mu)=1-exp(-mu)`
- `REP-1`: Highest-reputation selection must exclude self (`C\\{i}`)
- `REP-2`: Reputation update must follow Eq. (9) additive form (`avg + delta_v`)
- `REP-3`: Remove extra pairwise gossip pass in the same timestep
- `ROLE-1`: Non-followers must bootstrap reputation-role entry from observed reputation signal
- `ROLE-2`: Remove extra `max_rep >= B_i` gate from Step-1 follow condition
- `ROLE-3`: Redirect if selected follow target is itself a follower

Legacy mapping from old inline labels:

- old `Bug 1` -> `ROLE-1`
- old `Bug 2` -> `ROLE-2`
- old `Bug 3` -> `REP-2`
- old `Bug 4` -> `REP-3`
- old `Bug 5` -> `ROLE-3`

## By Group

### Interaction Rates

#### IR-1 (High)
Active actor/participant sampling used raw `mu` instead of `theta(mu)=1-exp(-mu)`.

### Reputation Learning

#### REP-1 (High)
Highest-reputation selection allowed self as candidate.

#### REP-2 (High)
Reputation update deviated from Eq. (9): implementation used EMA-style updates instead of `avg + delta_v`.

#### REP-3 (Medium)
A second pairwise gossip update occurred in Phase 5 after Phase 4 already applied gossip updates.

### Role Updates

#### ROLE-1 (High)
Step-1 follow decision used `estimated_reward_rep` for non-followers, preventing reputation-role bootstrap.

#### ROLE-2 (High)
Step-1 follow condition added extra gate `max_rep >= B_i`, which is not in Section 7.3.

#### ROLE-3 (Medium)
Indirect-follow redirect logic failed to redirect to the leader of a follower target.

## Note on Fixed Code

`src/code_debugged.py` uses these canonical IDs in inline comments (e.g., `[IR-1]`, `[REP-2]`, `[ROLE-3]`) so each fix point maps back to this catalog unambiguously.
