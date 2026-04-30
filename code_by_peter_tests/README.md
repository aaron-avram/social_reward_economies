# code_by_peter Test Suite

Validates all bug fixes implemented in `src/code_debugged.py` relative to `src/code_old.py`.

## Files

- `test_code_debugged.py` — 187 tests across 5 sections (see below)
- `_shared.py` — shared module loader used by all tests

## Coverage

All 14 bugs documented in `doc/BUGS.md` are covered:

| Section | Bugs covered |
|---|---|
| Gossip Oracles and Reputation Learning | REP-1, REP-3, REP-4, REP-5 |
| Interaction Rates, Status, and Role Switching | IR-1, REP-1, REP-2, REP-4, REP-5, REP-6, REP-7, ROLE-1–5, STATUS-1 |
| Async Role Switching and Scheduler | ROLE-2, ROLE-3, ROLE-4, REP-6 |
| Perturbation and Recovery | experiment D harness; config parsing; seed selection |
| Additional Gossip, Role, and Estimate Tests | REP-1, REP-4, REP-5, ROLE-1–5, STATUS-1 |

## Run

```bash
pytest -q code_by_peter_tests/
```
