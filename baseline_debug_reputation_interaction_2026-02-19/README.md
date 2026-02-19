# Baseline Debug Package (Reputation + Interaction Rates)

This directory contains a focused debug package for `doc/code_old.py`, centered on:

- Reputation learning and role-switching logic
- Interaction-rate sampling and update dynamics

The tests are written in `pytest` style and were designed using:

- `doc/learning_paper_newest_ver_transcription.md` (Sections 6.2, 6.4, 6.7, 7.3)
- `doc/Social_Reward_Economies_transcription.md` (conceptual framing)
- Patterns from `single-reward-economy/reputation_tests/`

## Contents

- `test_reputation_behavior.py`: 4 reputation-focused tests
- `test_interaction_rates.py`: 4 interaction-rate-focused tests
- `test_existing_bug_comments.py`: 3 targeted tests for comment-labeled BUG 1/4/5
- `BUG_REPORT.md`: validated defect report with evidence and impact
- `test_run_2026-02-19.txt`: captured pytest output
- `test_bug_comments_run_2026-02-19.txt`: captured run for BUG 1/4/5 checks

## Run

```bash
cd /Users/xia/social_reward_economies
pytest -q baseline_debug_reputation_interaction_2026-02-19
```

## Current Status

The current baseline fails multiple spec-aligned tests (see `BUG_REPORT.md` and `test_run_2026-02-19.txt`).
