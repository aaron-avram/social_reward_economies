# Follow/Hysteresis Audit Report

Date: 2026-03-26

This audit reviewed the March 20 supervisor guidance against the current paper and synchronous implementation. Existing files under `doc/` were not edited.

## Sources Reviewed

- Supervisor intent: [ROP meeting transcript 3.20.txt](/Users/xia/social_reward_economies/doc/ROP%20meeting%20transcript%203.20.txt)
- Paper, authoritative notation: [learning_paper_newest_ver.pdf](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver.pdf)
- Paper, searchable text: [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2824
- Synchronous implementation: [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py):823
- Regression tests: [tests.py](/Users/xia/social_reward_economies/src/tests.py):118

## Paper Verdicts

1. Does an eligible agent always use the current highest-reputation agent at each update epoch?

- Evidence: the formal Step 1 says the agent follows the one it believes currently has the highest reputation and then sets `k_hat = L_i(s_n)`. See [learning_paper_newest_ver.pdf](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver.pdf) Section 7.3 Step 1 and [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2915.
- Verdict: Yes. The formal algorithm uses the current `L_i(s_n)` at each role-update epoch.

2. Does hysteresis only change `B_i` (`B_R` vs `B_F`)?

- Evidence: Section 7.1.3 introduces `B_R` and `B_F`, and Step 1 defines `B_i = B_F` for `C_r(s_n)` and `B_i = B_R` for `C_pu(s_n)` before using the current `L_i(s_n)`. See [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2824 and [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2928.
- Verdict: Yes in the formal step-by-step algorithm.

3. Can the paper be read as locking an agent to the previously followed leader?

- Evidence: Section 7.1.3 says that once an agent is a follower, it will continue following as long as the reputation of the agent with the highest reputation remains above `B_F`. See [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2827.
- Evidence: Step 1 later clarifies that the target is the current `L_i(s_n)`, not an earlier leader choice. See [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2934 and [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2998.
- Verdict: A reader could misread Section 7.1.3 in isolation as sticky-leader behavior. Read together with Step 1, the intended reading is not sticky. The paper is therefore mildly ambiguous and should be treated as needing clarification in a later revision.

4. Does chain-redirection rely on current-pass follower sets rather than stale `t-1` relations?

- Evidence: the overview says the procedure copies the current follower sets and then applies a sequential, in-place update in which each agent observes the most up-to-date follower sets. See [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2898.
- Evidence: Step 1 says that if `k_hat` is itself in `F_{k'}(s_n)`, then the agent follows `k'` instead. See [learning_paper_newest_ver_transcription.md](/Users/xia/social_reward_economies/doc/learning_paper_newest_ver_transcription.md):2999.
- Verdict: Yes. The paper intends current-pass chain redirection, not inherited stale leader identity.

## Code Verdicts and Actions Taken

1. Highest-reputation refresh before role updates

- The synchronous path already refreshes reputation estimates and recomputes `highest_rep_agent_estimate` before role updates. See [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py):823 and [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py):838.
- Verdict: Matches the intended synchronous algorithm.

2. Hysteresis logic

- The previous code applied `B_F` only when there was exactly one opinion leader. That extra condition was not supported by the March 20 guidance or the formal Step 1 in the paper.
- Action: removed the single-leader gate so hysteresis now depends only on whether the agent is already in `C_r`. See [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py):1016.

3. Threshold signal source

- The previous code compared against `max(reputation_estimates.values())`, which could include self even though target selection excludes self.
- Action: the follow threshold now uses the same selected non-self target `L_i(t)` that is used for the actual follow decision. See [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py):1029.

4. Chain redirection

- The code already maintained a copied follower graph and redirected away from follower targets during the same pass.
- Verdict: This part already matched the paper’s sequential current-pass logic. See [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py):924 and [code_debugged.py](/Users/xia/social_reward_economies/src/code_debugged.py):1061.

## Test Coverage Updated

- Repaired chain-redirection coverage so it explicitly sets the highest-reputation estimate before calling the role update. See [tests.py](/Users/xia/social_reward_economies/src/tests.py):118.
- Added a regression showing that `B_F` still applies when there are multiple current leaders. See [tests.py](/Users/xia/social_reward_economies/src/tests.py):189.
- Added a regression for switching from an old leader to the current highest-reputation leader. See [tests.py](/Users/xia/social_reward_economies/src/tests.py):219.
- Added a regression for current-pass chain redirection using updated same-pass follower state. See [tests.py](/Users/xia/social_reward_economies/src/tests.py):247.
- Added a regression showing that inflated self-reputation does not trigger false follow entry. See [tests.py](/Users/xia/social_reward_economies/src/tests.py):281.
- Corrected the status test fixture so it satisfies the current `ceil(cN)` threshold rule. See [tests.py](/Users/xia/social_reward_economies/src/tests.py):320.

## Verification

The updated suite passed with:

```bash
MPLCONFIGDIR=/tmp/mpl XDG_CACHE_HOME=/tmp python3 /Users/xia/social_reward_economies/src/tests.py
```

## Recommended Later Paper Clarification

No existing paper file was modified in this audit. If the paper is revised later, the wording should be made explicit that:

- at each role-update epoch, agent `i` compares `gamma * J_i^r(s_n)` against `max(B_i, J_i^pu(s_n))` using the current target `L_i(s_n)`;
- prior following status only determines whether `B_i = B_F` or `B_i = B_R`;
- if `L_i(s_n)` is itself a follower in the current sequential pass, the agent follows that target’s current leader instead.
