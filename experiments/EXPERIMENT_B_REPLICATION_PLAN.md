# Experiment B Replication Plan (Reputation Scaling)

## Goal
Use Experiment B (`gamma > 0`, `kappa = 0`) as the bridge to reproduce the **Reputation Scaling** findings in:

- `/Users/xia/Downloads/Norm__Working_Copy__2_-16.pdf` (Section 5)

and align with Alex's implementation context in:

- `/Users/xia/social_reward_economies/single-reward-economy/norm/train.py`

## Current Status
- B can run independently from A.
- Latest B run (`N=8`, `gamma=2`, `kappa=0`) produced:
  - 1 opinion leader with 7 followers
  - roles: 1 PU, 7 Reputation, 0 Status
  - plot written to `/Users/xia/social_reward_economies/experiments/outputs/exp_B_reputation_only.png`
- This is qualitatively consistent with "high-`gamma` reputation-only consolidation."

## Target Outputs to Replicate (PDF Section 5)
1. **Gamma sweep** with `kappa = 0`, `N=100`:
   - `gamma` in `{0, 1, 1.25, 1.5, 1.75, 2, 3, 5}`
2. Two switching variants:
   - static
   - asynchronous
3. Outputs:
   - table of top follower counts vs gamma (Table 1/2 style)
   - follower progression graphs with leader-switch blips (Figure 3/4 style)
   - text summary of phase-transition point

## Execution Plan

### Phase 1: Lock Down B Verification (done enough to proceed)
Purpose: verify B behavior and capture reproducible baseline.

Tasks:
- Run B-only command and archive output logs.
- Record final leader/follower count and welfare summary.

Acceptance:
- B runs without dependency on A.
- Plot artifact is generated in `experiments/outputs`.

### Phase 2: Build Dedicated Reputation-Scaling Harness (next)
Purpose: make Section 5 runs reproducible and scriptable.

Deliverable file:
- `/Users/xia/social_reward_economies/experiments/reputation_scaling.py`

Required CLI options:
- `--mode {static,async}`
- `--gammas "0,1,1.25,1.5,1.75,2,3,5"`
- `--num-agents 100`
- `--num-states <int>`
- `--num-actions <int>`
- `--num-steps <int>`
- `--seeds <int>`
- `--kappa 0`
- `--output-dir <path>`

Required outputs:
- per-run CSV: one row per `(mode, gamma, seed)` with:
  - final top follower count
  - leader id
  - time-to-90%-followers (if reached)
  - number of leader switches
  - tail welfare
- aggregate CSV:
  - mean, std, CI per `(mode, gamma)`
- plots:
  - follower progression (top agent) by gamma
  - final top-follower vs gamma curve

Acceptance:
- One command runs full gamma sweep and writes all artifacts.

### Phase 3: Implement/Validate Static vs Async Update Modes
Purpose: match PDF's static/asynchronous comparisons.

Tasks:
- Add mode toggle in experiment runner logic:
  - static: synchronized global update intervals
  - async: agent-level independent update timing
- Keep all other parameters fixed when comparing modes.

Acceptance:
- Same gamma sweep runs in both modes, with only `--mode` changed.

### Phase 4: Scale to 100 Agents + Expanded Social State Space
Purpose: reproduce reputation scaling at target size.

Tasks:
- Run sweeps at `N=100`.
- Increase states/actions beyond toy defaults as required by study design.
- Use at least 20 seeds per gamma per mode for robust summary.

Acceptance:
- Tables and curves show expected monotonic/fase-transition behavior.
- Near-full consolidation at high gamma (around 3+) appears consistently.

### Phase 5: Add More Complex Reward Functions
Purpose: move from baseline B to richer environments while preserving comparability.

Tasks:
- Introduce reward model abstraction in experiment pipeline:
  - baseline utility (current)
  - unimodal variance model
  - bimodal/polarized model
- Keep same metrics/reporting format.

Acceptance:
- Same harness executes all reward models with comparable outputs.

### Phase 6: Replicate Figures and Findings
Purpose: produce supervisor-ready replication package.

Deliverables:
- `reputation_scaling_static.csv`
- `reputation_scaling_async.csv`
- figure set (Table 1/2 and Figure 3/4 analogs)
- short memo: matched/partial/unmatched findings with explanations

Acceptance:
- Reported trends and thresholds are quantitatively close to PDF findings.

## Immediate Next Actions
1. Implement Phase 2 harness (`reputation_scaling.py`).
2. Then execute Phase 3 and Phase 4 together (mode validation + `N=100` sweeps).

## Notes on Alex Code Alignment
Reference implementation context is in:
- `/Users/xia/social_reward_economies/single-reward-economy/norm/train.py`

Important Alex-side defaults observed:
- `N=100`
- long horizons (`timesteps ~ 50000`)
- reputation/status factors (`gamma`, `kappa`) and threshold knobs
- async/static labeling logic

Use these as calibration points, but keep your replication harness in `experiments/` to avoid coupling to legacy code paths.
