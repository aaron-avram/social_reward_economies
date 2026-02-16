# Quick Start Guide

## Run All Tests

```bash
cd single-reward-economy/reputation_tests

# Run all 4 tests in sequence (~4 minutes total)
python3 test_1_reputation_baseline.py && \
python3 test_2_differential_reputation.py && \
python3 test_3_gossip_simple.py && \
python3 test_4_influencer_gossip.py
```

## View Results

All test results are saved in `Results/` subdirectories:

```bash
# View Test 4 results (PRIMARY validation)
open Results/test_4_influencer_gossip/follower_dynamics.png
open Results/test_4_influencer_gossip/reputation_distribution.png
```

## Key Plots for PDF Comparison

**Compare these with PDF figures**:

1. **Follower Dynamics**: `Results/test_4_influencer_gossip/follower_dynamics.png`
   - Shows winner-takes-all emergence
   - Top influencer (Agent 19) reaches 19 followers

2. **Reputation Distribution**: `Results/test_4_influencer_gossip/reputation_distribution.png`
   - Shows gossip-driven convergence
   - High-reputation agents emerge as influencers

## Test Status

✓ **Test 1** - Reputation Baseline: PASSED
✓ **Test 2** - Differential Reputation: PASSED
✓ **Test 3** - Gossip Mechanism: PASSED
✓ **Test 4** - Influencer + Gossip: PASSED

## Expected Results

### Test 1
- PR → 1.000, R → 9.000, beta → 1.000
- Formula validation error: 0.000

### Test 2
- Policy π(a=1) → 1.000
- All agents learn optimal action

### Test 3
- Gossip convergence: variance → 0.000
- Perfect consensus on reputation values

### Test 4 (PRIMARY)
- Winner-takes-all: Agent 19 with 19 followers
- Gossip convergence: max divergence = 0.000
- System utility: 0.95 (95% of optimal)

## Troubleshooting

### If tests fail
1. Check Python version (requires 3.x)
2. Verify dependencies: `pip install numpy torch matplotlib`
3. Check random seed is set (34243)

### If results differ from PDF
1. Check hyperparameters match (gamma=20, kappa=0)
2. Verify gossip starts at t=50
3. Check influencer switching interval (3000 timesteps)

## Next Steps

1. ✓ Compare Test 4 plots with PDF figures
2. ⏸ Run reputation_scaling with Test 4 parameters
3. ⏸ Implement Test 5 (status optimization)

See `VALIDATION_REPORT.md` for detailed analysis.
