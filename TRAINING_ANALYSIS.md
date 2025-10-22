# Training Analysis: 100-Iteration Run with policy_weight=0.5

## Key Metrics Over Time

### Strategic Win Rate (tested every 10 iterations)
- **Iter 30:** 45.0% (baseline from previous run: 32.5%)
- **Iter 40:** 52.5% ← **PLATEAU BEGINS**
- **Iter 50:** 52.5% (no improvement)
- **Iter 60:** 47.5% (regression)
- **Iter 70:** 52.5% (recovery)
- **Iter 80:** 55.0%
- **Iter 90:** 57.5% (peak)
- **Iter 100:** 55.0% (slight drop)

**Gain from 40→100:** Only +2.5% improvement (52.5% → 55%)

### Value Correlation
- **Range:** 0.05 - 0.33 (highly unstable)
- **Common range:** 0.2 - 0.3
- **Peak moments:** Iter 7 (0.33), Iter 25 (0.33), Iter 62 (0.31), Iter 87 (0.33)
- **Final:** 0.262
- **Trend:** NO consistent improvement - oscillates throughout

### Loss Progression
- **Iter 1:** 1.75
- **Iter 30:** 0.64
- **Iter 50:** 0.60
- **Iter 100:** 0.46

Steady decrease, but doesn't translate to value learning.

## Diminishing Returns Analysis

### Time Investment vs Gain
- **Iterations 1-40:** 6 hours → 52.5% Strategic win rate
- **Iterations 40-100:** 9.8 hours → +2.5% gain (55% final)

**Efficiency:**
- First 40 iters: 8.75% gain per hour
- Last 60 iters: 0.26% gain per hour
- **33x less efficient after iter 40!**

### The Value Head Problem
Value correlation **never improves beyond 0.3** regardless of training time:
- Early training (1-30): oscillates 0.05-0.33
- Mid training (30-60): oscillates 0.1-0.32
- Late training (60-100): oscillates 0.1-0.33

**This indicates policy_weight=0.5 is still too high.**

## Recommended Training Duration

### Sweet Spot: **Iteration 40-50**

**Why stop at 40-50:**
1. Strategic win rate plateaus at ~52.5%
2. Value correlation shows no upward trend
3. 60% time savings (6h vs 15h)
4. Minimal performance loss (52.5% vs 55% = -2.5%)

### Early Stopping Criteria

Instead of just checking Strategic win rate, add:

```python
# Stop if value correlation hasn't improved in 20 iterations
if iteration >= 40:
    recent_correlations = last_20_iterations_correlations
    if max(recent_correlations) < 0.35:
        print("Value head not learning - adjust policy_weight")
        stop_and_restart_with_lower_policy_weight()
```

## Recommendations

### For Next Run (policy_weight=0.2)

1. **Max iterations: 50** (not 100)
   - Evaluate at iter 30, 40, 50
   - Stop early if plateau detected

2. **Value correlation target: 0.4+**
   - If not reached by iter 40, policy_weight still too high
   - Try 0.1 if 0.2 doesn't work

3. **Plateau detection:**
   ```
   if (iter >= 40 and
       win_rate_change_last_20_iters < 5% and
       correlation_max_last_20_iters < 0.35):
       stop_early()
   ```

### Time Savings

- Old approach: 100 iterations = ~16 hours
- New approach: 40-50 iterations = ~6-8 hours
- **Savings: 8-10 hours per run** (50-60% reduction)

## Summary

The 100-iteration run revealed:
- ✓ Strategic performance plateaus by iteration 40-50
- ✗ Value correlation never improves (stuck at 0.2-0.3)
- ✗ Diminishing returns: 33x less efficient after iter 40
- ✗ policy_weight=0.5 still too high for value head learning

**Action:** For policy_weight=0.2 run, use **max 50 iterations** with early stopping based on value correlation plateau.
