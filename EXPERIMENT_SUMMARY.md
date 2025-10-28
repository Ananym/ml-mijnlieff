# TicTacDo AI Training Experiments - Complete Summary

## Goal
Achieve 75% win rate against Strategic opponent using AlphaZero-style reinforcement learning.

## Key Discoveries

### 1. Correlation Bug (CRITICAL)
**Discovery**: The Pearson correlation metric was computing cosine similarity instead of proper correlation due to missing mean-centering terms.

**Impact**: Severely underestimated value head learning quality (reported 0.05-0.14 when actually ~0.3-0.4).

**Fix**: Implemented proper Pearson correlation formula:
```python
r = (n*Σxy - Σx*Σy) / (sqrt(n*Σx² - (Σx)²) * sqrt(n*Σy² - (Σy)²))
```

**Verification**: Exp 4 achieved 0.408 correlation (proving fix works), but this had WORST Strategic performance (37.5%).

**Conclusion**: **Value head quality is NOT the bottleneck** for beating Strategic opponent.

### 2. High Training Variance
**Discovery**: Exp 6 used identical configuration to Exp 1 but achieved 47.5% vs 60%.

**Implications**:
- Significant random variation in training outcomes
- Exp 1's 60% may have been fortunate initialization
- Consistent 40-50% ceiling across all experiments
- Small hyperparameter changes have unpredictable effects

---

## Experiment Results Table

| Exp | MCTS | Buffer Balance | Bootstrap | Strategic Curriculum | Policy Weight | Strategic Win Rate | Value Corr | Training Time | Status |
|-----|------|----------------|-----------|---------------------|---------------|-------------------|------------|---------------|--------|
| **1** | 600-1200 | ❌ | 0.0 | 70%→10% (30 iter) | 0.2 | **60%** | 0.14 (buggy) | ~10-12h | ✓ Best |
| **2** | 600-1200 | ✅ | 0.0 | 70%→10% (30 iter) | 0.05 | 40% | 0.03 | - | Policy starved |
| **3** | 600-1200 | ? | 0.8 | 0%→0% (pure self-play) | 0.2 | 50% | ? | - | Bootstrap hurt |
| **4** | 800-1600 | ✅ | 0.2 | 80%→20% (40 iter) | 0.2 | 37.5% | **0.408** | 14h40m | ✗ Worst |
| **5** | 800-1600 | ✅ | 0.0 | 70%→10% (30 iter) | 0.2 | 42.5% | 0.358 | 14h40m | Disappointing |
| **6** | 600-1200 | ❌ | 0.0 | 70%→10% (30 iter) | 0.2 | 47.5% | 0.342 | 15h36m | Reversion failed |

---

## Detailed Experiment Analysis

### Experiment 1 - Original Baseline (60% Strategic)
**Configuration:**
- MCTS: 600-1200 simulations
- Buffer balancing: Disabled (natural ~62% draws)
- Bootstrap: 0.0 (pure game outcomes)
- Strategic curriculum: 70%→10% over 30 iterations
- Policy weight: 0.2 adaptive
- Correlation metric: BUGGY (cosine similarity)

**Results:**
- **Strategic win rate: 60%** (best result)
- Value correlation: 0.14 (buggy, likely ~0.3 actual)
- Draw rates: High (~62% in buffer)
- Training time: ~10-12 hours (estimated)

**Why it worked:**
- Balanced strategic opponent exposure (70%→10%)
- Pure game outcomes (no MCTS value noise)
- Adaptive policy weighting
- Natural draw distribution (no forced balancing)

**Limitations:**
- Correlation metric was broken (monitoring only)
- Cannot reproduce results (Exp 6 tried and failed)
- Unclear if 60% was luck or optimal config

---

### Experiment 2 - Policy Starvation (40% Strategic)
**Configuration:**
- MCTS: 600-1200 simulations
- Buffer balancing: Enabled (20% draw cap)
- Bootstrap: 0.0
- Strategic curriculum: 70%→10% over 30 iterations
- Policy weight: **0.05** (too low!)
- Draw penalty: -0.6 (aggressive)

**Results:**
- Strategic win rate: 40%
- Value correlation: 0.03 (very poor)
- Policy loss dominated by value loss

**Why it failed:**
- **Policy weight too low (0.05)** - policy network didn't get enough gradient signal
- Extreme draw penalty may have distorted learning
- Buffer balancing might have created distribution mismatch

**Lessons:**
- Policy network needs substantial gradient signal (0.2 minimum)
- Extreme penalties can hurt more than help

---

### Experiment 3 - Pure Self-Play with Bootstrapping (50% Strategic)
**Configuration:**
- MCTS: 600-1200 simulations
- Buffer balancing: Unknown
- Bootstrap: **0.8** (heavy MCTS value weighting)
- Strategic curriculum: **0%→0%** (pure self-play, no Strategic opponent)
- Policy weight: 0.2

**Results:**
- Strategic win rate: 50%
- Insufficient strategic opponent exposure during training

**Why it failed:**
- **No Strategic opponent during training** - model had no benchmark
- Heavy bootstrapping (0.8) - learned from noisy MCTS estimates
- Pure self-play can lead to local optima

**Lessons:**
- Strategic curriculum is valuable (need opponent as benchmark)
- Bootstrap weight 0.8 too high - prefer pure game outcomes (0.0)

---

### Experiment 4 - Too Much Strategic Opponent (37.5% Strategic)
**Configuration:**
- MCTS: 800-1600 simulations (stronger)
- Buffer balancing: Enabled (20% draw cap)
- Bootstrap: 0.2 (light)
- Strategic curriculum: **80%→20% over 40 iterations** (very heavy)
- Policy weight: 0.2 adaptive
- Correlation metric: **FIXED**

**Results:**
- Strategic win rate: **37.5% (WORST)**
- Value correlation: **0.408 (BEST)** - proves fix works
- Draw rates: 42-61% (very high despite balancing)
- P1/P2 asymmetry: 15% vs 24% (9% gap)

**Why it failed:**
- **Too much Strategic curriculum (80%→20%)** - prevented self-play exploration
- Model learned defensive/draw-heavy patterns instead of winning strategies
- Distribution mismatch: trained on forced 20% draws, evaluated on 42-61% draws

**Critical Insight:**
- **Best value correlation (0.408) produced WORST Strategic win rate (37.5%)**
- **Value head quality is NOT the bottleneck** for beating Strategic
- Policy network is what beats Strategic, largely independent of value quality

**Lessons:**
- More strategic opponent ≠ better (80%→20% too heavy)
- Need balance between benchmark (Strategic) and exploration (self-play)
- Value head metrics don't predict policy performance

---

### Experiment 5 - Optimized Attempt (42.5% Strategic)
**Configuration:**
- MCTS: 800-1600 simulations (stronger than Exp 1)
- Buffer balancing: Enabled (20% draw cap)
- Bootstrap: 0.0 (reverted to Exp 1)
- Strategic curriculum: 70%→10% over 30 iterations (reverted to Exp 1)
- Policy weight: 0.2 adaptive (same as Exp 1)
- Correlation metric: Fixed

**Results:**
- Strategic win rate: 42.5%
- Value correlation: 0.358 peak → 0.142 final (value head collapsed)
- Draw rates: ended at 56% (very high)
- Training time: 14h40m

**Why it failed:**
- **Stronger MCTS (800-1600) = fewer iterations** in same time
- Fewer diverse positions seen, less exploration
- Value head collapsed after early peak

**Hypothesis tested:**
Should combine Exp 1's working baseline with improvements (stronger MCTS, buffer balancing). **FAILED - performed worse than Exp 1 (42.5% vs 60%).**

**Lessons:**
- Stronger MCTS may hurt training efficiency (fewer iterations per hour)
- Value head collapse is persistent problem
- "Improvements" can make things worse

---

### Experiment 6 - Systematic Reversion (47.5% Strategic)
**Configuration:**
- MCTS: 600-1200 simulations (reverted to Exp 1)
- Buffer balancing: **Disabled** (reverted to Exp 1)
- Bootstrap: 0.0 (same as Exp 1)
- Strategic curriculum: 70%→10% over 30 iterations (same as Exp 1)
- Policy weight: 0.2 adaptive (same as Exp 1)
- Correlation metric: Fixed (only difference from Exp 1)

**Results:**
- Strategic win rate: **47.5%**
- Value correlation: 0.342 peak, 0.245 final
- Draw rates: ended at 48% (healthier than Exp 5)
- Buffer reached natural 49% draws
- Training time: 15h36m

**CRITICAL FINDING:**
**Exp 6 configuration is IDENTICAL to Exp 1, but achieved 47.5% vs 60%!**

**Why the hypothesis was wrong:**
- Stronger MCTS (800-1600) was NOT the problem
- Buffer balancing was NOT the problem
- Exp 1's 60% may have been **lucky initialization/random variance**

**Implications:**
- **High training variance** - same config can yield 47.5% to 60%
- All experiments after Exp 1 consistently underperform (37.5% - 50%)
- 40-50% may be more realistic ceiling with current architecture
- Random seed/initialization heavily influences outcome

**Lessons:**
- Need multiple runs per configuration to account for variance
- Single "best result" may not be reproducible
- Current approach may be hitting fundamental ceiling

---

## Pattern Analysis

### What Doesn't Help (Tested & Failed)
1. ❌ **Stronger MCTS (800-1600)** - slightly worse, slower training (Exp 5 vs Exp 6)
2. ❌ **Buffer balancing (20% draw cap)** - no clear benefit (Exp 5 vs Exp 6)
3. ❌ **Heavy Strategic curriculum (80%→20%)** - much worse, prevents exploration (Exp 4)
4. ❌ **Bootstrapping (0.2-0.8)** - hurts learning from true outcomes (Exp 3, Exp 4)
5. ❌ **Low policy weight (0.05)** - policy network gets starved (Exp 2)
6. ❌ **Pure self-play (0% Strategic)** - no benchmark, worse performance (Exp 3)
7. ❌ **Aggressive draw penalty (-0.6)** - distorts learning (Exp 2)

### What Might Help (Baseline from Exp 1)
1. ✓ **Moderate Strategic curriculum (70%→10%)** - balanced benchmark + exploration
2. ✓ **Pure game outcomes (bootstrap=0.0)** - learn from truth, not noisy MCTS
3. ✓ **Adaptive policy weighting (0.2)** - gives policy sufficient gradient signal
4. ✓ **Moderate MCTS (600-1200)** - balance between quality and iteration speed
5. ⚠️ **Natural draw distribution** - worked in Exp 1, but unclear if necessary

### Surprising Findings
1. **Best value correlation ≠ Best Strategic performance** (Exp 4: 0.408 corr, 37.5% win)
2. **Identical configs yield different results** (Exp 1: 60%, Exp 6: 47.5%)
3. **All "improvements" made things worse** (37.5% - 50% vs Exp 1's 60%)
4. **High random variance** - suggests current approach hitting ceiling

---

## Remaining Questions

### 1. Was Exp 1's 60% Reproducible?
**Answer**: NO - Exp 6 used identical config and got 47.5%

**Implication**: High variance in training, need multiple runs to assess true performance

### 2. What Is the True Ceiling?
**Current evidence**: 40-50% with high variance, 60% as lucky outlier

**Needs**: Multiple runs of same config to establish confidence intervals

### 3. Why Do "Improvements" Make Things Worse?
**Possible reasons:**
- Fragile training dynamics - small changes disrupt learning
- Overfitting to Exp 1's lucky run
- Fundamental architectural limitations
- Complex interaction effects between hyperparameters

### 4. How To Break Past 60%?
**Options to explore:**
1. **Architectural changes:**
   - Deeper/wider network
   - Attention mechanisms
   - Separate policy/value networks

2. **Training paradigm shifts:**
   - Two-phase training (policy first, then value)
   - Curriculum learning (easy → hard opponents)
   - Ensemble methods

3. **Better exploration:**
   - Temperature scheduling
   - Diversity bonuses
   - Novelty search

4. **Data quality:**
   - Filter low-quality games
   - Prioritized experience replay
   - Data augmentation (symmetries, rotations)

---

## Recommendations

### Short Term: Establish Baseline Variance
1. **Run Exp 1 config 3 more times** to establish variance bounds
2. **Track all hyperparameters** including random seed
3. **Compute confidence intervals** for "true" performance

### Medium Term: Incremental Testing
If 60% is reproducible, try **ONE change at a time**:
1. Policy weight: 0.2 → 0.25 (slightly more policy gradient)
2. Iterations: 50 → 75 (more training time)
3. Temperature scheduling (lower temp at eval time)

If 60% was lucky (likely), accept 45-50% baseline and try:
1. **Larger model** (2x parameters)
2. **Different optimizer** (Adam → AdamW, learning rate search)
3. **Curriculum opponents** (Random → Easy → Strategic)

### Long Term: Architectural Innovation
Current approach may be hitting ceiling. Consider:
1. **Two-phase training:**
   - Phase 1 (iter 1-35): Train policy only (freeze value head)
   - Phase 2 (iter 36-50): Train value head only (freeze policy)

2. **Multi-opponent training:**
   - Strategic, Random, and self simultaneously
   - Learn to adapt to different play styles

3. **Search-time improvements:**
   - AlphaZero used 1600 sims at evaluation (we use 800-1600 during training but may need more at eval)
   - Temperature tuning at search time
   - Bias toward aggressive/winning moves vs draws

---

## File Changes Log

### train.py
**Correlation Bug Fix (Lines 1467-1490):**
- Added tracking for `value_actual_sum` and `value_pred_sum_for_corr`
- Rewrote formula to proper Pearson correlation with mean-centering
- Added correlation tracking in training loop (lines 520-525, 1120-1132, 1161-1174)

**Hyperparameter changes across experiments:**
- MCTS sims: 600-1200 → 800-1600 → 600-1200
- Buffer balancing: disabled → enabled → disabled
- Bootstrap weight: 0.0 → 0.2 → 0.0
- Strategic curriculum: 70%→10% → 80%→20% → 70%→10%
- Policy weight: 0.2 → 0.05 → 0.2

### debug_tools/test_perspective_bug.py
**Created**: Complete diagnostic tool with 4 tests for player perspective bugs

**Tests:**
1. Symmetric position test
2. Player bias test (mean/std analysis)
3. MCTS perspective test
4. Absolute vs subjective encoding test

**Result**: 4/4 tests passed - no perspective bugs found

### Documentation Files Created
1. **CORRELATION_BUG_INVESTIGATION.md** - Detailed bug analysis and fix
2. **DRAW_DEATH_SPIRAL_FIX.md** - Draw bias analysis
3. **VALUE_BOOTSTRAPPING_FIX.md** - Bootstrapping experiments
4. **EXPERIMENT_SUMMARY.md** (this file) - Complete experimental record

---

## Conclusion

After 6 comprehensive experiments:
1. **Achieved**: 60% Strategic win rate (Exp 1), though not reproducible
2. **Discovered**: Critical correlation bug, value head ≠ Strategic performance
3. **Learned**: High variance, fragile training dynamics, possible architectural ceiling
4. **Status**: 40-50% appears more realistic with current approach

**Next Steps**: Establish baseline variance with multiple runs, then consider architectural changes if ceiling confirmed.

---

## Experiment Quick Reference

```
Exp 1: 60%  - Original lucky run ✓ BEST (not reproducible)
Exp 2: 40%  - Policy weight too low (0.05)
Exp 3: 50%  - Heavy bootstrap (0.8), no Strategic
Exp 4: 37.5% - Too much Strategic (80%→20%) ✗ WORST
Exp 5: 42.5% - Stronger MCTS, buffer balance (overcorrection)
Exp 6: 47.5% - Exact reversion to Exp 1 (proves high variance)
```

**Key Insight**: Value head quality (0.408 corr) ≠ Strategic performance (37.5% win). Policy network is the bottleneck.
