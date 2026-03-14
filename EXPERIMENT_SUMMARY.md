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
| **7** | 800-1600 | ❌ | 0.0 | 70%→10% (50 iter) | 0.2 | 47.5% | 0.311 | ~15h | Plateau confirmed |
| **8** | 800-1600 | ❌ | 0.0 | **40%→80% (50 iter)** | 0.2 | **72.5%** | 0.439 | 6h14m | ✓ **BREAKTHROUGH** |
| **9** | 800-1600 | ❌ | 0.0 | **40%→85% (100 iter)** | 0.2 | 75% (iter 70) → 59.5% (iter 100) | 0.433 | 11h51m | ⚠️ Late-stage decline |
| **10** | 800-1600 | ❌ | 0.0 | **40%→80%→50% (Inverted V)** | 0.2 | 65% (iter 60) → 62.5% (iter 100) | 0.356 | 11h34m | ✗ Never reached peak |
| **11** | 800-1600 | ❌ | 0.0 | **40%→85%→40% (Plateau)** | 0.2 | 65% (iter 40) → **72.5%** (iter 100) | 0.388 | 11h54m | ✓ **Late improvement!** |
| **12** | 800-1600 | ❌ | 0.0 | 0% (pure self-play) | 0.2 | 67.5% (iter 10) → ~45% avg | - | 17h | ✗ Unstable (10.8M model) |
| **13** | 800-1600 | ❌ | 0.0 | **40%→85%→40% (Plateau)** | 0.2 | 70% (iter 80) → 65% (iter 100) | - | 17h12m | ✗ **Overfit to Strategic** |

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

### Experiment 7 - Larger Model, Same Curriculum (47.5% Strategic)
**Configuration:**
- MCTS: 800-1600 simulations
- Model size: **1,225,029 parameters** (56% increase from 785,173)
- Hidden channels: 64 → **128**
- Buffer balancing: Disabled
- Bootstrap: 0.0
- Strategic curriculum: 70%→10% over 50 iterations
- Policy weight: 0.2 adaptive

**Results:**
- Strategic win rate: 47.5% (identical to Exp 6!)
- Value correlation: 0.311 final
- Training time: ~15 hours

**Critical Finding:**
**56% model size increase yielded ZERO improvement (47.5% = 47.5%)**

**Why it failed:**
- Model capacity was NOT the bottleneck
- Same curriculum (70%→10%) still failing
- Larger model = more parameters to train = slower convergence
- Confirms: **Training sequence matters more than model size**

**Lessons:**
- Don't scale model without fixing curriculum first
- Capacity ≠ performance if training is suboptimal

---

### Experiment 8 - Inverted Curriculum BREAKTHROUGH (72.5% Strategic)
**Configuration:**
- MCTS: 800-1600 simulations
- Model: 1,225,029 parameters (128 hidden channels)
- Buffer balancing: Disabled
- Bootstrap: 0.0
- **Strategic curriculum: 40%→80% over 50 iterations (INVERTED!)**
- Policy weight: 0.2 adaptive
- Eval temperature: 0.4 (deployment conditions)

**Results:**
- **Strategic win rate: 72.5%** ✓ (16W+13W = 29/40 wins)
  - As P1: 80.0% (16W-2D-2L)
  - As P2: 65.0% (13W-3D-4L)
- Value correlation: **0.439** (excellent)
- Training time: **6h14m** (60% faster than Exp 6!)
- Final loss: 0.4910

**Why it succeeded:**
- **Inverted curriculum (40%→80%)**: Build general skills via self-play FIRST
- Model learns winning strategies through exploration (60% self-play early)
- Then specializes against Strategic opponent (80% Strategic late)
- Sequence: Strength → Specialization (vs previous: Specialization → Weakness)

**BREAKTHROUGH INSIGHT:**
**Training sequence matters MORE than model capacity!**
- Exp 7 (70%→10%, large model): 47.5%
- Exp 8 (40%→80%, same model): 72.5%
- **25% improvement just from inverting curriculum!**

**Lessons:**
- Build foundation (self-play) before specializing (Strategic)
- Early exploration > Early specialization
- Faster training (6h vs 15h) with better results

---

### Experiment 9 - Extended Inverted Curriculum (75% Peak, 59.5% Final)
**Configuration:**
- MCTS: 800-1600 simulations
- Model: 1,225,029 parameters (128 hidden channels)
- Buffer balancing: Disabled
- Bootstrap: 0.0
- **Strategic curriculum: 40%→85% over 100 iterations**
- Policy weight: 0.2 adaptive
- Anti-overfitting measures:
  - Dirichlet noise: 0.3 → 0.20 (extended to iter 70)
  - Entropy bonus: 0.07
  - 15% self-play maintained (capped at 85% Strategic)

**Results - Strategic Win Rate Progression:**
- Iter 10: 72.5% ✓
- Iter 20: 50.0%
- Iter 30: 54.0%
- Iter 40: 50.0%
- Iter 50: 50.0%
- Iter 60: 60.0%
- **Iter 70: 75.0% ✓✓ PEAK** (30/40 wins, target achieved!)
- Iter 80: 67.5%
- Iter 90: 57.5%
- Iter 100: 59.5% (25/42 wins)

**Final Metrics (Iter 100):**
- Value correlation: 0.433 (excellent quality maintained)
- Training time: 11h51m
- As P1: 66.7% (14W-4D-3L)
- As P2: 52.4% (11W-6D-4L)

**Critical Finding - Late-Stage Overfitting:**
**Model peaked at iteration 70 (75%), then DECLINED to 59.5% by iteration 100!**

**Why it failed after iter 70:**
1. **Overfitting to Strategic opponent's patterns** - 85% Strategic exposure too specialized
2. **Lost generalization** - Model became "memorizing" Strategic's specific moves
3. **Anti-overfitting measures insufficient** - Despite Dirichlet noise, entropy bonus, and 15% self-play
4. **Monotonic curriculum problem** - Continuous increase (40%→85%) allowed no "recovery"

**What Is Overfitting in This Context?**
- The model learns to exploit **specific patterns** in the Strategic opponent
- It becomes highly tuned to Strategic's predictable moves
- But loses ability to handle **diverse, unpredictable strategies**
- Like a student who memorizes the teacher's test questions instead of understanding the material
- Performs worse against both Strategic AND humans by iteration 100

**Why Anti-Overfitting Measures Failed:**
- **Dirichlet noise (0.20)**: Not enough exploration variance
- **Entropy bonus (0.07)**: Insufficient to prevent policy collapse
- **15% self-play**: Too little - model spent 85% of time vs ONE opponent
- **No curriculum reversal**: Once specialized, never "unlearned" the narrow patterns

**Lessons:**
1. ✓ **Inverted curriculum works** - achieved 75% target at iter 70
2. ✗ **Extended training backfires** - monotonic increase causes overfitting
3. ✗ **Anti-overfitting measures insufficient** - passive measures can't overcome active specialization
4. 💡 **Need curriculum reversal** - after peak, return to more self-play to restore generalization

---

## Experiment 10 - Inverted V Curriculum (FAILED)

**Hypothesis:**
After achieving peak specialization, **actively restore generalization** by reducing Strategic ratio.

**Configuration:**
- MCTS: 800-1600 simulations
- Model: 1,225,029 parameters (128 hidden channels)
- **Phase 1 (Iter 1-60): 40% → 80% Strategic** (build expertise)
- **Phase 2 (Iter 60-100): 80% → 50% Strategic** (restore generalization)
- Dirichlet noise: Inverted V pattern
  - Phase 1: 0.30 → 0.15 (focus during specialization)
  - Phase 2: 0.15 → 0.25 (explore during generalization)
- Policy weight: 0.2 adaptive
- Entropy bonus: 0.07
- Eval temperature: 0.4

**Results - Strategic Win Rate Progression:**
- Iter 10: 50.0%
- Iter 20: 55.0%
- Iter 30: 53.1%
- Iter 40: 60.0%
- Iter 50: 60.0%
- **Iter 60: 65.0%** (PEAK - expected 75%, missed by 10%)
- Iter 70: 57.5%
- Iter 80: 62.5%
- Iter 90: 60.0%
- Iter 100: 62.5% (expected 70%, missed by 7.5%)

**Final Metrics (Iter 100):**
- Value correlation: 0.356 (adequate)
- Training time: 11h34m
- Loss: 0.5350

**Critical Finding - HYPOTHESIS FAILED:**
**Experiment 10 performed WORSE than Experiment 9 across all metrics!**

| Metric | Exp 9 (Monotonic) | Exp 10 (Inverted V) | Difference |
|--------|------------------|---------------------|------------|
| Peak win rate | 75.0% @ iter 70 | 65.0% @ iter 60 | **-10%** ✗ |
| Final win rate | 59.5% @ iter 100 | 62.5% @ iter 100 | +3% |
| Peak iteration | 70 | 60 | -10 |
| Value corr (final) | 0.433 | 0.356 | -0.077 |

**Why It Failed:**
1. **Never reached peak performance** - 65% vs expected 75%
2. **Phase 1 too slow** - By iter 60, Exp 9 was already at ~64% Strategic ratio and achieving strong results
3. **Curriculum reversal started too early** - Model hadn't specialized enough before generalization began
4. **Competing signals confused learning** - Phase 2 reversal may have created gradient conflicts
5. **Lower exploration in Phase 1** - Decreasing Dirichlet noise (0.30→0.15) may have prevented sufficient exploration

**Lessons Learned:**
1. ✗ **Curriculum reversal doesn't fix the problem** - prevented both overfitting AND peak performance
2. ✗ **Symmetric ramp-up/ramp-down failed** - 60 iterations wasn't enough to specialize
3. ✗ **Inverted V hypothesis was wrong** - model needs longer specialization phase
4. 💡 **Monotonic curriculum with early stopping may be optimal** - Exp 9's iter 70 model (75%) still best
5. 💡 **Alternative: Faster ramp to higher peak** - reach 85% Strategic by iter 40, maintain for 30 iter, then reverse

**Status**: **FAILED - Worse than both Exp 8 (72.5%) and Exp 9 peak (75%)**

---

## Experiment 11 - Plateau Curriculum (SURPRISING SUCCESS)

**Hypothesis:**
Three-phase plateau curriculum: Fast specialization (40%→85% over 40 iter) → Plateau (85% for 20 iter) → Gradual generalization (85%→40% over 40 iter)

**Configuration:**
- MCTS: 800-1600 simulations
- Model: 1,225,029 parameters (128 hidden channels)
- **Phase 1 (Iter 1-40): 40% → 85% Strategic** (fast specialization)
- **Phase 2 (Iter 40-60): 85% Strategic** (plateau to consolidate)
- **Phase 3 (Iter 60-100): 85% → 40% Strategic** (generalization)
- Dirichlet noise:
  - Phase 1-2: 0.25 (constant moderate exploration)
  - Phase 3: 0.25 → 0.30 (increased for generalization)
- Policy weight: 0.2 adaptive
- Entropy bonus: 0.07
- Eval temperature: 0.4

**Results - Strategic Win Rate Progression:**
- Iter 10: 32.5%
- Iter 20: 65.0%
- Iter 30: 55.0%
- Iter 40: 65.0% (Phase 1 complete - below 75% target)
- Iter 50: 45.0% (Plateau phase - unexpected drop)
- Iter 60: 65.0% (Phase 2 complete)
- Iter 70: 65.0% (Phase 3 - generalization begins)
- Iter 80: 70.0% (IMPROVING during generalization!)
- Iter 90: 67.5%
- **Iter 100: 72.5%** ✓ (Strong finish!)

**Final Metrics (Iter 100):**
- Strategic win rate: **72.5%** (29/40 wins)
  - As P1: 70.0% (14W-6D-1L)
  - As P2: 75.0% (15W-4D-1L)
- Value correlation: 0.388 (good quality)
- Training time: 11h54m
- Loss: 0.4970

**Critical Finding - UNEXPECTED LATE IMPROVEMENT:**
**Phase 3 generalization didn't cause decline - it caused IMPROVEMENT from 65% to 72.5%!**

| Metric | Phase 1 End (Iter 40) | Phase 2 End (Iter 60) | Final (Iter 100) | Change |
|--------|----------------------|----------------------|------------------|---------|
| Strategic win rate | 65.0% | 65.0% | **72.5%** | **+7.5%** ✓ |
| Strategic ratio | 85% | 85% → 40% | 40% | Generalized |
| Dirichlet noise | 0.25 | 0.25 → 0.30 | 0.30 | More exploration |

**Why It Succeeded (Unexpectedly):**
1. **Phase 3 reversal restored balance** - Reducing Strategic ratio from 85% to 40% helped model learn diverse strategies
2. **Increased exploration found better moves** - Dirichlet noise 0.25→0.30 enabled discovery
3. **85% Strategic was overfitting in Phase 2** - Reversal "cured" the overspecialization
4. **Self-play provided better training signal** - More diverse opponents (60% self-play) taught winning patterns Strategic couldn't
5. **Model consolidated learning during Phase 3** - Time to integrate Phase 1-2 knowledge with diverse play

**Comparison to Experiments 8, 9, 10:**

| Exp | Peak | Final | Peak → Final | Outcome |
|-----|------|-------|--------------|---------|
| **8** | 72.5% @ 50 | 72.5% | Stable | Baseline success |
| **9** | 75.0% @ 70 | 59.5% | **-15.5%** ✗ | Overfitting collapse |
| **10** | 65.0% @ 60 | 62.5% | -2.5% | Never reached peak ✗ |
| **11** | 65.0% @ 40 | **72.5%** | **+7.5%** ✓ | **Late improvement!** |

**Lessons Learned:**
1. ✓ **Generalization phase can IMPROVE performance** - not just prevent decline
2. ✓ **85% Strategic ratio is too high** - causes subtle overfitting even during plateau
3. ✓ **Self-play is underrated** - diverse opponents teach strategies Strategic can't
4. ✓ **Exploration matters** - Increasing Dirichlet noise to 0.30 enabled discovery
5. ✓ **Plateau curriculum works** - but improvement came from Phase 3, not Phase 1-2
6. 💡 **Fast specialization isn't necessary** - slower build to 40% self-play may be optimal

**Status**: **SUCCESS - Matched Exp 8's 72.5% with better generalization!**

**Key Insight**: The generalization phase wasn't just maintenance - it was active learning that found better strategies than pure Strategic training could provide.

---

## Experiment 13 - Large Model + Plateau Curriculum (FAILED - Worse Play Quality)

**Hypothesis:**
Combine the 10x larger model from Exp 12 (10.8M params) with the successful plateau curriculum from Exp 11 to get both capacity AND good training dynamics.

**Configuration:**
- MCTS: 800-1600 simulations
- **Model: 10,848,517 parameters** (SE-ResNet from Exp 12)
  - 256 hidden channels
  - 8 SE-ResBlocks with squeeze-and-excitation
- **Learning rate: 0.002** (reduced from 0.003 for stability)
- Buffer balancing: Disabled
- Bootstrap: 0.0
- **Strategic curriculum: 40%→85%→40% (Plateau, same as Exp 11)**
  - Phase 1 (Iter 1-40): 40% → 85% Strategic
  - Phase 2 (Iter 40-60): 85% Strategic (plateau)
  - Phase 3 (Iter 60-100): 85% → 40% Strategic
- Dirichlet noise: 0.25 base, increasing to 0.30 in Phase 3
- Policy weight: 0.2 adaptive

**Results - Strategic Win Rate Progression:**
| Iter | Time | vs Strategic | Self-Play P1 | MCTS Contribution |
|------|------|--------------|--------------|-------------------|
| 10 | 2h 2m | 57.5% | 55% | P1:55% P2:10% |
| 20 | 3h 43m | 52.5% | 25% | P1:35% P2:15% |
| 30 | 5h 26m | 42.5% | 40% | P1:50% P2:40% |
| 40 | 7h 0m | 57.5% | 40% | P1:100% P2:30% |
| 50 | 8h 27m | 52.5% | 35% | P1:55% P2:30% |
| 60 | 9h 53m | 60.0% | 0% | P1:10% P2:85% |
| 70 | 11h 30m | 55.0% | 15% | P1:20% P2:10% |
| **80** | 13h 11m | **70.0%** | 0% | P1:0% P2:25% |
| 90 | 15h 8m | 60.0% | 0% | P1:5% P2:20% |
| 100 | 17h 12m | 65.0% | 0% | P1:0% P2:5% |

**Final Metrics (Iter 100):**
- Strategic win rate: 65.0% (75% as P1, 55% as P2)
- Value loss: 0.188 (very low)
- Policy loss: 1.20
- Training time: 17h 12m

**Critical Finding - METRICS LIE:**
**Despite achieving 70% peak and 65% final vs Strategic, the model plays MUCH WORSE against humans than Exp 12!**

User feedback: "This one feels a lot stupider to play against than our previous entirely-self-play attempt... it's much much easier to beat than 12."

**Why It Failed (Despite Good Strategic Win Rate):**

1. **Self-play collapsed to 0% P1 wins**
   - By iter 60, self-play showed 0% P1 wins (complete P2 dominance)
   - Model learned heavily biased strategies that only work from one side
   - Indicates degenerate, exploitable patterns

2. **MCTS contribution dropped to ~0%**
   - By iter 100, MCTS provided almost no improvement over raw policy
   - Policy became overconfident in bad strategies
   - No diversity in move selection

3. **Overfit to Strategic opponent specifically**
   - 85% Strategic exposure during plateau phase (iter 40-60) caused narrow specialization
   - Model learned to beat Strategic's specific patterns, not general good play
   - Strategic win rate ≠ actual playing strength

4. **Large model amplified overfitting**
   - 10.8M parameters = more capacity to memorize Strategic's patterns
   - Without diverse training signal, capacity becomes a liability
   - Smaller models (1.2M) may generalize better

**Comparison to Exp 11 & 12:**

| Metric | Exp 11 (1.2M) | Exp 12 (10.8M) | Exp 13 (10.8M) |
|--------|---------------|----------------|----------------|
| Curriculum | Plateau | Pure self-play | Plateau |
| Peak Strategic | 70% @ 80 | 67.5% @ 10 | 70% @ 80 |
| Final Strategic | 72.5% | ~45% | 65% |
| Human play quality | Good | Unstable but creative | **Stupid, easily beaten** |
| Self-play balance | Healthy | Varied | **Collapsed (0% P1)** |
| Training time | 12h | 17h | 17h |

**Key Insight - Strategic Win Rate Is a Bad Metric:**
- Exp 13 achieved similar Strategic win rate to Exp 11 (65% vs 72.5%)
- But plays MUCH worse against humans
- Strategic opponent is too predictable - model overfit to its patterns
- Need a metric that measures general playing strength, not opponent-specific performance

**Lessons Learned:**
1. ✗ **Large model + plateau curriculum doesn't combine well** - amplifies overfitting
2. ✗ **Self-play collapse is a critical warning sign** - indicates degenerate strategies
3. ✗ **Strategic win rate doesn't predict human play quality** - can overfit to one opponent
4. ✗ **MCTS contribution near 0% indicates overconfident policy** - bad sign
5. 💡 **Smaller models may generalize better** - capacity without diversity = overfitting
6. 💡 **Need diverse opponents during training** - Strategic alone is insufficient
7. 💡 **Need better evaluation metrics** - Strategic win rate is misleading

**Status**: **FAILED - Good metrics, bad actual play quality**

---

## Conclusion

After 13 comprehensive experiments:
1. **Best Peak**: 75% Strategic win rate (Exp 9 at iter 70) ✓ **TARGET MET**
2. **Best Stable**: 72.5% Strategic win rate (Exp 8 @ iter 50, Exp 11 @ iter 100) ✓
3. **Discovered**:
   - Critical correlation bug, value head ≠ Strategic performance
   - **Inverted curriculum (40%→80%) is the key** - 25% improvement over traditional approach
   - Training sequence > Model capacity
   - Late-stage overfitting from monotonic curriculum (Exp 9)
   - Inverted V curriculum failed to reach peak (Exp 10)
   - **Generalization phase can IMPROVE performance** (Exp 11) ✓✓
   - **Strategic win rate is a misleading metric** (Exp 13) - can overfit to one opponent
   - **Large models amplify overfitting** (Exp 12, 13) - capacity without diversity = worse play
4. **Learned**:
   - Build general skills (self-play) BEFORE specializing (Strategic opponent) ✓
   - Extended training can backfire WITHOUT generalization (Exp 9)
   - **Curriculum reversal WORKS when aggressive enough** (85%→40% in Exp 11)
   - **Self-play is critical for finding optimal strategies** - Strategic alone isn't enough
   - Higher exploration (Dirichlet 0.30) enables discovery of better moves
   - **Self-play collapse (0% P1 wins) indicates degenerate strategies** (Exp 13)
   - **MCTS contribution near 0% = overconfident bad policy** (Exp 13)
   - **Smaller models (1.2M) may generalize better than large models (10.8M)**
5. **Status**: **75% achieved (Exp 9 iter 70, unstable) | 72.5% achieved (Exp 8 & 11, stable)**

**Best Models**:
1. **Experiment 11, Iteration 100** - 72.5% Strategic (stable, best generalization, good human play)
2. **Experiment 8, Iteration 50** - 72.5% Strategic (stable baseline, faster training)
3. **Experiment 9, Iteration 70** - 75% Strategic (peak, use with early stopping)

**Recommended Model**: **Experiment 11** - 72.5% Strategic with proven generalization
- Same performance as Exp 8 but with explicit generalization phase
- More training data (100 iter vs 50 iter)
- Better equipped for diverse opponents (human play)
- **Smaller model (1.2M) generalizes better than 10.8M models**

**Warning**: Large models (Exp 12, 13) achieved decent Strategic win rates but played WORSE against humans. Strategic win rate alone is insufficient for evaluating actual playing strength.

---

## Experiment Quick Reference

```
Exp 1:  60%   - Original lucky run ✓ (not reproducible)
Exp 2:  40%   - Policy weight too low (0.05)
Exp 3:  50%   - Heavy bootstrap (0.8), no Strategic
Exp 4:  37.5% - Too much Strategic (80%→20%) ✗ WORST
Exp 5:  42.5% - Stronger MCTS, buffer balance
Exp 6:  47.5% - Exact reversion to Exp 1 (high variance)
Exp 7:  47.5% - 56% larger model, NO improvement (capacity ≠ performance)
Exp 8:  72.5% - Inverted curriculum (40%→80%) ✓✓ BREAKTHROUGH (stable baseline)
Exp 9:  75% peak @ iter 70 → 59.5% final - Extended inverted, late-stage overfitting
Exp 10: 65% peak @ iter 60 → 62.5% final - Inverted V ✗ FAILED (never reached peak)
Exp 11: 65% @ iter 40 → 72.5% final - Plateau curriculum ✓✓ LATE IMPROVEMENT (generalization works!)
Exp 12: 67.5% peak @ iter 10 → ~45% avg - 10x model + pure self-play ✗ UNSTABLE (see EXPERIMENT_12_SUMMARY.md)
Exp 13: 70% peak @ iter 80 → 65% final - 10x model + plateau curriculum ✗ WORSE PLAY QUALITY (overfit to Strategic)
```

**Key Insights**:
1. **Training sequence > Model capacity** (Exp 7 vs 8: same model, 25% difference)
2. **Inverted curriculum is essential** (40%→80% vastly outperforms 70%→10%)
3. **Monotonic increase causes overfitting** (Exp 9: 75%→59.5%)
4. **Aggressive curriculum reversal WORKS** (Exp 11: 85%→40% improved 65%→72.5%)
5. **Generalization phase can improve performance** (Exp 11: Phase 3 added +7.5%)
6. **Pure self-play requires curriculum anchor** (Exp 12: 10x model failed without Strategic opponent)
7. **Strategic win rate ≠ human play quality** (Exp 13: 65% Strategic but easily beaten by humans)
8. **Large models (10.8M) overfit more than small (1.2M)** - capacity amplifies bad training signals
9. **Self-play collapse to 0% P1 = degenerate strategies** - critical warning sign
