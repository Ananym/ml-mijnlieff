# Experiment 11: Plateau Curriculum
## Fast Specialization → Plateau → Generalization

**Date**: 2024-01-19
**Goal**: Achieve and maintain 75% Strategic win rate by reaching peak quickly, consolidating, then generalizing
**Status**: READY TO RUN

---

## Problem Being Solved

Experiment 10 revealed that the Inverted V curriculum failed:
- **Peaked at only 65%** (expected 75%, missed by 10%) ✗
- **Final at 62.5%** (expected 70%, missed by 7.5%) ✗
- **Root causes**:
  - Phase 1 too slow (60 iterations to reach only 60% Strategic ratio)
  - Reversal started too early (before model specialized enough)
  - Dirichlet noise decreased too much (0.30→0.15 hurt exploration)
  - Competing gradients in Phase 2 confused learning

**New Insight**: We need to reach peak performance FIRST, THEN focus on generalization.

---

## Hypothesis

**A three-phase plateau curriculum will achieve 75% peak AND maintain 70%+ with generalization:**

1. **Phase 1 (Fast Specialization)**: Ramp aggressively to 85% Strategic over 40 iterations
2. **Phase 2 (Plateau)**: Maintain 85% Strategic for 20 iterations to consolidate learning
3. **Phase 3 (Generalization)**: Gradually reduce to 40% Strategic over 40 iterations

This should:
- Reach 75% by iteration 40-50 (based on Exp 9 trajectory)
- Solidify peak performance during plateau (iter 40-60)
- Restore generalization without losing too much performance (iter 60-100)
- Final target: 70-73% Strategic win rate (stable and robust)

---

## Configuration Changes from Experiment 10

### Strategic Opponent Curriculum

**Experiment 10 (Inverted V - FAILED)**:
```
Phase 1 (Iter 1-60): 40% → 80% Strategic (too slow)
Phase 2 (Iter 60-100): 80% → 50% Strategic (reversal)
Result: Peak 65% @ iter 60, Final 62.5%
```

**Experiment 11 (Plateau Curriculum)**:
```
Phase 1 - Fast Specialization (Iter 1-40):
  40% → 85% Strategic  (faster ramp, higher peak)

Phase 2 - Plateau (Iter 40-60):
  85% Strategic (constant - consolidate learning)

Phase 3 - Generalization (Iter 60-100):
  85% → 40% Strategic (gradual reversal)
```

**Key Differences**:
1. **Faster Phase 1**: 40 iterations to reach 85% (vs 60 iterations to 80%)
2. **Higher peak**: 85% Strategic (vs 80%)
3. **Plateau phase**: 20 iterations at 85% to solidify (NEW!)
4. **Longer Phase 3**: 40 iterations for generalization (vs 40 in Exp 10)
5. **More aggressive final**: Return to 40% Strategic (vs 50% in Exp 10)

### Dirichlet Noise Curriculum

**Experiment 10 (Inverted V - FAILED)**:
```
Phase 1 (Iter 1-60):  0.30 → 0.15 (decreasing - hurt exploration)
Phase 2 (Iter 60-100): 0.15 → 0.25 (increasing)
```

**Experiment 11 (Plateau Curriculum)**:
```
Phase 1 (Iter 1-40):  0.25 (constant moderate exploration)
Phase 2 (Iter 40-60): 0.25 (constant)
Phase 3 (Iter 60-100): 0.25 → 0.30 (gradually increase for generalization)
```

**Key Differences**:
1. **Constant moderate noise in Phases 1 & 2**: Avoids Exp 10's exploration problem
2. **0.25 baseline**: Higher than Exp 10's minimum (0.15), ensures diverse strategies
3. **Gradual increase in Phase 3**: Smooth transition, not abrupt

### Other Configuration (Unchanged)

- **Model**: 1,225,029 parameters (128 hidden channels, 3 residual blocks)
- **MCTS**: 800-1600 simulations
- **Policy weight**: 0.2 (adaptive)
- **Entropy bonus**: 0.07
- **Buffer balancing**: Disabled
- **Bootstrap**: 0.0
- **Eval temperature**: 0.4
- **Max iterations**: 100
- **Evaluation frequency**: Every 10 iterations

---

## Expected Results

### Phase 1: Fast Specialization (Iterations 1-40)

**Strategic Opponent Exposure**: Rapidly increasing (40% → 85%)
**Dirichlet Noise**: Constant 0.25 (moderate exploration)

**Expected Behavior**:
- Early (iter 1-20): Build general TicTacDo skills, start learning Strategic patterns
- Mid (iter 20-30): Rapid improvement against Strategic opponent
- Late (iter 30-40): Approaching peak performance

**Expected Performance** (based on Exp 9 trajectory with faster ramp):
- Iter 10: ~60% Strategic win rate (building)
- Iter 20: ~65% Strategic win rate (improving)
- Iter 30: ~70% Strategic win rate (approaching peak)
- Iter 40: ~73-75% Strategic win rate (near/at peak)

### Phase 2: Plateau (Iterations 40-60)

**Strategic Opponent Exposure**: Constant 85%
**Dirichlet Noise**: Constant 0.25

**Expected Behavior**:
- Consolidate learning against Strategic opponent
- Refine strategies without new curriculum changes
- Allow model to "settle" at peak performance
- No conflicting gradients (unlike Exp 10's immediate reversal)

**Expected Performance**:
- Iter 50: ~75% Strategic win rate (peak achieved and maintained)
- Iter 60: ~75% Strategic win rate (stable peak)

**Why Plateau Matters**:
- Exp 10 reversed immediately after reaching 65% @ iter 60
- Plateau phase allows model to solidify peak before generalization
- Prevents gradient conflicts from rapid curriculum changes

### Phase 3: Generalization (Iterations 60-100)

**Strategic Opponent Exposure**: Gradually decreasing (85% → 40%)
**Dirichlet Noise**: Gradually increasing (0.25 → 0.30)

**Expected Behavior**:
- Early (iter 60-70): Maintain peak while starting to diversify
- Mid (iter 70-85): Active generalization - broaden strategies
- Late (iter 85-100): Stabilize at balanced, robust play

**Expected Performance**:
- Iter 70: ~73-74% Strategic win rate (slight drop acceptable)
- Iter 85: ~71-72% Strategic win rate (gradual decline)
- Iter 100: ~70-72% Strategic win rate (STABLE, robust)

**Target Trade-off**:
- Sacrifice 3-5% Strategic performance (75% → 70-72%)
- Gain strong generalization vs unpredictable opponents
- More stable than Exp 9 (which dropped to 59.5%)
- Better peak than Exp 10 (which only reached 65%)

### Final Model (Iteration 100)

**Target Metrics**:
- Strategic win rate: **70-72%** (stable, robust)
- Value correlation: **>0.35** (maintained quality)
- Training time: **~11-12 hours**

**Key Success Criteria**:
1. ✓ Reach 75% by iteration 40-50 (unlike Exp 10's 65%)
2. ✓ Maintain 75% during plateau phase (iter 40-60)
3. ✓ Graceful decline to 70-72% during generalization (not collapse like Exp 9)
4. ✓ Strong generalization (robust vs diverse play)
5. ✓ Value head quality maintained (correlation >0.35)

---

## Success vs Failure Criteria

### SUCCESS Indicators

1. **Phase 1 Success**:
   - Reach ≥73% Strategic win rate by iteration 40
   - Proves fast ramp works better than Exp 10's slow ramp

2. **Phase 2 Success**:
   - Maintain ≥73% Strategic win rate during plateau (iter 40-60)
   - Proves consolidation phase stabilizes learning

3. **Phase 3 Success**:
   - Final ≥70% Strategic win rate at iteration 100
   - Decline <10% from peak to final
   - Better than Exp 9's 59.5% final and Exp 10's 65% peak

4. **Overall Success**:
   - Peak ≥75% (match Exp 9's best)
   - Final ≥70% (beat Exp 9's 59.5% and Exp 10's 62.5%)
   - Stable performance curve (no sudden drops)
   - Value correlation >0.35

### FAILURE Indicators

1. **Phase 1 Failure**:
   - Can't reach 70% by iteration 40
   - Suggests fast ramp too fast, model can't learn quickly enough

2. **Phase 2 Failure**:
   - Performance drops during plateau phase
   - Suggests 85% Strategic ratio too high, causes instability

3. **Phase 3 Failure**:
   - Final <65% Strategic win rate
   - Decline >15% from peak
   - Suggests reversal too aggressive or generalization harmful

4. **Comparison Failure**:
   - Peak <75% (worse than Exp 9)
   - Final <65% (worse than Exp 10)
   - Either means plateau curriculum doesn't work

---

## Comparison to Previous Experiments

| Experiment | Curriculum | Peak | Final | Outcome |
|------------|-----------|------|-------|---------|
| **8** | 40%→80% (50 iter) | 72.5% @ 50 | 72.5% | Stable, conservative |
| **9** | 40%→85% (100 iter) | 75.0% @ 70 | 59.5% | Peak achieved, overfitting |
| **10** | 40%→80%→50% (Inverted V) | 65.0% @ 60 | 62.5% | Never reached peak ✗ |
| **11** | 40%→85%→40% (Plateau) | **75%? @ 40-50** | **70-72%?** | **Testing now** |

**Key Advantages over Exp 9**:
- Plateau phase prevents immediate overfitting after peak
- Gradual Phase 3 reversal maintains more Strategic performance
- Should avoid the 75%→59.5% collapse

**Key Advantages over Exp 10**:
- Faster Phase 1 reaches 85% Strategic by iter 40 (vs 60% @ iter 60)
- Higher peak ratio (85% vs 80%)
- Plateau phase allows consolidation before reversal
- Constant moderate Dirichlet noise (0.25) maintains exploration

**Key Advantages over Exp 8**:
- Higher peak (targeting 75% vs 72.5%)
- Explicit generalization phase for human-play robustness
- More training data for value head

---

## Risks and Mitigation

### Risk 1: Fast Ramp Too Aggressive

**Problem**: 40 iterations might not be enough to reach 85% Strategic ratio effectively

**Mitigation**:
- Based on Exp 9 data, model was at 56% Strategic ratio by iter 40
- Faster ramp (40 vs 100 iter) should reach 85% by design
- Plateau phase (iter 40-60) provides buffer to catch up if needed

**Fallback**: If peak not reached by iter 60, extend plateau phase mentally and evaluate iter 70 model

### Risk 2: Plateau Phase Causes Overfitting

**Problem**: 20 iterations at 85% Strategic might cause early overfitting

**Mitigation**:
- 85% still includes 15% self-play (diversity maintained)
- Constant 0.25 Dirichlet noise ensures exploration
- Shorter than Exp 9's continuous ramp (which took 100 iter)

**Fallback**: If overfitting detected during plateau, use iter 40-50 model instead of iter 60

### Risk 3: Phase 3 Reversal Too Aggressive

**Problem**: 85%→40% over 40 iterations might be too steep, causing performance collapse

**Mitigation**:
- Gradual 40-iteration slope smoother than Exp 10 (which had immediate drop)
- Increasing Dirichlet noise (0.25→0.30) eases transition
- Model has 20-iteration plateau to solidify learning first

**Fallback**: If final <65%, use iter 60 model (peak + plateau) instead of iter 100

### Risk 4: Still Doesn't Solve Generalization

**Problem**: Model might still overfit like Exp 9 despite plateau

**Mitigation**:
- Phase 3 actively restores self-play (85%→40%)
- Higher noise in Phase 3 (0.30) than Exp 9 (0.20)
- Return to 40% Strategic (vs Exp 9's final 85%)

**Fallback**: Accept Exp 9 iter 70 model as best (75% Strategic, no generalization phase)

---

## Implementation Details

### Code Changes (train.py)

**1. Strategic Opponent Ratio Configuration** (lines 122-128):
```python
# Experiment 11: PLATEAU CURRICULUM
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.40
PEAK_STRATEGIC_OPPONENT_RATIO = 0.85
FINAL_STRATEGIC_OPPONENT_RATIO = 0.40
PHASE_1_END = 40
PHASE_2_END = 60
OPPONENT_TRANSITION_ITERATIONS = 100
```

**2. Three-Phase get_opponent_ratios()** (lines 873-906):
```python
def get_opponent_ratios(iteration):
    if iteration <= PHASE_1_END:
        # Phase 1: Fast ramp to peak
        progress = iteration / PHASE_1_END
        strategic_ratio = INITIAL + progress * (PEAK - INITIAL)
    elif iteration <= PHASE_2_END:
        # Phase 2: Plateau
        strategic_ratio = PEAK
    else:
        # Phase 3: Gradual reversal
        progress = (iteration - PHASE_2_END) / (MAX - PHASE_2_END)
        strategic_ratio = PEAK + progress * (FINAL - PEAK)
    return 0.0, strategic_ratio
```

**3. Three-Phase get_dirichlet_scale()** (lines 100-116):
```python
def get_dirichlet_scale(iteration):
    PHASE_1_END = 40
    PHASE_2_END = 60

    if iteration <= PHASE_2_END:
        # Phases 1 & 2: Constant moderate noise
        return 0.25
    else:
        # Phase 3: Increase for generalization
        progress = (iteration - PHASE_2_END) / (MAX - PHASE_2_END)
        return 0.25 + progress * (0.30 - 0.25)
```

---

## Expected Training Timeline

**Estimated Duration**: 11-12 hours
**Evaluation Points**: Every 10 iterations (11 evaluations total)

**Progress Checkpoints**:
- **1.5 hours** (iter 10): Should show ~60% Strategic
- **3 hours** (iter 20): Should reach ~65% Strategic
- **4.5 hours** (iter 30): Should approach ~70% Strategic
- **6 hours** (iter 40): PHASE 1 COMPLETE - should reach ~73-75% Strategic ✓
- **7.5 hours** (iter 50): PLATEAU - should maintain ~75% Strategic
- **9 hours** (iter 60): PHASE 2 COMPLETE - should maintain ~75% Strategic ✓
- **10.5 hours** (iter 80): PHASE 3 - should be ~71-72% Strategic
- **12 hours** (iter 100): FINAL - target 70-72% Strategic (stable) ✓

---

## Post-Experiment Analysis Plan

After completion, analyze:

1. **Phase 1 Performance** (iter 1-40):
   - Did we reach 75% by iter 40?
   - Compare to Exp 9 and Exp 10 at same iterations
   - Validate fast ramp hypothesis

2. **Phase 2 Stability** (iter 40-60):
   - Did performance maintain or improve during plateau?
   - Any signs of overfitting during constant 85% ratio?
   - Value correlation during this phase

3. **Phase 3 Generalization** (iter 60-100):
   - Graceful decline or collapse?
   - Final vs peak performance gap
   - Compare to Exp 9's collapse (75%→59.5%)

4. **Curriculum Comparison**:
   - Plot all three curves: Exp 9, Exp 10, Exp 11
   - Identify which phase transitions worked
   - Determine best model iteration

5. **Value Head Analysis**:
   - Value correlation throughout training
   - Does plateau phase help value learning?
   - Compare to Exp 9 and Exp 10

---

## Summary

Experiment 11 tests whether a **three-phase plateau curriculum** can achieve both peak performance AND stable generalization:

**Key Innovation**:
- **Fast specialization** (40% → 85% over 40 iter)
- **Plateau consolidation** (85% constant for 20 iter) ← NEW!
- **Gradual generalization** (85% → 40% over 40 iter)

**Expected Trade-off**:
- Reach 75% by iteration 40-50 (match Exp 9's peak)
- Maintain through plateau (iter 40-60)
- Sacrifice 3-5% for generalization (75% → 70-72%)
- Gain robust, generalizable play

**Success Metric**:
- Peak ≥75% by iter 40-50
- Final 70-72% with <10% decline from peak
- Better than Exp 9 final (59.5%) and Exp 10 peak (65%)

**If Successful**: Validates plateau curriculum as optimal approach - achieves peak quickly, consolidates, then generalizes gracefully.

**If Unsuccessful**: Revert to Exp 9 iter 70 model (75% Strategic) with early stopping, or accept Exp 8's stable 72.5%.
