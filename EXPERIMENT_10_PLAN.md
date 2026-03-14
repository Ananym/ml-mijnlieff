# Experiment 10: Inverted V Curriculum
## Testing Active Generalization Recovery

**Date**: 2024-01-18
**Goal**: Achieve 70-75% Strategic win rate with strong generalization by implementing curriculum reversal
**Status**: READY TO RUN

---

## Problem Being Solved

Experiment 9 revealed a critical flaw:
- **Achieved 75% Strategic win rate at iteration 70** ✓
- **Declined to 59.5% by iteration 100** ✗ (late-stage overfitting)
- **Cause**: Monotonic curriculum (40%→85%) with no recovery period

**Root Issue**: Continuous specialization without "de-specialization" → brittle, overfitted model

---

## Hypothesis

**After achieving peak specialization, actively restore generalization by REDUCING Strategic opponent ratio.**

This "Inverted V" pattern should:
1. Build expertise against Strategic opponent (Phase 1)
2. Restore flexible, generalizable play (Phase 2)
3. Maintain stable performance without late-stage collapse

---

## Configuration Changes from Experiment 9

### Strategic Opponent Curriculum

**Experiment 9 (Monotonic)**:
```
Iter 1:   40% Strategic → 60% self-play
Iter 50:  62.5% Strategic → 37.5% self-play
Iter 100: 85% Strategic → 15% self-play
```
Result: 75% peak → 59.5% final (overfitting)

**Experiment 10 (Inverted V)**:
```
PHASE 1 - Specialization (Iter 1→60):
Iter 1:  40% Strategic → 60% self-play
Iter 30: 60% Strategic → 40% self-play
Iter 60: 80% Strategic → 20% self-play (PEAK)

PHASE 2 - Generalization (Iter 60→100):
Iter 70: 72.5% Strategic → 27.5% self-play
Iter 85: 61.25% Strategic → 38.75% self-play
Iter 100: 50% Strategic → 50% self-play (BALANCED)
```

**Key Change**: After iter 60, Strategic ratio DECREASES instead of continuing to increase!

### Dirichlet Noise Curriculum

**Experiment 9 (Monotonic Decrease)**:
```
Iter 1:   0.30 (high exploration)
Iter 70:  0.20 (lower exploration)
Iter 100: 0.20 (maintained)
```

**Experiment 10 (Inverted V)**:
```
PHASE 1 - Focus (Iter 1→60):
Iter 1:  0.30 (high exploration)
Iter 30: 0.225 (decreasing)
Iter 60: 0.15 (MINIMUM - focused learning)

PHASE 2 - Explore (Iter 60→100):
Iter 70: 0.175 (increasing!)
Iter 85: 0.2125 (more exploration)
Iter 100: 0.25 (HIGH - restore diversity)
```

**Key Change**: After iter 60, exploration INCREASES to encourage diverse strategies!

### Other Configuration (Unchanged from Exp 9)

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

## Implementation Details

### Code Changes (train.py)

**1. Strategic Opponent Ratio** (lines 113-118):
```python
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.40  # Start balanced
PEAK_STRATEGIC_OPPONENT_RATIO = 0.80     # Peak at iter 60
FINAL_STRATEGIC_OPPONENT_RATIO = 0.50    # End balanced (KEY!)
PEAK_ITERATION = 60                       # When to reverse
OPPONENT_TRANSITION_ITERATIONS = 100
```

**2. Two-Phase get_opponent_ratios()** (lines 863-892):
```python
def get_opponent_ratios(iteration):
    if iteration <= PEAK_ITERATION:
        # Phase 1: Ramp UP to peak
        progress = iteration / PEAK_ITERATION
        strategic_ratio = INITIAL + progress * (PEAK - INITIAL)
    else:
        # Phase 2: Ramp DOWN from peak
        progress = (iteration - PEAK_ITERATION) / (MAX - PEAK_ITERATION)
        strategic_ratio = PEAK + progress * (FINAL - PEAK)
    return 0.0, strategic_ratio
```

**3. Two-Phase get_dirichlet_scale()** (lines 100-117):
```python
def get_dirichlet_scale(iteration):
    DIRICHLET_MID_SCALE = 0.15
    DIRICHLET_PHASE2_SCALE = 0.25
    PEAK_ITERATION = 60

    if iteration <= PEAK_ITERATION:
        # Phase 1: Decrease (focus)
        progress = iteration / PEAK_ITERATION
        return 0.30 - progress * (0.30 - 0.15)
    else:
        # Phase 2: Increase (explore)
        progress = (iteration - PEAK_ITERATION) / (MAX - PEAK_ITERATION)
        return 0.15 + progress * (0.25 - 0.15)
```

---

## Expected Results

### Phase 1: Specialization (Iterations 1-60)

**Strategic Opponent Exposure**: Gradually increasing (40% → 80%)
**Dirichlet Noise**: Gradually decreasing (0.30 → 0.15)

**Expected Behavior**:
- Early (iter 1-20): Build general TicTacDo skills through self-play
- Mid (iter 20-40): Learn Strategic opponent patterns
- Late (iter 40-60): Specialize against Strategic opponent

**Expected Performance**:
- Iter 10: ~70% Strategic win rate (similar to Exp 9)
- Iter 30: ~55% Strategic win rate (building phase)
- Iter 60: ~75% Strategic win rate (peak specialization)

### Phase 2: Generalization (Iterations 60-100)

**Strategic Opponent Exposure**: Gradually decreasing (80% → 50%)
**Dirichlet Noise**: Gradually increasing (0.15 → 0.25)

**Expected Behavior**:
- Early (iter 60-70): Maintain specialization while starting to diversify
- Mid (iter 70-85): Active generalization - "un-learn" narrow patterns
- Late (iter 85-100): Stabilize at balanced, robust play

**Expected Performance**:
- Iter 70: ~73% Strategic win rate (slight drop from peak, acceptable)
- Iter 85: ~71% Strategic win rate (continued slight decline)
- Iter 100: ~70% Strategic win rate (STABLE, robust)

### Final Model (Iteration 100)

**Target Metrics**:
- Strategic win rate: **70-72%** (stable)
- Value correlation: **>0.40** (maintained quality)
- Training time: **~11-12 hours**

**Key Success Criteria**:
1. ✓ No late-stage collapse (unlike Exp 9's 75%→59.5%)
2. ✓ Stable performance (±2% in last 30 iterations)
3. ✓ Strong generalization (robust vs unpredictable play)
4. ✓ Value head quality maintained (correlation >0.40)

---

## Success vs Failure Criteria

### SUCCESS Indicators

1. **Stable Performance Curve**:
   - Peak at iter 50-70
   - Decline <10% from peak to iter 100
   - Final >65% Strategic win rate

2. **Maintained Value Quality**:
   - Value correlation >0.35 throughout
   - No collapse in later iterations

3. **Phase 2 Works**:
   - Strategic win rate decreases gracefully (not collapse)
   - Final model shows diverse move selection
   - Buffer maintains healthy diversity (not all Strategic patterns)

### FAILURE Indicators

1. **Phase 2 Collapse**:
   - Strategic win rate drops >15% from peak
   - Final <60% Strategic win rate
   - Suggests too much generalization, not enough retention

2. **Phase 1 Never Peaks**:
   - Can't reach 70% by iteration 60
   - Suggests curriculum ramps too slowly

3. **Still Overfits**:
   - Performance continues declining iter 60-100
   - Suggests Phase 2 reversal insufficient

---

## Comparison to Previous Experiments

| Experiment | Curriculum | Peak | Final | Outcome |
|------------|-----------|------|-------|---------|
| **8** | 40%→80% (50 iter) | 72.5% @ 50 | 72.5% | Stable, but short |
| **9** | 40%→85% (100 iter) | 75.0% @ 70 | 59.5% | Overfitting |
| **10** | 40%→80%→50% (100 iter) | 75% @ 60? | **70%?** | **Testing now** |

**Key Advantage over Exp 9**:
- Active recovery period prevents overfitting
- Maintains Strategic performance while gaining generalization
- Should be robust to longer training (could extend to 150 iter if needed)

**Key Advantage over Exp 8**:
- Longer training for more refinement
- Explicit generalization phase for human-play robustness
- More data for value head training

---

## Risks and Mitigation

### Risk 1: Too Much Generalization

**Problem**: Phase 2 reversal too aggressive → forget how to beat Strategic

**Mitigation**:
- Only reverse to 50% (not back to 40%)
- Gradual slope (40 iterations to reverse)
- Model retains Strategic patterns, just adds diversity

**Fallback**: If final <65%, use iteration 60-70 model instead

### Risk 2: Dirichlet Noise Too High

**Problem**: 0.25 final noise too much → unstable learning in Phase 2

**Mitigation**:
- Monitor loss progression
- If loss stops decreasing in Phase 2 → noise might be too high
- Can reduce to 0.20 in future runs

**Fallback**: Current Exp 9 config had 0.20 final noise and worked well until overfitting

### Risk 3: Conflict Between Phases

**Problem**: Phase 1 and Phase 2 gradients conflict → confusion

**Mitigation**:
- Gradual transitions (not abrupt)
- 40-iteration Phase 2 allows smooth adaptation
- Replay buffer maintains continuity

**Fallback**: Early stopping at iteration 60-70 if conflict evident

---

## Post-Experiment Analysis Plan

After completion, analyze:

1. **Performance Trajectory**:
   - Plot Strategic win rate vs iteration
   - Identify peak iteration
   - Measure decline from peak to final
   - Compare to Exp 9 trajectory

2. **Phase Transition Point**:
   - Does iter 60 show visible inflection point?
   - Is decline gradual or sudden?
   - Does Phase 2 achieve stabilization?

3. **Value Head Behavior**:
   - Value correlation throughout training
   - Does Phase 2 help or hurt value learning?
   - Compare to Exp 9 value correlation

4. **Buffer Composition**:
   - Win/loss/draw distribution over time
   - Does Phase 2 change buffer composition?
   - Signs of healthy diversity?

5. **Model Comparison**:
   - Evaluate iter 60, 70, 80, 90, 100 models
   - Which iteration is truly best?
   - Trade-off analysis: Strategic vs generalization

---

## Timeline and Expectations

**Estimated Duration**: 11-12 hours
**Evaluation Points**: Every 10 iterations (11 evaluations total)
**Expected Completion**: ~12 hours from start

**Progress Checkpoints**:
- **2 hours** (iter 20): Should show ~60-70% Strategic
- **4 hours** (iter 40): Should approach ~70% Strategic
- **6 hours** (iter 60): PEAK - should reach ~75% Strategic
- **8 hours** (iter 80): Phase 2 active - should be ~71-72% Strategic
- **12 hours** (iter 100): Final - target 70% Strategic (stable)

---

## Summary

Experiment 10 tests whether **active curriculum reversal** can prevent late-stage overfitting while maintaining high Strategic win rate.

**Key Innovation**: After achieving peak specialization (80% Strategic @ iter 60), actively restore generalization by reducing to 50% Strategic by iter 100, with matching increase in exploration noise.

**Expected Trade-off**: Sacrifice 5% Strategic performance (75% → 70%) to gain robust, generalizable play suitable for unpredictable human opponents.

**Success Metric**: Final Strategic win rate 70%±2% with <10% decline from peak.

**If Successful**: Validates Inverted V curriculum as the optimal training approach for this domain. Can be applied to future experiments and potentially extended (e.g., 150 iterations with longer Phase 2).

**If Unsuccessful**: Revert to Experiment 8 approach (stop at 50 iterations, 72.5% Strategic) or use early stopping at detected peak (iter 60-70, ~75% Strategic).
