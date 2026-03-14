# Experiment 13: Large Model + Plateau Curriculum

## Goal
Combine the 10x larger model from Exp 12 with the proven plateau curriculum from Exp 11, aiming for sustained improvement throughout training rather than early peaking.

## Rationale

**What failed in Exp 12:**
- Pure self-play (0% Strategic) caused instability and strategy cycling
- Peaked at 67.5% on iteration 10, collapsed to 25% by iteration 60
- No external benchmark to anchor learning
- Model capacity wasn't the issue - training curriculum was

**What worked in Exp 11:**
- Plateau curriculum (40%→85%→40%) achieved stable 72.5%
- Generalization phase (85%→40%) actually *improved* performance
- Late improvement from iter 65→100 shows proper learning dynamics
- Strategic opponent provided necessary grounding

**Hypothesis:**
The large 10.8M parameter model will benefit from Exp 11's proven curriculum, achieving higher peak performance than the 1.2M model while maintaining stability.

## Model Architecture (Same as Exp 12)

| Parameter | Value |
|-----------|-------|
| Hidden channels | 256 |
| Residual blocks | 8 (SE-ResBlocks) |
| Policy channels | 128 |
| Value channels | 128 |
| SE attention | Yes (reduction=16) |
| **Total params** | **~10.8M** |

## Training Configuration

### Curriculum (Adapted from Exp 11)

Three-phase plateau curriculum over 100 iterations:

| Phase | Iterations | Strategic % | Dirichlet | Purpose |
|-------|------------|-------------|-----------|---------|
| 1 - Ramp | 1-40 | 40% → 85% | 0.25 | Build foundation via self-play, then specialize |
| 2 - Plateau | 40-60 | 85% | 0.25 | Consolidate expertise against Strategic |
| 3 - Generalize | 60-100 | 85% → 40% | 0.25 → 0.30 | Restore generalization, find better strategies |

### Key Parameters

```python
# Curriculum schedule
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.40
PEAK_STRATEGIC_OPPONENT_RATIO = 0.85    # Reached at iter 40
FINAL_STRATEGIC_OPPONENT_RATIO = 0.40   # Back down by iter 100

# Plateau phase
PLATEAU_START_ITER = 40
PLATEAU_END_ITER = 60

# No random opponent (pure self-play + Strategic mix)
INITIAL_RANDOM_OPPONENT_RATIO = 0.0
FINAL_RANDOM_OPPONENT_RATIO = 0.0

# Exploration
DIRICHLET_SCALE = 0.25  # Increase to 0.30 in Phase 3

# Learning rate (slightly lower for larger model stability)
max_lr = 0.002          # was 0.003 in Exp 12
min_lr = 0.0002
weight_decay = 3e-5

# Buffer (larger for 10x model)
DEFAULT_BUFFER_SIZE = 8000

# MCTS
DEFAULT_MIN_MCTS_SIMS = 800
DEFAULT_MAX_MCTS_SIMS = 1600

# Policy weight
POLICY_WEIGHT = 0.2     # Proven optimal across experiments

# Entropy bonus
ENTROPY_BONUS = 0.07    # From Exp 11
```

### Training Duration
- **100 iterations** (same as Exp 11, 12)
- **Expected time**: ~20-24 hours (similar to Exp 12)
- **Per iteration**: ~12-15 minutes

## Changes from Exp 12

| Setting | Exp 12 | Exp 13 |
|---------|--------|--------|
| Strategic curriculum | 0% constant | 40%→85%→40% |
| Learning rate | 0.003 | 0.002 |
| Dirichlet schedule | 0.30 constant | 0.25→0.30 (Phase 3) |
| Random opponent | 0% | 0% |

## Changes from Exp 11

| Setting | Exp 11 | Exp 13 |
|---------|--------|--------|
| Model size | 1.2M | 10.8M |
| Hidden channels | 128 | 256 |
| Residual blocks | 3 | 8 (SE) |
| Learning rate | 0.005 | 0.002 |
| Buffer size | 6000 | 8000 |

## Expected Outcomes

### Success Criteria
1. **No early peaking**: Performance should improve through at least iteration 60
2. **Phase 3 improvement**: Win rate should increase during generalization (iter 60-100)
3. **Final win rate**: Target 75%+ (vs Exp 11's 72.5%)
4. **Stability**: No >15% drops between evaluations

### Evaluation Schedule
Every 10 iterations:
1. vs Strategic as P1 (20 games)
2. vs Strategic as P2 (20 games)
3. Self-play balance check
4. MCTS contribution check

### Warning Signs
- Peak before iteration 40 → Model may be overfitting despite curriculum
- Decline during Phase 2 (iter 40-60) → Plateau too long, reduce to 15 iterations
- No improvement in Phase 3 → Dirichlet noise may be too low

## Implementation Checklist

### model.py
- [ ] Verify SE-ResBlock class exists (from Exp 12)
- [ ] Confirm 256 hidden channels, 8 blocks
- [ ] No changes needed if Exp 12 model.py is intact

### train.py
- [ ] Update `get_strategic_opponent_ratio()` for plateau curriculum:
  ```python
  def get_strategic_opponent_ratio(iteration):
      if iteration <= PLATEAU_START_ITER:
          # Phase 1: Ramp up 40% → 85%
          progress = iteration / PLATEAU_START_ITER
          return 0.40 + (0.45 * progress)
      elif iteration <= PLATEAU_END_ITER:
          # Phase 2: Plateau at 85%
          return 0.85
      else:
          # Phase 3: Ramp down 85% → 40%
          progress = (iteration - PLATEAU_END_ITER) / (100 - PLATEAU_END_ITER)
          return 0.85 - (0.45 * progress)
  ```
- [ ] Update `get_dirichlet_scale()` for Phase 3 increase:
  ```python
  def get_dirichlet_scale(iteration):
      if iteration <= PLATEAU_END_ITER:
          return 0.25
      else:
          # Phase 3: 0.25 → 0.30
          progress = (iteration - PLATEAU_END_ITER) / (100 - PLATEAU_END_ITER)
          return 0.25 + (0.05 * progress)
  ```
- [ ] Set max_lr = 0.002
- [ ] Set ENTROPY_BONUS = 0.07
- [ ] Verify DEFAULT_BUFFER_SIZE = 8000

## Risk Mitigation

### If training is too slow (>20 min/iter)
- Reduce buffer size to 6000
- Reduce MCTS sims to 600-1200

### If model peaks early (before iter 40)
- Reduce Phase 1 ramp speed (40 iter → 50 iter)
- Lower initial Strategic ratio to 30%

### If Phase 3 causes decline
- Cap Dirichlet increase at 0.27
- Slow down Strategic reduction (85%→60% instead of 85%→40%)

## Comparison Targets

| Experiment | Model | Curriculum | Peak | Final | Notes |
|------------|-------|-----------|------|-------|-------|
| Exp 8 | 1.2M | 40%→80% | 72.5% | 72.5% | Stable baseline |
| Exp 9 | 1.2M | 40%→85% | 75% | 59.5% | Overfitting |
| Exp 11 | 1.2M | Plateau | 65% | 72.5% | Late improvement |
| Exp 12 | 10.8M | 0% (self-play) | 67.5% | ~45% | Unstable |
| **Exp 13** | **10.8M** | **Plateau** | **75%+** | **75%+** | **Target** |

## Files to Backup Before Starting
- `model.py` → `model_exp12.py`
- `train.py` → `train_exp12.py`

## Summary

Experiment 13 combines:
1. **Large model capacity** (10.8M params, SE attention) from Exp 12
2. **Proven curriculum** (plateau with generalization) from Exp 11
3. **Lower learning rate** for large model stability
4. **Strategic grounding** to prevent self-play instability

The hypothesis is that Exp 12 failed due to curriculum, not capacity. By applying Exp 11's curriculum to the larger model, we expect to exceed the 72.5% ceiling achieved by the smaller model.
