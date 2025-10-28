# Draw Death Spiral: Analysis and Fix

## Problem Statement

Training run with adaptive gradient weighting achieved only **60% vs Strategic** (target: 75%) due to a catastrophic feedback loop.

## The Vicious Cycle

```
71% draws in self-play
    ↓
Replay buffer: 62% draws, 20% wins, 18% losses
    ↓
Value head learns to predict ~0.0 for all positions
    ↓
Policy network sees "everything is a draw" from value head
    ↓
Policy learns ultra-conservative play (avoid all risk)
    ↓
EVEN MORE draws in self-play (71%!)
    ↓
CYCLE REPEATS - STUCK IN LOCAL MINIMUM
```

## Evidence from Training Run (50 iterations, discrete_mild, adaptive weighting)

### Value Head Collapse
- **Iteration 1**: mean=0.001, std=0.164, correlation=0.146
- **Iteration 17**: mean=0.029, std=0.152, correlation=**0.330** ← PEAK
- **Iteration 50**: mean=0.001, std=**0.085**, correlation=0.136 ← COLLAPSED

**Critical observation**: Standard deviation HALVED (0.164 → 0.085). The value head compressed all predictions toward zero, essentially giving up on distinguishing winning from losing positions.

### Draw Rate Explosion
| Metric | Value | Impact |
|--------|-------|--------|
| Self-play draw rate (iter 50) | 71% | Model playing ultra-conservatively |
| Replay buffer draws | 62% | Training data heavily biased |
| Expected draw rate | ~20-30% | Normal for balanced play |
| **Excess draws** | **+41%** | Clear pathology |

### Gradient Imbalance Persisted
- Adaptive weighting adjusted from 0.32 → 0.405 (trying to compensate)
- But gradient ratio stayed 1.4-2.0x throughout (policy still dominating)
- Even at 68% value weight, value gradients remained small
- **Conclusion**: Value head stuck in local minimum where gradients naturally small

### Strategic Win Rate Instability
| Iteration | Win Rate | Change |
|-----------|----------|--------|
| 10 | 55% | Baseline |
| 20 | 35% | -20% (major drop!) |
| 30 | 55% | +20% (recovered) |
| 40 | 42.5% | -12.5% (dropped again) |
| **50** | **60%** | **+17.5% (best, but still 15% below target)** |

High variance indicates unstable learning. Model can't find consistent strategy.

## Root Cause Analysis

### Why Did Value Head Collapse?

1. **Easy Optimization**: Predicting zero for everything gives low MSE loss when 62% of training data is draws (value ≈ -0.3)
2. **Gradient Magnitude**: Once stuck near zero, gradients become small (derivatives of MSE near zero are small)
3. **Policy Dominance**: Policy loss gradients 1.4-2.0x larger, even with 68% value weight
4. **No Escape**: Adaptive weighting couldn't compensate enough - max value weight was 68%, but value head needed 90%+

### Why Did Draws Dominate?

1. **Initial Conservative Bias**: Model starts uncertain, conservative play is "safe"
2. **Value Head Feedback**: Value predicting near-zero reinforces conservative play
3. **Draw Penalty Too Mild**: -0.3 penalty not strong enough to break the cycle
4. **No Buffer Filtering**: 62% draws in buffer means value head sees mostly draws

## Three-Pronged Solution

### 1. Drastically Increase Draw Penalty
**Change**: `discrete_mild` (-0.3) → **`discrete_heavy` (-0.6)**

- Draw now almost as bad as loss (-0.6 vs -1.0)
- Forces model to seek decisive play
- Breaks the "draws are acceptable" mindset

**Expected Impact**:
- Immediate reduction in draw rate from 71% → 40-50%
- More informative training signal for value head
- Policy learns to take calculated risks

### 2. Force Value Head Learning with Very Low Policy Weight
**Change**: `policy_weight = 0.2 (adaptive)` → **`policy_weight = 0.05 (fixed)`**

- Gives **95% of loss signal to value head** (vs 68% max with adaptive)
- Disables adaptive weighting to prevent drift back up
- Value head MUST learn even with small gradients

**Expected Impact**:
- Value gradients effectively amplified 1.4x (95%/68%)
- Value head forced out of local minimum
- Correlation should improve from 0.14 → 0.4+

### 3. Filter Replay Buffer to Cap Draws
**Change**: `BALANCE_REPLAY_BUFFER = False` → **`True`**

- Target distribution: 40% wins (20% P1 + 20% P2), 40% losses, **20% draws**
- Current: 62% draws → New: **20% draws** (3x reduction!)
- Prioritizes decisive games for training

**Expected Impact**:
- Value head sees 3x fewer draws in training
- Better balance of winning/losing examples
- Prevents value head from learning "everything is a draw"

## Implementation Details

### Files Modified: `train.py`

#### New Reward Configuration
```python
elif reward_config == "discrete_heavy":
    return (1.0, -1.0, -0.6)  # Strong draw penalty
```

#### Fixed Low Policy Weight
```python
DEFAULT_POLICY_WEIGHT = 0.05  # Was 0.2
# Gives 95% of loss to value head, 5% to policy
```

#### Buffer Balancing Enabled
```python
BALANCE_REPLAY_BUFFER = True  # Was False
# Target: 20% P1 wins, 20% P2 wins, 20% P1 loss, 20% P2 loss, 10% P1 draw, 10% P2 draw
```

#### Adaptive Weighting Disabled
```python
# DISABLED: Keep policy_weight fixed at 0.05
# adaptive_policy_weight = 0.7 * adaptive_policy_weight + 0.3 * avg_recommended
```

## Expected Results

### Draw Rate
- **Current**: 71% self-play draws, 62% buffer draws
- **Target**: 35-45% self-play draws, 20% buffer draws (enforced by balancing)

### Value Head Performance
- **Current**: correlation=0.14, std=0.085 (collapsed)
- **Target**: correlation=0.4+, std=0.15+ (healthy spread)

### Strategic Win Rate
- **Current**: 60% (15 points below target)
- **Target**: 75%+ (decisive improvement)

### Gradient Balance
- **Current**: policy/value ratio 1.4-2.0x, adaptive weight maxed at 0.405
- **Expected**: Ratio may stay similar, but fixed 0.05 weight forces value learning anyway

## Metrics to Monitor

1. **Draw rate trajectory**: Should drop from 71% → 40-50% by iteration 20
2. **Value correlation**: Should exceed 0.3 by iteration 20, target 0.4+ by iteration 40
3. **Value std**: Should increase from 0.085 → 0.15+ (more diverse predictions)
4. **Strategic win rate**: Should exceed 60% by iteration 20, target 75% by iteration 50
5. **Buffer composition**: Should quickly converge to 20% draws (enforced by balancing)

## Risk Assessment

### Potential Issues

1. **Policy degradation**: With only 5% of loss signal, policy may get worse initially
   - **Mitigation**: Strong MCTS (600-1200 sims) provides good policy targets

2. **Value overfitting**: With 95% weight, value head might overfit
   - **Mitigation**: L2 regularization (weight_decay=2e-5) still active

3. **Extreme risk-taking**: -0.6 draw penalty might cause suicidal play
   - **Monitoring**: Watch for win rate collapse (both players losing to Strategic)

4. **Training instability**: Drastic changes might cause wild oscillations
   - **Mitigation**: Still using 20 epochs, OneCycleLR scheduler for stability

### Fallback Plan

If after 20 iterations:
- Strategic win rate < 40%: Reduce draw penalty to -0.4
- Value correlation < 0.2: Increase policy_weight to 0.1
- Draw rate > 60%: Increase draw penalty to -0.8

## Training Configuration

```
Mode: stable
Episodes per iteration: 100
MCTS sims: 600-1200
Iterations: 50
Batch size: 256
Buffer size: 6000
Epochs per iteration: 20
Reward: discrete_heavy (Win: +1.0, Loss: -1.0, Draw: -0.6)
Policy weight: 0.05 (fixed, non-adaptive)
Buffer balancing: ENABLED (20% draws max)
Device: CPU (faster than GPU for this workload)
```

## Hypothesis

The draw death spiral is the PRIMARY bottleneck preventing 75% Strategic win rate. By simultaneously:
1. Making draws nearly as bad as losses (-0.6 penalty)
2. Forcing value head to learn with 95% of loss signal
3. Filtering replay buffer to cap draws at 20%

We will break the vicious cycle and enable the value head to learn meaningful position evaluations, which will drive policy improvement and reach the 75% target.

**Success Criteria**: Strategic win rate ≥ 75% within 50 iterations, with draw rate ≤ 40% and value correlation ≥ 0.4.
