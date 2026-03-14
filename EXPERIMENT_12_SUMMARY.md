# Experiment 12: Large Model + Pure Self-Play

## Overview

Experiment 12 tested a radically different approach: a 10x larger model (~10.8M parameters vs ~1.2M baseline) trained with 100% pure self-play (no Strategic opponent during training). The hypothesis was that sufficient model capacity combined with pure self-play exploration might discover strategies that curriculum-based training missed.

## Configuration

### Model Architecture
| Parameter | Baseline (Exp 11) | Experiment 12 |
|-----------|-------------------|---------------|
| Hidden channels | 128 | 256 |
| Residual blocks | 3 (standard) | 8 (SE-ResBlocks) |
| Policy channels | 64 | 128 |
| Value channels | 64 | 128 |
| SE attention | No | Yes (reduction=16) |
| **Total params** | ~1.2M | **~10.8M** (9x) |

### Training Configuration
- **MCTS**: 800-1600 simulations per move
- **Opponent**: 100% self-play (0% Strategic, 0% Random)
- **Dirichlet noise**: 0.30 constant (higher for exploration without curriculum)
- **Learning rate**: 0.003 max, 0.0003 final (OneCycleLR)
- **Weight decay**: 3e-5 (increased for larger model)
- **Buffer size**: 8000 (increased for larger model)
- **Policy weight**: 0.2 adaptive
- **Iterations**: 100 (same as previous experiments)

### Extended Evaluation Suite
Every 10 iterations, 5 evaluation types:
1. Self-play balance (no MCTS) - P1 vs P2 win rates
2. MCTS contribution as P1 - MCTS wins over raw policy
3. MCTS contribution as P2 - MCTS wins over raw policy
4. Policy vs Strategic as P1 - real-world performance proxy
5. Policy vs Strategic as P2 - real-world performance proxy

## Results

### Strategic Win Rate Progression

| Iteration | vs Strategic | Self-play P1/P2 | MCTS Contribution | Notes |
|-----------|--------------|-----------------|-------------------|-------|
| 10 | **67.5%** | 60%/10% | 13/40 | Strong early peak |
| 20 | 32.5% | 45%/25% | 13/40 | Sharp decline |
| 30 | 47.5% | 70%/15% | 18/40 | Partial recovery |
| 40 | 50.0% | 55%/20% | 19/40 | MCTS strengthening |
| 50 | 37.5% | 30%/35% | 16/40 | Dip, balanced P1/P2 |
| 60 | **25.0%** | 45%/10% | 24/40 | **Minimum**, barely passed early stopping |
| 70 | **60.0%** | 30%/35% | 18/40 | Strong recovery |
| 80 | TBD | TBD | TBD | |
| 90 | TBD | TBD | TBD | |
| 100 | TBD | TBD | TBD | |

### Training Time
- **Total**: ~20+ hours on CPU
- **Per iteration**: ~12-17 minutes (significantly longer due to 10x model size)

### Training Dynamics
- **Loss**: Started at 1.36, dropped rapidly to ~0.6 by iteration 70
- **Policy loss**: Decreased from 5.8 to ~1.2 (good convergence)
- **Value loss**: Relatively stable around 0.17-0.23
- **Value correlation**: Low and unstable (0.11-0.37), never reached Exp 11's 0.4+ levels

## Analysis

### Observations

1. **Early Strong Performance (Iter 10: 67.5%)**
   - Model showed immediate capability, matching Exp 11's final result
   - Suggests large model learns basic strategy quickly
   - But this was unstable - not maintained

2. **Dramatic Instability**
   - Performance swung between 25% and 67.5%
   - Much higher variance than curriculum-trained models
   - Pure self-play creates unstable learning dynamics

3. **Self-Play Balance Issues**
   - P1/P2 balance varied wildly (60/10, 45/25, 30/35)
   - Model struggled to learn symmetric play
   - Pure self-play didn't correct this automatically

4. **MCTS Contribution**
   - MCTS became stronger over time (13→24 wins over policy)
   - But raw policy performance degraded
   - Model learned to rely on search rather than intuition

5. **Minimum at Iteration 60 (25%)**
   - Barely passed early stopping threshold
   - Shows pure self-play can lead to catastrophic forgetting
   - No Strategic curriculum means no "anchor" to prevent drift

### Key Finding: Pure Self-Play Failed

Despite 10x more parameters and SE attention blocks, pure self-play **performed worse** than curriculum-trained models:

| Experiment | Model Size | Training Approach | Final vs Strategic |
|------------|------------|-------------------|-------------------|
| Exp 8 | 1.2M | 40%→80% Strategic | **72.5%** |
| Exp 11 | 1.2M | Plateau curriculum | **72.5%** |
| **Exp 12** | **10.8M** | 100% Self-play | **~45%** average |

### Why Pure Self-Play Failed

1. **No External Benchmark**
   - Self-play creates closed-loop feedback
   - Model can develop strategies that beat itself but fail against external opponents
   - Strategic opponent provides grounding

2. **Strategy Cycling**
   - Model discovers strategy A, then counters with B, then C...
   - Can cycle through strategies without converging on robust play
   - Curriculum training prevents this by forcing adaptation to fixed opponent

3. **Draw Spiral**
   - High draw rates (40-54% in buffer)
   - Self-play can converge to defensive/drawing strategies
   - No incentive to find winning lines against passive play

4. **Value Head Quality**
   - Low correlation (0.1-0.3) suggests poor position evaluation
   - Without Strategic opponent, model doesn't learn which positions are actually winning
   - Value head needs diverse, known-quality training signals

5. **Model Capacity Not the Bottleneck**
   - 10x parameters didn't help
   - Confirms Exp 7's finding: training sequence > model size
   - Adding SE attention didn't overcome training dynamics issues

## Comparison to AlphaZero

AlphaZero famously used pure self-play, but with key differences:

| Factor | AlphaZero | Experiment 12 |
|--------|-----------|---------------|
| MCTS sims/move | 800-1600 | 800-1600 (same) |
| Training games | Millions | 10,000 (100 iter × 100 games) |
| Compute | Thousands of TPUs | Single CPU |
| Game complexity | Chess (10^43 positions) | TicTacDo (much smaller) |
| Diversity | Massive parallelism | Sequential learning |

**Key Insight**: Pure self-play requires massive scale to work. Our scale is insufficient.

## Conclusions

### What We Learned

1. **Model capacity alone insufficient**: 10x parameters with SE attention didn't overcome training issues
2. **Pure self-play unstable**: High variance, strategy cycling, draw spirals
3. **Curriculum training essential**: Strategic opponent provides stability and grounding
4. **MCTS can mask poor policy**: Model learned to rely on search, hiding policy weaknesses

### Recommendations

1. **Keep curriculum training**: 40%→80% (Exp 8) or plateau (Exp 11) approach works
2. **Model scaling requires training changes**: Don't just scale model without fixing training
3. **Some Strategic exposure needed**: Even 15-20% Strategic helps stabilize learning
4. **Consider hybrid approach**: Self-play for exploration + Strategic for grounding

### Future Directions

If pursuing large model + self-play again:

1. **Phased approach**: Start with curriculum, switch to self-play after convergence
2. **Population-based training**: Multiple agents with diversity pressure
3. **Regularization**: Entropy bonuses, policy distillation to prevent collapse
4. **Longer training**: Current 100 iterations may be insufficient for large model convergence

## Final Verdict

**Experiment 12: FAILED**

Pure self-play with a large model did not outperform curriculum-trained smaller models. The 10.8M parameter SE-ResNet achieved worse and more unstable results than the 1.2M parameter baseline with inverted curriculum.

**Best models remain**:
- Experiment 11 @ iteration 100: 72.5% (stable, good generalization)
- Experiment 8 @ iteration 50: 72.5% (stable baseline)
- Experiment 9 @ iteration 70: 75% (peak, requires early stopping)

---

## Appendix: Training Configuration Diff

### model.py changes

```python
# New SE-ResBlock class added
class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation attention"""
    def __init__(self, channels: int, reduction: int = 16):
        # ... SE attention implementation

# PolicyValueNet changes
hidden_channels = 256  # was 128
num_res_blocks = 8     # was 3, now SEResBlock
policy_channels = 128  # was 64
value_channels = 128   # was 64
```

### train.py changes

```python
# Pure self-play configuration
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.0
PEAK_STRATEGIC_OPPONENT_RATIO = 0.0
FINAL_STRATEGIC_OPPONENT_RATIO = 0.0
INITIAL_RANDOM_OPPONENT_RATIO = 0.0
FINAL_RANDOM_OPPONENT_RATIO = 0.0

# Higher exploration for self-play
DIRICHLET_SCALE = 0.30  # constant

# Adjusted for larger model
DEFAULT_BUFFER_SIZE = 8000
max_lr = 0.003
weight_decay = 3e-5
```

### eval_model.py additions

```python
def extended_evaluation(model, rng, iteration, num_games=20, mcts_simulations=100):
    """5-metric evaluation suite for pure self-play training"""
    # 1. Self-play balance (no MCTS)
    # 2. MCTS contribution as P1
    # 3. MCTS contribution as P2
    # 4. vs Strategic as P1
    # 5. vs Strategic as P2
```
