# Experiment 14: Large Model + Mostly Self-Play with Strategic Anchor

## Hypothesis

Combine Exp 12's strengths (large model, self-play, good human play) with a small Strategic anchor for stability:
- **85% self-play** to preserve the diverse, creative strategies that made Exp 12 good against humans
- **15% Strategic** as a stability anchor (not a learning target)
- **10.8M SE-ResNet** since large model + self-play produced the best human-facing model

Goals:
1. Maintain Exp 12's human play quality ("pretty good, beatable if you know what you're doing")
2. Add stability from Strategic anchor to reduce training variance
3. Avoid the overfitting that ruined Exp 13 (which had 40-85% Strategic exposure)

## Skepticism

User concern: "I have a hard time believing that a self-play-trained small model is going to be able to beat a human with any reliability"

**Key insight from user**: Human play quality ranking is:
1. **Exp 12** (10.8M, pure self-play) - "pretty good, but beatable if you know what you're doing"
2. **Exp 11** (1.2M, plateau curriculum) - "could barely play the game in real terms"
3. **Exp 13** (10.8M, plateau curriculum) - "stupid", easiest to beat

This is shocking:
- **Exp 11 has 72.5% Strategic win rate but "could barely play the game"**
- **Strategic win rate is almost meaningless** for human play quality
- **Pure self-play (Exp 12) produced the only decent human-facing model**
- **Strategic curriculum actively hurts** regardless of model size

Implication: Maybe we should stick with the **large model** since Exp 12 was the best human-facing model. The question becomes: can we stabilize Exp 12's training while preserving its human play quality?

## Configuration

### Model
- **Size**: 10,848,517 parameters (Exp 12/13 SE-ResNet architecture)
- **Hidden channels**: 256
- **Blocks**: 8 SE-ResBlocks with squeeze-and-excitation
- **Learning rate**: 0.002 (reduced for large model stability)

### Curriculum
- **Self-play**: 85% constant throughout
- **Strategic**: 15% constant throughout
- **No curriculum ramp** - flat ratio to isolate the effect of high self-play

### Training
- **Iterations**: 100
- **Episodes per iteration**: 100
- **MCTS simulations**: 800-1600 (scaling with iteration)
- **Dirichlet noise**: 0.25 (moderate exploration)
- **Policy weight**: 0.2 (80% value, 20% policy)
- **Bootstrap**: 0.0 (pure game outcomes)

### New Health Metrics
Track these in checkpoint metadata for early warning:
1. **Self-play P1 win rate** - should stay 30-70%, not collapse to 0% or 100%
2. **MCTS contribution** - should stay >10%, not drop to 0%
3. **Self-play draw rate** - monitor for draw spiral

### Early Stopping Criteria
Consider stopping if:
- Self-play P1 wins < 20% for 3 consecutive evals (degenerate P2 strategy)
- Self-play P1 wins > 80% for 3 consecutive evals (degenerate P1 strategy)
- MCTS contribution < 5% for 3 consecutive evals (overconfident bad policy)

## Code Changes Required

### train.py
```python
# Experiment 14: Mostly Self-Play
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.15  # Constant 15%
PEAK_STRATEGIC_OPPONENT_RATIO = 0.15     # No ramp
FINAL_STRATEGIC_OPPONENT_RATIO = 0.15    # Stay at 15%

# Simplified - no phases, just constant ratio
def get_opponent_ratios(iteration):
    return 0.0, 0.15  # 0% random, 15% strategic, 85% self-play
```

### model.py
- Keep 10.8M SE-ResNet (256 hidden channels, 8 SE-ResBlocks)
- Learning rate: 0.002 (same as Exp 13)

### Checkpoint metadata
Add to eval_summary:
- `self_play_p1_winrate` (already tracked)
- `self_play_draw_rate`
- Flag if health metrics breach thresholds

## Expected Outcomes

### Optimistic
- Model matches or exceeds Exp 12's human play quality
- 15% Strategic anchor adds stability without hurting creativity
- Achieves 50-60% vs Strategic (better than Exp 12's ~45%)
- Self-play stays balanced throughout
- Best of both worlds: good human play + stable training

### Pessimistic
- 15% Strategic is enough to cause overfitting (like Exp 13)
- Loses Exp 12's creative play while not gaining Strategic stability
- Worst of both worlds

### Middle ground
- Similar to Exp 12 in human play quality
- Slightly better Strategic stability but not dramatic
- Confirms that self-play is the key, Strategic is mostly noise

### Either way, we learn something
- If it works: 15% Strategic is the sweet spot for stability without overfitting
- If it matches Exp 12: Strategic anchor is irrelevant, pure self-play is fine
- If it's worse: even 15% Strategic causes overfitting in large models

## Comparison Targets

| Metric | Exp 11 (plateau) | Exp 12 (pure self-play) | Exp 13 (plateau+big) | Exp 14 (target) |
|--------|------------------|-------------------------|----------------------|-----------------|
| Model size | 1.2M | 10.8M | 10.8M | **10.8M** |
| Strategic exposure | 40-85-40% | 0% | 40-85-40% | **15% constant** |
| Peak Strategic WR | 70% | 67.5% | 70% | 50-60%? |
| Final Strategic WR | 72.5% | ~45% | 65% | 50-60%? |
| Human play quality | Barely plays | Pretty good | Stupid | **Good+stable?** |
| Self-play balance | Healthy | Varied | Collapsed | Healthy? |

## Training Time Estimate
- 10.8M model on CPU
- ~17 hours for 100 iterations (similar to Exp 12/13)

## Success Criteria
1. **Primary**: Plays well against humans (subjective but important)
2. **Secondary**: Self-play balance stays healthy (P1 30-70%)
3. **Tertiary**: Achieves >55% vs Strategic
