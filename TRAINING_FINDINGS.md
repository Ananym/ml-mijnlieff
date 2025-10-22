# TicTacDo AI Training Findings & Recommendations

**Date:** 2025-10-05
**Training Duration:** ~12+ hours (280+ iterations)
**Status:** **Not viable with current hyperparameters**

---

## Executive Summary

After extensive hyperparameter tuning and training runs, the AlphaZero-style approach is **not learning effectively** with the current configuration. At iteration 10 (after ~20 minutes), the policy network achieved only **~30% win rate vs MCTS** (43% as P1, 27% as P2), and did not show meaningful improvement through iteration 280+.

**Early stopping has been implemented at iteration 30** to prevent wasting compute on non-viable training runs.

---

## Key Findings

### 1. Performance Metrics (Iteration 10)
- **Raw policy vs MCTS:** ~30% combined win rate
  - As P1: 3/7 wins (43%)
  - As P2: 3/11 wins (27%)
- **Expected for viable training:** >40% by iteration 30, trending upward

### 2. Root Causes Identified

#### A. **Training Signal Quality**
- **MCTS simulations too weak (50-100)** during self-play
  - Cut from 100-200 for speed, but this weakened the training targets
  - AlphaZero needs strong MCTS guidance to learn from
  - Recommendation: Restore to 100-300 range minimum

#### B. **Insufficient Training Iterations Per Update**
- **Epochs: 15** (reduced from 20)
- **Episodes: 100** (reduced from 150)
- **Batch size: 256** (reduced from 512)
- These cuts prioritized iteration speed over learning depth
- Recommendation: Restore epochs to 20-30, episodes to 150-200

#### C. **Model Capacity**
- **235K parameters** may be insufficient for this game's complexity
- Hidden channels: 64 (was 128)
- Residual blocks: 2 (was 3)
- Recommendation: Test original model size (400K params)

#### D. **CPU Performance Bottleneck**
- **GPU was 3-4x slower than CPU** for this small model
  - GPU overhead dominated compute time
  - Forced CPU training, limiting throughput
- Recommendation: Either scale up model for GPU, or accept CPU training

---

## Hyperparameter Change History

### Most Recent Configuration (Current)
```python
# Model
Hidden channels: 64
Residual blocks: 2
Policy head channels: 32
Total params: ~235K

# Training
Episodes per iteration: 100
MCTS simulations: 50-100 (scaled by iteration)
Training epochs: 15
Batch size: 256
Buffer size: 4000
Policy weight: 0.8

# Evaluation
Eval interval: 10 iterations
Eval games: 40 vs MCTS, 40 vs Strategic
```

### Previous Configuration (Before Speed Optimization)
```python
# Model
Hidden channels: 128
Residual blocks: 3
Policy head channels: 64
Total params: ~400K

# Training
Episodes per iteration: 150
MCTS simulations: 100-200
Training epochs: 20
Batch size: 512
Buffer size: 5000
Policy weight: 1.0

# Evaluation
Eval interval: 20 iterations
Eval games: 60 vs MCTS, 60 vs Strategic
```

---

## Recommendations for Next Steps

### Immediate Actions (High Priority)

1. **Restore MCTS Strength**
   ```python
   DEFAULT_MIN_MCTS_SIMS = 100  # Was 50
   DEFAULT_MAX_MCTS_SIMS = 300  # Was 100
   ```
   - Strong MCTS is critical for quality training data
   - Accept slower iteration time for better learning

2. **Increase Training Depth**
   ```python
   DEFAULT_EPISODES = 150  # Was 100
   DEFAULT_NUM_EPOCHS = 25  # Was 15
   DEFAULT_BATCH_SIZE = 512  # Was 256
   ```
   - More training per iteration = better policy extraction

3. **Restore Model Capacity**
   ```python
   # In model.py
   hidden_channels = 128  # Was 64
   num_residual_blocks = 3  # Was 2
   policy_channels = 64  # Was 32
   ```
   - Larger model can capture more game patterns
   - ~400K params is reasonable for this complexity

4. **Use Early Stopping**
   - Now implemented: stops at iteration 30 if win rate < 40%
   - Saves compute on failed training runs
   - Adjust thresholds based on empirical results

### Infrastructure Improvements (Medium Priority)

5. **GPU Training Optimization**
   - Options:
     - A. Accept CPU training with current model size
     - B. Scale up model to 1M+ params to justify GPU overhead
     - C. Investigate batched MCTS for GPU parallelism
   - Recommendation: Start with B (larger model on GPU)

6. **Distributed Training**
   - Self-play can be parallelized across multiple CPUs/GPUs
   - Network training stays on single GPU
   - Significant speedup for iteration time

7. **Checkpoint Management**
   - Save models every 10 iterations
   - Track evaluation metrics in JSON/CSV for analysis
   - Enable resuming from best checkpoint, not just latest

8. **Logging & Visualization**
   - Export training curves (loss, win rate) to TensorBoard or WandB
   - Plot MCTS value prediction vs actual outcomes
   - Track policy entropy over time

### Methodology Improvements (Lower Priority)

9. **Curriculum Learning**
   - Start with simpler opponents (random, basic strategic)
   - Gradually increase opponent strength
   - May help initial learning phase

10. **Learning Rate Scheduling**
    - Current: OneCycleLR with max_lr=0.005
    - Test: Cosine annealing with warm restarts
    - Monitor for overfitting vs underfitting

11. **Exploration Tuning**
    - Current: Dirichlet noise α=0.15, entropy bonus=0.05
    - May need more exploration early, less later
    - Track policy diversity metrics

12. **Value Target Bootstrapping**
    - Current: No bootstrapping (weight=0.0)
    - Test: Gradual increase from 0.0 to 0.3
    - Can provide smoother learning signal

---

## Proposed Next Experiment

**Goal:** Validate if stronger MCTS + larger model = viable learning

### Configuration
```python
# Model (restore original size)
Hidden channels: 128
Residual blocks: 3
Policy head channels: 64
Total params: ~400K

# Training (strengthen MCTS, reduce episodes if needed)
Episodes per iteration: 100  # Keep for speed
MCTS simulations: 150-400  # 3-4x stronger
Training epochs: 25
Batch size: 512
Policy weight: 0.8

# Early stopping
Check at iteration: 30
Minimum win rate: 45%  # Slightly higher threshold
```

### Expected Outcomes
- **Success:** >45% win rate by iteration 30, trending upward
- **Failure:** <40% win rate → reassess approach entirely

### Estimated Runtime
- **Per iteration:** ~8-10 minutes (vs current ~2-3 minutes)
- **To iteration 30:** ~4-5 hours
- **Trade-off:** 2x slower, but much better learning signal

---

## Alternative Approaches (If AlphaZero Continues to Fail)

### 1. **Supervised Learning + Fine-tuning**
- Pre-train on games from StrategicOpponent vs itself
- Bootstrap policy network with known-good moves
- Then switch to AlphaZero-style reinforcement learning

### 2. **Simpler RL Algorithms**
- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)
- Less sample-efficient but more robust

### 3. **Hybrid Approach**
- Use MCTS for actual play (it's working well)
- Train policy network purely for move ordering/pruning
- Reduces burden on network to "solve" the game

---

## Minimum Viable Testing Protocol

Going forward, use this protocol to quickly validate training runs:

1. **Iteration 10:** Check win rate ≥ 35%
   - If failed: Hyperparameters are severely broken
   - Runtime: ~20 minutes

2. **Iteration 20:** Check win rate ≥ 40% and improving
   - If failed: Learning is too slow
   - Runtime: ~40 minutes

3. **Iteration 30:** Check win rate ≥ 45% (early stopping trigger)
   - If failed: Stop training, adjust hyperparameters
   - If passed: Continue to iteration 100
   - Runtime: ~1 hour

4. **Iteration 100:** Target win rate ≥ 55%
   - Competitive with MCTS
   - Runtime: ~3-4 hours

**Total validation time:** 1 hour to know if run is viable, vs 12+ hours wasted

---

## Conclusion

The current training configuration prioritized iteration speed at the expense of learning quality. The model is iterating quickly but not learning effectively from weak MCTS targets (50-100 sims) and insufficient training depth (15 epochs, 100 episodes).

**Recommended next action:** Restore stronger MCTS (150-400 sims) and larger model (400K params), accepting slower iteration time (~8 min vs 2 min) in exchange for viable learning. Use early stopping at iteration 30 to validate within ~4-5 hours instead of running for 12+ hours.

---

## Files Modified

- `train.py:34-37` - Added early stopping constants
- `train.py:1134-1161` - Implemented early stopping logic
- Model retains current 235K param configuration (pending restoration)

## Next Session Checklist

- [ ] Restore model to 400K params (hidden=128, res_blocks=3, policy=64)
- [ ] Increase MCTS sims to 150-400 range
- [ ] Increase epochs to 25, batch to 512
- [ ] Run training with early stopping enabled
- [ ] Monitor iteration 30 checkpoint
- [ ] If viable, let run to iteration 100 and reassess
