# TicTacDo AI - Hyperparameter Search Iteration Log

**Goal:** Achieve high-quality policy network that performs well vs MCTS
**Target:** >55% win rate vs MCTS by iteration 50
**Early Stop:** Iteration 30 if win rate < 40%

---

## Run 1: Current Configuration (Baseline - Failed)
**Status:** FAILED - 30% win rate at iteration 10
**Date:** 2025-10-05
**Iterations:** 280+ (stopped early due to poor performance)

### Configuration
```python
# Model
Hidden channels: 64
Residual blocks: 2
Policy channels: 32
Value channels: 32
Total params: ~235K

# Training
Episodes: 100
MCTS sims: 50-100
Epochs: 15
Batch size: 256
Buffer: 4000
Policy weight: 0.8

# Device: CPU (GPU was 3-4x slower)
```

### Results
- Iteration 10: ~30% win rate vs MCTS (P1: 43%, P2: 27%)
- No improvement through iteration 280+
- Training time per iteration: ~2-3 minutes

### Analysis
**Root causes:**
1. MCTS too weak (50-100 sims) - poor training signal
2. Model too small (235K params) - insufficient capacity
3. Training too shallow (15 epochs) - not extracting enough from data
4. GPU overhead dominated small model (forced CPU training)

**Conclusion:** Configuration optimized for speed at expense of learning quality

---

## Run 2: Restored Model + Strong MCTS (CPU)
**Status:** FAILED - 20% win rate at iteration 30
**Date:** 2025-10-05
**Duration:** ~4.5 hours (01:42 - 06:12 UTC)
**Iterations:** 30 (early stopped)

### Configuration
```python
# Model (RESTORED to original)
Hidden channels: 128
Residual blocks: 3
Policy channels: 64
Value channels: 64
Total params: 1,225,029 (larger than expected ~400K)

# Training (STRENGTHENED)
Episodes: 100 (kept for reasonable speed)
MCTS sims: 200-400 (INCREASED 3-4x from Run 1)
Epochs: 25 (INCREASED from 15)
Batch size: 512 (INCREASED from 256)
Buffer: 5000 (INCREASED from 4000)
Policy weight: 0.8

# Learning rate
Mode: stable
Max LR: 0.005

# Limits
Max iterations: 50
Early stop: iteration 30, < 40% win rate
Eval interval: 10
```

### GPU Test Results
- Tested 1.2M param model on GPU vs CPU
- CPU: 2.1ms per prediction
- GPU: 11.0ms per prediction
- Result: **GPU is 5x SLOWER, using CPU**
- Reason: Model still too small, GPU overhead dominates

### Results
- **Iteration 30**: 20.0% win rate vs MCTS (8/40 wins)
  - As Player 1: 0/20 wins (0%)
  - As Player 2: 8/20 wins (40%)
  - **EARLY STOPPING TRIGGERED** (20% < 40% threshold)
- **Final Loss**: 1.5347 (decreased from 2.86 at iter 1)
  - Policy loss: 1.9143
  - Value loss: 0.0247
- **Training time**: ~9 minutes per iteration
- **Total runtime**: 4.5 hours

### Analysis
**Hypothesis REJECTED:** Stronger MCTS did NOT improve learning

**Critical Findings:**
1. **Worse than Run 1**: 20% vs 30% - went backwards despite improvements
2. **Asymmetric Performance**:
   - Player 1 (first): 0% win rate (catastrophic)
   - Player 2 (second): 40% win rate (reasonable)
   - Suggests position-specific learning failure
3. **Loss decreased normally**: 2.86 → 1.53 indicates training is happening
4. **MCTS too strong**: Network can't learn from opponents it never beats
5. **Model size mismatch**: 1.2M params much larger than calculated ~400K

**Root Causes:**
1. **MCTS strength paradox**: 200-400 sims may be too strong for early iterations
   - Network starts weak, faces unbeatable opponent, learns helplessness
   - No positive reinforcement to guide learning
2. **Insufficient exploration**: Network may be converging to local minimum
   - Low Dirichlet noise (0.15) may not provide enough exploration
   - Needs more randomness in early training
3. **Player 1 failure**: Complete inability to win as P1 suggests fundamental issue
   - May need position-specific training or curriculum learning

**Conclusion:** "Stronger is better" doesn't apply - need graduated difficulty

### Next Steps
**Strategy: Curriculum Learning with Adaptive MCTS Strength**

1. **Graduated MCTS Difficulty:**
   - Start weak (50-100 sims) for first 10 iterations
   - Gradually increase to 200-400 over iterations 10-30
   - Allow network to build confidence before facing strong opponents

2. **Increase Exploration:**
   - Dirichlet noise: 0.15 → 0.3 for first 15 iterations
   - Higher temperature in early training
   - More diversity in training signal

3. **Model Size Verification:**
   - Investigate why model is 1.2M instead of ~400K params
   - May need to reduce model capacity to match training data

4. **Player Position Balance:**
   - Track P1 vs P2 performance separately
   - Add position-specific training weights if asymmetry persists

**Recommended Run 3 Configuration:**
- Adaptive MCTS: Start 50, end 200 (not 400)
- Dirichlet noise: 0.3 → 0.15 over 30 iterations
- Model: Reduce to actual ~400K params
- Keep: CPU, 100 episodes, 25 epochs, batch 512

### Issues Discovered
- **BUG 1 - FIXED:** Training loop didn't check MAX_ITERATIONS
  - Fixed in train.py:641 - added `and iteration < MAX_ITERATIONS` to while condition

- **BUG 2 - FIXED:** Unicode encoding crash with Greek letter μ in logging
  - Error: 'charmap' codec can't encode character '\u03bc'
  - Fixed in train.py:173 - changed "μ=" to "mean="

- **BUG 3 - FIXED:** Emoji characters crashed on Windows console
  - Error: 'charmap' codec can't encode character (emoji/warning symbols)
  - Fixed in train.py:185-187 - changed ⚠ to "WARNING -"
  - Fixed in autonomous_monitor.py:64 - removed 🛑 emoji
  - Fixed in check_training.py:37,39 - removed ⚠ and ✓ emoji
  - Training restarted at ~01:42 UTC with all Unicode issues resolved

---

## Run 3: Curriculum Learning - Graduated Difficulty (ABORTED)
**Status:** ABORTED - Value scale bug found
**Date:** 2025-10-05
**Strategy:** Adaptive MCTS strength + adaptive exploration
**Note:** Discovered critical bug where MCTS used [-0.8, 0.8] scale but training targets used [-1.0, 1.0] scale. This likely prevented effective learning in all previous runs.

---

## Run 4: Curriculum Learning with Fixed Value Scale (ABORTED)
**Status:** ABORTED - Critical perspective bug found at iteration 30
**Date:** 2025-10-05
**Iterations:** 30 (early stopped)
**Strategy:** Adaptive MCTS strength + adaptive exploration + FIXED VALUE SCALE

### Bug Fix Applied
**Critical:** Fixed value scale mismatch between MCTS and training targets
- MCTS `_get_game_value()`: Changed from score_diff/8 clamped to [-0.8, 0.8] → score_diff/6 clamped to [-1.0, 1.0]
- MCTS `_estimate_rollout_value()`: Updated to use score_diff/6 and [-0.7, 0.7] range
- Now matches `get_adjusted_value()` which uses score_diff/6 in [-1.0, 1.0] range
- **Expected impact:** Significantly improved value learning consistency

### Configuration
```python
# Model (Same as Run 2 & 3)
Hidden channels: 128
Residual blocks: 3
Policy channels: 64
Value channels: 64
Total params: 1,225,029

# Training (CURRICULUM LEARNING)
Episodes: 100
MCTS sims: 50→200 (ADAPTIVE - increases over iterations)
Dirichlet noise: 0.3→0.15 (ADAPTIVE - decreases over 30 iterations)
Epochs: 25
Batch size: 512
Buffer: 5000
Policy weight: 0.8

# Learning rate
Mode: stable
Max LR: 0.005

# Limits
Max iterations: 50
Early stop: iteration 30, < 40% win rate
Eval interval: 10
```

### Hypothesis
**Graduated Difficulty Approach:**
1. Start weak MCTS (50 sims) → network can win, learns from success
2. Gradually strengthen (→200 sims) as network improves
3. High exploration early (Dirichlet 0.3) → find good strategies
4. Reduce exploration later (→0.15) → refine strategies

**Expected Outcome:** Network builds confidence early, learns effectively through gradual challenge increase

### Results
- **Iteration 30:** Early stopping triggered (2.5% win rate vs MCTS)
  - Raw policy vs MCTS: 1 win, 10 draws, 69 losses = 2.5% win rate
  - Raw policy vs Strategic (P1): 17 wins, 3 draws, 6 losses = **65.4% win rate** ✅
  - Raw policy vs Strategic (P2): 4 wins, 3 draws, 7 losses = 28.6% win rate

### Critical Discovery
**Paradox Found:** Raw policy performs well vs Strategic opponent (65.4%) but terribly vs MCTS (2.5%)

**Investigation revealed SECOND CRITICAL BUG:**
- MCTS calls `get_game_state_representation()` without `subjective=True` (mcts.py:442, 539)
- Training uses `subjective=True` everywhere
- **MCTS interprets network predictions with wrong perspective!**
- Network learned: "positive value = good for current player"
- MCTS interprets: "positive value = good for Player ONE (absolute)"

This perspective mismatch causes MCTS to misuse the value head during tree search, explaining why MCTS destroys the learned policy even though the policy head works correctly.

### Analysis
**Two bugs found in Run 4:**
1. ✅ Value scale mismatch (FIXED before Run 4)
2. 🔴 Perspective mismatch (FOUND at iteration 30)

The value scale fix allowed the network to learn (65.4% vs Strategic proves this), but the perspective bug prevented MCTS from using the learned values correctly.

---

## Run 5: Both Fixes Applied (FAILED)
**Status:** FAILED - Early stopped at iteration 30
**Date:** 2025-10-05
**Iterations:** 30
**Strategy:** Adaptive MCTS strength + adaptive exploration + BOTH BUG FIXES

### Bugs Fixed
1. **Value scale fix** (from Run 4): MCTS now uses [-1.0, 1.0] scale matching training targets
   - mcts.py:709-710: Changed from `/8.0` and `[-0.8, 0.8]` → `/6.0` and `[-1.0, 1.0]`

2. **Perspective fix** (NEW in Run 5): MCTS now uses subjective state representation
   - mcts.py:442: `get_game_state_representation(subjective=True)` (root expansion)
   - mcts.py:539: `get_game_state_representation(subjective=True)` (leaf expansion)

### Configuration
```python
# Model (Same as Runs 2-4)
Hidden channels: 128
Residual blocks: 3
Policy channels: 64
Value channels: 64
Total params: 1,225,029

# Training (CURRICULUM LEARNING - same as Run 4)
Episodes: 100
MCTS sims: 50→200 (ADAPTIVE)
Dirichlet noise: 0.3→0.15 (ADAPTIVE)
Epochs: 25
Batch size: 512
Buffer: 5000
Policy weight: 0.8

# Learning rate
Mode: stable
Max LR: 0.005

# Limits
Max iterations: 50
Early stop: iteration 30, < 40% win rate
Eval interval: 10
```

### Hypothesis
**With both fixes applied:**
1. Value head will correctly guide MCTS search (perspective fix)
2. Value predictions on correct scale (value scale fix)
3. MCTS and network will be aligned (both use subjective perspective)
4. Raw policy win rate vs MCTS should gradually improve as network learns to mimic MCTS
5. Both raw policy and MCTS performance vs Strategic should improve

**Expected Results:**
- Early iterations: MCTS much stronger than raw policy (correct, as it should be)
- Mid iterations: Raw policy catches up as it learns MCTS patterns
- Target: >40% raw policy win rate vs MCTS by iteration 30

### Results
- **Iteration 10:** 72.5% win rate vs MCTS ✅ (81% P1, 63% P2) - Bugs confirmed fixed!
  - Raw policy vs Strategic: 9.5% P1, 5.3% P2
- **Iteration 20:** 60% win rate vs MCTS (68% P1, 52% P2) - Still aligned
  - Raw policy vs Strategic: 9.1% P1, 5.6% P2
- **Iteration 30:** 0% win rate vs MCTS - Early stopping triggered
  - Raw policy vs Strategic: **0% P1, 0% P2** 💀 Complete collapse

### Analysis
**Good news:** The bug fixes work! Iteration 10 showed 72.5% alignment (above 40% threshold).

**Bad news:** Network overfitted to weak MCTS:
1. Successfully learned to mimic MCTS (72.5% → 60% shows alignment)
2. But MCTS (50-200 sims) plays worse than Strategic opponent
3. Network learned bad strategies from bad teacher
4. Complete performance collapse by iteration 30

**Root cause:** MCTS with 50-200 simulations is not a strong enough teacher. Network is learning to copy MCTS's mistakes rather than discovering good strategies.

**Hypothesis:** Either MCTS needs significantly more simulations to play well, OR self-play alone is insufficient and we need to train against Strategic opponent.

---

## Run 6: Increased MCTS Strength Test (FAILED)
**Status:** FAILED - Early stopped at iteration 30
**Date:** 2025-10-05
**Iterations:** 30
**Strategy:** Test if stronger MCTS (100→300 sims) helps

### Change from Run 5
- MCTS sims: 100→300 (was 50→200)
- 50% increase to test if direction is useful
- All other params same as Run 5

### Configuration
```python
# Model (Same as Runs 2-5)
Hidden channels: 128
Residual blocks: 3
Policy channels: 64
Value channels: 64
Total params: 1,225,029

# Training
Episodes: 100
MCTS sims: 100→300 (INCREASED from 50→200)
Dirichlet noise: 0.3→0.15 (ADAPTIVE)
Epochs: 25
Batch size: 512
Buffer: 5000
Policy weight: 0.8

# Learning rate
Mode: stable
Max LR: 0.005

# Limits
Max iterations: 50
Early stop: iteration 30, < 40% win rate
Eval interval: 10
```

### Hypothesis
If MCTS with more sims is a better teacher:
- Raw policy vs MCTS: Should stay >40% (network can still learn to mimic)
- Raw policy vs Strategic: Should IMPROVE (learning better strategies)

If not helpful:
- Same pattern as Run 5 (good alignment, bad performance)
- Need different approach (train vs Strategic, not self-play)

### Results
- **Iteration 10:** 68.5% win rate vs MCTS (vs 72.5% in Run 5)
  - Raw policy vs Strategic: **5.3% P1, 4.8% P2** (vs 9.5% / 5.3% in Run 5) ❌ WORSE
- **Iteration 30:** 2.5% win rate vs MCTS - Early stopping triggered
  - Raw policy vs Strategic: **4.8% P1, 0% P2** 💀

### Analysis
**Conclusion: More MCTS sims made things WORSE**

Increasing MCTS from 50→200 to 100→300 degraded performance vs Strategic opponent. This proves:
1. MCTS self-play is converging to local strategies that work against itself but not real opponents
2. The Strategic opponent uses better tactics that MCTS isn't discovering
3. Deeper search doesn't help if searching in wrong direction

**Key insight discovered:** Network performance DEGRADES over iterations:
- Run 5: 9.5% → 9.1% → 0% vs Strategic
- Run 6: 5.3% → 0% vs Strategic

This is **overfitting**, not "MCTS is bad teacher". Network learns then forgets due to:
1. **Too many training epochs** (25) - overfits to replay buffer
2. **Learning rate too aggressive** (0.005) - causes instability
3. **Value head undertrained** (0.8 policy weight) - MCTS gets bad guidance

---

## Run 7: Anti-Overfitting Configuration
**Status:** READY (not started yet)
**Date:** 2025-10-05
**Strategy:** Fix overfitting with conservative training hyperparameters

### Changes from Run 6
1. **Epochs: 25 → 10** - Prevent overfitting to replay buffer
2. **Max LR: 0.005 → 0.003** - More stable training
3. **Policy weight: 0.8 → 1.0** - Equal value/policy importance, better MCTS guidance
4. **Buffer: 5000 → 8000** - More diverse experiences
5. **MCTS sims: 100→300 → 50→200** - Revert to Run 5 (better baseline)

### Configuration
```python
# Model (Same as Runs 2-6)
Hidden channels: 128
Residual blocks: 3
Policy channels: 64
Value channels: 64
Total params: 1,225,029

# Training (ANTI-OVERFITTING)
Episodes: 100
MCTS sims: 50→200 (reverted from 100→300)
Dirichlet noise: 0.3→0.15 (ADAPTIVE)
Epochs: 10 (REDUCED from 25)
Batch size: 512
Buffer: 8000 (INCREASED from 5000)
Policy weight: 1.0 (INCREASED from 0.8)

# Learning rate
Mode: stable
Max LR: 0.003 (REDUCED from 0.005)

# Limits
Max iterations: 50
Early stop: iteration 30, < 40% win rate
Eval interval: 10
```

### Hypothesis
**If overfitting was the problem:**
- Iteration 10: Should match/exceed Run 5 (9.5% vs Strategic)
- Iteration 20-30: Should MAINTAIN or IMPROVE (not degrade to 0%)
- Value head better trained → MCTS gets better guidance

**Expected results:**
- More stable learning curve
- Performance doesn't collapse over time
- Better MCTS alignment maintained throughout training

### Training Progress
- Awaiting user approval to start...

---

## Template for Future Runs

## Run N: [Name]
**Status:** [PLANNED/IN PROGRESS/COMPLETED/FAILED]
**Date:** YYYY-MM-DD
**Iterations:** X

### Configuration
```python
# Model
[params]

# Training
[params]
```

### Results
- Iteration 10: X% win rate
- Iteration 20: X% win rate
- Iteration 30: X% win rate
- Iteration 50: X% win rate

### Analysis
[What worked, what didn't, why]

### Next Steps
[What to try based on these results]

---
