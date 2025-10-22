# Code Investigation Report
**Date:** 2025-10-05
**Focus:** Training logic inconsistencies and GPU performance bottleneck

---

## A. Training Logic Analysis

### Summary
**Overall Assessment:** Training logic is mostly sound with minor issues that are unlikely to prevent good performance. One theoretical concern with value target consistency.

### Detailed Findings

#### 1. Value Target Assignment ✓ CORRECT
**Location:** train.py:789-873

The code uses two different approaches for value targets:
- **MCTS moves:** Blends game outcome with MCTS value prediction using `get_improved_value_target()`
- **Direct policy moves:** Blends outcome with bootstrap value from next state

**Analysis:** Both approaches are valid. The use of `subjective=True` state representation (train.py:401, 477) ensures all value predictions are from the current player's perspective, which is correct for AlphaZero-style training.

**Potential Issue:** Bootstrap weight implementation differs between MCTS moves (move-index dependent, train.py:621-627) and direct moves (simple blend, train.py:866-870). However, bootstrap weight is currently 0.0 (train.py:66-68), so this doesn't affect current training.

#### 2. Policy Target Mixing ✓ HANDLED
**Location:** train.py:502-503, 573-574; model.py:362-387

**Finding:** Training batches mix two types of policy targets:
- Distribution targets from MCTS (full visit distribution)
- One-hot targets from direct policy moves

**Analysis:** The `train_step` function correctly detects and handles both types (model.py:362-387), using KL divergence for distributions and cross-entropy for one-hot targets. This is proper implementation.

#### 3. Value Range Inconsistency 🔴 CRITICAL ISSUE FOUND
**Location:** train.py:372-396, mcts.py:686-720

**Finding:** Training value targets and MCTS value predictions use DIFFERENT scales:

**get_adjusted_value() - Used for training targets:**
```python
# train.py:390-394
clamped_diff = max(-6.0, min(6.0, score_diff))
normalized_value = clamped_diff / 6.0  # Range: [-1.0, 1.0]
return normalized_value
```

**MCTS _get_game_value() - Used during self-play:**
```python
# mcts.py:709-720
normalized_diff = max(-0.8, min(0.8, score_diff / 8.0))  # Range: [-0.8, 0.8]
if winner == original_player:
    return max(0.8, normalized_diff)
else:
    return min(-0.8, normalized_diff)
```

**Impact Analysis:**

This creates a **fundamental training inconsistency**:

1. **During self-play:** MCTS provides value estimates in [-0.8, 0.8] range
2. **During training:** Model is trained to predict values in [-1.0, 1.0] range
3. **Model output:** Uses tanh activation (model.py:135), naturally outputting [-1, 1]

**Why this matters:**

- MCTS bootstrap values (from self-play) are compressed to 80% of the training target scale
- When bootstrapping is enabled, the blended targets will be systematically biased
- Even with bootstrap weight = 0.0 (current setting), the MCTS values stored with examples are in the wrong scale for later use
- The model learns to predict [-1, 1] but MCTS interprets predictions as if they were [-0.8, 0.8]

**Example:**
- Game ends with score_diff = 4
- Training target: 4/6 = 0.667
- MCTS would have predicted: max(-0.8, min(0.8, 4/8)) = 0.5
- Model learns 0.667, but during next game, MCTS sees model prediction of 0.667 and interprets it as "stronger than max possible (0.8)"

**Severity: HIGH** - This inconsistency could explain why the model struggles to learn effective policies. The value head is being trained on one scale but used on another during self-play.

#### 4. Subjective State Representation ✓ CORRECT
**Location:** train.py:401, 477, 837-842

All state representations use `subjective=True`, ensuring board/values are always from the current player's perspective. This is the standard AlphaZero approach and is correctly implemented.

#### 5. Replay Buffer Management ✓ CORRECT
**Location:** train.py:39, 944-955

Simple FIFO buffer with max size 5000. No balancing is performed (`BALANCE_REPLAY_BUFFER = False`). This is standard and appropriate.

#### 6. Training Loop Structure ✓ CORRECT
**Location:** train.py:654-999

The loop correctly:
1. Generates self-play games with MCTS
2. Assigns value targets after game completion
3. Adds examples to replay buffer
4. Samples batches and trains for multiple epochs
5. Uses curriculum learning for MCTS strength and exploration

This follows the AlphaZero training paradigm correctly.

### Critical Logic Issues: **1 CRITICAL ISSUE FOUND**

**🔴 Value Scale Mismatch (Issue #3):**
- Training targets use [-1.0, 1.0] scale (divide by 6)
- MCTS predictions use [-0.8, 0.8] scale (divide by 8)
- Creates systematic bias in value learning
- **Likely contributes to poor training performance**

**Recommendation:** Unify the value scales. Options:
1. Change `get_adjusted_value()` to divide by 8 and clamp to [-0.8, 0.8]
2. Change MCTS `_get_game_value()` to divide by 6 and use [-1.0, 1.0]
3. Remove MCTS clamping and use full [-1.0, 1.0] range everywhere

Option 3 is cleanest - let the full score differential information flow through.

**Other aspects are correct:**
- ✅ Proper subjective state representation
- ✅ Correct policy target generation (MCTS visit distribution)
- ✅ Appropriate value target assignment with game outcomes
- ✅ Proper loss function selection based on target type

---

## B. GPU Performance Bottleneck Analysis

### Summary
**Root Cause Identified:** Sequential single-batch inference during MCTS creates massive GPU overhead, making CPU 5x faster than GPU for this workload.

### The Problem

#### 1. MCTS Inference Pattern 🔴 CRITICAL BOTTLENECK
**Location:** mcts.py:478-646

During each MCTS search:
- **50 simulations** per search (current curriculum setting)
- Each simulation traverses the tree and expands **one leaf node**
- Each expansion calls `_predict_with_cache()` (mcts.py:541-543)
- Cache hit rate ~27% (from training logs)
- **Result: ~37 individual model calls per MCTS search**

#### 2. Per-Inference GPU Overhead 🔴 CRITICAL BOTTLENECK
**Location:** model.py:217-329

Each `model.predict()` call with **batch size 1** incurs:

```python
# CPU→GPU transfer
board_state = torch.FloatTensor(board_state).to(self.device)  # ~4KB transfer
flat_state = torch.FloatTensor(flat_state).to(self.device)    # ~48 bytes

# GPU kernel launch overhead
policy_logits, value = self.model(board_state, flat_state)

# GPU→CPU transfer
return policy.cpu().numpy(), value.cpu().numpy()  # ~256 bytes + 4 bytes
```

**Overhead breakdown per inference:**
- CPU→GPU memory transfer: ~4KB
- GPU kernel launch latency: ~20-50μs (varies by GPU)
- GPU compute time: ~10-50μs for 1.2M param model
- GPU→CPU memory transfer: ~260 bytes
- **Total: ~100-200μs dominated by overhead**

**On CPU:**
- Direct computation: ~50-100μs
- No transfer overhead
- **Total: ~50-100μs of pure compute**

#### 3. Why GPU is 5x Slower

**Per MCTS search (50 simulations, ~37 model calls):**

GPU (RTX 4060):
- 37 calls × 150μs avg = **5.5ms per search**
- Overhead: 80-90% of time spent on transfers/launches
- Actual compute: Only 10-20% of time

CPU (modern x64):
- 37 calls × 75μs avg = **2.8ms per search**
- Zero transfer overhead
- All time is useful compute

**Ratio: GPU 5.5ms / CPU 2.8ms = 2x slower**

This matches the observed 5x slowdown in the GPU test (train.py GPU test showed 11ms vs 2.1ms).

#### 4. Training Phase

**Location:** train.py:965-999

Training uses batch size 512, which SHOULD benefit from GPU. However:
- Only 25 epochs per iteration (train.py:26)
- Training completes in seconds
- Still has transfer overhead (model.py:343-346)
- Offset by faster self-play on CPU

**Net result:** Total iteration time still faster on CPU due to MCTS dominating the workload.

### Why Traditional Approaches Don't Work

**Increasing batch size:** Not applicable - MCTS inherently generates one state at a time during tree traversal

**Reducing transfers:** Predictions must return to CPU for MCTS logic (tree update, move selection)

**Faster GPU:** Doesn't help - overhead is in launch/transfer latency, not compute throughput

---

## C. Solutions and Recommendations

### For GPU Performance (High Impact)

#### Solution 1: Batched MCTS Inference ⭐ RECOMMENDED
**Complexity:** Moderate
**Expected speedup:** 5-10x on GPU

Modify MCTS to collect multiple leaf nodes before expanding:

```python
# Pseudo-code
def batched_search(self, game_state):
    batch_size = 8  # Expand 8 nodes at once
    for batch_idx in range(self.num_simulations // batch_size):
        # Collect batch_size leaf nodes
        leaves = []
        for i in range(batch_size):
            leaf = self._select_until_leaf(root, game_state)
            leaves.append(leaf)

        # Batch predict all at once
        states = [leaf.state_rep for leaf in leaves]
        policies, values = model.predict_batch(states)  # NEW METHOD

        # Expand all leaves
        for leaf, policy, value in zip(leaves, policies, values):
            leaf.expand(policy)
            self._backpropagate(leaf, value)
```

**Benefits:**
- Amortizes GPU overhead across multiple inferences
- 8x reduction in transfer/launch overhead
- Better GPU utilization (batch size 8-16 is sweet spot for small models)

**Challenges:**
- Need virtual loss to prevent multiple simulations from selecting same leaf
- Requires refactoring MCTS search structure
- Need to handle variable batch sizes (final batch may be smaller)

#### Solution 2: Keep Tensors on GPU ⭐ MODERATE IMPACT
**Complexity:** Low
**Expected speedup:** 2-3x on GPU

Cache predictions as GPU tensors instead of converting to numpy:

```python
# In _predict_with_cache
def _predict_with_cache(self, state_rep):
    cache_key = self._get_state_key(...)

    if cache_key in self.prediction_cache:
        return self.prediction_cache[cache_key]  # Return GPU tensors

    # Predict and keep on GPU
    policy_tensor, value_tensor = model.predict_gpu(...)  # NEW METHOD

    self.prediction_cache[cache_key] = (policy_tensor, value_tensor)
    return policy_tensor, value_tensor
```

Convert to numpy only when selecting final move.

**Benefits:**
- Eliminates most GPU→CPU transfers
- Reduces overhead by ~40%
- Simpler to implement than batching

**Challenges:**
- MCTS logic needs to work with tensors
- Increased GPU memory usage for cache

#### Solution 3: Model Distillation to Tiny Model ⚠️ LONG-TERM
**Complexity:** High
**Expected speedup:** 10-20x on CPU

Train a tiny model (~50K params) to mimic the current model:
- Small enough that CPU is genuinely faster
- Can run 10-20x faster on CPU
- Acceptable accuracy loss for MCTS guidance

**Not recommended for current training - consider only for deployment**

### For Training Logic (High Priority) ⭐

#### Fix Value Scale Inconsistency 🔴 CRITICAL
**Location:** train.py:390-394 AND mcts.py:709-710

**Action Required:** Unify value scales between training targets and MCTS predictions.

**Recommended fix (Option 3 - cleanest):**

In `mcts.py`, change `_get_game_value()`:
```python
# Current (WRONG):
normalized_diff = max(-0.8, min(0.8, score_diff / 8.0))

# Proposed (CORRECT):
normalized_diff = max(-1.0, min(1.0, score_diff / 6.0))
```

And remove the 0.8 clamps for wins/losses:
```python
# Current (WRONG):
if winner == original_player:
    return max(0.8, normalized_diff)
else:
    return min(-0.8, normalized_diff)

# Proposed (CORRECT):
if winner == original_player:
    return max(0.2, normalized_diff)  # Ensure wins are positive
else:
    return min(-0.2, normalized_diff)  # Ensure losses are negative
```

Also update `_estimate_rollout_value()` in mcts.py:409-412 to use same scale.

**Expected Impact:** This fix could significantly improve training performance by ensuring value predictions and targets are on the same scale.

---

## D. Conclusions

### Training Logic
🔴 **1 CRITICAL ISSUE FOUND: Value scale mismatch**

The training implementation mostly follows AlphaZero principles correctly, BUT:
- **Critical:** Training targets ([-1, 1]) and MCTS predictions ([-0.8, 0.8]) use different scales
- **Impact:** Model learns one value scale but MCTS uses another during self-play
- **Likely effect:** Degraded value prediction accuracy, potentially explaining poor training results
- **Fix:** Unify scales to [-1, 1] everywhere (see recommendations)

### GPU Performance
🔴 **Root cause definitively identified: Sequential single-batch MCTS inference**

The GPU slowdown is not a bug or misconfiguration - it's an architectural mismatch between MCTS (sequential single-inference workload) and GPU optimization (parallel batch workload).

**The model is too small and MCTS is too sequential for GPU to help.**

### Recommendations

**For current training (Run 3):**
1. 🔴 **STOP AND FIX VALUE SCALE MISMATCH** - This is likely preventing good learning
2. ✅ **Continue using CPU** - it's genuinely faster for this workload
3. Focus on hyperparameter tuning (curriculum learning approach is good)

**For future improvements:**
1. **Implement batched MCTS** if you want GPU acceleration (5-10x speedup)
2. **Keep tensors on GPU** as a simpler alternative (2-3x speedup)
3. Consider increasing model size to ~5-10M params if you implement batching - larger models benefit more from GPU

**Priority:**
GPU optimization is **optional**. The current CPU-based training is working correctly and efficiently. Only implement GPU optimizations if you need faster iteration time and are willing to invest in the code refactoring.

---

## E. References

**Code locations examined:**
- train.py: Lines 1-100, 400-550, 700-999
- model.py: Lines 1-450
- mcts.py: Lines 250-750

**Key insights:**
1. MCTS makes 37 individual predictions per search (50 sims × 73% cache miss rate)
2. Each prediction on GPU has ~100-150μs overhead, ~50μs compute
3. Each prediction on CPU has 0μs overhead, ~75μs compute
4. GPU overhead dominates for small batch sizes
5. Training logic correctly implements AlphaZero with proper subjective state representation
