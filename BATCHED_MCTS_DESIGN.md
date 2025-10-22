# Batched MCTS Design Document

**Goal:** Enable GPU-accelerated MCTS by batching multiple leaf node evaluations together

**Expected Speedup:** 5-10x on GPU (vs current 5x slower)

---

## Problem Statement

Current MCTS makes ~37 individual model predictions per search (50 simulations × 73% cache miss rate). Each prediction:
- Has batch size 1
- Incurs ~100-150μs GPU overhead (transfers + kernel launch)
- Only ~50μs actual compute

**Result:** GPU is 5x slower than CPU due to overhead domination.

---

## Solution: Batched Leaf Expansion

### Core Concept

Instead of expanding one leaf node at a time, collect multiple leaf nodes and evaluate them in a single batched prediction:

```python
# Current (Sequential)
for sim in range(num_simulations):
    leaf = select_leaf()
    policy, value = model.predict(leaf)  # Batch size 1
    expand_and_backprop(leaf, policy, value)

# Proposed (Batched)
batch_size = 8
for batch_idx in range(num_simulations // batch_size):
    leaves = []
    for i in range(batch_size):
        leaf = select_leaf()
        leaves.append(leaf)

    # Single batched prediction
    policies, values = model.predict_batch(leaves)  # Batch size 8

    for leaf, policy, value in zip(leaves, policies, values):
        expand_and_backprop(leaf, policy, value)
```

---

## Key Challenges

### 1. Leaf Selection Conflicts ⚠️ CRITICAL

**Problem:** Multiple simulations might select the same leaf node

**Solution:** Virtual Loss
- When selecting a leaf, add a temporary "virtual loss" to discourage other simulations from selecting it
- Remove virtual loss after expansion

```python
class MCTSNode:
    def __init__(self):
        self.virtual_losses = 0  # NEW

    def get_value(self):
        if self.visits == 0:
            return 0.0
        # Subtract virtual losses from visit count to make node less attractive
        effective_visits = self.visits + self.virtual_losses
        return self.value_sum / effective_visits
```

### 2. State Management

**Problem:** Need to maintain separate game states for each simulation in the batch

**Current approach:** Single shared state that's modified and undone
**Batched approach:** Multiple cloned states

```python
class BatchedMCTS:
    def search(self, game_state):
        batch_size = 8

        # Clone state for each simulation in batch
        batch_states = [clone_game_state(game_state) for _ in range(batch_size)]
        batch_paths = [[] for _ in range(batch_size)]  # Track moves for each
        batch_leaves = [None] * batch_size

        # Select leaves in parallel
        for i in range(batch_size):
            node, path = self._select_leaf(root, batch_states[i])
            batch_leaves[i] = node
            batch_paths[i] = path
```

### 3. Batched Model Inference

**Need:** New method to predict multiple states at once

```python
class ModelWrapper:
    def predict_batch(self, state_reps, legal_moves_list=None):
        """Predict for multiple states in one forward pass"""
        batch_size = len(state_reps)

        # Stack all boards and flat states
        board_batch = np.stack([sr.board for sr in state_reps])
        flat_batch = np.stack([sr.flat_values for sr in state_reps])

        # Single GPU call
        board_tensor = torch.FloatTensor(board_batch).to(self.device)
        flat_tensor = torch.FloatTensor(flat_batch).to(self.device)

        # Batch forward pass
        with torch.no_grad():
            policies, values = self.model(board_tensor, flat_tensor)

        # Return as list of numpy arrays
        return policies.cpu().numpy(), values.cpu().numpy()
```

### 4. Cache Compatibility

**Issue:** Prediction cache uses single-state keys

**Solution:** Check cache before batching, only batch uncached states

```python
def _batch_predict_with_cache(self, state_reps):
    results = [None] * len(state_reps)
    uncached_indices = []
    uncached_states = []

    # Check cache
    for i, state_rep in enumerate(state_reps):
        cache_key = self._get_state_key(state_rep.board, state_rep.flat_values)
        if cache_key in self.prediction_cache:
            results[i] = self.prediction_cache[cache_key]
        else:
            uncached_indices.append(i)
            uncached_states.append(state_rep)

    # Batch predict uncached
    if uncached_states:
        policies, values = self.model.predict_batch(uncached_states)
        for idx, policy, value in zip(uncached_indices, policies, values):
            results[idx] = (policy, value)
            # Update cache
            cache_key = self._get_state_key(uncached_states[idx].board, ...)
            self.prediction_cache[cache_key] = (policy, value)

    return results
```

---

## Implementation Plan

### Phase 1: Core Batched MCTS (New File)
**File:** `mcts_batched.py`
**Estimated effort:** 4-6 hours

```python
class BatchedMCTS:
    def __init__(self, model=None, num_simulations=100, batch_size=8, c_puct=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.batch_size = batch_size  # NEW
        self.c_puct = c_puct
        # ... other params same as MCTS

    def search(self, game_state):
        """Main entry point - same interface as MCTS"""
        root = MCTSNode(move=None, parent=None)

        # Expand root
        policy, value = self._expand_node(root, game_state)

        # Run batched simulations
        num_batches = self.num_simulations // self.batch_size
        for batch_idx in range(num_batches):
            self._run_batch_simulation(root, game_state)

        # Return same format as MCTS
        return self._get_visit_distribution(root), root

    def _run_batch_simulation(self, root, game_state):
        """Run batch_size simulations in parallel"""
        # 1. Select batch_size leaves with virtual loss
        batch_leaves, batch_states, batch_paths = self._select_batch_leaves(
            root, game_state, self.batch_size
        )

        # 2. Batch expand all leaves
        policies, values = self._batch_expand_nodes(batch_leaves, batch_states)

        # 3. Backpropagate and remove virtual losses
        for leaf, value, path in zip(batch_leaves, values, batch_paths):
            self._backpropagate(leaf, value, remove_virtual_loss=True)
```

### Phase 2: Virtual Loss Implementation
**Estimated effort:** 1-2 hours

Add virtual loss to MCTSNode:
```python
class MCTSNode:
    def __init__(self, ...):
        self.virtual_losses = 0

    def apply_virtual_loss(self):
        self.virtual_losses += 1

    def remove_virtual_loss(self):
        self.virtual_losses = max(0, self.virtual_losses - 1)

    def get_value(self):
        # Adjust for virtual losses
        effective_visits = self.visits + self.virtual_losses
        if effective_visits == 0:
            return 0.0
        return self.value_sum / effective_visits
```

### Phase 3: Batched Model Interface
**Estimated effort:** 2-3 hours

Add to `ModelWrapper`:
```python
def predict_batch(self, board_states, flat_states, legal_moves_list=None):
    """Batch prediction for multiple states"""
    # Implementation as shown above
```

### Phase 4: Integration & Testing
**Estimated effort:** 2-3 hours

- Add `--mcts-type` flag to train.py
- Test GPU speedup vs CPU
- Verify same search quality as sequential MCTS
- Benchmark different batch sizes (4, 8, 16)

---

## Expected Performance

### Batch Size Analysis

**Batch size 4:**
- 50 sims ÷ 4 = 13 batches
- ~37 predictions → 10 batches (with cache)
- GPU time: 10 × 300μs = 3ms (vs 5.5ms sequential)
- **Speedup: ~1.8x**

**Batch size 8:**
- 50 sims ÷ 8 = 7 batches
- ~37 predictions → 5 batches (with cache)
- GPU time: 5 × 500μs = 2.5ms
- **Speedup: ~2.2x vs sequential GPU, ~1.1x vs CPU**

**Batch size 16:**
- 50 sims ÷ 16 = 4 batches
- ~37 predictions → 3 batches (with cache)
- GPU time: 3 × 800μs = 2.4ms
- **Speedup: ~2.3x vs sequential GPU, ~1.2x vs CPU**

### GPU vs CPU with Batching

**GPU (batched, size 8):**
- Per MCTS: ~2.5ms
- 100 games × ~2.5ms = 250ms per iteration
- **3.7x faster than current GPU (9.3ms)**
- **1.1x faster than current CPU (2.8ms)**

**Conclusion:** Batching makes GPU competitive with CPU, slight edge to GPU

---

## Risks & Mitigations

### Risk 1: Virtual loss breaks convergence
**Mitigation:** Use small virtual loss value (0.1-1.0), tune empirically

### Risk 2: Batch size too large → quality degradation
**Mitigation:** Start with batch_size=4, benchmark quality vs sequential

### Risk 3: Implementation complexity
**Mitigation:** Create separate file, keep original MCTS as fallback

### Risk 4: Minimal speedup if cache hit rate increases
**Mitigation:** Even with 50% cache hit rate, batching still helps

---

## Recommendation

### For Current Run 4:
**Don't implement yet** - Let Run 4 complete with the value scale fix to validate the bug fix effectiveness

### For Future:
**Implement if:**
1. Run 4 shows significantly improved learning (validates value scale fix)
2. You want to train with stronger MCTS (200-400 sims) where GPU would help more
3. You're willing to invest 8-12 hours in implementation

**Prototype approach:**
1. Create `mcts_batched.py` as separate file
2. Implement with batch_size=8 as default
3. Add `--mcts-batched` flag to train.py
4. Benchmark GPU vs CPU with both implementations
5. Compare policy quality (visit distributions should be similar)

---

## Alternative: Simpler GPU Optimization

If batched MCTS is too complex, consider:

**Keep tensors on GPU between predictions** (2-3 hours effort):
- Don't convert predictions to numpy until final move selection
- Cache GPU tensors instead of numpy arrays
- Expected speedup: 2-3x on GPU
- Makes GPU ~1.5x faster than CPU

This is much simpler and still provides meaningful speedup.
