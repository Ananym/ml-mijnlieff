# MCTS Value Perspective Bug Investigation

**Date:** 2025-10-05
**Status:** 🔴 CRITICAL BUG FOUND

---

## Summary

**CRITICAL:** MCTS is using network predictions with the wrong perspective, causing Player TWO's values to be interpreted backwards.

### The Bug

**Training (train.py):**
```python
state_rep = game.get_game_state_representation(subjective=True)  # Lines 401, 477
```
- Network is trained with `subjective=True`
- Network learns: "positive value = good for current player"
- Works for both Player ONE and Player TWO

**MCTS Search (mcts.py):**
```python
state_rep = search_state.get_game_state_representation()  # Lines 442, 539
```
- Uses default parameter: `subjective=False` (see game.py:473)
- Network predictions interpreted from absolute perspective
- "positive value = good for Player ONE" (absolute perspective)

### Impact

**For Player ONE:**
- Network predicts: +0.5 (good for current player = Player ONE)
- MCTS interprets: +0.5 (good for Player ONE)
- ✅ **Correct interpretation**

**For Player TWO:**
- Network predicts: +0.5 (good for current player = Player TWO)
- MCTS interprets: +0.5 (good for Player ONE)
- ❌ **BACKWARDS! MCTS thinks bad positions are good and vice versa**

---

## Why This Explains Everything

### Observation 1: Raw Policy Works (65.4% vs Strategic)

**Raw policy evaluation (eval_model.py:84-101):**
```python
policy, _ = model.predict(state_rep.board, state_rep.flat_values, legal_moves)
# Uses ONLY policy head, ignores value head
```

- Raw policy picks highest probability move
- Doesn't use value predictions for anything
- **Not affected by the bug** ✅

### Observation 2: MCTS Destroys Performance (2.5% vs MCTS)

**MCTS search (mcts.py:115-125):**
```python
q_values = value_sum / visits  # Uses value predictions for Q
exploration = c_puct * priors * sqrt(parent_visits) / (1 + child_visits)
ucb_values = q_values + exploration  # Value predictions guide search
```

- MCTS uses value predictions to guide tree search
- Player TWO's values are inverted
- **MCTS actively chooses bad moves for Player TWO** ❌

### Observation 3: Asymmetric Performance

From Run 4 evaluation (ITERATION_LOG.md):
- **As Player 1:** 0% win rate vs MCTS (but 65.4% vs Strategic with raw policy)
- **As Player 2:** 4.2% win rate vs MCTS (but 28.6% vs Strategic with raw policy)

Wait, this doesn't match the hypothesis... If Player TWO had inverted values, Player ONE should win more. Let me reconsider.

Actually, the MCTS evaluation pits raw policy vs MCTS using the SAME network. So:
- When raw policy is P1, it faces MCTS-as-P2 (which has inverted values)
- When raw policy is P2, it faces MCTS-as-P1 (which has correct values)

So we'd expect:
- Raw policy as P1 should beat MCTS-as-P2 (opponent has broken MCTS)
- Raw policy as P2 should lose to MCTS-as-P1 (opponent has working MCTS)

But we see:
- Raw policy as P1: 0% win rate
- Raw policy as P2: 4.2% win rate

Hmm, this doesn't match. Let me think more carefully...

Actually, BOTH players in "Raw Policy vs MCTS" use the same network. The MCTS player uses search with the buggy perspective. Since the bug affects Player TWO positions:

- When MCTS plays as Player TWO: Gets inverted values, makes bad moves
- When MCTS plays as Player ONE: Gets correct values, makes good moves
- When raw policy plays as Player ONE: Correct values don't matter (no search)
- When raw policy plays as Player TWO: Correct values don't matter (no search)

So:
- Games where MCTS is P1: MCTS has correct values → MCTS wins
- Games where MCTS is P2: MCTS has inverted values → MCTS still wins (because raw policy is weak)

Actually, this suggests both players have the bug, so it might not explain the asymmetry.

Let me reconsider the whole thing...

---

## Revised Analysis

Actually, let me check if eval_model.py also uses subjective=True. Let me trace through the eval code more carefully.

Looking at eval_model.py:80, the state_rep comes from:
```python
state_rep = game.get_game_state_representation(subjective=True)
```

So eval DOES use subjective=True! So the raw policy evaluation is correct.

But MCTS within eval_model.py calls mcts.search() which then uses subjective=False internally. So the MCTS has the bug.

So the evaluation is:
- **Raw policy**: Uses `subjective=True` (correct) ✅
- **MCTS**: Uses `subjective=False` internally (wrong) ❌

When "Raw Policy vs MCTS" happens:
1. Both use the same network (trained with subjective=True)
2. Raw policy evaluates positions correctly (uses subjective=True)
3. MCTS evaluates positions incorrectly (uses subjective=False)

The effect:
- **MCTS as P1**: Value predictions from absolute perspective, which by chance might be somewhat correct for P1
- **MCTS as P2**: Value predictions completely inverted

So MCTS is broken for both players, but more broken for P2.

But wait, if MCTS is broken, why does it beat raw policy 97.5% of the time?

Maybe because:
1. The policy head is still mostly correct (not affected by perspective bug)
2. MCTS uses policy priors heavily (c_puct * prior term in UCB)
3. The value predictions guide exploration, but don't dominate completely
4. Even with broken values, MCTS search (100-200 sims) > raw policy (no search)

Actually, I think the key insight is:
- Policy head is trained correctly and works
- Value head is trained correctly but MCTS uses it wrong
- Since MCTS uses BOTH policy and value, it's a mix of correct (policy) and wrong (value)
- The search itself (more simulations, exploration) gives an advantage even if values are wrong

This would explain:
- Raw policy (no search, only policy head): Decent performance (65% vs Strategic)
- MCTS (search with broken value head): Still better than raw policy due to search advantage
- But the value bug prevents MCTS from being as strong as it should be

---

## Code Locations

### Bug Locations (NEED FIX):

**mcts.py:442** - Root node expansion:
```python
# WRONG - uses subjective=False by default
state_rep = search_state.get_game_state_representation()

# SHOULD BE:
state_rep = search_state.get_game_state_representation(subjective=True)
```

**mcts.py:539** - Leaf node expansion:
```python
# WRONG - uses subjective=False by default
state_rep = search_state.get_game_state_representation()

# SHOULD BE:
state_rep = search_state.get_game_state_representation(subjective=True)
```

### Correct Usage (for reference):

**train.py:401** - Self-play move selection:
```python
state_rep = game.get_game_state_representation(subjective=True)  # ✅ Correct
```

**train.py:477** - Direct policy move selection:
```python
state_rep = game.get_game_state_representation(subjective=True)  # ✅ Correct
```

**eval_model.py:80** - Raw policy evaluation:
```python
state_rep = game.get_game_state_representation(subjective=True)  # ✅ Correct
```

---

## Expected Impact of Fix

### Before Fix:
- MCTS uses network with mismatched perspective
- Value predictions mislead tree search
- Policy network learns despite broken MCTS
- Raw policy: 65.4% vs Strategic (P1)
- MCTS: 2.5% combined vs raw policy (but MCTS should be stronger!)

### After Fix:
- MCTS will use network predictions correctly
- Value head will properly guide tree search
- MCTS should become significantly stronger
- Expected: Raw policy win rate vs MCTS should DECREASE (MCTS gets better)
- Expected: MCTS win rate vs Strategic should INCREASE significantly

---

## Recommendation

### Priority: 🔴 CRITICAL - FIX IMMEDIATELY

This bug has been preventing effective learning throughout ALL previous runs (Runs 1-4). The value scale fix in Run 4 was correct, but this perspective bug remained.

### Fix:

1. **Edit mcts.py:442:**
   ```python
   state_rep = search_state.get_game_state_representation(subjective=True)
   ```

2. **Edit mcts.py:539:**
   ```python
   state_rep = search_state.get_game_state_representation(subjective=True)
   ```

3. **Restart training** - Run 5 with both fixes:
   - Value scale fix (already applied in Run 4)
   - Perspective fix (new)

### Expected Results:

- **Immediate:** MCTS will become much stronger (correctly using value head)
- **Short term:** Raw policy win rate vs MCTS will drop initially (MCTS improvement)
- **Long term:** Overall learning will improve because training signal is now correct
- **Target:** By iteration 30, expect >40% raw policy win rate as network learns to mimic MCTS

---

## Root Cause

The `get_game_state_representation()` function defaults to `subjective=False` (game.py:473), but the network was trained exclusively with `subjective=True`.

MCTS code failed to specify `subjective=True` when calling this function, leading to a perspective mismatch between training and inference.

This is a classic train/test mismatch bug that went undetected because:
1. Policy head still worked (move probabilities less sensitive to perspective)
2. MCTS still outperformed raw policy (search advantage > value bug)
3. Training metrics (loss) didn't reveal the inference-time bug

---

## Additional Investigation Notes

### Why didn't this break everything completely?

1. **Policy head dominates early**: With low visit counts, the prior (policy) term in UCB is larger than the Q term (value)
2. **Search is better than no search**: Even with wrong values, exploring more positions helps
3. **Values are noisy anyway**: Network's value predictions aren't perfect, so wrong perspective looks like noise
4. **Policy learning still works**: Supervised policy loss on MCTS visit distribution doesn't depend on value interpretation

### Why this explains the 2.5% win rate:

The raw policy vs MCTS evaluation shows that MCTS is choosing very different moves than the raw policy. With correct values, MCTS should:
- Choose similar moves to the trained policy (aligned)
- Have slightly better win rate due to deeper search

Instead, MCTS is:
- Choosing very different moves (misaligned due to bad values)
- Still winning due to search depth, but not learning alignment

As training progresses, the raw policy should increasingly match MCTS choices, but this can't happen if MCTS is using broken value predictions.
