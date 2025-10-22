# Debug Tools

Diagnostic utilities for TicTacDo AI training.

## Tools

### debug_player_bias.py
Comprehensive diagnostic for P1/P2 win rate asymmetry in self-play.
- **Test 1:** Self-play win rates with stochastic sampling (200 games)
- **Test 2:** Policy distribution analysis by player position
- **Test 3:** Position symmetry checks
- **Usage:** `python debug_tools/debug_player_bias.py`
- **Output:** Detects severe (>15%), moderate (>8%), or slight (>3%) bias

### test_value_symmetry.py
Tests if value head has inherent P1/P2 bias.
- **Purpose:** Verify model predicts symmetric values for symmetric positions
- **Method:** Compares value predictions for flipped board positions
- **Usage:** `python debug_tools/test_value_symmetry.py`
- **Expected:** Mean difference < 0.02 (model is unbiased)

### test_mcts_vs_strategic.py
Quick evaluation against Strategic opponent.
- **Purpose:** Rapid performance check without full training
- **Method:** Runs N games of trained model vs Strategic opponent
- **Usage:** `python debug_tools/test_mcts_vs_strategic.py`
- **Output:** Win/Loss/Draw percentages

### check_training.py
Quick training progress checker.
- **Purpose:** Monitor training metrics without running full training
- **Method:** Loads checkpoint and displays key metrics
- **Usage:** `python debug_tools/check_training.py`

### analyze_replay_buffer.py
Analyzes replay buffer composition.
- **Purpose:** Inspect distribution of wins/losses/draws in training data
- **Method:** Loads replay buffer and computes statistics
- **Usage:** `python debug_tools/analyze_replay_buffer.py`
- **Output:** Buffer composition, value distribution, outcome percentages
