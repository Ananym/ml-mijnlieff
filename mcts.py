import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from game import GameState, Move, PieceType, Player, TurnResult
import time

# Global flag to enable/disable timing measurements
# Set this to True before running to enable timing measurements
# Example:
#   import mcts
#   mcts.ENABLE_TIMING = True  # Enable timing
#   my_game = GameState()
#   my_mcts = mcts.MCTS(num_simulations=100)
#   my_mcts.search(my_game)  # Will print timing information
#
# Alternatively, use the run_benchmark function:
#   mcts.run_benchmark(num_simulations=50, num_runs=2)
ENABLE_TIMING = False

# Noise control - synced with train.py
# Lower values reduce exploration noise
DIRICHLET_SCALE = 0.15  # Scale factor for Dirichlet noise (increased from 0.1)

# Random number generator with fixed seed for reproducibility
rng = np.random.Generator(np.random.PCG64(seed=42))


def clone_game_state(game_state):
    """Create a deep copy of a game state for search"""
    clone = GameState()

    # Fast array copying with NumPy
    clone.board = game_state.board.copy()
    clone.move_history = game_state.move_history.copy()

    # Simple attribute copying - use correct attribute names!
    clone.current_player = game_state.current_player
    clone.is_over = game_state.is_over
    clone.last_move = game_state.last_move
    clone.winner = game_state.winner
    clone.move_count = game_state.move_count

    # Copy piece counts (nested dict)
    clone.piece_counts = {
        Player.ONE: {pt: game_state.piece_counts[Player.ONE][pt] for pt in PieceType},
        Player.TWO: {pt: game_state.piece_counts[Player.TWO][pt] for pt in PieceType},
    }

    return clone


class MCTSNode:
    """Represents a node in the Monte Carlo search tree"""

    def __init__(self, game_state=None, parent=None, move=None):
        self.game_state = game_state  # Current game state at this node
        self.parent = parent  # Parent node
        self.move = move  # Move that led to this state
        self.children = []  # Child nodes
        self.visits = 0  # Number of visits to this node
        self.value_sum = 0.0  # Sum of values from simulations
        self.prior = 0.0  # Prior probability from policy network
        self.expanded = False  # Whether node has been expanded
        self.predicted_value = None  # Value predicted by neural network

    def expand(self, policy, game_state):
        """Expand node by creating children for all legal moves"""
        # prevent double expansion
        if self.expanded:
            return

        # Find all legal move indices at once
        legal_moves = game_state.get_legal_moves()
        legal_indices = np.argwhere(legal_moves)
        if len(legal_indices) == 0:
            self.expanded = True
            return

        # Pre-allocate children array (much faster than appending)
        self.children = [None] * len(legal_indices)

        # Batch process all moves
        for i, (x, y, piece_type_idx) in enumerate(legal_indices):
            # Reuse Move objects for common piece types to reduce allocation
            move = Move(x, y, PieceType(piece_type_idx))

            # Direct assignment instead of append (avoids list resizing)
            self.children[i] = MCTSNode(parent=self, move=move)
            self.children[i].prior = policy[x, y, piece_type_idx]

        self.expanded = True

    def select_child(self, c_puct=1.0):
        """Select child with highest UCB value using vectorized operations"""
        # Quick returns for edge cases
        if not self.children:
            return None

        if self.visits == 0 or len(self.children) == 1:
            return self.children[0]

        # Vectorized implementation using NumPy
        # Extract child properties as arrays
        child_count = len(self.children)
        visits = np.empty(child_count, dtype=np.float32)
        values = np.empty(child_count, dtype=np.float32)
        priors = np.empty(child_count, dtype=np.float32)

        # Populate arrays (faster than list comprehensions)
        for i, child in enumerate(self.children):
            visits[i] = child.visits
            values[i] = child.value_sum
            priors[i] = child.prior

        # Calculate UCB values in a vectorized way
        # Q values must be negated because child values are from child's perspective,
        # but we need parent's perspective for action selection
        with np.errstate(divide="ignore", invalid="ignore"):  # Handle division by zero
            q_values = np.divide(
                -values, visits, out=np.zeros_like(values), where=visits != 0
            )

        # Calculate exploration bonus
        exploration = c_puct * priors * np.sqrt(self.visits) / (1.0 + visits)

        # Find index of max UCB value
        ucb_values = q_values + exploration
        best_idx = np.argmax(ucb_values)

        return self.children[best_idx]

    def update(self, value):
        """Update node statistics"""
        # simple version - we can optimize this later if needed
        self.visits += 1
        # incremental update formula: https://math.stackexchange.com/questions/106700/incremental-average
        self.value_sum += value

    def is_terminal(self, game_state):
        """Check if node represents game end state"""
        return game_state.is_over

    def get_value(self):
        """Get mean value of node"""
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


def add_dirichlet_noise(
    policy_flat, legal_moves_flat, iteration, move_count, max_iterations=150, dirichlet_scale=None
):
    """add dirichlet noise to root policy with dynamic parameters based on training progress

    args:
        policy_flat: flattened policy vector
        legal_moves_flat: flattened legal moves mask
        iteration: current training iteration
        move_count: current move count in the game
        max_iterations: maximum training iterations
        dirichlet_scale: optional scale factor (uses global DIRICHLET_SCALE if None)

    returns:
        modified policy with noise added
    """
    legal_indices = np.nonzero(legal_moves_flat > 0)[0]

    if len(legal_indices) == 0:
        return policy_flat

    # Use provided scale or fall back to global constant
    scale = dirichlet_scale if dirichlet_scale is not None else DIRICHLET_SCALE

    # If noise is completely disabled, return original policy
    if scale <= 0.0:
        return policy_flat

    # calculate training progress (0 to 1)
    progress = min(1.0, iteration / max_iterations)

    # dynamic concentration parameter:
    # - early training: lower alpha (0.15) creates more diverse/spiky distribution
    # - late training: higher alpha (0.4) creates more uniform noise
    # this helps explore more aggressively early but refine choices later
    alpha_param = 0.15 + 0.25 * progress

    # for early game positions, use more aggressive noise
    if move_count < 5:
        alpha_param *= 0.8  # reduce alpha for more exploration in opening moves

    # generate dirichlet noise
    noise = rng.dirichlet([alpha_param] * len(legal_indices))

    # dynamic noise weight:
    # - start with 0.25 (25% noise) early in training
    # - decrease to 0.10 (10% noise) by the end
    noise_weight = max(0.10, 0.25 - 0.15 * progress)

    # Apply scale factor to reduce noise
    noise_weight *= scale

    # apply noise only to legal moves (vectorized)
    policy_with_noise = policy_flat.copy()
    policy_with_noise[legal_indices] = (1 - noise_weight) * policy_flat[legal_indices] + noise_weight * noise

    return policy_with_noise


class MCTS:
    """Monte Carlo Tree Search implementation that can work with or without neural network policy guidance"""

    # Class-level cache that persists across instances
    prediction_cache = {}
    max_cache_size = 250000
    cache_hits = 0
    cache_misses = 0

    @classmethod
    def clear_cache(cls):
        """Clear the prediction cache and reset statistics"""
        cls.prediction_cache.clear()
        cls.cache_hits = 0
        cls.cache_misses = 0

    def __init__(self, model=None, num_simulations=100, c_puct=1.0, dirichlet_scale=None, enable_early_stopping=True):
        self.model = model  # Neural network model (optional)
        self.num_simulations = num_simulations  # Number of simulations per search
        self.c_puct = c_puct  # Exploration constant
        self.temperature = 1.0  # Temperature for move selection
        self.iteration = 0  # Current training iteration
        self.move_count = 0  # Current move count in the game
        self.dirichlet_scale = dirichlet_scale  # Exploration noise scale (None uses default)
        self.enable_early_stopping = enable_early_stopping  # Whether to allow early stopping

        # Timing statistics
        self.timing_stats = {
            "rollout_time": 0.0,
            "selection_time": 0.0,
            "expansion_time": 0.0,
            "model_prediction_time": 0.0,
            "backprop_time": 0.0,
            "undo_time": 0.0,
            "total_search_time": 0.0,
            "get_legal_moves_time": 0.0,
            "make_move_time": 0.0,
            "clone_time": 0.0,
        }
        self.timing_counts = {k: 0 for k in self.timing_stats}

    def _get_state_key(self, board, flat_values):
        """Create a hashable key for the game state"""
        # Convert board and flat values to bytes for hashing
        return (board.tobytes(), flat_values.tobytes())

    def _predict_with_cache(self, state_rep, legal_moves=None):
        """Get model predictions, using cache if available"""
        if self.model is None:
            return None, None

        # Create cache key from state representation
        cache_key = self._get_state_key(state_rep.board, state_rep.flat_values)

        # Try to get from cache
        cached = self.prediction_cache.get(cache_key)
        if cached is not None:
            self.cache_hits += 1
            policy, value = cached
            return policy, value

        self.cache_misses += 1

        # Get fresh prediction
        policy, value = self.model.predict(
            state_rep.board, state_rep.flat_values, legal_moves
        )

        # Store in cache
        self.prediction_cache[cache_key] = (policy, value)

        # Limit cache size by removing oldest entries
        if len(self.prediction_cache) > self.max_cache_size:
            # Remove an arbitrary key (first one)
            self.prediction_cache.pop(next(iter(self.prediction_cache)))

        return policy, value

    def print_timing_stats(self):
        """Print timing statistics if enabled"""
        if not ENABLE_TIMING:
            return

        print("\n--- MCTS Timing Statistics ---")
        for stat, time_value in self.timing_stats.items():
            count = self.timing_counts[stat]
            if count > 0:
                avg_time = time_value / count
                print(
                    f"{stat}: {time_value:.6f}s total, {avg_time:.6f}s avg ({count} calls)"
                )
            else:
                print(f"{stat}: {time_value:.6f}s total (0 calls)")

        # Print cache stats if model is being used
        if self.model is not None:
            total = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
            print(f"\nPrediction Cache Stats:")
            print(f"  Cache hits: {self.cache_hits}")
            print(f"  Cache misses: {self.cache_misses}")
            print(f"  Hit rate: {hit_rate:.1f}%")
            print(f"  Cache size: {len(self.prediction_cache)}")

        print("-----------------------------\n")

    def reset_timing_stats(self):
        """Reset all timing statistics"""
        self.timing_stats = {k: 0.0 for k in self.timing_stats}
        self.timing_counts = {k: 0 for k in self.timing_stats}
        # Reset cache stats too
        self.cache_hits = 0
        self.cache_misses = 0

    def _rollout(self, state: GameState) -> float:
        """Perform a random rollout from the given state until game end"""
        start_time = time.perf_counter() if ENABLE_TIMING else 0

        # create a backup state for rollout
        rollout_state = clone_game_state(state)

        # limit depth for performance
        max_rollout_depth = 20
        depth = 0

        while not rollout_state.is_over and depth < max_rollout_depth:
            if ENABLE_TIMING:
                legal_moves_start = time.perf_counter()

            legal_moves = rollout_state.get_legal_moves()

            if ENABLE_TIMING:
                self.timing_stats["get_legal_moves_time"] += (
                    time.perf_counter() - legal_moves_start
                )
                self.timing_counts["get_legal_moves_time"] += 1

            if not np.any(legal_moves):
                rollout_state.pass_turn()
                depth += 1  # count passes towards depth too for consistency
                continue

            # Get all legal move positions
            legal_positions = np.argwhere(legal_moves)
            if len(legal_positions) == 0:
                break

            # Choose random move
            idx = rng.integers(len(legal_positions))
            x, y, piece_type = legal_positions[idx]
            move = Move(x, y, PieceType(piece_type))

            # Make the move - pass turns are handled inside make_move if needed
            if ENABLE_TIMING:
                make_move_start = time.perf_counter()

            result = rollout_state.make_move(move)

            if ENABLE_TIMING:
                self.timing_stats["make_move_time"] += (
                    time.perf_counter() - make_move_start
                )
                self.timing_counts["make_move_time"] += 1

            depth += 1  # count this move

            # if a forced pass happened, count it towards depth too
            if result == TurnResult.OPPONENT_WAS_FORCED_TO_PASS:
                depth += 1

        # If we reached max depth without game end, estimate value
        if not rollout_state.is_over and depth >= max_rollout_depth:
            # Simple heuristic based on piece counts and board position
            result = self._estimate_rollout_value(rollout_state, state.current_player)
        else:
            # Return game outcome from perspective of original player
            result = self._get_game_value(rollout_state, state.current_player)

        if ENABLE_TIMING:
            self.timing_stats["rollout_time"] += time.perf_counter() - start_time
            self.timing_counts["rollout_time"] += 1

        return result

    def _estimate_rollout_value(
        self, state: GameState, perspective_player: Player
    ) -> float:
        """Estimate position value using score differential"""
        # Get scores from both players
        scores = state.get_scores()
        opponent = Player.ONE if perspective_player == Player.TWO else Player.TWO

        # Calculate score differential from perspective player's view
        score_diff = scores[perspective_player] - scores[opponent]

        # Add smaller weight for piece advantage (piece count matters less than actual score)
        pieces_left = sum(state.piece_counts[perspective_player].values())
        opp_pieces_left = sum(state.piece_counts[opponent].values())
        piece_advantage = (pieces_left - opp_pieces_left) * 0.05

        # Normalize to [-0.7, 0.7] range for non-terminal estimates (FIXED: was [-0.6, 0.6])
        # More conservative than terminal values, using same scale as training targets
        combined_value = score_diff / 6.0 + piece_advantage
        return max(-0.7, min(0.7, combined_value))

    def search(self, game_state: GameState) -> np.ndarray:
        """Perform MCTS search from given state and return move probabilities"""
        if ENABLE_TIMING:
            search_start = time.perf_counter()
            self.reset_timing_stats()

            # Measure clone time
            clone_start = time.perf_counter()

        # Create root node
        root = MCTSNode(move=None, parent=None)

        # Use a single shared game state that we'll modify during search
        # Clone it once at the beginning to avoid modifying the original
        search_state = clone_game_state(game_state)

        if ENABLE_TIMING:
            self.timing_stats["clone_time"] += time.perf_counter() - clone_start
            self.timing_counts["clone_time"] += 1

        # Pre-allocate move_sequence with a reasonable capacity
        # to avoid resizing (most games won't exceed 16 moves in a single path)
        max_seq_len = 16
        move_sequence = [None] * max_seq_len
        seq_len = 0

        # Initialize root node's predicted value if model exists
        if self.model is not None:
            state_rep = search_state.get_game_state_representation(subjective=True)
            legal_moves = search_state.get_legal_moves()
            policy, value_pred = self._predict_with_cache(state_rep, legal_moves)
            policy = policy.squeeze(0)
            # Convert tensor to float
            root.predicted_value = value_pred.squeeze().item()

            # Add Dirichlet noise to root policy for exploration regardless of player
            legal_flat = legal_moves.flatten()
            policy_flat = policy.flatten()

            # Apply noise with dynamic parameters for both players
            policy_flat = add_dirichlet_noise(
                policy_flat,
                legal_flat,
                self.iteration,
                self.move_count,
                max_iterations=150,
                dirichlet_scale=self.dirichlet_scale,
            )

            policy = policy_flat.reshape(policy.shape)
        else:
            # If no model, use uniform policy over legal moves
            legal_moves = search_state.get_legal_moves()
            policy = legal_moves / (np.sum(legal_moves) + 1e-8)

        # Expand root with policy
        root.expand(policy, search_state)

        # Early stopping variables - track if best move is stable
        prev_best_visits = 0
        best_child_idx = -1
        stable_count = 0

        # Run simulations
        for sim_idx in range(self.num_simulations):
            node = root
            # Reset sequence length counter
            seq_len = 0

            # SELECTION: traverse tree to find leaf node
            if ENABLE_TIMING:
                selection_start = time.perf_counter()

            while node.expanded and not node.is_terminal(search_state):
                node = node.select_child(self.c_puct)
                # Apply the move to our shared state
                if node.move:
                    if ENABLE_TIMING:
                        make_move_start = time.perf_counter()

                    # Track if the move results in a forced pass
                    result = search_state.make_move(node.move)

                    if ENABLE_TIMING:
                        self.timing_stats["make_move_time"] += (
                            time.perf_counter() - make_move_start
                        )
                        self.timing_counts["make_move_time"] += 1

                    # Add to move sequence, resizing if needed
                    if seq_len >= max_seq_len:
                        # Double the size if we need more space
                        move_sequence.extend([None] * max_seq_len)
                        max_seq_len *= 2

                    move_sequence[seq_len] = node.move
                    seq_len += 1

                    if result == TurnResult.OPPONENT_WAS_FORCED_TO_PASS:
                        # Also track the automatic pass that occurred
                        if seq_len >= max_seq_len:
                            move_sequence.extend([None] * max_seq_len)
                            max_seq_len *= 2

                        move_sequence[seq_len] = None  # None represents a pass
                        seq_len += 1

            if ENABLE_TIMING:
                self.timing_stats["selection_time"] += (
                    time.perf_counter() - selection_start
                )
                self.timing_counts["selection_time"] += 1

            # EXPANSION and SIMULATION
            if ENABLE_TIMING:
                model_pred_start = time.perf_counter()
                expansion_start = time.perf_counter()  # Initialize here for all paths

            # If node is terminal, use its game outcome for backpropagation
            if node.is_terminal(search_state):
                value = self._get_game_value(search_state, search_state.current_player)
            else:
                # EXPANSION: use policy network or uniform policy to expand node
                if self.model is not None:
                    # Use model for policy and value
                    state_rep = search_state.get_game_state_representation(subjective=True)
                    legal_moves = search_state.get_legal_moves()
                    policy, value_pred = self._predict_with_cache(
                        state_rep, legal_moves
                    )
                    policy = policy.squeeze(0)
                    # Convert tensor to float
                    value = value_pred.squeeze().item()
                    node.predicted_value = value

                    if ENABLE_TIMING:
                        self.timing_stats["model_prediction_time"] += (
                            time.perf_counter() - model_pred_start
                        )
                        self.timing_counts["model_prediction_time"] += 1
                else:
                    # Use uniform policy and rollout for value
                    if ENABLE_TIMING:
                        legal_moves_start = time.perf_counter()

                    legal_moves = search_state.get_legal_moves()

                    if ENABLE_TIMING:
                        self.timing_stats["get_legal_moves_time"] += (
                            time.perf_counter() - legal_moves_start
                        )
                        self.timing_counts["get_legal_moves_time"] += 1

                    policy = legal_moves / (np.sum(legal_moves) + 1e-8)
                    value = self._rollout(search_state)

                # Expand node with chosen policy
                node.expand(policy, search_state)

            if ENABLE_TIMING:
                self.timing_stats["expansion_time"] += (
                    time.perf_counter() - expansion_start
                )
                self.timing_counts["expansion_time"] += 1

            # BACKPROPAGATION: update all nodes in path
            if ENABLE_TIMING:
                backprop_start = time.perf_counter()

            current_node = node
            current_value = value

            while True:
                current_node.update(current_value)
                if current_node.parent is None:
                    break
                # Flip value when moving to parent (opponent's perspective)
                current_value = -current_value
                current_node = current_node.parent

            if ENABLE_TIMING:
                self.timing_stats["backprop_time"] += (
                    time.perf_counter() - backprop_start
                )
                self.timing_counts["backprop_time"] += 1

            # Undo all moves (including passes) to restore original state
            if ENABLE_TIMING:
                undo_start = time.perf_counter()

            for i in range(seq_len):
                search_state.undo_move()

            if ENABLE_TIMING:
                self.timing_stats["undo_time"] += time.perf_counter() - undo_start
                self.timing_counts["undo_time"] += 1

            # Check for early stopping - if best move hasn't changed for several iterations
            if self.enable_early_stopping and sim_idx > 20 and sim_idx % 5 == 0:
                # Find current best child
                max_visits = 0
                current_best_idx = 0
                for i, child in enumerate(root.children):
                    if child.visits > max_visits:
                        max_visits = child.visits
                        current_best_idx = i

                # Check if best child is stable
                if current_best_idx == best_child_idx:
                    # Check if visit count has increased significantly
                    if max_visits > 0 and prev_best_visits > 0:
                        # If visits haven't increased by at least 5%, count as stable
                        if (max_visits - prev_best_visits) / prev_best_visits < 0.05:
                            stable_count += 1
                            # If stable for 3 consecutive checks, stop early
                            if (
                                stable_count >= 3
                                and sim_idx > self.num_simulations // 2
                            ):
                                if ENABLE_TIMING:
                                    print(f"Early stopping at {sim_idx} simulations")
                                break
                        else:
                            stable_count = 0
                else:
                    stable_count = 0
                    best_child_idx = current_best_idx

                prev_best_visits = max_visits

            # Print progress every 20 simulations if timing is enabled
            if ENABLE_TIMING and (sim_idx + 1) % 20 == 0:
                print(f"Completed {sim_idx + 1}/{self.num_simulations} simulations")

        # Calculate move probabilities based on visit counts more efficiently
        visit_counts = np.zeros((4, 4, 4), dtype=np.float32)

        # No need to check node.move - root children should all have moves
        for child in root.children:
            if child.move:  # Still check to be safe
                x, y, piece_type = (
                    child.move.x,
                    child.move.y,
                    child.move.piece_type.value,
                )
                visit_counts[x, y, piece_type] = child.visits

        # Apply temperature
        if abs(self.temperature) < 1e-9:  # epsilon check instead of exact equality
            # Choose most visited move deterministically
            best_idx = np.unravel_index(np.argmax(visit_counts), visit_counts.shape)
            probs = np.zeros_like(visit_counts)
            probs[best_idx] = 1.0
        else:
            # Convert visits to probabilities with temperature
            if self.temperature != 1.0:
                visit_counts = np.power(visit_counts, 1.0 / self.temperature)

            sum_visits = np.sum(visit_counts)
            if sum_visits > 0:
                probs = visit_counts / sum_visits
            else:
                # Fallback to uniform distribution if no visits
                legal_moves = game_state.get_legal_moves()
                probs = legal_moves / np.sum(legal_moves)

        if ENABLE_TIMING:
            self.timing_stats["total_search_time"] = time.perf_counter() - search_start
            self.timing_counts["total_search_time"] = 1
            self.print_timing_stats()

        return probs, root

    def _get_game_value(self, game_state, original_player):
        """Calculate the game value from original player's perspective using score differential"""
        # Validate the game is actually over
        if not game_state.is_over:
            # If the game isn't over, estimate using score difference
            scores = game_state.get_scores()
            player_score = scores[original_player]
            opponent = Player.ONE if original_player == Player.TWO else Player.TWO
            opponent_score = scores[opponent]
            score_diff = player_score - opponent_score

            # normalize to [-0.7, 0.7] range for non-terminal estimates
            # more conservative than terminal values
            return max(-0.7, min(0.7, score_diff / 6.0))

        # For terminal states, calculate based on winner and score differential
        winner = game_state.get_winner()
        scores = game_state.get_scores()
        player_score = scores[original_player]
        opponent = Player.ONE if original_player == Player.TWO else Player.TWO
        opponent_score = scores[opponent]
        score_diff = player_score - opponent_score

        # normalize to [-1.0, 1.0] range (FIXED: was [-0.8, 0.8] with /8.0)
        normalized_diff = max(-1.0, min(1.0, score_diff / 6.0))

        if winner is None:
            # Game is a draw, use normalized score difference
            return normalized_diff
        elif winner == original_player:
            # Win: ensure positive but consider margin
            return max(0.2, normalized_diff)
        else:
            # Loss: ensure negative but consider margin
            return min(-0.2, normalized_diff)

    def set_temperature(self, temperature):
        """Set temperature for move selection"""
        self.temperature = temperature

    def set_iteration(self, iteration):
        """Set current training iteration"""
        self.iteration = iteration

    def set_move_count(self, move_count):
        """Set current move count in the game"""
        self.move_count = move_count


def run_benchmark(num_simulations=100, num_runs=3):
    """Run a performance benchmark of MCTS search

    Args:
        num_simulations: Number of MCTS simulations per search
        num_runs: Number of benchmark runs
    """
    print(
        f"Running MCTS benchmark with {num_simulations} simulations x {num_runs} runs..."
    )

    # Enable timing
    global ENABLE_TIMING
    old_setting = ENABLE_TIMING
    ENABLE_TIMING = True
    GameState.enable_timing = True

    # Reset timing stats
    GameState.reset_timing()
    mcts = MCTS(model=None, num_simulations=num_simulations)
    mcts.reset_timing_stats()

    # Create a test game state
    game = GameState()

    # Make a couple of moves to create a more realistic state
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.ORTHO))

    # Run benchmark
    total_time = 0
    for i in range(num_runs):
        print(f"\nRun {i+1}/{num_runs}:")
        start = time.perf_counter()

        # Run MCTS search
        mcts.search(game)

        elapsed = time.perf_counter() - start
        total_time += elapsed
        print(f"Run completed in {elapsed:.4f} seconds")

        # Print detailed stats for the first run
        if i == 0:
            GameState.print_timing()

    # Print summary
    avg_time = total_time / num_runs
    print(f"\nBenchmark summary:")
    print(f"Average time: {avg_time:.4f} seconds per search")
    print(f"Simulations per second: {num_simulations / avg_time:.1f}")

    # Restore previous timing setting
    ENABLE_TIMING = old_setting
    GameState.enable_timing = old_setting

    return avg_time
