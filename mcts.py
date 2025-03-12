import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from game import GameState, Move, PieceType, Player, TurnResult

# Random number generator with fixed seed for reproducibility
rng = np.random.Generator(np.random.PCG64(seed=42))


def clone_game_state(state: GameState) -> GameState:
    """Create a new GameState with the same properties without using deepcopy"""
    new_state = GameState()
    # Copy board state (2D array)
    new_state.board = state.board.copy()
    # Copy current player (enum)
    new_state.current_player = state.current_player
    # Copy last move (immutable so direct reference is fine)
    new_state.last_move = state.last_move
    # Copy game over state
    new_state.is_over = state.is_over
    # Copy winner
    new_state.winner = state.winner

    # Copy piece counts (nested dict)
    new_state.piece_counts = {
        Player.ONE: {pt: state.piece_counts[Player.ONE][pt] for pt in PieceType},
        Player.TWO: {pt: state.piece_counts[Player.TWO][pt] for pt in PieceType},
    }

    return new_state


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

        legal_moves = game_state.get_legal_moves()
        # For each legal move, create a child node
        for move_idx in np.argwhere(legal_moves):
            x, y, piece_type_idx = move_idx
            move = Move(x, y, PieceType(piece_type_idx))

            # Create child node without its own game state
            child = MCTSNode(parent=self, move=move)

            # Set prior probability from policy network
            child.prior = policy[x, y, piece_type_idx]

            self.children.append(child)

        self.expanded = True

    def select_child(self, c_puct=1.0):
        """Select child with highest UCB value"""
        # UCB formula: Q(s,a) + c_puct * P(s,a) * sqrt(sum(N(s,b))) / (1 + N(s,a))
        # where Q is the mean value, P is prior probability, N is visit count

        # If node hasn't been visited, prioritize exploration
        if self.visits == 0:
            return rng.choice(self.children)

        ucb_values = []
        for child in self.children:
            # Exploitation term: average value
            q_value = 0.0 if child.visits == 0 else child.value_sum / child.visits

            # Exploration term: prior scaled by visits
            u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)

            ucb_values.append(q_value + u_value)

        # Select child with highest UCB value
        return self.children[np.argmax(ucb_values)]

    def update(self, value):
        """Update node statistics after a simulation"""
        self.visits += 1
        self.value_sum += value

    def is_terminal(self, game_state):
        """Check if node represents game end state"""
        return game_state.is_over

    def get_value(self):
        """Get mean value of node"""
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


def add_dirichlet_noise(
    policy_flat, legal_moves_flat, iteration, move_count, max_iterations=150
):
    """add dirichlet noise to root policy with dynamic parameters based on training progress

    args:
        policy_flat: flattened policy vector
        legal_moves_flat: flattened legal moves mask
        iteration: current training iteration
        move_count: current move count in the game
        max_iterations: maximum training iterations

    returns:
        modified policy with noise added
    """
    legal_indices = np.where(legal_moves_flat > 0)[0]

    if len(legal_indices) == 0:
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

    # apply noise only to legal moves
    policy_with_noise = policy_flat.copy()
    for i, idx in enumerate(legal_indices):
        policy_with_noise[idx] = (1 - noise_weight) * policy_flat[
            idx
        ] + noise_weight * noise[i]

    return policy_with_noise


class MCTS:
    """Monte Carlo Tree Search implementation that can work with or without neural network policy guidance"""

    def __init__(self, model=None, num_simulations=100, c_puct=1.0):
        self.model = model  # Neural network model (optional)
        self.num_simulations = num_simulations  # Number of simulations per search
        self.c_puct = c_puct  # Exploration constant
        self.temperature = 1.0  # Temperature for move selection
        self.iteration = 0  # Current training iteration
        self.move_count = 0  # Current move count in the game

    def _rollout(self, state: GameState) -> float:
        """Perform a random rollout from the given state until game end"""
        # create a backup state for rollout
        rollout_state = clone_game_state(state)

        # limit depth for performance
        max_rollout_depth = 20
        depth = 0

        while not rollout_state.is_over and depth < max_rollout_depth:
            legal_moves = rollout_state.get_legal_moves()
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
            # and will increment depth implicitly
            result = rollout_state.make_move(move)
            depth += 1  # count this move

            # if a forced pass happened, count it towards depth too
            if result == TurnResult.OPPONENT_WAS_FORCED_TO_PASS:
                depth += 1

        # If we reached max depth without game end, estimate value
        if not rollout_state.is_over and depth >= max_rollout_depth:
            # Simple heuristic based on piece counts and board position
            return self._estimate_rollout_value(rollout_state, state.current_player)

        # Return game outcome from perspective of original player
        return self._get_game_value(rollout_state, state.current_player)

    def _estimate_rollout_value(
        self, state: GameState, perspective_player: Player
    ) -> float:
        """Simple heuristic for rollout termination before game end"""
        # Compare scores
        scores = state.get_scores()
        score_diff = (
            scores[perspective_player]
            - scores[Player.ONE if perspective_player == Player.TWO else Player.TWO]
        )

        # Add piece advantage
        pieces_left = sum(state.piece_counts[perspective_player].values())
        opp_pieces_left = sum(
            state.piece_counts[
                Player.ONE if perspective_player == Player.TWO else Player.TWO
            ].values()
        )
        piece_advantage = (pieces_left - opp_pieces_left) * 0.1

        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, (score_diff * 0.2 + piece_advantage) / 2.0))

    def search(self, game_state: GameState) -> np.ndarray:
        """Perform MCTS search from given state and return move probabilities"""
        # Create root node
        root = MCTSNode(move=None, parent=None)

        # Use a single shared game state that we'll modify during search
        # Clone it once at the beginning to avoid modifying the original
        search_state = clone_game_state(game_state)

        # Store move sequence during traversal (including passes as None)
        move_sequence = []

        # Initialize root node's predicted value if model exists
        if self.model is not None:
            state_rep = search_state.get_game_state_representation()
            _, value_pred = self.model.predict(state_rep.board, state_rep.flat_values)
            root.predicted_value = value_pred.squeeze()

            # Get policy from model for root node
            legal_moves = search_state.get_legal_moves()
            policy, _ = self.model.predict(
                state_rep.board, state_rep.flat_values, legal_moves
            )
            policy = policy.squeeze(0)

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
            )

            policy = policy_flat.reshape(policy.shape)
        else:
            # If no model, use uniform policy over legal moves
            legal_moves = search_state.get_legal_moves()
            policy = legal_moves / (np.sum(legal_moves) + 1e-8)

        # Expand root with policy
        root.expand(policy, search_state)

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            # Clear move sequence for new simulation
            move_sequence.clear()

            # SELECTION: traverse tree to find leaf node
            while node.expanded and not node.is_terminal(search_state):
                node = node.select_child(self.c_puct)
                # Apply the move to our shared state
                if node.move:
                    # Track if the move results in a forced pass
                    result = search_state.make_move(node.move)
                    move_sequence.append(node.move)

                    if result == TurnResult.OPPONENT_WAS_FORCED_TO_PASS:
                        # Also track the automatic pass that occurred
                        move_sequence.append(None)  # None represents a pass

            # If node is terminal, use its game outcome for backpropagation
            if node.is_terminal(search_state):
                value = self._get_game_value(search_state, game_state.current_player)
            else:
                # EXPANSION: use policy network or uniform policy to expand node
                if self.model is not None:
                    # Use model for policy and value
                    state_rep = search_state.get_game_state_representation()
                    legal_moves = search_state.get_legal_moves()
                    policy, value_pred = self.model.predict(
                        state_rep.board, state_rep.flat_values, legal_moves
                    )
                    policy = policy.squeeze(0)
                    value = value_pred.squeeze()
                    node.predicted_value = value
                else:
                    # Use uniform policy and rollout for value
                    legal_moves = search_state.get_legal_moves()
                    policy = legal_moves / (np.sum(legal_moves) + 1e-8)
                    value = self._rollout(search_state)

                # Expand node with chosen policy
                node.expand(policy, search_state)

            # BACKPROPAGATION: update all nodes in path
            current_node = node
            current_value = value

            while current_node is not None:
                current_node.update(current_value)
                current_node = current_node.parent
                if current_node is not None:
                    # Flip value when moving to parent (opponent's perspective)
                    current_value = -current_value

            # Undo all moves (including passes) to restore original state
            for _ in range(len(move_sequence)):
                search_state.undo_move()

        # Calculate move probabilities based on visit counts
        visit_counts = np.zeros((4, 4, 4), dtype=np.float32)

        for child in root.children:
            if child.move:
                x, y, piece_type = (
                    child.move.x,
                    child.move.y,
                    child.move.piece_type.value,
                )
                visit_counts[x, y, piece_type] = child.visits

        # Apply temperature
        if self.temperature == 0:
            # Choose most visited move deterministically
            best_idx = np.unravel_index(np.argmax(visit_counts), visit_counts.shape)
            probs = np.zeros_like(visit_counts)
            probs[best_idx] = 1.0
        else:
            # Convert visits to probabilities with temperature
            visit_counts = np.power(visit_counts, 1.0 / self.temperature)
            sum_visits = np.sum(visit_counts)
            if sum_visits > 0:
                probs = visit_counts / sum_visits
            else:
                # Fallback to uniform distribution if no visits
                legal_moves = game_state.get_legal_moves()
                probs = legal_moves / np.sum(legal_moves)

        return probs, root

    def _get_game_value(
        self, game_state: GameState, perspective_player: Player
    ) -> float:
        """Get terminal game value from perspective of the given player"""
        winner = game_state.get_winner()

        if winner is None:  # Draw
            return 0.0

        # Binary win/loss signal with score bonus
        if winner == perspective_player:
            base_value = 1.0  # Win
        else:
            base_value = -1.0  # Loss

        # Add small score difference bonus to encourage larger wins
        score_one = game_state._calculate_score(Player.ONE)
        score_two = game_state._calculate_score(Player.TWO)

        score_diff = (
            score_one - score_two
            if perspective_player == Player.ONE
            else score_two - score_one
        )

        # Small bonus based on score difference (max 0.5)
        score_bonus = 0.03 * max(-0.5, min(0.5, score_diff / 5.0))

        return base_value + score_bonus

    def set_temperature(self, temperature):
        """Set temperature for move selection"""
        self.temperature = temperature

    def set_iteration(self, iteration):
        """Set current training iteration"""
        self.iteration = iteration

    def set_move_count(self, move_count):
        """Set current move count in the game"""
        self.move_count = move_count
