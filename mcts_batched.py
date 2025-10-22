"""
Batched MCTS implementation that batches neural network evaluations.

This is the AlphaZero approach: run multiple MCTS searches concurrently,
and when they need neural network evaluation, batch all the requests together
for efficient GPU inference.
"""

import numpy as np
from typing import List, Tuple
from mcts import MCTS, MCTSNode, clone_game_state
from game import GameState
import torch


class BatchedMCTS(MCTS):
    """
    MCTS with batched neural network evaluation.

    Instead of evaluating nodes one at a time, this implementation runs
    multiple searches concurrently and batches all pending neural network
    evaluations together.
    """

    def search_batch(self, game_states: List[GameState]) -> List[Tuple[np.ndarray, MCTSNode]]:
        """
        Perform MCTS search on multiple game states, batching NN evaluations.

        Args:
            game_states: List of game states to search

        Returns:
            List of (move probabilities, root node) tuples
        """
        batch_size = len(game_states)

        # Create root nodes and search states for each game
        roots = []
        search_states = []

        for game_state in game_states:
            root = MCTSNode(move=None, parent=None)
            search_state = clone_game_state(game_state)
            roots.append(root)
            search_states.append(search_state)

        # Initialize root nodes with batched prediction
        if self.model is not None:
            self._batch_expand_roots(roots, search_states, game_states)

        # Run simulations with batched evaluation
        for sim_idx in range(self.num_simulations):
            # For each game, run one simulation step
            self._batch_simulation_step(roots, search_states, game_states)

        # Convert visit counts to probabilities for each game
        results = []
        for i, (root, game_state) in enumerate(zip(roots, game_states)):
            probs = self._visits_to_probs(root, game_state)
            results.append((probs, root))

        return results

    def _batch_expand_roots(self, roots: List[MCTSNode], search_states: List[GameState],
                           original_states: List[GameState]):
        """Initialize all root nodes with batched neural network evaluation."""
        batch_size = len(roots)

        # Collect state representations
        board_inputs = []
        flat_inputs = []
        legal_moves_list = []

        for state in search_states:
            state_rep = state.get_game_state_representation(subjective=True)
            legal_moves = state.get_legal_moves()
            board_inputs.append(state_rep.board)
            flat_inputs.append(state_rep.flat_values)
            legal_moves_list.append(legal_moves)

        # Batch predict
        board_batch = np.stack(board_inputs)
        flat_batch = np.stack(flat_inputs)

        # Convert to tensors
        board_tensor = torch.tensor(board_batch, dtype=torch.float32, device=self.model.device)
        flat_tensor = torch.tensor(flat_batch, dtype=torch.float32, device=self.model.device)

        # Get batched predictions
        policies, values = self.model.predict(board_tensor, flat_tensor, None)

        # Process each root
        for i, (root, state) in enumerate(zip(roots, search_states)):
            # Model.predict already returns numpy arrays
            policy = policies[i] if isinstance(policies[i], np.ndarray) else policies[i].cpu().numpy()
            value = values[i].item() if hasattr(values[i], 'item') else float(values[i])

            root.predicted_value = value

            # Apply Dirichlet noise to root policy
            legal_flat = legal_moves_list[i].flatten()
            policy_flat = policy.flatten()

            from mcts import add_dirichlet_noise
            policy_flat = add_dirichlet_noise(
                policy_flat,
                legal_flat,
                self.iteration,
                self.move_count,
                max_iterations=150,
                dirichlet_scale=self.dirichlet_scale,
            )

            policy = policy_flat.reshape(policy.shape)
            root.expand(policy, state)

    def _batch_simulation_step(self, roots: List[MCTSNode], search_states: List[GameState],
                               original_states: List[GameState]):
        """
        Run one simulation step for each search, batching NN evaluations.

        Important: search_states should be clones that we can freely modify.
        They get reset at the start of each simulation by undoing all moves.
        """
        batch_size = len(roots)

        # Track simulation state for each game
        current_nodes = []
        move_sequences = []
        needs_expansion = []

        # Selection phase: traverse to leaf nodes
        for i, (root, search_state) in enumerate(zip(roots, search_states)):
            # Ensure we're starting from a clean state
            # (moves from previous simulation should already be undone)
            node = root

            # Track how many moves we add to history (including forced passes)
            initial_history_len = len(search_state.move_history)

            # Select until we hit an unexpanded node or terminal
            while node.expanded and not node.is_terminal(search_state):
                node = node.select_child(self.c_puct)
                if node.move:
                    result = search_state.make_move(node.move)
                    # Note: make_move might add 2 items to history if there's a forced pass

            # Count actual moves made (including any forced passes)
            moves_made = len(search_state.move_history) - initial_history_len

            current_nodes.append(node)
            move_sequences.append(moves_made)  # Store count instead of list
            needs_expansion.append(not node.is_terminal(search_state))

        # Expansion phase: batch evaluate all leaf nodes that need expansion
        if self.model is not None and any(needs_expansion):
            # Collect states that need evaluation
            eval_indices = [i for i, needs in enumerate(needs_expansion) if needs]

            if eval_indices:
                board_inputs = []
                flat_inputs = []
                legal_moves_list = []

                for idx in eval_indices:
                    state = search_states[idx]
                    state_rep = state.get_game_state_representation(subjective=True)
                    legal_moves = state.get_legal_moves()
                    board_inputs.append(state_rep.board)
                    flat_inputs.append(state_rep.flat_values)
                    legal_moves_list.append(legal_moves)

                # Batch predict
                board_batch = np.stack(board_inputs)
                flat_batch = np.stack(flat_inputs)

                board_tensor = torch.tensor(board_batch, dtype=torch.float32, device=self.model.device)
                flat_tensor = torch.tensor(flat_batch, dtype=torch.float32, device=self.model.device)

                policies, values = self.model.predict(board_tensor, flat_tensor, None)

                # Distribute results and expand nodes
                for j, idx in enumerate(eval_indices):
                    node = current_nodes[idx]
                    state = search_states[idx]

                    # Model.predict already returns numpy arrays
                    policy = policies[j] if isinstance(policies[j], np.ndarray) else policies[j].cpu().numpy()
                    value = values[j].item() if hasattr(values[j], 'item') else float(values[j])

                    node.predicted_value = value
                    node.expand(policy, state)

        # Backpropagation phase
        for i, (node, moves_count) in enumerate(zip(current_nodes, move_sequences)):
            search_state = search_states[i]

            # Get value for backpropagation
            if node.is_terminal(search_state):
                value = self._get_game_value(search_state, search_state.current_player)
            else:
                value = node.predicted_value if node.predicted_value is not None else 0.0

            # Backpropagate
            current_node = node
            current_value = value

            while True:
                current_node.update(current_value)
                if current_node.parent is None:
                    break
                current_value = -current_value
                current_node = current_node.parent

            # Undo moves to restore state (including any forced passes)
            for j in range(moves_count):
                success = search_state.undo_move()
                if not success:
                    raise RuntimeError(f"Failed to undo move {j} of {moves_count} in game {i}")

    def _visits_to_probs(self, root: MCTSNode, game_state: GameState) -> np.ndarray:
        """Convert root visit counts to move probabilities."""
        visit_counts = np.zeros((4, 4, 4), dtype=np.float32)

        for child in root.children:
            if child.move:
                x, y, piece_type = child.move.x, child.move.y, child.move.piece_type.value
                visit_counts[x, y, piece_type] = child.visits

        # Apply temperature
        if abs(self.temperature) < 1e-9:
            best_idx = np.unravel_index(np.argmax(visit_counts), visit_counts.shape)
            probs = np.zeros_like(visit_counts)
            probs[best_idx] = 1.0
        else:
            if self.temperature != 1.0:
                visit_counts = np.power(visit_counts, 1.0 / self.temperature)

            sum_visits = np.sum(visit_counts)
            if sum_visits > 0:
                probs = visit_counts / sum_visits
            else:
                legal_moves = game_state.get_legal_moves()
                probs = legal_moves / np.sum(legal_moves)

        return probs

    def search(self, game_state: GameState) -> Tuple[np.ndarray, MCTSNode]:
        """
        Single-game search (for compatibility with MCTS interface).
        Just calls batch search with a single game.
        """
        results = self.search_batch([game_state])
        return results[0]
