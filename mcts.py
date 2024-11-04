import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from game import GameState, Move, Player, PieceType
import time


@dataclass
class MCTSNode:
    state: GameState
    parent: Optional['MCTSNode']
    prior_probability: float
    player_perspective: Player  # Player whose perspective we're evaluating from

    def __post_init__(self):
        self.visit_count: int = 0
        self.value_sum: float = 0
        self.children: Dict[Move, MCTSNode] = {}
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def get_ucb_score(self,
                      parent_visit_count: int,
                      c_puct: float = 1.0) -> float:
        """Calculate the Upper Confidence Bound score for this node."""
        if self.visit_count == 0:
            return float('inf')

        # Q-value term (exploitation)
        q_value = self.value

        # U-value term (exploration)
        u_value = (c_puct * self.prior_probability *
                   math.sqrt(parent_visit_count) / (1 + self.visit_count))

        return q_value + u_value


class MCTS:

    def __init__(self,
                 network_wrapper,
                 num_simulations: int = 100,
                 c_puct: float = 1.0):
        self.network_wrapper = network_wrapper
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.debug = False  # Enable debug output

    def _expand_node(self, node: MCTSNode) -> None:
        """Expand a node using the policy network."""
        if node.is_expanded:
            return

        # Get possible moves
        possible_moves = node.state.get_possible_next_states()
        if not possible_moves:
            if self.debug:
                print("No possible moves available for expansion")
            return

        if self.debug:
            print(f"Expanding node with {len(possible_moves)} possible moves")

        # Prepare input for network
        grid_input = np.expand_dims(
            node.state.get_board_state_representation(), axis=0)
        flat_input = np.expand_dims(
            node.state.get_pieces_state_representation(), axis=0)

        # Get policy and value predictions
        policy_probs, value_pred = self.network_wrapper.predict(
            grid_input, flat_input)
        policy_probs = policy_probs[0]  # Remove batch dimension

        # Create child nodes for each possible move
        total_prob = 0
        valid_moves = set()

        for future_state in possible_moves:
            move = future_state.move
            valid_moves.add((move.x, move.y, move.piece_type.value))
            prob = policy_probs[move.x, move.y, move.piece_type.value]
            total_prob += prob

            # Create new game state for child
            new_state = GameState()
            new_state.board = node.state.board.copy()
            new_state.terrain = node.state.terrain
            new_state.current_player = node.state.current_player
            new_state.piece_counts = {
                Player.ONE: node.state.piece_counts[Player.ONE].copy(),
                Player.TWO: node.state.piece_counts[Player.TWO].copy()
            }
            new_state.make_move(move)

            child = MCTSNode(state=new_state,
                             parent=node,
                             prior_probability=prob,
                             player_perspective=node.player_perspective)
            node.children[move] = child

        # Renormalize probabilities if needed
        if total_prob > 0:
            for child in node.children.values():
                child.prior_probability /= total_prob

        node.is_expanded = True

        if self.debug:
            print(f"Node expanded with {len(node.children)} children")

    def _select_child(self, node: MCTSNode) -> Tuple[Move, MCTSNode]:
        """Select the best child node according to UCB score."""
        best_score = float('-inf')
        best_move = None
        best_child = None

        for move, child in node.children.items():
            score = child.get_ucb_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        if self.debug and best_move is None:
            print("Warning: No child selected in _select_child")

        return best_move, best_child

    def _simulate(self, node: MCTSNode) -> float:
        """Get a value estimate from the value network."""
        grid_input = np.expand_dims(
            node.state.get_board_state_representation(), axis=0)
        flat_input = np.expand_dims(
            node.state.get_pieces_state_representation(), axis=0)

        _, value = self.network_wrapper.predict(grid_input, flat_input)
        value = value[0]

        # Flip value if we're player 2
        if node.player_perspective == Player.TWO:
            value = 1 - value

        return value

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Update statistics for all nodes up to the root."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent
            # Flip value when moving up the tree (alternating players)
            value = 1 - value

    def get_action_probabilities(self,
                                 state: GameState,
                                 temperature: float = 1.0) -> np.ndarray:
        """Run MCTS simulations and return action probabilities."""
        search_start = time.time()
        root = MCTSNode(state=state,
                        parent=None,
                        prior_probability=1.0,
                        player_perspective=state.current_player)

        # Run simulations
        for i in range(self.num_simulations):
            if self.debug and i % 20 == 0:
                print(f"Starting simulation {i}/{self.num_simulations}")

            node = root
            search_path = [node]

            # Selection
            while node.is_expanded and node.children:
                move, node = self._select_child(node)
                if node is None:
                    break
                search_path.append(node)

            # Expansion and simulation
            if not node.is_expanded and not node.state.is_over:
                self._expand_node(node)

            # Get value from network for current position
            value = self._simulate(node)

            # Backpropagate
            self._backpropagate(node, value)

        # Calculate action probabilities based on visit counts
        policy = np.zeros((4, 4, 4), dtype=np.float32)

        if temperature == 0:  # Deterministic
            visits = {
                move: child.visit_count
                for move, child in root.children.items()
            }
            if visits:  # Check if there are any visited nodes
                best_move = max(visits.items(), key=lambda x: x[1])[0]
                policy[best_move.x, best_move.y,
                       best_move.piece_type.value] = 1
        else:
            # Calculate softmax of visit counts
            visits = np.array(
                [child.visit_count for child in root.children.values()])
            moves = list(root.children.keys())
            if len(visits) > 0:  # Check if there are any visited nodes
                visits = visits**(1 / temperature)
                visits_sum = visits.sum()
                if visits_sum > 0:
                    probs = visits / visits_sum
                    for move, prob in zip(moves, probs):
                        policy[move.x, move.y, move.piece_type.value] = prob

        if self.debug:
            print(
                f"MCTS search completed in {time.time() - search_start:.2f} seconds"
            )
            print(f"Root node visit count: {root.visit_count}")

        return policy

    def get_best_move(self, state: GameState) -> Move:
        """Get the best move according to MCTS simulations."""
        policy = self.get_action_probabilities(state, temperature=0)
        move_coords = np.unravel_index(policy.argmax(), policy.shape)
        return Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
