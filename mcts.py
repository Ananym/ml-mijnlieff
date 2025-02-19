import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List
from game import GameState, Move, Player, PieceType


@dataclass
class MCTSNode:
    state: GameState
    parent: Optional["MCTSNode"]
    prior_prob: float

    def __post_init__(self):
        self.children: Dict[Move, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def get_ucb_score(self, parent_visit_count: int, c_puct: float = 1.0) -> float:
        """UCB score calculation following AlphaZero methodology"""
        if self.visit_count == 0:
            return float("inf")

        # Q value (exploitation)
        q_value = self.value
        # U value (exploration)
        u_value = (
            c_puct
            * self.prior_prob
            * math.sqrt(parent_visit_count)
            / (1 + self.visit_count)
        )

        return q_value + u_value


class MCTS:
    def __init__(self, model, num_simulations: int = 100, c_puct: float = 1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def _expand(self, node: MCTSNode) -> None:
        """Expand a node using the policy network"""
        if node.is_expanded:
            return

        # Get policy and value predictions from the model
        state_rep = node.state.get_game_state_representation()
        policy, _ = self.model.predict(state_rep.board, state_rep.flat_values)
        policy = policy[0]  # Remove batch dimension

        # Create child nodes for all legal moves
        legal_moves = node.state.get_legal_moves()
        policy_sum = 1e-8

        for move_coords in np.argwhere(legal_moves):
            x, y, piece_type = move_coords
            move = Move(x, y, PieceType(piece_type))

            # Create new state
            new_state = GameState()
            new_state.board = node.state.board.copy()
            new_state.terrain = node.state.terrain
            new_state.current_player = node.state.current_player
            new_state.piece_counts = {
                Player.ONE: node.state.piece_counts[Player.ONE].copy(),
                Player.TWO: node.state.piece_counts[Player.TWO].copy(),
            }

            # Make move in new state
            new_state.make_move(move)

            # Create child node
            prior_prob = policy[x, y, piece_type]
            policy_sum += prior_prob

            child = MCTSNode(state=new_state, parent=node, prior_prob=prior_prob)
            node.children[move] = child

        # Normalize probabilities
        for child in node.children.values():
            child.prior_prob /= policy_sum

        node.is_expanded = True

    def _select_child(self, node: MCTSNode) -> tuple[Move, MCTSNode]:
        """Select the child with the highest UCB score"""
        best_score = float("-inf")
        best_move = None
        best_child = None

        for move, child in node.children.items():
            score = child.get_ucb_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def _simulate(self, node: MCTSNode) -> float:
        """Get a value estimate from the value network"""
        state_rep = node.state.get_game_state_representation()
        _, value = self.model.predict(state_rep.board, state_rep.flat_values)
        return float(value[0])

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Update statistics for all nodes up to the root"""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent
            value = 1.0 - value  # Flip value for opponent

    def get_action_probs(
        self, state: GameState, temperature: float = 1.0
    ) -> np.ndarray:
        """Run MCTS simulations and return action probabilities"""
        root = MCTSNode(state=state, parent=None, prior_prob=1.0)

        # Run simulations
        for _ in range(self.num_simulations):
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
                self._expand(node)

            value = self._simulate(node)
            self._backpropagate(node, value)

        # Calculate action probabilities
        policy = np.zeros((4, 4, 4), dtype=np.float32)

        if temperature == 0:  # Deterministic
            visits = {move: child.visit_count for move, child in root.children.items()}
            if visits:
                best_move = max(visits.items(), key=lambda x: x[1])[0]
                policy[best_move.x, best_move.y, best_move.piece_type.value] = 1.0
        else:
            # Calculate softmax of visit counts
            visits = np.array([child.visit_count for child in root.children.values()])
            moves = list(root.children.keys())

            if len(visits) > 0:
                visits = visits ** (1.0 / temperature)
                visits_sum = visits.sum()
                if visits_sum > 0:
                    probs = visits / visits_sum
                    for move, prob in zip(moves, probs):
                        policy[move.x, move.y, move.piece_type.value] = prob

        return policy

    def get_best_move(self, state: GameState) -> Move:
        """Get the best move according to MCTS simulations"""
        policy = self.get_action_probs(state, temperature=0)
        move_coords = np.unravel_index(policy.argmax(), policy.shape)
        return Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
