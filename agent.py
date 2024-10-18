import numpy as np
import torch
from game import GameState, GameStateRepresentation, Move, PieceType
from model import ValueNetworkWrapper
from typing import List


class RLAgent:

    def __init__(
        self, epsilon=0.2, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.value_network = ValueNetworkWrapper(device)
        self.epsilon = epsilon

    def select_move(self, legal_moves, game_state_representation):
        if np.random.random() < self.epsilon:
            # print("Random move")
            move = self.random_move(legal_moves)
        else:
            move = self.best_move(legal_moves, game_state_representation)

        return move

    def random_move(self, legal_moves):
        valid_moves = np.array(np.where(legal_moves == 1)).T
        if len(valid_moves) == 0:
            return None
        move = valid_moves[np.random.choice(len(valid_moves))]
        return Move(x=move[0], y=move[1], piece_type=PieceType(move[2]))

    def best_move(
        self,
        legal_moves_grid: np.ndarray,
        game_state_representation: GameStateRepresentation,
    ):
        if np.all(legal_moves_grid == 0):
            raise ValueError(
                "No legal moves available: all positions in legal_moves are zero"
            )

        # Transpose the grid input to (batch, channel, height, width)
        grid_input = np.transpose(game_state_representation.board, (2, 0, 1))
        grid_input = np.expand_dims(grid_input, axis=0)
        flat_input = np.expand_dims(
            game_state_representation.player_piece_counts, axis=0
        )

        # do we need to untranspose the output?

        probabilities = self.value_network.predict(grid_input, flat_input)[0]
        valid_probabilities = probabilities * legal_moves_grid

        # print_probs(valid_probabilities)

        x, y, piece_type = np.unravel_index(
            np.argmax(valid_probabilities), valid_probabilities.shape
        )

        # max_val = np.max(valid_probabilities)
        # max_indices = np.where(valid_probabilities == max_val)
        # if len(max_indices[0]) == 1:
        #     x, y, piece_type = [index[0] for index in max_indices]
        # else:
        #     random_choice = np.random.choice(len(max_indices[0]))
        #     x, y, piece_type = [index[random_choice] for index in max_indices]

        move = Move(x=x, y=y, piece_type=PieceType(piece_type))
        # print(f"Best move: {move}")
        return move

    def train(
        self,
        states: List[GameStateRepresentation],
        moves: List[Move],
        outcomes: List[int],
    ):
        grid_inputs = np.array(
            [np.transpose(state.board, (2, 0, 1)) for state in states]
        )
        flat_inputs = np.array([state.player_piece_counts for state in states])

        # Create target tensors
        targets = np.zeros((len(states), 4, 4, 4))
        for i, (move, outcome) in enumerate(zip(moves, outcomes)):
            targets[i, move.x, move.y, move.piece_type.value] = outcome

        return self.value_network.train(grid_inputs, flat_inputs, targets)

    def save(self, path):
        self.value_network.save(path)

    def load(self, path):
        self.value_network.load(path)


def print_probs(probs):
    for pt in PieceType:
        print(f"{pt.name}:")
        if not np.all(probs[:, :, pt.value] == 0):
            for y in range(4):
                line = []
                for x in range(4):
                    line.append(str(round(probs[x][y][pt.value], 2)))
                print(" ".join(line))
        else:
            print("No pieces available")
        print("\n")
