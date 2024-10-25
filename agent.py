from dataclasses import dataclass
import random
import numpy as np
import torch
from game import (
    GameStateRepresentation,
    PossibleFutureGameStateRepresentation,
    score_position,
)

from model import ValueNetworkWrapper
from typing import List, Optional


@dataclass
class FutureStateWithOutcomePrediction(PossibleFutureGameStateRepresentation):
    predicted_outcome: Optional[float]


class RLAgent:

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.value_network = ValueNetworkWrapper(device)

    # def select_move(
    #     self,
    #     possible_next_states: List[PossibleFutureGameStateRepresentation],
    #     target_low: bool,
    #     epsilon=0.2,
    # ):
    #     if random.random() < epsilon:
    #         next_state = self.random_move(possible_next_states)
    #     else:
    #         next_state = self.best_move(possible_next_states, target_low)

    #     return next_state

    def select_move(self, possible_next_states, target_low, epsilon=0.2):
        if random.random() < epsilon:
            # Smart random - prefer moves that build lines or block opponent
            scores = []
            for state in possible_next_states:
                board = state.representation.board
                score = score_position(board[:, :, 0 if target_low else 1])
                scores.append(score)

            # Softmax the scores
            scores = np.exp(scores) / np.sum(np.exp(scores))
            return np.random.choice(possible_next_states, p=scores)
        else:
            return self.best_move(possible_next_states, target_low)

    def random_move(
        self, possible_next_states: List[PossibleFutureGameStateRepresentation]
    ):
        next_state = random.choice(possible_next_states)
        return FutureStateWithOutcomePrediction(
            next_state.representation, next_state.move, None
        )

    def best_move(
        self,
        possible_next_states: List[PossibleFutureGameStateRepresentation],
        target_low: bool = True,
    ):
        strongest_predicted_outcome = None
        strongest_state = None
        grid_inputs = []
        flat_inputs = []

        for possible_state in possible_next_states:
            grid_inputs.append(possible_state.representation.board)
            flat_inputs.append(possible_state.representation.flat_values)

        predicted_outcomes = self.value_network.predict(
            np.array(grid_inputs), np.array(flat_inputs)
        )

        for i, possible_state in enumerate(possible_next_states):
            predicted_outcome = predicted_outcomes[i]
            if (
                strongest_predicted_outcome is None
                or (target_low and predicted_outcome < strongest_predicted_outcome)
                or (not target_low and predicted_outcome > strongest_predicted_outcome)
            ):
                strongest_predicted_outcome = predicted_outcome
                strongest_state = possible_state

        # print(
        #     f"Target low: {target_low}, predicted outcome: {strongest_predicted_outcome}"
        # )
        return FutureStateWithOutcomePrediction(
            strongest_state.representation,
            strongest_state.move,
            strongest_predicted_outcome,
        )

    def train(
        self,
        states: List[GameStateRepresentation],
        outcomes: List[float],
    ):
        grid_inputs = np.array([state.board for state in states])
        flat_inputs = np.array([state.flat_values for state in states])

        return self.value_network.train(grid_inputs, flat_inputs, outcomes)

    def save(self, path):
        self.value_network.save(path)

    def load(self, path):
        self.value_network.load(path)


# def print_probs(probs):
#     for pt in PieceType:
#         print(f"{pt.name}:")
#         if not np.all(probs[:, :, pt.value] == 0):
#             for y in range(4):
#                 line = []
#                 for x in range(4):
#                     line.append(str(round(probs[x][y][pt.value], 2)))
#                 print(" ".join(line))
#         else:
#             print("No pieces available")
#         print("\n")
