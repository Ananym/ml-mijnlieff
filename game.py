import copy
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from enum import Enum
from typing import Dict, Optional, List
from numpy.typing import NDArray


class PieceType(Enum):
    DIAG = 0
    ORTHO = 1
    NEAR = 2
    FAR = 3


class Player(Enum):
    ONE = 0
    TWO = 1


class TurnResult(Enum):
    NORMAL = 0
    OPPONENT_WAS_FORCED_TO_PASS = 1
    GAME_OVER = 2


@dataclass(frozen=True)  # Making it immutable and hashable
class Move:
    __slots__ = ["x", "y", "piece_type"]
    x: int
    y: int
    piece_type: PieceType

    def __hash__(self):
        return hash((self.x, self.y, self.piece_type))


@dataclass
class GameStateRepresentation:
    __slots__ = ["board", "flat_values"]
    # 4x4x10
    board: np.ndarray
    # 8x1
    flat_values: np.ndarray


@dataclass
class PossibleFutureGameStateRepresentation:
    __slots__ = ["representation", "move"]
    representation: GameStateRepresentation
    move: Move


class IllegalMoveException(Exception):

    def __init__(self, player, move: Move):
        self.player = player
        self.x, self.y, self.piece_type = move.x, move.y, move.piece_type
        super().__init__(self.__str__())

    def __str__(self):
        return f"{self.player.name} made an illegal move: ({self.x}, {self.y}, {self.piece_type.name})"


class GameState:
    verbose_output = False  # Class variable to toggle output

    def __init__(self):
        self.board: NDArray[np.int8] = np.zeros((4, 4, 2), dtype=np.int8)
        self.terrain: NDArray[np.int8] = np.zeros((4, 4, 4), dtype=np.int8)
        self.current_player = Player.ONE
        self.piece_counts: Dict[Player, Dict[PieceType, int]] = {
            Player.ONE: {pt: 2 for pt in PieceType},
            Player.TWO: {pt: 2 for pt in PieceType},
        }
        self.last_move: Optional[Move] = None
        self.is_over = False
        self.winner: Player | None = None

    def make_move(self, move: Move) -> TurnResult:
        x, y, piece_type = move.x, move.y, move.piece_type
        if not self.is_valid_move(move):
            raise IllegalMoveException(self.current_player, move)

        player_channel = self.current_player.value
        self.board[x, y, player_channel] = 1
        self.piece_counts[self.current_player][piece_type] -= 1
        self.last_move = move

        if self.verbose_output:
            self.print_move_info(move)

        self_has_pieces = any(
            self.piece_counts[self.current_player][pt] > 0 for pt in PieceType
        )
        self.current_player = (
            Player.ONE if self.current_player == Player.TWO else Player.TWO
        )
        opp_has_pieces = any(
            self.piece_counts[self.current_player][pt] > 0 for pt in PieceType
        )
        opp_legal_moves = self.get_legal_moves()
        opp_must_pass = False
        if np.all(opp_legal_moves[:, :, :] == 0):
            opp_must_pass = True

        if self_has_pieces and opp_must_pass:
            self.pass_turn()
            return TurnResult.OPPONENT_WAS_FORCED_TO_PASS
        elif (not self_has_pieces and opp_must_pass) or not opp_has_pieces:
            self.is_over = True
            self.winner = self.get_winner()
            return TurnResult.GAME_OVER
        else:
            return TurnResult.NORMAL

    def print_move_info(self, move: Move):
        x, y, piece_type = move.x, move.y, move.piece_type
        print(
            f"Player {self.current_player.name} played a {piece_type.name} piece at position ({x}, {y})"
        )
        print(f"Remaining pieces for Player {self.current_player.name}:")
        for pt in PieceType:
            print(f"  {pt.name}: {self.piece_counts[self.current_player][pt]}")
        print()

    def is_valid_move(self, move: Move) -> bool:
        x, y, piece_type = move.x, move.y, move.piece_type

        if np.all(self.board[0:4, 0:4, 0:2] == 0):
            if x not in [0, 3] and y not in [0, 3]:
                return False

        piece_is_on_board = (0 <= x < 4) and (0 <= y < 4)
        piece_is_available = (
            self.piece_counts[self.current_player][PieceType(piece_type)] > 0
        )
        cell_is_not_empty = (self.board[x, y, 0] == 0) and (self.board[x, y, 1] == 0)

        validAndEmpty = piece_is_on_board and piece_is_available and cell_is_not_empty

        if self.last_move is None:
            return validAndEmpty

        last_x, last_y, last_piece_type = (
            self.last_move.x,
            self.last_move.y,
            self.last_move.piece_type,
        )

        if last_piece_type == PieceType.DIAG:
            if abs(x - last_x) != abs(y - last_y) or x == last_x:
                return False
        elif last_piece_type == PieceType.ORTHO:
            if x != last_x and y != last_y:
                return False
        elif last_piece_type == PieceType.NEAR:
            if abs(x - last_x) > 1 or abs(y - last_y) > 1:
                return False
        elif last_piece_type == PieceType.FAR:
            if abs(x - last_x) < 2 and abs(y - last_y) < 2:
                return False

        return validAndEmpty

    def check_game_over(self) -> bool:
        return all(
            count == 0 for count in self.piece_counts[self.current_player].values()
        )

    def pass_turn(self):
        self.last_move = None
        if self.verbose_output:
            print("Passing the turn of " + self.current_player.name)
        self.current_player = (
            Player.ONE if self.current_player == Player.TWO else Player.TWO
        )

    def get_winner(self) -> Optional[Player]:
        score_one = self._calculate_score(Player.ONE)
        score_two = self._calculate_score(Player.TWO)

        if score_one > score_two:
            return Player.ONE
        elif score_two > score_one:
            return Player.TWO
        else:
            return None  # Draw

    def _calculate_score(self, player: Player) -> int:

        def get_value_or_none(arr, indices):
            try:
                return arr[indices]
            except IndexError:
                return None

        score = 0
        board = self.board[:, :, player.value]
        for x in range(0, 4):
            for y in range(0, 4):
                for delta in ((1, 0), (0, 1), (1, 1), (-1, 1)):
                    if get_value_or_none(board, (x - delta[0], y - delta[1])) == 1:
                        # previous cell in line is occupied, so this is part of an existing line, skip
                        continue
                    lineLength = 0
                    lineContinuationIndex = 0
                    while True:
                        lineCellVal = get_value_or_none(
                            board,
                            (
                                x + lineContinuationIndex * delta[0],
                                y + lineContinuationIndex * delta[1],
                            ),
                        )
                        if lineCellVal is None:
                            # exited the board
                            break
                        if lineCellVal == 1:
                            # found a piece in the line
                            lineLength += 1
                        lineContinuationIndex += 1
                    if lineLength == 4:
                        score += 2
                    elif lineLength == 3:
                        score += 1
        return score

    def has_legal_moves(self) -> bool:
        return np.any(self.get_legal_moves())

    # def get_legal_moves(self) -> np.ndarray:
    #     legal_moves = np.zeros((4, 4, 4), dtype=np.int64)

    #     for x in range(4):
    #         for y in range(4):
    #             for piece_type in range(4):
    #                 if self.is_valid_move(Move(x, y, piece_type)):
    #                     legal_moves[x, y, piece_type] = 1

    #     return legal_moves

    def get_legal_moves(self) -> np.ndarray:
        # Initialize legal moves array
        legal_moves = np.zeros((4, 4, 4), dtype=np.int64)

        # Check piece availability for each type
        available_pieces = np.array(
            [self.piece_counts[self.current_player][PieceType(i)] > 0 for i in range(4)]
        )

        # If no pieces are available, return all zeros
        if not np.any(available_pieces):
            return legal_moves

        # Create a mask for empty cells
        empty_cells = (self.board[:, :, 0] == 0) & (self.board[:, :, 1] == 0)

        # Handle the first move of the game
        if np.all(self.board == 0):
            edge_cells = np.ones((4, 4), dtype=bool)
            edge_cells[1:3, 1:3] = False  # Set middle 2x2 to False
            for piece_type in range(4):
                if available_pieces[piece_type]:
                    legal_moves[:, :, piece_type] = edge_cells & empty_cells
            return legal_moves

        # Apply constraints based on the last move
        if self.last_move is not None:
            last_x, last_y, last_piece_type = (
                self.last_move.x,
                self.last_move.y,
                self.last_move.piece_type,
            )

            x_coords, y_coords = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")

            if last_piece_type == PieceType.DIAG:
                valid_cells = np.abs(x_coords - last_x) == np.abs(y_coords - last_y)
            elif last_piece_type == PieceType.ORTHO:
                valid_cells = (x_coords == last_x) | (y_coords == last_y)
            elif last_piece_type == PieceType.NEAR:
                valid_cells = (np.abs(x_coords - last_x) <= 1) & (
                    np.abs(y_coords - last_y) <= 1
                )
            elif last_piece_type == PieceType.FAR:
                valid_cells = (np.abs(x_coords - last_x) >= 2) | (
                    np.abs(y_coords - last_y) >= 2
                )

            for piece_type in range(4):
                if available_pieces[piece_type]:
                    legal_moves[:, :, piece_type] = valid_cells & empty_cells
        else:
            # If there's no last move (e.g., after a pass), all empty cells are valid
            for piece_type in range(4):
                if available_pieces[piece_type]:
                    legal_moves[:, :, piece_type] = empty_cells

        return legal_moves

    def get_board_state_representation(self) -> np.ndarray:
        representation = np.zeros((4, 4, 10), dtype=self.board.dtype)
        representation[:, :, :2] = self.board

        move_representation = np.zeros((4, 4, 4))
        if self.last_move is not None:
            move_representation[
                (self.last_move.x, self.last_move.y, self.last_move.piece_type.value)
            ] = 1

        representation[:, :, 2:6] = move_representation
        representation[:, :, 6:10] = self.terrain

        return representation

    def get_pieces_state_representation(self) -> np.ndarray:
        return np.array(
            [
                1 if self.current_player == Player.ONE else 0,
                1 if self.current_player == Player.TWO else 0,
                self.piece_counts[Player.ONE][PieceType.DIAG],
                self.piece_counts[Player.ONE][PieceType.ORTHO],
                self.piece_counts[Player.ONE][PieceType.NEAR],
                self.piece_counts[Player.ONE][PieceType.FAR],
                self.piece_counts[Player.TWO][PieceType.DIAG],
                self.piece_counts[Player.TWO][PieceType.ORTHO],
                self.piece_counts[Player.TWO][PieceType.NEAR],
                self.piece_counts[Player.TWO][PieceType.FAR],
            ]
        )

    def get_game_state_representation(self) -> GameStateRepresentation:
        return GameStateRepresentation(
            board=self.get_board_state_representation(),
            flat_values=self.get_pieces_state_representation(),
        )

    def get_possible_next_states(self) -> List[PossibleFutureGameStateRepresentation]:
        next_states = []
        legal_moves = self.get_legal_moves()
        for index in np.argwhere(legal_moves):
            move = Move(index[0], index[1], PieceType(index[2]))
            # Instead of deep copying, we'll create a new state and update only what's necessary
            new_state = GameState()
            new_state.board = self.board.copy()
            new_state.terrain = self.terrain
            new_state.current_player = self.current_player
            new_state.piece_counts = {
                Player.ONE: self.piece_counts[Player.ONE].copy(),
                Player.TWO: self.piece_counts[Player.TWO].copy(),
            }
            new_state.make_move(move)
            possible_state_package = PossibleFutureGameStateRepresentation(
                new_state.get_game_state_representation(), move
            )
            next_states.append(possible_state_package)
        return next_states


def count_legal_move_positions(legal_moves):
    return np.sum(legal_moves[:, :, 0])


def print_legal_moves(legal_moves):
    for y in range(4):
        line = []
        for x in range(4):
            cellVal = np.sum(legal_moves[x, y, :])
            line.append(str(cellVal))
        print(" ".join(line))


def print_full_legal_moves(legal_moves):
    for pt in PieceType:
        print(f"{pt.name}:")
        for y in range(4):
            line = []
            for x in range(4):
                cellVal = legal_moves[x, y, pt.value]
                line.append(str(cellVal))
            print(" ".join(line))
        print("\n")


def print_piece_counts(piece_counts):
    for player in [Player.ONE, Player.TWO]:
        for piece_type in PieceType:
            print(f"{piece_type.name}: {piece_counts[player][piece_type]}")


# Bonus rules
# 1 Can be played into by both players
# 2 +1 point to line using
# 3 blocks lines
# 4 instant +1 point but no restrictions
