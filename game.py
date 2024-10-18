from dataclasses import dataclass
import numpy as np
from enum import Enum
from typing import Dict, Optional
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
    OPPONENT_MUST_PASS = 1
    GAME_OVER = 2


@dataclass
class Move:
    x: int
    y: int
    piece_type: PieceType


@dataclass
class GameStateRepresentation:
    # 4x4x6
    board: np.ndarray
    # 8x1
    player_piece_counts: np.ndarray


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
        self.current_player = Player.ONE
        self.piece_counts: Dict[Player, Dict[PieceType, int]] = {
            Player.ONE: {pt: 2 for pt in PieceType},
            Player.TWO: {pt: 2 for pt in PieceType},
        }
        self.last_move: Optional[Move] = None

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
            return TurnResult.OPPONENT_MUST_PASS
        elif not self_has_pieces and opp_must_pass:
            return TurnResult.GAME_OVER
        elif not opp_has_pieces:
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

        # print(
        #     f"Checking if move ({x}, {y}, {PieceType(piece_type).name}) is valid..."
        # )

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
        score_one = self.calculate_score(Player.ONE)
        score_two = self.calculate_score(Player.TWO)

        if score_one > score_two:
            return Player.ONE
        elif score_two > score_one:
            return Player.TWO
        else:
            return None  # Draw

    def calculate_score(self, player: Player) -> int:

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

    def get_legal_moves(self) -> np.ndarray:
        legal_moves = np.zeros((4, 4, 4), dtype=np.int64)

        for x in range(4):
            for y in range(4):
                for piece_type in range(4):
                    if self.is_valid_move(Move(x, y, piece_type)):
                        legal_moves[x, y, piece_type] = 1

        return legal_moves

    # def get_legal_moves(self) -> np.ndarray:
    #     legal_moves = np.zeros((4, 4, 4), dtype=np.int64)

    #     # Check if any pieces are available
    #     pieces_available = np.array([
    #         self.piece_counts[self.current_player][pt] > 0 for pt in PieceType
    #     ],
    #                                 dtype=bool)

    #     if not np.any(pieces_available):
    #         return legal_moves  # No legal moves if no pieces are available

    #     # Check which cells are empty
    #     empty_cells = (self.board[:, :, 0] == 0) & (self.board[:, :, 1] == 0)

    #     # Create edge mask
    #     edge_mask = np.ones((4, 4), dtype=bool)
    #     edge_mask[1:3, 1:3] = False

    #     if self.last_move is None:
    #         # First move: only edge cells are valid for all available piece types
    #         for pt in PieceType:
    #             if pieces_available[pt.value]:
    #                 legal_moves[:, :, pt.value] = empty_cells & edge_mask
    #     else:
    #         last_x, last_y, last_piece_type = self.last_move.x, self.last_move.y, self.last_move.piece_type

    #         # Create base masks for each move type
    #         diag_mask = np.abs(np.subtract.outer(
    #             np.arange(4),
    #             last_x)) == np.abs(np.subtract.outer(np.arange(4), last_y))
    #         np.fill_diagonal(diag_mask, False)  # Exclude the cell itself

    #         ortho_mask = (np.arange(4)
    #                       == last_x)[:, np.newaxis] | (np.arange(4) == last_y)

    #         near_mask = (np.abs(np.subtract.outer(np.arange(4), last_x))
    #                      <= 1) & (np.abs(
    #                          np.subtract.outer(np.arange(4), last_y)) <= 1)
    #         near_mask[last_x, last_y] = False  # Exclude the cell itself

    #         far_mask = ~near_mask
    #         far_mask[last_x,
    #                  last_y] = True  # Include the cell itself in far moves

    #         # Apply restrictions based on the last piece type
    #         if last_piece_type == PieceType.DIAG:
    #             move_mask = diag_mask
    #         elif last_piece_type == PieceType.ORTHO:
    #             move_mask = ortho_mask
    #         elif last_piece_type == PieceType.NEAR:
    #             move_mask = near_mask
    #         elif last_piece_type == PieceType.FAR:
    #             move_mask = far_mask
    #         else:
    #             raise ValueError(f"Invalid last piece type: {last_piece_type}")

    #         # Apply the move mask to all piece types
    #         for pt in PieceType:
    #             if pieces_available[pt.value]:
    #                 legal_moves[:, :, pt.value] = empty_cells & move_mask

    #         # Apply the edge rule only if not all cells are empty
    #         if not np.all(empty_cells):
    #             legal_moves[:, :, :] &= edge_mask[:, :, np.newaxis]

    #     return legal_moves

    def get_board_state_representation(self) -> np.ndarray:
        # return self.board
        current_shape = self.board.shape
        new_shape = (current_shape[0], current_shape[1], current_shape[2] + 4)
        new_board = np.zeros(new_shape, dtype=self.board.dtype)
        new_board[:, :, : current_shape[2]] = self.board
        return new_board

    def get_pieces_state_representation(self) -> np.ndarray:
        # return self.piece_counts
        return np.array(
            [
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
            player_piece_counts=self.get_pieces_state_representation(),
        )


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
