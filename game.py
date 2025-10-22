import copy
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from enum import Enum
from typing import Dict, Optional, List
from numpy.typing import NDArray
import time


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


@dataclass  # Removed frozen=True to allow pickling in multiprocessing
class Move:
    __slots__ = ["x", "y", "piece_type"]
    x: int
    y: int
    piece_type: PieceType

    def __hash__(self):
        return hash((self.x, self.y, self.piece_type))

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return self.x == other.x and self.y == other.y and self.piece_type == other.piece_type


@dataclass
class GameStateRepresentation:
    __slots__ = ["board", "flat_values"]
    # 4x4x6
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
    enable_timing = False  # Class variable to toggle timing measurements

    # Timing data dictionary to store cumulative times
    timing_data = {
        "get_legal_moves": 0.0,
        "make_move": 0.0,
        "undo_move": 0.0,
        "pass_turn": 0.0,
        "_calculate_score": 0.0,
    }
    # Count of calls to each method for averaging
    timing_counts = {k: 0 for k in timing_data}

    @classmethod
    def reset_timing(cls):
        """Reset all timing counters"""
        cls.timing_data = {k: 0.0 for k in cls.timing_data}
        cls.timing_counts = {k: 0 for k in cls.timing_data}

    @classmethod
    def print_timing(cls):
        """Print timing statistics if enabled"""
        if not cls.enable_timing:
            return

        print("\n--- GameState Timing Statistics ---")
        for method, time_value in cls.timing_data.items():
            count = cls.timing_counts[method]
            if count > 0:
                avg_time = time_value / count
                print(
                    f"{method}: {time_value:.6f}s total, {avg_time:.6f}s avg ({count} calls)"
                )
            else:
                print(f"{method}: {time_value:.6f}s total (0 calls)")
        print("--------------------------------\n")

    def __init__(self):
        self.board: NDArray[np.int8] = np.zeros((4, 4, 2), dtype=np.int8)
        self.current_player = Player.ONE
        self.piece_counts: Dict[Player, Dict[PieceType, int]] = {
            Player.ONE: {pt: 2 for pt in PieceType},
            Player.TWO: {pt: 2 for pt in PieceType},
        }
        self.last_move: Optional[Move] = None
        self.is_over = False
        self.winner: Player | None = None
        self.move_count = 0  # Track number of moves made
        self.move_history = []  # track history of moves for undo

    def make_move(self, move: Move) -> TurnResult:
        start_time = time.perf_counter() if self.enable_timing else 0

        x, y, piece_type = move.x, move.y, move.piece_type
        if not self.is_valid_move(move):
            raise IllegalMoveException(self.current_player, move)

        # save previous state info for undo
        prev_state = {
            "player": self.current_player,
            "last_move": self.last_move,
            "is_over": self.is_over,
            "winner": self.winner,
        }
        self.move_history.append((move, prev_state))

        player_channel = self.current_player.value
        self.board[x, y, player_channel] = 1
        self.piece_counts[self.current_player][piece_type] -= 1
        self.last_move = move
        self.move_count += 1  # Increment move counter when a move is made

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

        # Time legal moves calculation separately
        legal_start = time.perf_counter() if self.enable_timing else 0
        opp_legal_moves = self.get_legal_moves()
        if self.enable_timing:
            # Don't double-count this get_legal_moves call
            self.timing_data["get_legal_moves"] -= time.perf_counter() - legal_start
            self.timing_counts["get_legal_moves"] -= 1

        opp_must_pass = False
        if np.all(opp_legal_moves[:, :, :] == 0):
            opp_must_pass = True

        result = TurnResult.NORMAL
        if self_has_pieces and opp_must_pass:
            self.pass_turn()
            result = TurnResult.OPPONENT_WAS_FORCED_TO_PASS
        elif (not self_has_pieces and opp_must_pass) or not opp_has_pieces:
            self.is_over = True
            self.winner = self.get_winner()
            result = TurnResult.GAME_OVER

        if self.enable_timing:
            self.timing_data["make_move"] += time.perf_counter() - start_time
            self.timing_counts["make_move"] += 1

        return result

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
        start_time = time.perf_counter() if self.enable_timing else 0

        # save previous state info for undo
        prev_state = {
            "player": self.current_player,
            "last_move": self.last_move,
            "is_over": self.is_over,
            "winner": self.winner,
        }
        # use None as move to indicate a pass
        self.move_history.append((None, prev_state))

        self.last_move = None
        if self.verbose_output:
            print("Passing the turn of " + self.current_player.name)
        self.current_player = (
            Player.ONE if self.current_player == Player.TWO else Player.TWO
        )

        if self.enable_timing:
            self.timing_data["pass_turn"] += time.perf_counter() - start_time
            self.timing_counts["pass_turn"] += 1

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
        start_time = time.perf_counter() if self.enable_timing else 0

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

        if self.enable_timing:
            self.timing_data["_calculate_score"] += time.perf_counter() - start_time
            self.timing_counts["_calculate_score"] += 1

        return score

    def has_legal_moves(self) -> bool:
        return np.any(self.get_legal_moves())

    def get_legal_moves(self) -> np.ndarray:
        start_time = time.perf_counter() if self.enable_timing else 0

        # Initialize legal moves array
        legal_moves = np.zeros((4, 4, 4), dtype=np.int64)

        # Check piece availability for each type
        available_pieces = np.array(
            [self.piece_counts[self.current_player][PieceType(i)] > 0 for i in range(4)]
        )

        # If no pieces are available, return all zeros
        if not np.any(available_pieces):
            if self.enable_timing:
                self.timing_data["get_legal_moves"] += time.perf_counter() - start_time
                self.timing_counts["get_legal_moves"] += 1
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

            if self.enable_timing:
                self.timing_data["get_legal_moves"] += time.perf_counter() - start_time
                self.timing_counts["get_legal_moves"] += 1
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

        if self.enable_timing:
            self.timing_data["get_legal_moves"] += time.perf_counter() - start_time
            self.timing_counts["get_legal_moves"] += 1

        return legal_moves

    def get_board_state_representation(self, subjective: bool = False) -> np.ndarray:
        representation = np.zeros((4, 4, 6), dtype=self.board.dtype)

        if not subjective:
            # Standard representation - player one then player two
            representation[:, :, :2] = self.board
        else:
            # Subjective representation - current player then opponent
            current_idx = self.current_player.value
            opponent_idx = 1 - current_idx  # flip between 0 and 1

            # Current player's pieces in first layer
            representation[:, :, 0] = self.board[:, :, current_idx]
            # Opponent's pieces in second layer
            representation[:, :, 1] = self.board[:, :, opponent_idx]

        # Last move representation stays the same
        last_move_representation = np.zeros((4, 4, 4))
        if self.last_move is not None:
            last_move_representation[
                (self.last_move.x, self.last_move.y, self.last_move.piece_type.value)
            ] = 1

        representation[:, :, 2:6] = last_move_representation

        return representation

    def get_pieces_state_representation(self, subjective: bool = False) -> np.ndarray:
        # add player scores to the flat representation
        player_one_score = self._calculate_score(Player.ONE)
        player_two_score = self._calculate_score(Player.TWO)

        if not subjective:
            # traditional ordering: player one then player two
            return np.array(
                [
                    (
                        1 if self.current_player == Player.ONE else 0
                    ),  # single value for current player
                    self.piece_counts[Player.ONE][PieceType.DIAG],
                    self.piece_counts[Player.ONE][PieceType.ORTHO],
                    self.piece_counts[Player.ONE][PieceType.NEAR],
                    self.piece_counts[Player.ONE][PieceType.FAR],
                    self.piece_counts[Player.TWO][PieceType.DIAG],
                    self.piece_counts[Player.TWO][PieceType.ORTHO],
                    self.piece_counts[Player.TWO][PieceType.NEAR],
                    self.piece_counts[Player.TWO][PieceType.FAR],
                    player_one_score,
                    player_two_score,
                    self.move_count,  # track how many moves have been made
                ]
            )
        else:
            # subjective ordering: current player then opponent
            current = self.current_player
            opponent = Player.TWO if current == Player.ONE else Player.ONE
            current_score = (
                player_one_score if current == Player.ONE else player_two_score
            )
            opponent_score = (
                player_two_score if current == Player.ONE else player_one_score
            )

            return np.array(
                [
                    (
                        1 if current == Player.ONE else 0
                    ),  # still indicate which player is current
                    self.piece_counts[current][PieceType.DIAG],
                    self.piece_counts[current][PieceType.ORTHO],
                    self.piece_counts[current][PieceType.NEAR],
                    self.piece_counts[current][PieceType.FAR],
                    self.piece_counts[opponent][PieceType.DIAG],
                    self.piece_counts[opponent][PieceType.ORTHO],
                    self.piece_counts[opponent][PieceType.NEAR],
                    self.piece_counts[opponent][PieceType.FAR],
                    current_score,
                    opponent_score,
                    self.move_count,  # track how many moves have been made
                ]
            )

    def get_game_state_representation(
        self, subjective: bool = False
    ) -> GameStateRepresentation:
        return GameStateRepresentation(
            board=self.get_board_state_representation(subjective),
            flat_values=self.get_pieces_state_representation(subjective),
        )

    def get_new_game_state_representation(self) -> GameStateRepresentation:
        # use the existing board representation
        board_representation = self.get_board_state_representation()

        # calculate scores for flat representation
        player_one_score = self._calculate_score(Player.ONE)
        player_two_score = self._calculate_score(Player.TWO)

        # create flat representation with single current player indicator
        # (1 for Player.ONE, 0 for Player.TWO)
        flat_representation = np.array(
            [
                (
                    1 if self.current_player == Player.ONE else 0
                ),  # single value for current player
                self.piece_counts[Player.ONE][PieceType.DIAG],
                self.piece_counts[Player.ONE][PieceType.ORTHO],
                self.piece_counts[Player.ONE][PieceType.NEAR],
                self.piece_counts[Player.ONE][PieceType.FAR],
                self.piece_counts[Player.TWO][PieceType.DIAG],
                self.piece_counts[Player.TWO][PieceType.ORTHO],
                self.piece_counts[Player.TWO][PieceType.NEAR],
                self.piece_counts[Player.TWO][PieceType.FAR],
                player_one_score,
                player_two_score,
            ]
        )

        return GameStateRepresentation(
            board=board_representation,
            flat_values=flat_representation,
        )

    def get_possible_next_states(self) -> List[PossibleFutureGameStateRepresentation]:
        next_states = []
        legal_moves = self.get_legal_moves()
        for index in np.argwhere(legal_moves):
            move = Move(index[0], index[1], PieceType(index[2]))
            # create a new state and update only what's necessary
            new_state = GameState()
            new_state.board = self.board.copy()
            # terrain is removed
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

    def get_scores(self) -> Dict[Player, int]:
        """Returns current scores for both players"""
        return {
            Player.ONE: self._calculate_score(Player.ONE),
            Player.TWO: self._calculate_score(Player.TWO),
        }

    def undo_move(self) -> bool:
        """Undo the last move, restoring the previous game state

        returns:
            bool: True if a move was undone, False if there was no move to undo
        """
        start_time = time.perf_counter() if self.enable_timing else 0

        if not self.move_history:
            if self.enable_timing:
                self.timing_data["undo_move"] += time.perf_counter() - start_time
                self.timing_counts["undo_move"] += 1
            return False

        # pop the last move and its previous state
        last_move, prev_state = self.move_history.pop()

        # check if it was a pass (move is None)
        if last_move is not None:
            # restore the piece to the player's supply
            player = prev_state["player"]
            piece_type = last_move.piece_type
            self.piece_counts[player][piece_type] += 1

            # remove the piece from the board
            x, y = last_move.x, last_move.y
            self.board[x, y, player.value] = 0

            # only decrement move count for actual moves, not passes
            self.move_count -= 1

        # restore previous game state
        self.current_player = prev_state["player"]
        self.last_move = prev_state[
            "last_move"
        ]  # this is the move before the current one
        self.is_over = prev_state["is_over"]
        self.winner = prev_state["winner"]

        if self.enable_timing:
            self.timing_data["undo_move"] += time.perf_counter() - start_time
            self.timing_counts["undo_move"] += 1

        return True


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
