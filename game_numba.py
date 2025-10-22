"""
Experimental Numba-optimized versions of game logic functions.

These are drop-in replacements for hot path functions in game.py.
The original functions remain unchanged for safety.
"""

import numpy as np
from numba import njit, int8, int64, boolean
from game import PieceType, Player

# Compile-time constants for Numba
PIECE_DIAG = 0
PIECE_ORTHO = 1
PIECE_NEAR = 2
PIECE_FAR = 3

PLAYER_ONE = 0
PLAYER_TWO = 1


@njit(cache=True)
def _numba_get_legal_moves(
    board: np.ndarray,
    current_player_value: int,
    piece_counts: np.ndarray,
    last_move_x: int,
    last_move_y: int,
    last_move_piece_type: int,
    has_last_move: bool
) -> np.ndarray:
    """
    Numba-optimized version of get_legal_moves.

    Args:
        board: 4x4x2 board array
        current_player_value: 0 for Player.ONE, 1 for Player.TWO
        piece_counts: 4-element array of piece counts for current player
        last_move_x, last_move_y, last_move_piece_type: Last move info
        has_last_move: Whether there was a last move

    Returns:
        4x4x4 legal moves array
    """
    legal_moves = np.zeros((4, 4, 4), dtype=np.int64)

    # Check if any pieces available
    has_pieces = False
    for i in range(4):
        if piece_counts[i] > 0:
            has_pieces = True
            break

    if not has_pieces:
        return legal_moves

    # Create empty cells mask
    empty_cells = np.zeros((4, 4), dtype=np.int64)
    for x in range(4):
        for y in range(4):
            if board[x, y, 0] == 0 and board[x, y, 1] == 0:
                empty_cells[x, y] = 1

    # Check if board is empty (first move of game)
    board_is_empty = True
    for x in range(4):
        for y in range(4):
            if board[x, y, 0] != 0 or board[x, y, 1] != 0:
                board_is_empty = False
                break
        if not board_is_empty:
            break

    if board_is_empty:
        # First move: only edge cells
        for x in range(4):
            for y in range(4):
                is_edge = (x == 0 or x == 3 or y == 0 or y == 3)
                if is_edge and empty_cells[x, y]:
                    for piece_type in range(4):
                        if piece_counts[piece_type] > 0:
                            legal_moves[x, y, piece_type] = 1
        return legal_moves

    # Apply constraints based on last move
    if has_last_move:
        for x in range(4):
            for y in range(4):
                if not empty_cells[x, y]:
                    continue

                is_valid = False

                if last_move_piece_type == PIECE_DIAG:
                    # Diagonal: abs(dx) == abs(dy) and not same position
                    if abs(x - last_move_x) == abs(y - last_move_y) and x != last_move_x:
                        is_valid = True
                elif last_move_piece_type == PIECE_ORTHO:
                    # Orthogonal: same row or column
                    if x == last_move_x or y == last_move_y:
                        is_valid = True
                elif last_move_piece_type == PIECE_NEAR:
                    # Near: within 1 step
                    if abs(x - last_move_x) <= 1 and abs(y - last_move_y) <= 1:
                        is_valid = True
                elif last_move_piece_type == PIECE_FAR:
                    # Far: at least 2 steps in one direction
                    if abs(x - last_move_x) >= 2 or abs(y - last_move_y) >= 2:
                        is_valid = True

                if is_valid:
                    for piece_type in range(4):
                        if piece_counts[piece_type] > 0:
                            legal_moves[x, y, piece_type] = 1
    else:
        # No last move (after pass): all empty cells valid
        for x in range(4):
            for y in range(4):
                if empty_cells[x, y]:
                    for piece_type in range(4):
                        if piece_counts[piece_type] > 0:
                            legal_moves[x, y, piece_type] = 1

    return legal_moves


@njit(cache=True)
def _numba_calculate_score(board_slice: np.ndarray) -> int:
    """
    Numba-optimized version of _calculate_score.

    Args:
        board_slice: 4x4 array for a single player (board[:,:,player_value])

    Returns:
        Score for this player
    """
    score = 0

    # Check all possible line directions
    deltas = [(1, 0), (0, 1), (1, 1), (-1, 1)]

    for x in range(4):
        for y in range(4):
            for delta_idx in range(4):
                dx = deltas[delta_idx][0]
                dy = deltas[delta_idx][1]

                # Check if previous cell in line is occupied (would make this part of existing line)
                prev_x = x - dx
                prev_y = y - dy
                if 0 <= prev_x < 4 and 0 <= prev_y < 4:
                    if board_slice[prev_x, prev_y] == 1:
                        continue

                # Count line length
                line_length = 0
                continuation_idx = 0

                while True:
                    check_x = x + continuation_idx * dx
                    check_y = y + continuation_idx * dy

                    # Check bounds
                    if check_x < 0 or check_x >= 4 or check_y < 0 or check_y >= 4:
                        break

                    # Check if cell has piece
                    if board_slice[check_x, check_y] == 1:
                        line_length += 1

                    continuation_idx += 1

                # Add score for line
                if line_length == 4:
                    score += 2
                elif line_length == 3:
                    score += 1

    return score


@njit(cache=True)
def _numba_is_valid_first_move(x: int, y: int) -> bool:
    """Check if position is valid for first move (edge cells only)."""
    return x in (0, 3) or y in (0, 3)


@njit(cache=True)
def _numba_is_valid_move(
    board: np.ndarray,
    current_player_value: int,
    piece_counts: np.ndarray,
    x: int,
    y: int,
    piece_type: int,
    last_move_x: int,
    last_move_y: int,
    last_move_piece_type: int,
    has_last_move: bool
) -> bool:
    """
    Numba-optimized version of is_valid_move.

    Returns True if the move is valid, False otherwise.
    """
    # Check bounds
    if x < 0 or x >= 4 or y < 0 or y >= 4:
        return False

    # Check if piece is available
    if piece_counts[piece_type] <= 0:
        return False

    # Check if cell is empty
    if board[x, y, 0] != 0 or board[x, y, 1] != 0:
        return False

    # Check if board is empty (first move)
    board_is_empty = True
    for i in range(4):
        for j in range(4):
            if board[i, j, 0] != 0 or board[i, j, 1] != 0:
                board_is_empty = False
                break
        if not board_is_empty:
            break

    if board_is_empty:
        # First move: must be edge cell
        return _numba_is_valid_first_move(x, y)

    if not has_last_move:
        return True

    # Check constraints based on last move's piece type
    if last_move_piece_type == PIECE_DIAG:
        # Diagonal: abs(dx) == abs(dy) and not same position
        if abs(x - last_move_x) != abs(y - last_move_y) or x == last_move_x:
            return False
    elif last_move_piece_type == PIECE_ORTHO:
        # Orthogonal: same row or column
        if x != last_move_x and y != last_move_y:
            return False
    elif last_move_piece_type == PIECE_NEAR:
        # Near: within 1 step
        if abs(x - last_move_x) > 1 or abs(y - last_move_y) > 1:
            return False
    elif last_move_piece_type == PIECE_FAR:
        # Far: at least 2 steps in one direction
        if abs(x - last_move_x) < 2 and abs(y - last_move_y) < 2:
            return False

    return True


def get_legal_moves_numba(game_state) -> np.ndarray:
    """
    Wrapper function to call Numba-optimized get_legal_moves.

    This is a drop-in replacement for GameState.get_legal_moves().
    """
    # Extract current player's piece counts as array
    piece_counts = np.array([
        game_state.piece_counts[game_state.current_player][PieceType(i)]
        for i in range(4)
    ], dtype=np.int64)

    # Extract last move info
    if game_state.last_move is not None:
        last_move_x = game_state.last_move.x
        last_move_y = game_state.last_move.y
        last_move_piece_type = game_state.last_move.piece_type.value
        has_last_move = True
    else:
        last_move_x = 0
        last_move_y = 0
        last_move_piece_type = 0
        has_last_move = False

    return _numba_get_legal_moves(
        game_state.board,
        game_state.current_player.value,
        piece_counts,
        last_move_x,
        last_move_y,
        last_move_piece_type,
        has_last_move
    )


def calculate_score_numba(game_state, player) -> int:
    """
    Wrapper function to call Numba-optimized _calculate_score.

    This is a drop-in replacement for GameState._calculate_score().
    """
    board_slice = game_state.board[:, :, player.value].copy()
    return _numba_calculate_score(board_slice)


def is_valid_move_numba(game_state, move) -> bool:
    """
    Wrapper function to call Numba-optimized is_valid_move.

    This is a drop-in replacement for GameState.is_valid_move().
    """
    # Extract piece counts
    piece_counts = np.array([
        game_state.piece_counts[game_state.current_player][PieceType(i)]
        for i in range(4)
    ], dtype=np.int64)

    # Extract last move info
    if game_state.last_move is not None:
        last_move_x = game_state.last_move.x
        last_move_y = game_state.last_move.y
        last_move_piece_type = game_state.last_move.piece_type.value
        has_last_move = True
    else:
        last_move_x = 0
        last_move_y = 0
        last_move_piece_type = 0
        has_last_move = False

    return _numba_is_valid_move(
        game_state.board,
        game_state.current_player.value,
        piece_counts,
        move.x,
        move.y,
        move.piece_type.value,
        last_move_x,
        last_move_y,
        last_move_piece_type,
        has_last_move
    )


# Monkey-patch helper for easy testing
def apply_numba_optimizations(game_state):
    """
    Apply Numba optimizations to a GameState instance.

    This replaces the instance methods with Numba-optimized versions.
    """
    # Store original methods for restoration if needed
    game_state._original_get_legal_moves = game_state.get_legal_moves
    game_state._original_calculate_score = game_state._calculate_score
    game_state._original_is_valid_move = game_state.is_valid_move

    # Replace with Numba versions
    game_state.get_legal_moves = lambda: get_legal_moves_numba(game_state)
    game_state._calculate_score = lambda player: calculate_score_numba(game_state, player)
    game_state.is_valid_move = lambda move: is_valid_move_numba(game_state, move)

    return game_state


def restore_original_methods(game_state):
    """Restore original non-Numba methods."""
    if hasattr(game_state, '_original_get_legal_moves'):
        game_state.get_legal_moves = game_state._original_get_legal_moves
        game_state._calculate_score = game_state._original_calculate_score
        game_state.is_valid_move = game_state._original_is_valid_move
    return game_state
