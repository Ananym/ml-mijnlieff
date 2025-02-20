import numpy as np
from game import GameState, Move, PieceType, Player


class RandomOpponent:
    def get_move(self, game_state: GameState) -> Move:
        """Choose a random legal move"""
        legal_moves = game_state.get_legal_moves()
        if not np.any(legal_moves):
            return None

        legal_positions = np.argwhere(legal_moves)
        idx = np.random.randint(len(legal_positions))
        x, y, piece_type = legal_positions[idx]
        return Move(x, y, PieceType(piece_type))


class StrategicOpponent:
    def get_move(self, game_state: GameState) -> Move:
        """Strategic opponent that prioritizes central control"""
        legal_moves = game_state.get_legal_moves()
        if not np.any(legal_moves):
            return None

        # Define central squares
        central_squares = {(1, 1), (1, 2), (2, 1), (2, 2)}

        # Get all possible legal moves
        possible_moves = [(x, y, pt) for x, y, pt in np.argwhere(legal_moves)]

        # First priority: If opponent has a piece in center, block their moves
        opponent = Player.TWO if game_state.current_player == Player.ONE else Player.ONE
        opponent_channel = opponent.value

        # Check if opponent has pieces in center
        for cx, cy in central_squares:
            if game_state.board[cx, cy, opponent_channel] == 1:
                # Find moves that would block most future moves to center
                best_blocking_moves = []
                max_blocked = 0

                for x, y, pt in possible_moves:
                    # Create a test state
                    test_state = GameState()
                    test_state.board = game_state.board.copy()
                    test_state.current_player = game_state.current_player
                    test_state.piece_counts = {
                        Player.ONE: game_state.piece_counts[Player.ONE].copy(),
                        Player.TWO: game_state.piece_counts[Player.TWO].copy(),
                    }

                    # Try the move
                    test_state.make_move(Move(x, y, PieceType(pt)))

                    # Count how many central moves this blocks
                    blocked_moves = 0
                    for tx, ty in central_squares:
                        if not any(
                            test_state.is_valid_move(Move(tx, ty, PieceType(p)))
                            for p in range(4)
                        ):
                            blocked_moves += 1

                    if blocked_moves > max_blocked:
                        max_blocked = blocked_moves
                        best_blocking_moves = [(x, y, pt)]
                    elif blocked_moves == max_blocked:
                        best_blocking_moves.append((x, y, pt))

                if best_blocking_moves:
                    x, y, pt = best_blocking_moves[
                        np.random.randint(len(best_blocking_moves))
                    ]
                    return Move(x, y, PieceType(pt))

        # Second priority: Play in center if possible
        center_moves = [
            (x, y, pt) for x, y, pt in possible_moves if (x, y) in central_squares
        ]
        if center_moves:
            x, y, pt = center_moves[np.random.randint(len(center_moves))]
            return Move(x, y, PieceType(pt))

        # Last resort: Random move
        x, y, pt = possible_moves[np.random.randint(len(possible_moves))]
        return Move(x, y, PieceType(pt))
