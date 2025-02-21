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
        """Strategic opponent with sophisticated prioritization:
        1. Moves that force opponent to skip their turn (no valid moves)
        2. Moves that prevent opponent from placing in center next turn
        3. Moves that complete a four-line (for points)
        4. Moves in center squares
        5. Moves that complete a three-line
        6. Any other legal move
        """
        legal_moves = game_state.get_legal_moves()
        if not np.any(legal_moves):
            return None

        # Define central squares
        central_squares = {(1, 1), (1, 2), (2, 1), (2, 2)}
        possible_moves = [(x, y, pt) for x, y, pt in np.argwhere(legal_moves)]

        # Priority 1: Find moves that would leave opponent with no valid moves
        forcing_moves = []
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

            # Check if opponent would have any legal moves
            opponent_legal_moves = test_state.get_legal_moves()
            if not np.any(opponent_legal_moves):
                forcing_moves.append((x, y, pt))

        if forcing_moves:
            x, y, pt = forcing_moves[np.random.randint(len(forcing_moves))]
            return Move(x, y, PieceType(pt))

        # Priority 2: Prevent opponent from placing in center next turn
        blocking_moves = []
        max_blocked_center = 0
        for x, y, pt in possible_moves:
            test_state = GameState()
            test_state.board = game_state.board.copy()
            test_state.current_player = game_state.current_player
            test_state.piece_counts = {
                Player.ONE: game_state.piece_counts[Player.ONE].copy(),
                Player.TWO: game_state.piece_counts[Player.TWO].copy(),
            }

            test_state.make_move(Move(x, y, PieceType(pt)))
            blocked_center_moves = 0

            # Count how many center moves would be blocked
            for cx, cy in central_squares:
                if not any(
                    test_state.is_valid_move(Move(cx, cy, PieceType(p)))
                    for p in range(4)
                ):
                    blocked_center_moves += 1

            if blocked_center_moves > max_blocked_center:
                max_blocked_center = blocked_center_moves
                blocking_moves = [(x, y, pt)]
            elif (
                blocked_center_moves == max_blocked_center and blocked_center_moves > 0
            ):
                blocking_moves.append((x, y, pt))

        if blocking_moves:
            x, y, pt = blocking_moves[np.random.randint(len(blocking_moves))]
            return Move(x, y, PieceType(pt))

        # Priority 3: Complete a four-line for points
        four_line_moves = []
        for x, y, pt in possible_moves:
            test_state = GameState()
            test_state.board = game_state.board.copy()
            test_state.current_player = game_state.current_player
            test_state.piece_counts = {
                Player.ONE: game_state.piece_counts[Player.ONE].copy(),
                Player.TWO: game_state.piece_counts[Player.TWO].copy(),
            }

            test_state.make_move(Move(x, y, PieceType(pt)))

            # Check if this move created any four-lines
            player_channel = game_state.current_player.value
            board_2d = np.sum(test_state.board[:, :, player_channel], axis=2)

            # Check rows, columns, and diagonals for four-in-a-row
            has_four = False
            # Rows and columns
            for i in range(4):
                if np.sum(board_2d[i, :]) == 4 or np.sum(board_2d[:, i]) == 4:
                    has_four = True
                    break
            # Diagonals
            if not has_four:
                if (
                    np.sum(np.diagonal(board_2d)) == 4
                    or np.sum(np.diagonal(np.fliplr(board_2d))) == 4
                ):
                    has_four = True

            if has_four:
                four_line_moves.append((x, y, pt))

        if four_line_moves:
            x, y, pt = four_line_moves[np.random.randint(len(four_line_moves))]
            return Move(x, y, PieceType(pt))

        # Priority 4: Play in center if possible
        center_moves = [
            (x, y, pt) for x, y, pt in possible_moves if (x, y) in central_squares
        ]
        if center_moves:
            x, y, pt = center_moves[np.random.randint(len(center_moves))]
            return Move(x, y, PieceType(pt))

        # Priority 5: Complete a three-line
        three_line_moves = []
        for x, y, pt in possible_moves:
            test_state = GameState()
            test_state.board = game_state.board.copy()
            test_state.current_player = game_state.current_player
            test_state.piece_counts = {
                Player.ONE: game_state.piece_counts[Player.ONE].copy(),
                Player.TWO: game_state.piece_counts[Player.TWO].copy(),
            }

            test_state.make_move(Move(x, y, PieceType(pt)))

            # Check if this move created any three-lines
            player_channel = game_state.current_player.value
            board_2d = np.sum(test_state.board[:, :, player_channel], axis=2)

            # Check rows, columns, and diagonals for three-in-a-row
            has_three = False
            # Rows and columns
            for i in range(4):
                if np.sum(board_2d[i, :]) == 3 or np.sum(board_2d[:, i]) == 3:
                    has_three = True
                    break
            # Diagonals
            if not has_three:
                if (
                    np.sum(np.diagonal(board_2d)) == 3
                    or np.sum(np.diagonal(np.fliplr(board_2d))) == 3
                ):
                    has_three = True

            if has_three:
                three_line_moves.append((x, y, pt))

        if three_line_moves:
            x, y, pt = three_line_moves[np.random.randint(len(three_line_moves))]
            return Move(x, y, PieceType(pt))

        # Priority 6: Random legal move
        x, y, pt = possible_moves[np.random.randint(len(possible_moves))]
        return Move(x, y, PieceType(pt))
