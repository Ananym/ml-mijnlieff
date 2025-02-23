import os
import numpy as np
import torch
from game import GameState, Move, Player, PieceType, print_full_legal_moves
import random
from model import ModelWrapper

# Global model instance
_model = None


def get_model():
    """Get or initialize the model singleton"""
    global _model
    if _model is None:
        print("Initializing model...")
        device = "cpu" if os.getenv("FORCE_CPU") else "cuda"
        _model = ModelWrapper(device)
        model_path = os.getenv("TICTACDO_MODEL_PATH", "saved_models/model_latest.pth")
        _model.load(model_path)
        print("Model loaded successfully")
    return _model


def convert_frontend_state_to_game_state(frontend_state):
    game_state = GameState()

    # Convert board state to 4x4x6
    frontend_board = np.array(frontend_state["board"])
    for i in range(4):
        for j in range(4):
            piece_one, piece_two = frontend_board[i, j]
            if piece_one > 0:
                game_state.board[i, j, 0] = 1
            if piece_two > 0:
                game_state.board[i, j, 1] = 1

    # Set current player
    game_state.current_player = (
        Player.ONE if frontend_state["currentPlayer"] == 0 else Player.TWO
    )

    # Convert piece counts
    game_state.piece_counts = {
        Player.ONE: {
            PieceType(int(k) - 1): v
            for k, v in frontend_state["pieceCounts"]["0"].items()
        },
        Player.TWO: {
            PieceType(int(k) - 1): v
            for k, v in frontend_state["pieceCounts"]["1"].items()
        },
    }

    # Set last move
    if frontend_state["lastMove"]:
        x = frontend_state["lastMove"]["x"]
        y = frontend_state["lastMove"]["y"]
        piece_type = PieceType(frontend_state["lastMove"]["pieceType"] - 1)
        game_state.last_move = Move(x=x, y=y, piece_type=PieceType(piece_type))
    else:
        game_state.last_move = None

    return game_state


def get_ai_move_logic(frontend_state):
    # Get difficulty from request, default to hardest (0)
    difficulty = frontend_state.get("difficulty", 0)

    game_state = convert_frontend_state_to_game_state(frontend_state)

    legal_moves = game_state.get_legal_moves()
    if np.all(legal_moves == 0):
        raise ValueError("No legal moves available.")

    print(f"Current player: {game_state.current_player}")
    print(f"Difficulty: {ModelWrapper.DIFFICULTY_SETTINGS[difficulty]['name']}")
    print_full_legal_moves(legal_moves)

    # Get move probabilities from policy network with difficulty
    state_rep = game_state.get_game_state_representation()
    model = get_model()  # Get the singleton model instance
    policy, _ = model.predict(
        state_rep.board, state_rep.flat_values, legal_moves, difficulty=difficulty
    )

    print(
        f"Initial policy type: {type(policy)}, shape: {policy.shape if hasattr(policy, 'shape') else 'no shape'}"
    )

    # Remove batch dimension and get best move
    policy = policy.squeeze(0)
    print(
        f"After squeeze type: {type(policy)}, shape: {policy.shape if hasattr(policy, 'shape') else 'no shape'}"
    )

    # Convert policy to numpy array if it's a tensor
    if isinstance(policy, torch.Tensor):
        policy = policy.detach().cpu().numpy()
    print(
        f"After numpy conversion type: {type(policy)}, shape: {policy.shape if hasattr(policy, 'shape') else 'no shape'}"
    )

    # Find the best move coordinates
    try:
        flat_index = int(policy.argmax())
        print(f"Argmax result: {flat_index}, type: {type(flat_index)}")
        move_coords = np.unravel_index(flat_index, policy.shape)
        print(f"Move coords: {move_coords}, type: {type(move_coords)}")
        move = Move(
            int(move_coords[0]), int(move_coords[1]), PieceType(int(move_coords[2]))
        )
    except Exception as e:
        print(f"Error during move coordinate processing: {str(e)}")
        print(f"Policy content: {policy}")
        raise

    relevant_piece_counts = game_state.piece_counts[game_state.current_player]
    if relevant_piece_counts[move.piece_type] == 0:
        raise ValueError("Invalid move: piece type has no remaining pieces.")

    print(f"AI picked move: {move}")

    # Apply the move directly to get the next state
    game_state.make_move(move)

    # Get opponent's perspective value from the new state
    next_state_rep = game_state.get_game_state_representation()
    next_legal_moves = game_state.get_legal_moves()
    _, opponent_value = model.predict(
        next_state_rep.board,
        next_state_rep.flat_values,
        next_legal_moves,
        difficulty=0,  # Use highest difficulty for value prediction
    )

    # Opponent's value is from their perspective, so we invert it for AI's perspective
    ai_win_probability = 1.0 - float(opponent_value.squeeze())

    return {
        "x": int(move.x),
        "y": int(move.y),
        "pieceType": int(move.piece_type.value) + 1,
        "aiWinProbability": ai_win_probability,
    }
