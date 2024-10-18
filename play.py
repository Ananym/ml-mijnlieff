from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import numpy as np
from game import GameState, Move, Player, PieceType, print_full_legal_moves
from agent import RLAgent
import random

app = Flask(__name__)
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": ["http://localhost:5173", "http://127.0.0.1:5173"]
        }
    },
)

# Initialize the AI agents
first_player_agent = RLAgent()
second_player_agent = RLAgent()

# Load the pre-trained models
first_player_agent.load("final_agent1.pth")
second_player_agent.load("final_agent2.pth")


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
    game_state.current_player = Player.ONE if frontend_state[
        "currentPlayer"] == 0 else Player.TWO

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


@app.route("/api/get_ai_move", methods=["POST", "OPTIONS"])
def get_ai_move():
    if request.method == "OPTIONS":
        # Respond to preflight request
        return "", 204

    frontend_state = request.json
    game_state = convert_frontend_state_to_game_state(frontend_state)

    legal_moves_grid = game_state.get_legal_moves()
    if np.all(legal_moves_grid == 0):
        abort(400, "No legal moves available.")

    # Select the appropriate agent based on the current player
    agent = first_player_agent if game_state.current_player == Player.ONE else second_player_agent

    # Get the game state representation
    game_state_representation = game_state.get_game_state_representation()

    print(f"Current player: {game_state.current_player}")
    print_full_legal_moves(legal_moves_grid)

    move = agent.select_move(legal_moves_grid, game_state_representation)

    if move:

        relevent_piece_counts = game_state.piece_counts[
            game_state.current_player]
        if relevent_piece_counts[move.piece_type] == 0:
            abort(500, "Invalid move: piece type has no remaining pieces.")

        print(f"AI picked move: {move}")

        json_move = jsonify({
            "x": int(move.x),
            "y": int(move.y),
            "pieceType": int(move.piece_type.value) + 1
        })

        return json_move
    else:
        print("Model didn't pick a move - using random move")
        legal_moves = np.array(np.where(legal_moves_grid == 1)).T
        random_move = random.choice(legal_moves)
        return jsonify({
            "x": int(random_move[0]),
            "y": int(random_move[1]),
            "pieceType": int(random_move[2] + 1)
        })


if __name__ == "__main__":
    app.run(debug=True)
