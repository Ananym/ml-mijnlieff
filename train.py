import numpy as np
from game import (
    GameState,
    Player,
    GameStateRepresentation,
    TurnResult,
    Move,
    print_legal_moves,
    count_legal_move_positions,
)
from agent import RLAgent
from typing import List
from datetime import datetime, timedelta
import time
import signal
import sys

exit_flag = False


def signal_handler(sig, frame):
    global exit_flag
    print("\nCtrl-C received. Will exit after current game finishes...")
    exit_flag = True


signal.signal(signal.SIGINT, signal_handler)


def play_game(agent1: RLAgent, agent2: RLAgent, verbose: bool = False):
    game = GameState()
    game_states1: List[GameStateRepresentation] = []
    game_states2: List[GameStateRepresentation] = []
    moves1: List[Move] = []
    moves2: List[Move] = []
    move_count = 0
    while True:
        if move_count > 30:
            raise ValueError(
                "Game exceeded expected maximum moves. Ending prematurely."
            )

        game_state_representation = game.get_game_state_representation()
        legal_moves_grid = game.get_legal_moves()

        if np.all(legal_moves_grid == 0):
            raise ValueError("No legal moves available. Ending prematurely.")

        if game.current_player == Player.ONE:
            agent = agent1
            game_states1.append(game_state_representation)
        else:
            agent = agent2
            game_states2.append(game_state_representation)

        move: Move = agent.select_move(legal_moves_grid, game_state_representation)
        turn_result = game.make_move(move)

        # Store the move
        if game.current_player == Player.TWO:  # Player just switched after the move
            moves1.append(move)
        else:
            moves2.append(move)

        move_count += 1

        if verbose:
            player = Player.ONE if game.current_player == Player.TWO else Player.TWO
            remaining_pieces = game.piece_counts[player]
            remaining_piece_sum = sum(remaining_pieces.values())
            print(
                f"Player {player.name} played {move.x},{move.y},{move.piece_type.name} with result {turn_result.name} and has {remaining_piece_sum} pieces left"
            )

        if turn_result == TurnResult.NORMAL:
            continue
        elif turn_result == TurnResult.GAME_OVER:
            winner: Player | None = game.get_winner()
            return (game_states1, moves1), (game_states2, moves2), winner
        elif turn_result == TurnResult.OPPONENT_MUST_PASS:
            game.pass_turn()
            continue


def evaluate_agents(agent1: RLAgent, agent2: RLAgent, num_games: int = 1000):
    wins = {Player.ONE: 0, Player.TWO: 0, None: 0}

    for _ in range(num_games):
        _, _, winner = play_game(agent1, agent2)
        wins[winner] += 1

    print(f"Evaluation results after {num_games} games:")
    print(f"Agent 1 wins: {wins[Player.ONE]}")
    print(f"Agent 2 wins: {wins[Player.TWO]}")
    print(f"Draws: {wins[None]}")


def train_agents(
    training_minutes: int,
    save_interval_minutes: int = 15,
    load_agent1_path: str = None,
    load_agent2_path: str = None,
):
    global exit_flag

    if load_agent1_path:
        agent1 = RLAgent()
        agent1.load(load_agent1_path)
        print(f"Loaded Agent 1 from {load_agent1_path}")
    else:
        agent1 = RLAgent()

    if load_agent2_path:
        agent2 = RLAgent()
        agent2.load(load_agent2_path)
        print(f"Loaded Agent 2 from {load_agent2_path}")
    else:
        agent2 = RLAgent()

    game_index = 0
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=training_minutes)
    last_save_time = start_time

    while datetime.now() < end_time and not exit_flag:
        game_start = time.perf_counter()
        (game_states1, moves1), (game_states2, moves2), winner = play_game(
            agent1, agent2, verbose=False
        )

        if not len(moves1) and not len(moves2):
            print(f"Warning: Game {game_index + 1} ended without any moves.")
            continue

        if winner == Player.ONE:
            outcomes1 = [1] * len(game_states1)
            outcomes2 = [-1] * len(game_states2)
        elif winner == Player.TWO:
            outcomes1 = [-1] * len(game_states1)
            outcomes2 = [1] * len(game_states2)
        else:
            outcomes1 = [0] * len(game_states1)
            outcomes2 = [0] * len(game_states2)

        loss2 = agent2.train(game_states2, moves2, outcomes2)
        loss1 = agent1.train(game_states1, moves1, outcomes1)

        game_end = time.perf_counter()
        elapsed_time_ms = round((game_end - game_start) * 1000)
        if winner is None:
            winner_statement = "ended in a draw."
        else:
            winner_statement = f"player {winner.name} won."

        if game_index % 100 == 0:
            print(
                f"Game {game_index + 1} took {elapsed_time_ms}ms, {len(moves1)+len(moves2)} moves, and {winner_statement}"
            )

        current_time = datetime.now()
        if (
            current_time - last_save_time
        ).total_seconds() / 60 >= save_interval_minutes:
            save_path1 = f"agent1_{current_time.strftime('%Y%m%d_%H%M%S')}.pth"
            save_path2 = f"agent2_{current_time.strftime('%Y%m%d_%H%M%S')}.pth"
            agent1.save(save_path1)
            agent2.save(save_path2)
            print(f"Saved models at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            last_save_time = current_time

        game_index += 1

    return agent1, agent2


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train agents for the game.")
    parser.add_argument(
        "--training_minutes",
        type=int,
        default=60 * 24,
        help="Number of minutes to train",
    )
    parser.add_argument(
        "--save_interval_minutes",
        type=int,
        default=120,
        help="Number of minutes between saves",
    )
    parser.add_argument(
        "--load_agent1",
        type=str,
        default="final_agent1.pth",
        help="Path to load Agent 1 model",
    )
    parser.add_argument(
        "--load_agent2",
        type=str,
        default="final_agent2.pth",
        help="Path to load Agent 2 model",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the trained agents"
    )

    args = parser.parse_args()

    if args.evaluate:
        if args.load_agent1:
            agent1 = RLAgent()
            agent1.load(args.load_agent1)
            print(f"Loaded Agent 1 from {args.load_agent1}")
        else:
            agent1 = RLAgent()

        if args.load_agent2:
            agent2 = RLAgent()
            agent2.load(args.load_agent2)
            print(f"Loaded Agent 2 from {args.load_agent2}")
        else:
            agent2 = RLAgent()
        evaluate_agents(agent1, agent2)
        sys.exit(0)

    print("Starting training...")
    print("Press Ctrl-C to finish training after the current game.")
    trained_agent1, trained_agent2 = train_agents(
        training_minutes=args.training_minutes,
        save_interval_minutes=args.save_interval_minutes,
        load_agent1_path=args.load_agent1,
        load_agent2_path=args.load_agent2,
    )

    if not exit_flag:
        print("\nTraining completed. Evaluating agents...")
        evaluate_agents(trained_agent1, trained_agent2)

    print("\nSaving final models...")
    final_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    trained_agent1.save(f"final_agent1.pth")
    trained_agent2.save(f"final_agent2.pth")
    print("Training and evaluation completed.")

    if exit_flag:
        print("Training interrupted by user. Final models saved.")
        sys.exit(0)
