import os
import random
import numpy as np
from game import (
    GameState,
    Player,
    GameStateRepresentation,
    TurnResult,
    score_position,
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


def play_game(agent: RLAgent, p1_epsilon=0, p2_epsilon=0, verbose: bool = False):
    game = GameState()
    played_game_states: List[GameStateRepresentation] = []
    while True:

        current_player = game.current_player
        possible_next_states = game.get_possible_next_states()
        if len(possible_next_states) == 0:
            raise ValueError("No possible next states. Ending prematurely.")

        target_low = current_player == Player.ONE
        next_state = agent.select_move(
            possible_next_states,
            target_low,
            p1_epsilon if current_player == Player.ONE else p2_epsilon,
        )

        result = game.make_move(next_state.move)
        played_game_states.append(next_state.representation)

        if result == TurnResult.GAME_OVER:
            winner: Player | None = game.get_winner()
            return played_game_states, winner

        # if verbose:
        #     player = Player.ONE if game.current_player == Player.TWO else Player.TWO
        #     remaining_pieces = game.piece_counts[player]
        #     remaining_piece_sum = sum(remaining_pieces.values())
        #     print(
        #         f"Player {player.name} played {move.x},{move.y},{move.piece_type.name} with result {turn_result.name} and has {remaining_piece_sum} pieces left"
        #     )


def evaluate_agents(agent: RLAgent, num_games: int = 100):
    wins = {Player.ONE: 0, Player.TWO: 0, None: 0}

    for _ in range(num_games):
        _, winner = play_game(agent, 0, 0, False)
        wins[winner] += 1

    print(f"Evaluation results with zero epsilon after {num_games} games:")
    print(f"Agent 1 wins: {wins[Player.ONE]}")
    print(f"Agent 2 wins: {wins[Player.TWO]}")
    print(f"Draws: {wins[None]}")

    wins = {Player.ONE: 0, Player.TWO: 0, None: 0}
    for _ in range(num_games):
        _, winner = play_game(agent, 0.2, 0.2, False)
        wins[winner] += 1

    print(f"Evaluation results with small epsilon after {num_games} games:")
    print(f"Agent 1 wins: {wins[Player.ONE]}")
    print(f"Agent 2 wins: {wins[Player.TWO]}")
    print(f"Draws: {wins[None]}")

    wins = {Player.ONE: 0, Player.TWO: 0, None: 0}
    for _ in range(num_games):
        _, winner = play_game(agent, 0, 0.75, False)
        wins[winner] += 1

    print(f"Evaluation results with p2 0.75 epsilon after {num_games} games:")
    print(f"Agent 1 wins: {wins[Player.ONE]}")
    print(f"Agent 2 wins: {wins[Player.TWO]}")
    print(f"Draws: {wins[None]}")
    wins = {Player.ONE: 0, Player.TWO: 0, None: 0}
    for _ in range(num_games):
        _, winner = play_game(agent, 0.75, 0, False)
        wins[winner] += 1

    print(f"Evaluation results with p1 0.75 epsilon after {num_games} games:")
    print(f"Agent 1 wins: {wins[Player.ONE]}")
    print(f"Agent 2 wins: {wins[Player.TWO]}")
    print(f"Draws: {wins[None]}")


def get_training_epsilons():
    """
    Returns exploration rates for both players that help develop robust strategies.

    The system alternates between several training modes:
    - Exploration: High randomness to discover new strategies
    - Exploitation: Low randomness to refine known strategies
    - Asymmetric: One player explores while other exploits
    - Dynamic: Randomness based on position evaluation
    """
    # Base epsilon ranges
    low_epsilon = 0.1  # Minimum exploration
    high_epsilon = 0.7  # Maximum exploration

    # Randomly select a training mode
    mode = random.random()

    if mode < 0.3:
        # Exploration mode: Both players use high randomness
        # Good for discovering new strategies
        p1_epsilon = random.uniform(0.4, high_epsilon)
        p2_epsilon = random.uniform(0.4, high_epsilon)

    elif mode < 0.6:
        # Exploitation mode: Both players use low randomness
        # Good for refining known strategies
        p1_epsilon = random.uniform(low_epsilon, 0.3)
        p2_epsilon = random.uniform(low_epsilon, 0.3)

    elif mode < 0.8:
        # Asymmetric mode: One player explores while other exploits
        # Good for finding counter-strategies
        if random.random() < 0.5:
            p1_epsilon = random.uniform(0.4, high_epsilon)
            p2_epsilon = random.uniform(low_epsilon, 0.3)
        else:
            p1_epsilon = random.uniform(low_epsilon, 0.3)
            p2_epsilon = random.uniform(0.4, high_epsilon)

    else:
        # Random mode: Completely random epsilons
        # Adds variety to training
        p1_epsilon = random.uniform(low_epsilon, high_epsilon)
        p2_epsilon = random.uniform(low_epsilon, high_epsilon)

    return p1_epsilon, p2_epsilon


def get_state_outcome(game_states, winner, index):
    # Base outcome from winner (0 for P1 win, 1 for P2 win, 0.5 for draw)
    base = 0.5 if winner is None else (0 if winner == Player.ONE else 1)

    # Calculate score component
    state = game_states[index]
    p1_score = score_position(state.board[:, :, 0])
    p2_score = score_position(state.board[:, :, 1])

    # Normalize score difference to [-0.5, 0.5] range
    # Positive means P2 is winning, negative means P1 is winning
    score_component = (p2_score - p1_score) / (
        20
    )  # Divide by larger number to reduce impact

    # Time weight should pull uncertain positions toward final outcome
    time_weight = index / len(game_states)
    uncertainty = 1 - time_weight

    # Early in game: more weight on current board evaluation
    # Late in game: more weight on final outcome
    return base * time_weight + (0.5 + score_component) * uncertainty


def train_agents(
    training_minutes: int,
    save_interval_minutes: int = 15,
    load_agent_path: str = None,
):
    global exit_flag

    if load_agent_path and os.path.exists(load_agent_path):
        agent = RLAgent()
        agent.load(load_agent_path)
        print(f"Loaded Agent from {load_agent_path}")
    else:
        agent = RLAgent()

    game_index = 0
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=training_minutes)
    last_save_time = start_time

    while datetime.now() < end_time and not exit_flag:
        game_start = time.perf_counter()

        p1_epsilon, p2_epsilon = get_training_epsilons()

        game_states, winner = play_game(agent, p1_epsilon, p2_epsilon)

        # total_state_count = len(game_states)
        # if winner == Player.ONE:
        #     outcomes = [0] * total_state_count
        # elif winner == Player.TWO:
        #     outcomes = [1] * total_state_count
        # else:
        #     outcomes = [0.5] * total_state_count
        total_state_count = len(game_states)
        outcomes = [
            get_state_outcome(game_states, winner, index)
            for index in range(total_state_count)
        ]

        zipped_labelled_states = list(zip(game_states, outcomes))
        random.shuffle(zipped_labelled_states)
        shuffled_states = []
        shuffled_outcomes = []
        for state, outcome in zipped_labelled_states:
            shuffled_states.append(state)
            shuffled_outcomes.append(outcome)

        loss2 = agent.train(shuffled_states, shuffled_outcomes)

        game_end = time.perf_counter()
        elapsed_time_ms = round((game_end - game_start) * 1000)
        if winner is None:
            winner_statement = "ended in a draw."
        else:
            winner_statement = f"player {winner.name} won."

        if game_index % 100 == 0:
            print(
                f"Game {game_index + 1} took {elapsed_time_ms}ms, {total_state_count} moves, and {winner_statement}"
            )

        current_time = datetime.now()
        if (
            current_time - last_save_time
        ).total_seconds() / 60 >= save_interval_minutes:
            save_path = f"agent_{current_time.strftime('%Y%m%d_%H%M%S')}.pth"
            agent.save(save_path)
            print(f"Saved model at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            last_save_time = current_time

        game_index += 1

    return agent


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
        "--load_agent",
        type=str,
        default="final_agent.pth",
        help="Path to load Agent model",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate the trained agents"
    )

    args = parser.parse_args()

    if args.evaluate:
        if args.load_agent and args.load_agent != "None":
            agent = RLAgent()
            agent.load(args.load_agent)
            print(f"Loaded Agent from {args.load_agent}")
        else:
            agent = RLAgent()

        evaluate_agents(agent)
        sys.exit(0)

    print("Starting training...")
    print("Press Ctrl-C to finish training after the current game.")
    trained_agent = train_agents(
        training_minutes=args.training_minutes,
        save_interval_minutes=args.save_interval_minutes,
        load_agent_path=args.load_agent,
    )

    if not exit_flag:
        print("\nTraining completed. Evaluating agents...")
        evaluate_agents(trained_agent)

    print("\nSaving final models...")
    final_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    trained_agent.save(f"final_agent.pth")
    print("Training and evaluation completed.")

    if exit_flag:
        print("Training interrupted by user. Final models saved.")
        sys.exit(0)
