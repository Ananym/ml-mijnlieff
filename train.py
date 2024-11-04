import os
import random
import numpy as np
import torch
from game import (
    GameState,
    PieceType,
    Player,
    GameStateRepresentation,
    TurnResult,
    Move,
)
from typing import List, Tuple
from datetime import datetime, timedelta
import time
import signal
import sys
from dataclasses import dataclass
from mcts import MCTS

debugPrint = False


@dataclass
class TrainingExample:
    state_representation: GameStateRepresentation
    mcts_policy: np.ndarray  # 4x4x4 probabilities from MCTS
    value_target: float  # Final game outcome


def play_game_mcts(mcts_player: MCTS,
                   temperature_schedule: List[float]) -> List[TrainingExample]:
    """Play a complete game using MCTS, returning state/policy pairs for training."""
    game = GameState()
    training_examples = []
    move_count = 0
    game_start = time.time()

    if debugPrint:
        print(f"Starting new game...")

    while True:
        move_start = time.time()

        # Determine temperature based on move number
        temperature = temperature_schedule[min(move_count,
                                               len(temperature_schedule) - 1)]

        if debugPrint:
            print(f"\nMove {move_count + 1}")
            print(f"Current player: {game.current_player.name}")
            print(f"Temperature: {temperature}")

        # Get MCTS probabilities
        mcts_probs = mcts_player.get_action_probabilities(game, temperature)

        if debugPrint:
            print(
                f"MCTS move selection took {time.time() - move_start:.2f} seconds"
            )

        # Store the current state and MCTS probabilities
        training_examples.append(
            TrainingExample(
                state_representation=game.get_game_state_representation(),
                mcts_policy=mcts_probs,
                value_target=0.0  # Will be filled in later
            ))

        # Choose move based on MCTS probabilities
        if temperature == 0:
            move_coords = np.unravel_index(mcts_probs.argmax(),
                                           mcts_probs.shape)
        else:
            # Sample from the probability distribution
            move_coords = np.random.choice(64, p=mcts_probs.flatten())
            move_coords = np.unravel_index(move_coords, mcts_probs.shape)

        move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
        if debugPrint:
            print(
                f"Selected move: ({move.x}, {move.y}, {move.piece_type.name})")

        # Make the move
        result = game.make_move(move)
        move_count += 1

        if result == TurnResult.GAME_OVER:
            # Game is over, get the winner
            winner = game.get_winner()
            if debugPrint:
                print(f"\nGame over after {move_count} moves!")
                print(f"Winner: {winner.name if winner else 'Draw'}")
                print(
                    f"Total game time: {time.time() - game_start:.2f} seconds")

            # Set value targets based on game outcome
            for example in training_examples:
                if winner is None:
                    example.value_target = 0.5  # Draw
                else:
                    # 1 for P1 win, 0 for P2 win
                    example.value_target = 1.0 if winner == Player.ONE else 0.0

            return training_examples


def train_network(
        network_wrapper,
        training_minutes: int,
        save_interval_minutes: int = 15,
        batch_size: int = 128,
        num_simulations: int = 100,
        buffer_size: int = 10000,
        temperature_schedule: List[float] = None,
        save_callback=None,  # Add this parameter
):
    print("Starting training...")
    print(f"Training will run for {training_minutes} minutes")
    print(f"MCTS simulations per move: {num_simulations}")
    print(f"Batch size: {batch_size}")
    print(f"Buffer size: {buffer_size}")

    if temperature_schedule is None:
        temperature_schedule = [1.0] * 10 + [0.5] * 10 + [0.25] * 10 + [0.0]

    # Initialize MCTS
    mcts_player = MCTS(network_wrapper, num_simulations=num_simulations)

    # Initialize replay buffer
    replay_buffer: List[TrainingExample] = []

    # Training loop setup
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=training_minutes)
    last_save_time = start_time
    game_count = 0

    while datetime.now() < end_time:
        if debugPrint:
            print(f"\nStarting game {game_count + 1}")
        game_start = time.perf_counter()

        # Play a game and get training examples
        game_examples = play_game_mcts(mcts_player, temperature_schedule)

        game_duration = time.perf_counter() - game_start

        # Add new examples to replay buffer
        replay_buffer.extend(game_examples)

        # Keep buffer size in check
        if len(replay_buffer) > buffer_size:
            replay_buffer = replay_buffer[-buffer_size:]

        # Train on random minibatch from replay buffer
        if len(replay_buffer) >= batch_size:
            if debugPrint:
                print("Training on minibatch...")
            minibatch = random.sample(replay_buffer, batch_size)

            # Prepare training data
            grid_inputs = np.array(
                [ex.state_representation.board for ex in minibatch])
            flat_inputs = np.array(
                [ex.state_representation.flat_values for ex in minibatch])
            policy_targets = np.array([ex.mcts_policy for ex in minibatch])
            value_targets = np.array([ex.value_target for ex in minibatch])

            # Train network
            total_loss, policy_loss, value_loss = network_wrapper.train(
                grid_inputs, flat_inputs, policy_targets, value_targets)

            if debugPrint:
                print(
                    f"Game {game_count + 1} completed in {game_duration:.1f} seconds"
                )
                print(f"Collected {len(game_examples)} training examples")
            elif game_count % 5 == 0:
                print(
                    f"Game {game_count + 1} completed in {game_duration:.1f} seconds"
                )
                print(
                    f"Training losses - Total: {total_loss:.4f}, Policy: {policy_loss:.4f}, Value: {value_loss:.4f}"
                )

        game_count += 1

        # Save periodically
        current_time = datetime.now()
        if (current_time -
                last_save_time).total_seconds() / 60 >= save_interval_minutes:
            save_path = f"agent_{current_time.strftime('%Y%m%d_%H%M%S')}.pth"
            if save_callback:
                save_callback(network_wrapper, save_path)
            else:
                network_wrapper.save(save_path)
            print(
                f"\nModel saved at {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            last_save_time = current_time


def evaluate_model(
        network_wrapper,
        num_games: int = 100,
        num_simulations: int = 100,
        temperature: float = 0.0  # Use 0 for deterministic play in evaluation
):
    mcts_player = MCTS(network_wrapper, num_simulations=num_simulations)
    wins = {Player.ONE: 0, Player.TWO: 0, None: 0}

    for game_idx in range(num_games):
        game = GameState()
        while True:
            # Get move from MCTS
            move = mcts_player.get_best_move(game)

            # Make move
            result = game.make_move(move)

            if result == TurnResult.GAME_OVER:
                winner = game.get_winner()
                wins[winner] += 1
                break

        if game_idx % 10 == 0:
            print(f"Completed {game_idx + 1} evaluation games")

    print("\nEvaluation Results:")
    print(
        f"Player 1 wins: {wins[Player.ONE]} ({wins[Player.ONE]/num_games*100:.1f}%)"
    )
    print(
        f"Player 2 wins: {wins[Player.TWO]} ({wins[Player.TWO]/num_games*100:.1f}%)"
    )
    print(f"Draws: {wins[None]} ({wins[None]/num_games*100:.1f}%)")

    return wins


if __name__ == "__main__":
    import argparse
    from model import DualNetworkWrapper  # Import the new model

    parser = argparse.ArgumentParser(
        description="Train the dual network with MCTS.")
    parser.add_argument("--training_minutes",
                        type=int,
                        default=60,
                        help="Number of minutes to train")
    parser.add_argument("--save_interval_minutes",
                        type=int,
                        default=15,
                        help="Minutes between model saves")
    parser.add_argument("--num_simulations",
                        type=int,
                        default=100,
                        help="Number of MCTS simulations per move")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Training batch size")
    parser.add_argument("--buffer_size",
                        type=int,
                        default=10000,
                        help="Size of replay buffer")
    parser.add_argument("--load_model",
                        type=str,
                        default=None,
                        help="Path to load existing model")
    parser.add_argument("--evaluate",
                        action="store_true",
                        help="Run evaluation instead of training")

    args = parser.parse_args()

    # Initialize network
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network_wrapper = DualNetworkWrapper(device)

    if args.load_model and os.path.exists(args.load_model):
        network_wrapper.load(args.load_model)
        print(f"Loaded model from {args.load_model}")

    if args.evaluate:
        evaluate_model(network_wrapper, num_simulations=args.num_simulations)
    else:
        train_network(network_wrapper,
                      training_minutes=args.training_minutes,
                      save_interval_minutes=args.save_interval_minutes,
                      num_simulations=args.num_simulations,
                      batch_size=args.batch_size,
                      buffer_size=args.buffer_size)
