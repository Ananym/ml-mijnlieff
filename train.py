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
from model import ModelWrapper
from tqdm import tqdm, trange
from opponents import RandomOpponent, StrategicOpponent

debugPrint = False


@dataclass
class TrainingExample:
    state_rep: GameState
    policy: np.ndarray  # One-hot encoded actual move made
    value: float  # Game outcome


def self_play_game(
    model: ModelWrapper, temperature: float = 1.0, debug: bool = False
) -> List[TrainingExample]:
    """Play a game using the current policy, return state/action pairs"""
    game = GameState()
    examples = []
    move_count = 0

    while True:
        # Get legal moves
        legal_moves = game.get_legal_moves()
        if not np.any(legal_moves):
            # No legal moves, must pass
            game.pass_turn()
            continue

        # Get move probabilities from policy network
        state_rep = game.get_game_state_representation()
        policy, _ = model.predict(state_rep.board, state_rep.flat_values, legal_moves)

        # Remove batch dimension from policy
        policy = policy.squeeze(0)  # Now shape is (4, 4, 4)

        # Debug prints
        if debug:
            print(f"\nMove {move_count + 1}, Player {game.current_player.name}")
            print(f"Legal moves sum: {legal_moves.sum()}")
            print(f"Initial policy sum: {policy.sum()}")
            print(f"Policy shape: {policy.shape}")
            print(f"Legal moves shape: {legal_moves.shape}")

        # Ensure we only consider legal moves
        masked_policy = policy * legal_moves  # Mask out illegal moves
        masked_policy = masked_policy / (masked_policy.sum() + 1e-8)  # Renormalize

        if debug:
            print(f"Masked policy sum: {masked_policy.sum()}")

        # Verify the move is legal before choosing
        if not np.any(masked_policy):
            if debug:
                print("Warning: No valid moves in masked policy!")
            # Fallback to random legal move
            legal_positions = np.argwhere(legal_moves)
            idx = np.random.randint(len(legal_positions))
            move_coords = tuple(legal_positions[idx])
            if debug:
                print(f"Falling back to random legal move: {move_coords}")
        else:
            # Choose move (either greedily or with temperature)
            if (
                temperature == 0 or move_count >= 12
            ):  # Play deterministically in endgame
                move_coords = tuple(
                    np.unravel_index(masked_policy.argmax(), masked_policy.shape)
                )
            else:
                # Sample from the probability distribution
                policy_flat = masked_policy.flatten()
                move_idx = np.random.choice(len(policy_flat), p=policy_flat)
                move_coords = tuple(np.unravel_index(move_idx, masked_policy.shape))

        # Double check the move is legal
        x, y, piece_type = move_coords
        if not legal_moves[x, y, piece_type]:
            print(f"ERROR: Selected illegal move {move_coords}!")
            print("Legal moves shape:", legal_moves.shape)
            print("Policy shape:", policy.shape)
            print(
                "Selected move is legal according to mask:",
                legal_moves[x, y, piece_type],
            )
            raise ValueError(f"Attempted to make illegal move {move_coords}")

        move = Move(x, y, PieceType(piece_type))

        # Create one-hot encoded policy target
        policy_target = np.zeros_like(masked_policy)
        policy_target[x, y, piece_type] = 1.0

        # Store the example
        examples.append(
            TrainingExample(
                state_rep=state_rep,
                policy=policy_target,
                value=0.0,  # Will be filled in later
            )
        )

        # Make move
        result = game.make_move(move)
        move_count += 1

        if result == TurnResult.GAME_OVER:
            # Game is over, get the winner
            winner = game.get_winner()

            # Set value targets based on game outcome
            for example in examples:
                if winner is None:
                    example.value = 0.5  # Draw
                else:
                    # 1 for P1 win, 0 for P2 win from P1's perspective
                    example.value = 1.0 if winner == Player.ONE else 0.0

            return examples


def train_network(
    model: ModelWrapper,
    num_episodes: int = 100,
    batch_size: int = 256,  # Increased from 128
    save_interval: int = 10,
    num_checkpoints: int = 3,
    min_temp: float = 0.5,
    strategic_opponent_ratio: float = 0.3,  # 30% of games against strategic opponent
    debug: bool = False,
):
    """Main training loop that runs until interrupted"""
    replay_buffer = []
    running_loss = {"total": 0, "policy": 0, "value": 0}
    running_count = 0
    iteration = 0
    training_start = time.time()
    strategic_opponent = StrategicOpponent()

    # Keep track of last N checkpoints
    checkpoint_files = []

    # Setup interrupt handling
    interrupt_received = False

    def handle_interrupt(signum, frame):
        nonlocal interrupt_received
        if interrupt_received:  # Second interrupt, exit immediately
            print("\nForced exit...")
            sys.exit(1)
        print("\nInterrupt received, finishing current iteration...")
        interrupt_received = True

    signal.signal(signal.SIGINT, handle_interrupt)

    print(f"Starting infinite training loop")
    print(f"Episodes per iteration: {num_episodes}")
    print(f"Temperature range: {min_temp:.1f} - 2.0")
    print(f"Strategic opponent ratio: {strategic_opponent_ratio:.1%}")
    print(f"Batch size: {batch_size}")
    print(f"Press Ctrl+C to stop training gracefully")

    try:
        while not interrupt_received:
            iteration += 1
            iteration_start = time.time()

            # Self-play phase
            episode_pbar = tqdm(
                range(num_episodes),
                desc=f"Self-Play Games (Iter {iteration})",
                leave=False,
            )

            # Determine number of strategic opponent games
            strategic_games = int(num_episodes * strategic_opponent_ratio)
            selfplay_games = num_episodes - strategic_games

            # Self-play games
            for episode in range(selfplay_games):
                base_temp = max(min_temp, 2.0 - (1.5 * episode / num_episodes))
                temperature = base_temp * (0.8 + 0.4 * random.random())

                game_examples = self_play_game(model, temperature, debug)
                replay_buffer.extend(game_examples)
                episode_pbar.update(1)
                episode_pbar.set_postfix(
                    {
                        "buffer_size": len(replay_buffer),
                        "temp": f"{temperature:.2f}",
                        "type": "self-play",
                    }
                )

            # Strategic opponent games
            for episode in range(strategic_games):
                game = GameState()
                examples = []
                move_count = 0

                # Randomly decide if model plays as Player 1 or 2
                model_is_player_one = random.random() < 0.5

                while True:
                    legal_moves = game.get_legal_moves()
                    if not np.any(legal_moves):
                        game.pass_turn()
                        continue

                    is_model_turn = (
                        game.current_player == Player.ONE
                    ) == model_is_player_one

                    if is_model_turn:
                        # Model's turn
                        state_rep = game.get_game_state_representation()
                        policy, _ = model.predict(
                            state_rep.board, state_rep.flat_values, legal_moves
                        )

                        policy = policy.squeeze(0)
                        masked_policy = policy * legal_moves
                        masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

                        # Use temperature for exploration
                        temperature = max(min_temp, 1.0 - move_count / 12)
                        if temperature > 0:
                            policy_flat = masked_policy.flatten()
                            move_idx = np.random.choice(len(policy_flat), p=policy_flat)
                            move_coords = np.unravel_index(
                                move_idx, masked_policy.shape
                            )
                        else:
                            move_coords = np.unravel_index(
                                masked_policy.argmax(), masked_policy.shape
                            )

                        move = Move(
                            move_coords[0], move_coords[1], PieceType(move_coords[2])
                        )

                        # Store training example
                        policy_target = np.zeros_like(masked_policy)
                        policy_target[move_coords] = 1.0
                        examples.append(
                            TrainingExample(
                                state_rep=state_rep, policy=policy_target, value=0.0
                            )
                        )
                    else:
                        # Strategic opponent's turn
                        move = strategic_opponent.get_move(game)
                        if move is None:
                            game.pass_turn()
                            continue

                    result = game.make_move(move)
                    move_count += 1

                    if result == TurnResult.GAME_OVER:
                        winner = game.get_winner()
                        # Set value targets based on game outcome
                        for example in examples:
                            if winner is None:
                                example.value = 0.5
                            else:
                                example.value = (
                                    1.0
                                    if (winner == Player.ONE) == model_is_player_one
                                    else 0.0
                                )
                        break

                replay_buffer.extend(examples)
                episode_pbar.update(1)
                episode_pbar.set_postfix(
                    {"buffer_size": len(replay_buffer), "type": "strategic"}
                )

                # Keep buffer size in check
                if len(replay_buffer) > 10000:
                    replay_buffer = replay_buffer[-10000:]

            # Training phase
            train_pbar = tqdm(
                range(20), desc="Training Epochs", leave=False
            )  # Increased from 10
            epoch_losses = {"total": 0, "policy": 0, "value": 0}

            for _ in train_pbar:
                # Sample batch
                batch = random.sample(
                    replay_buffer, min(batch_size, len(replay_buffer))
                )

                # Prepare training data
                board_inputs = np.array([ex.state_rep.board for ex in batch])
                flat_inputs = np.array([ex.state_rep.flat_values for ex in batch])
                policy_targets = np.array([ex.policy for ex in batch])
                value_targets = np.array([ex.value for ex in batch])

                # Train network
                total_loss, policy_loss, value_loss = model.train_step(
                    board_inputs, flat_inputs, policy_targets, value_targets
                )

                # Update running averages
                running_count += 1
                running_loss["total"] = (
                    running_loss["total"] * (running_count - 1) + total_loss
                ) / running_count
                running_loss["policy"] = (
                    running_loss["policy"] * (running_count - 1) + policy_loss
                ) / running_count
                running_loss["value"] = (
                    running_loss["value"] * (running_count - 1) + value_loss
                ) / running_count

                # Update epoch losses
                epoch_losses["total"] += total_loss
                epoch_losses["policy"] += policy_loss
                epoch_losses["value"] += value_loss

                train_pbar.set_postfix(
                    {
                        "running_total": f"{running_loss['total']:.4f}",
                        "running_policy": f"{running_loss['policy']:.4f}",
                        "running_value": f"{running_loss['value']:.4f}",
                    }
                )

            # Print epoch summary
            avg_total = epoch_losses["total"] / 20
            avg_policy = epoch_losses["policy"] / 20
            avg_value = epoch_losses["value"] / 20
            elapsed_time = time.time() - training_start
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60

            tqdm.write(
                f"\nIteration {iteration} - Time: {int(hours)}h {int(minutes)}m"
                f"\nAvg Losses: Total: {avg_total:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}"
            )

            # Save checkpoint with rotation
            if iteration % save_interval == 0 or interrupt_received:
                save_path = f"model_iter_{iteration}.pth"
                model.save(save_path)
                checkpoint_files.append(save_path)

                # Remove old checkpoints if we have too many
                while len(checkpoint_files) > num_checkpoints:
                    old_checkpoint = checkpoint_files.pop(0)
                    try:
                        os.remove(old_checkpoint)
                        tqdm.write(f"Removed old checkpoint: {old_checkpoint}")
                    except OSError:
                        pass

                tqdm.write(f"Saved checkpoint: {save_path}")

            tqdm.write(
                f"Iteration completed in {time.time() - iteration_start:.1f} seconds\n"
            )

    finally:
        # Training summary
        elapsed_time = time.time() - training_start
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60

        print("\nTraining Summary:")
        print(f"Total training time: {int(hours)}h {int(minutes)}m")
        print(f"Iterations completed: {iteration}")
        print(f"Final running losses:")
        print(f"  Total: {running_loss['total']:.4f}")
        print(f"  Policy: {running_loss['policy']:.4f}")
        print(f"  Value: {running_loss['value']:.4f}")
        print(f"\nCheckpoints saved: {', '.join(checkpoint_files)}")


def evaluate_model(model: ModelWrapper, num_games: int = 100):
    """Evaluate model by playing against different opponents"""
    opponents = {"random": RandomOpponent(), "strategic": StrategicOpponent()}

    results = {
        "as_p1": {opp: {"wins": 0, "losses": 0, "draws": 0} for opp in opponents},
        "as_p2": {opp: {"wins": 0, "losses": 0, "draws": 0} for opp in opponents},
    }

    # Test model as both Player 1 and Player 2 against each opponent
    for opponent_name, opponent in opponents.items():
        # Model as Player 1
        for game_idx in tqdm(range(num_games), desc=f"Model (P1) vs {opponent_name}"):
            game = GameState()
            while True:
                if game.current_player == Player.ONE:
                    # Model's turn
                    legal_moves = game.get_legal_moves()
                    if not np.any(legal_moves):
                        game.pass_turn()
                        continue

                    state_rep = game.get_game_state_representation()
                    policy, _ = model.predict(
                        state_rep.board, state_rep.flat_values, legal_moves
                    )

                    # Apply legal moves mask
                    policy = policy.squeeze(0)
                    masked_policy = policy * legal_moves
                    masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

                    # Choose best legal move
                    move_coords = np.unravel_index(
                        masked_policy.argmax(), masked_policy.shape
                    )
                    move = Move(
                        move_coords[0], move_coords[1], PieceType(move_coords[2])
                    )
                else:
                    # Opponent's turn
                    move = opponent.get_move(game)
                    if move is None:
                        game.pass_turn()
                        continue

                result = game.make_move(move)
                if result == TurnResult.GAME_OVER:
                    winner = game.get_winner()
                    if winner == Player.ONE:
                        results["as_p1"][opponent_name]["wins"] += 1
                    elif winner == Player.TWO:
                        results["as_p1"][opponent_name]["losses"] += 1
                    else:
                        results["as_p1"][opponent_name]["draws"] += 1
                    break

        # Model as Player 2
        for game_idx in tqdm(range(num_games), desc=f"Model (P2) vs {opponent_name}"):
            game = GameState()
            while True:
                if game.current_player == Player.TWO:
                    # Model's turn
                    legal_moves = game.get_legal_moves()
                    if not np.any(legal_moves):
                        game.pass_turn()
                        continue

                    state_rep = game.get_game_state_representation()
                    policy, _ = model.predict(
                        state_rep.board, state_rep.flat_values, legal_moves
                    )

                    # Apply legal moves mask
                    policy = policy.squeeze(0)
                    masked_policy = policy * legal_moves
                    masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

                    # Choose best legal move
                    move_coords = np.unravel_index(
                        masked_policy.argmax(), masked_policy.shape
                    )
                    move = Move(
                        move_coords[0], move_coords[1], PieceType(move_coords[2])
                    )
                else:
                    # Opponent's turn
                    move = opponent.get_move(game)
                    if move is None:
                        game.pass_turn()
                        continue

                result = game.make_move(move)
                if result == TurnResult.GAME_OVER:
                    winner = game.get_winner()
                    if winner == Player.TWO:
                        results["as_p2"][opponent_name]["wins"] += 1
                    elif winner == Player.ONE:
                        results["as_p2"][opponent_name]["losses"] += 1
                    else:
                        results["as_p2"][opponent_name]["draws"] += 1
                    break

    # Print results
    print("\nEvaluation Results:")
    for role in ["as_p1", "as_p2"]:
        print(f"\nModel playing as {'Player 1' if role == 'as_p1' else 'Player 2'}:")
        for opp_name in opponents:
            r = results[role][opp_name]
            total = r["wins"] + r["losses"] + r["draws"]
            print(f"\nVs {opp_name} opponent:")
            print(f"Wins: {r['wins']} ({r['wins']/total*100:.1f}%)")
            print(f"Losses: {r['losses']} ({r['losses']/total*100:.1f}%)")
            print(f"Draws: {r['draws']} ({r['draws']/total*100:.1f}%)")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and evaluate the model")
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Episodes per iteration"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Training batch size"
    )
    parser.add_argument(
        "--save_interval", type=int, default=10, help="Save every N iterations"
    )
    parser.add_argument(
        "--num_checkpoints", type=int, default=3, help="Number of checkpoints to keep"
    )
    parser.add_argument(
        "--load_model", type=str, default=None, help="Path to model to load"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelWrapper(device)

    if args.load_model and os.path.exists(args.load_model):
        model.load(args.load_model)
        print(f"Loaded model from {args.load_model}")

    if args.mode == "train":
        train_network(
            model,
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            num_checkpoints=args.num_checkpoints,
            debug=args.debug,
        )
    else:
        evaluate_model(model)
