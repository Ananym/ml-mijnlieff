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
from typing import List, Tuple, Optional
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
    batch_size: int = 256,
    save_interval: int = 10,
    num_checkpoints: int = 3,
    min_temp: float = 0.5,
    strategic_opponent_ratio: float = 0.6,  # 50% strategic games
    random_opponent_ratio: float = 0.1,  # 10% random opponents
    buffer_size: int = 10000,
    policy_weight: float = 1.5,  # Increased from 1.0 to encourage decisive play
    num_epochs: int = 40,
    debug: bool = False,
):
    """Main training loop that runs until interrupted"""
    replay_buffer = []
    running_loss = {"total": 0, "policy": 0, "value": 0}
    running_count = 0
    iteration = 0
    training_start = time.time()
    strategic_opponent = StrategicOpponent()
    random_opponent = RandomOpponent()
    latest_model_path = "saved_models/model_latest.pth"

    # Base stats template
    stats_template = {
        "strategic_wins_as_p1": 0,
        "strategic_wins_as_p2": 0,
        "strategic_games_as_p1": 0,
        "strategic_games_as_p2": 0,
        "self_play_p1_wins": 0,
        "self_play_games": 0,
        "total_moves": 0,
        "total_games": 0,
        "central_moves": 0,
        "policy_confidence": 0,
        "moves_counted": 0,
    }

    # Stability metrics
    window_size = 50  # Window for moving averages
    stability_stats = {
        "policy_loss_window": [],
        "value_loss_window": [],
        "policy_confidence_window": [],
        "last_policy_loss": 0.0,
        "last_value_loss": 0.0,
        "total_variance_policy": 0.0,
        "total_variance_value": 0.0,
        "num_variance_samples": 0,
    }

    def update_stability_metrics(
        policy_loss: float, value_loss: float, policy_confidence: float
    ):
        """Update running stability statistics"""
        # Update loss windows
        stability_stats["policy_loss_window"].append(policy_loss)
        stability_stats["value_loss_window"].append(value_loss)
        if len(stability_stats["policy_loss_window"]) > window_size:
            stability_stats["policy_loss_window"].pop(0)
            stability_stats["value_loss_window"].pop(0)

        # Update policy confidence window
        stability_stats["policy_confidence_window"].append(policy_confidence)
        if len(stability_stats["policy_confidence_window"]) > window_size:
            stability_stats["policy_confidence_window"].pop(0)

        # Calculate loss variance
        if stability_stats["last_policy_loss"] != 0:  # Skip first sample
            policy_diff = policy_loss - stability_stats["last_policy_loss"]
            value_diff = value_loss - stability_stats["last_value_loss"]
            stability_stats["total_variance_policy"] += policy_diff * policy_diff
            stability_stats["total_variance_value"] += value_diff * value_diff
            stability_stats["num_variance_samples"] += 1

        stability_stats["last_policy_loss"] = policy_loss
        stability_stats["last_value_loss"] = value_loss

    def get_stability_metrics() -> dict:
        """Calculate current stability metrics"""
        if stability_stats["num_variance_samples"] == 0:
            return {
                "policy_loss_std": 0.0,
                "value_loss_std": 0.0,
                "policy_confidence_std": 0.0,
                "policy_loss_trend": 0.0,
                "value_loss_trend": 0.0,
            }

        # Calculate loss variance
        policy_variance = (
            stability_stats["total_variance_policy"]
            / stability_stats["num_variance_samples"]
        )
        value_variance = (
            stability_stats["total_variance_value"]
            / stability_stats["num_variance_samples"]
        )

        # Calculate policy confidence stability
        confidence_window = stability_stats["policy_confidence_window"]
        confidence_std = np.std(confidence_window) if confidence_window else 0.0

        # Calculate loss trends (negative means improving)
        policy_window = stability_stats["policy_loss_window"]
        value_window = stability_stats["value_loss_window"]
        policy_trend = (
            (np.mean(policy_window[-10:]) - np.mean(policy_window[:10]))
            if len(policy_window) >= 20
            else 0.0
        )
        value_trend = (
            (np.mean(value_window[-10:]) - np.mean(value_window[:10]))
            if len(value_window) >= 20
            else 0.0
        )

        return {
            "policy_loss_std": np.sqrt(policy_variance),
            "value_loss_std": np.sqrt(value_variance),
            "policy_confidence_std": confidence_std,
            "policy_loss_trend": policy_trend,
            "value_loss_trend": value_trend,
        }

    # Keep track of last N checkpoints
    checkpoint_files = []

    def get_temperature(
        move_count: int, total_moves: int = 12, is_strategic: bool = False
    ) -> float:
        """Higher temperature for opening moves to encourage exploration"""
        if move_count < 2:  # First move
            return 3.0  # Much higher temperature for first move
        elif move_count < 4:  # Early game
            return 2.0  # Still elevated for early moves
        else:  # Mid-late game
            return max(min_temp, 1.0)

    def get_adjusted_value(
        winner: Optional[Player], move_count: int, is_model_turn: bool
    ) -> float:
        """Simple value function with draw adjustment"""
        if winner is None:
            return 0.2  # Draws are less valuable
        else:
            return 1.0 if (winner == Player.ONE) == is_model_turn else 0.0

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
    print(f"Random opponent ratio: {random_opponent_ratio:.1%}")
    print(f"Batch size: {batch_size}")
    print(f"Replay buffer size: {buffer_size}")
    print(f"Policy weight: {policy_weight:.1f}")
    print(f"Press Ctrl+C to stop training gracefully")

    try:
        while not interrupt_received:
            iteration += 1
            iteration_start = time.time()

            # Reset iteration stats
            iter_stats = stats_template.copy()

            # Self-play phase
            episode_pbar = tqdm(
                range(num_episodes),
                desc=f"Self-Play Games (Iter {iteration})",
                leave=False,
            )

            # Determine number of opponent games
            strategic_games = int(num_episodes * strategic_opponent_ratio)
            random_games = int(num_episodes * random_opponent_ratio)
            selfplay_games = num_episodes - strategic_games - random_games

            # Self-play games
            for episode in range(selfplay_games):
                game = GameState()
                examples = []
                move_count = 0

                while True:
                    legal_moves = game.get_legal_moves()
                    if not np.any(legal_moves):
                        game.pass_turn()
                        continue

                    # Get move probabilities from policy network
                    state_rep = game.get_game_state_representation()
                    policy, _ = model.predict(
                        state_rep.board, state_rep.flat_values, legal_moves
                    )

                    # Remove batch dimension from policy
                    policy = policy.squeeze(0)  # Now shape is (4, 4, 4)

                    # Ensure we only consider legal moves
                    masked_policy = policy * legal_moves  # Mask out illegal moves
                    masked_policy = masked_policy / (
                        masked_policy.sum() + 1e-8
                    )  # Renormalize

                    # Use unified temperature schedule
                    temperature = get_temperature(move_count)

                    if temperature > 0:
                        # Sample from the probability distribution
                        policy_flat = masked_policy.flatten()
                        move_idx = np.random.choice(len(policy_flat), p=policy_flat)
                        move_coords = np.unravel_index(move_idx, masked_policy.shape)
                    else:
                        move_coords = np.unravel_index(
                            masked_policy.argmax(), masked_policy.shape
                        )

                    move = Move(
                        move_coords[0], move_coords[1], PieceType(move_coords[2])
                    )

                    # Create one-hot encoded policy target
                    policy_target = np.zeros_like(masked_policy)
                    policy_target[move_coords] = 1.0

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

                        # Set value targets with length penalty
                        for example in examples:
                            example.value = get_adjusted_value(winner, move_count, True)
                        break

                replay_buffer.extend(examples)
                episode_pbar.update(1)
                episode_pbar.set_postfix(
                    {
                        "buffer_size": len(replay_buffer),
                        "temp": f"{temperature:.2f}",
                        "type": "self-play",
                    }
                )

                # In self-play games, after game over:
                if result == TurnResult.GAME_OVER:
                    iter_stats["self_play_games"] += 1
                    if winner == Player.ONE:
                        iter_stats["self_play_p1_wins"] += 1
                    iter_stats["total_moves"] += move_count
                    iter_stats["total_games"] += 1

            # Random opponent games
            for episode in range(random_games):
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
                    state_rep = game.get_game_state_representation()

                    if is_model_turn:
                        # Model's turn
                        policy, _ = model.predict(
                            state_rep.board, state_rep.flat_values, legal_moves
                        )

                        policy = policy.squeeze(0)
                        masked_policy = policy * legal_moves
                        masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

                        # Use unified temperature schedule
                        temperature = get_temperature(move_count, is_strategic=False)

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

                        # Store training example for model's move
                        policy_target = np.zeros_like(masked_policy)
                        policy_target[move_coords] = 1.0
                        examples.append(
                            TrainingExample(
                                state_rep=state_rep, policy=policy_target, value=0.0
                            )
                        )
                    else:
                        # Random opponent's turn
                        move = random_opponent.get_move(game)
                        if move is None:
                            game.pass_turn()
                            continue

                    result = game.make_move(move)
                    move_count += 1

                    if result == TurnResult.GAME_OVER:
                        winner = game.get_winner()
                        # Set value targets based on game outcome with length penalty
                        for example in examples:
                            example.value = get_adjusted_value(
                                winner,
                                move_count,
                                (winner == Player.ONE) == model_is_player_one,
                            )
                        break

                replay_buffer.extend(examples)
                episode_pbar.update(1)
                episode_pbar.set_postfix(
                    {"buffer_size": len(replay_buffer), "type": "random"}
                )

                # Track stats for random opponent games
                if result == TurnResult.GAME_OVER:
                    iter_stats["total_moves"] += move_count
                    iter_stats["total_games"] += 1

                # Track model's moves
                if is_model_turn:
                    max_prob = masked_policy.max()
                    iter_stats["policy_confidence"] += max_prob
                    iter_stats["moves_counted"] += 1

                    if 1 <= move.x <= 2 and 1 <= move.y <= 2:
                        iter_stats["central_moves"] += 1

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
                    state_rep = (
                        game.get_game_state_representation()
                    )  # Get state before any moves

                    if is_model_turn:
                        # Model's turn
                        policy, _ = model.predict(
                            state_rep.board, state_rep.flat_values, legal_moves
                        )

                        policy = policy.squeeze(0)
                        masked_policy = policy * legal_moves
                        masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

                        # Use unified temperature schedule
                        temperature = get_temperature(move_count, is_strategic=True)

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

                        # Store training example for model's move
                        policy_target = np.zeros_like(masked_policy)
                        policy_target[move_coords] = 1.0
                        examples.append(
                            TrainingExample(
                                state_rep=state_rep, policy=policy_target, value=0.0
                            )
                        )
                    else:
                        # Strategic opponent's turn - now we store these moves as learning examples
                        move = strategic_opponent.get_move(game)
                        if move is None:
                            game.pass_turn()
                            continue

                        # Store the strategic opponent's move as a training example
                        policy_target = np.zeros_like(legal_moves, dtype=np.float32)
                        policy_target[move.x, move.y, move.piece_type.value] = 1.0
                        examples.append(
                            TrainingExample(
                                state_rep=state_rep, policy=policy_target, value=0.0
                            )
                        )

                    result = game.make_move(move)
                    move_count += 1

                    if result == TurnResult.GAME_OVER:
                        winner = game.get_winner()
                        # Set value targets based on game outcome with length penalty
                        for example in examples:
                            if winner is None:
                                example.value = get_adjusted_value(
                                    None, move_count, is_model_turn
                                )
                            else:
                                # Only learn from strategic opponent's winning moves
                                is_strategic_win = (
                                    winner == Player.ONE
                                ) != model_is_player_one
                                if is_strategic_win and not is_model_turn:
                                    # Learn from winning strategic moves
                                    example.value = get_adjusted_value(
                                        winner, move_count, False
                                    )
                                else:
                                    # Learn from model's moves only when it wins
                                    example.value = get_adjusted_value(
                                        winner, move_count, True
                                    )
                        break

                replay_buffer.extend(examples)
                episode_pbar.update(1)
                episode_pbar.set_postfix(
                    {"buffer_size": len(replay_buffer), "type": "strategic"}
                )

                # In strategic opponent games, after game over:
                if result == TurnResult.GAME_OVER:
                    if model_is_player_one:
                        iter_stats["strategic_games_as_p1"] += 1
                        if winner == Player.ONE:
                            iter_stats["strategic_wins_as_p1"] += 1
                    else:
                        iter_stats["strategic_games_as_p2"] += 1
                        if winner == Player.TWO:
                            iter_stats["strategic_wins_as_p2"] += 1
                    iter_stats["total_moves"] += move_count
                    iter_stats["total_games"] += 1

                # When making any move (both self-play and strategic):
                # Track policy confidence and central moves
                if is_model_turn:  # Only track model's moves
                    max_prob = masked_policy.max()
                    iter_stats["policy_confidence"] += max_prob
                    iter_stats["moves_counted"] += 1

                    # Check if move was in center
                    if 1 <= move.x <= 2 and 1 <= move.y <= 2:
                        iter_stats["central_moves"] += 1

                # Keep buffer size in check
                if len(replay_buffer) > buffer_size:
                    replay_buffer = replay_buffer[-buffer_size:]

            # Training phase
            train_pbar = tqdm(range(num_epochs), desc="Training Epochs", leave=False)
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

                # Train network with weighted losses
                total_loss, policy_loss, value_loss = model.train_step(
                    board_inputs,
                    flat_inputs,
                    policy_targets,
                    value_targets,
                    policy_weight=policy_weight,  # Pass weight to train_step
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
                        "ratio": f"{running_loss['policy']/running_loss['value']:.2f}",
                    }
                )

            # Calculate average losses for this epoch
            avg_total = epoch_losses["total"] / num_epochs
            avg_policy = epoch_losses["policy"] / num_epochs
            avg_value = epoch_losses["value"] / num_epochs

            # Update stability metrics after calculating averages
            update_stability_metrics(
                avg_policy,
                avg_value,
                iter_stats["policy_confidence"] / max(1, iter_stats["moves_counted"]),
            )
            stability_metrics = get_stability_metrics()

            # Print epoch summary with stability metrics
            elapsed_time = time.time() - training_start
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60

            tqdm.write(
                f"\nIteration {iteration} - Time: {int(hours)}h {int(minutes)}m"
                f"\nAvg Losses: Total: {avg_total:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}"
                f"\nStability Metrics:"
                f"\n  Loss Std Dev: Policy={stability_metrics['policy_loss_std']:.4f}, Value={stability_metrics['value_loss_std']:.4f}"
                f"\n  Policy Confidence Std Dev: {stability_metrics['policy_confidence_std']:.4f}"
                f"\n  Loss Trends: Policy={stability_metrics['policy_loss_trend']:.4f}, Value={stability_metrics['value_loss_trend']:.4f}"
            )

            # Update stats display
            tqdm.write(
                f"\nIteration {iteration} Statistics:"
                f"\n  Time: {int(hours)}h {int(minutes)}m"
                f"\n  Losses - Total: {avg_total:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}"
                f"\n  Strategic Performance:"
                f"\n    As P1: {iter_stats['strategic_wins_as_p1']}/{iter_stats['strategic_games_as_p1']} "
                f"({100 * iter_stats['strategic_wins_as_p1'] / max(1, iter_stats['strategic_games_as_p1']):.1f}%)"
                f"\n    As P2: {iter_stats['strategic_wins_as_p2']}/{iter_stats['strategic_games_as_p2']} "
                f"({100 * iter_stats['strategic_wins_as_p2'] / max(1, iter_stats['strategic_games_as_p2']):.1f}%)"
                f"\n  Self-play P1 Win Rate: {100 * iter_stats['self_play_p1_wins'] / max(1, iter_stats['self_play_games']):.1f}%"
                f"\n  Model Behavior:"
                f"\n    Avg Game Length: {iter_stats['total_moves'] / max(1, iter_stats['total_games']):.1f} moves"
                f"\n    Central Move Rate: {100 * iter_stats['central_moves'] / max(1, iter_stats['moves_counted']):.1f}%"
                f"\n    Avg Policy Confidence: {iter_stats['policy_confidence'] / max(1, iter_stats['moves_counted']):.3f}"
            )

            tqdm.write(
                f"Iteration completed in {time.time() - iteration_start:.1f} seconds\n"
            )

            # Save latest model after every iteration
            model.save(latest_model_path)

            # Periodic checkpoint saving
            if iteration % save_interval == 0:
                save_path = f"saved_models/model_iter_{iteration}.pth"
                model.save(save_path)
                checkpoint_files.append(save_path)

                # Remove old checkpoints if we have too many
                while len(checkpoint_files) > num_checkpoints:
                    old_checkpoint = checkpoint_files.pop(0)
                    if old_checkpoint != latest_model_path:  # Don't delete latest
                        try:
                            os.remove(old_checkpoint)
                            tqdm.write(f"Removed old checkpoint: {old_checkpoint}")
                        except OSError:
                            pass

                tqdm.write(f"Saved checkpoint: {save_path}")

    finally:
        # Calculate final elapsed time
        elapsed_time = time.time() - training_start
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60

        # Always save final model state
        final_path = f"saved_models/model_iter_final.pth"
        model.save(final_path)
        model.save(latest_model_path)  # Update latest as well
        print(f"\nFinal model saved as: {final_path}")
        print(f"Latest model saved as: {latest_model_path}")

        # Enhanced final summary
        print("\nTraining Summary:")
        print(f"Total training time: {int(hours)}h {int(minutes)}m")
        print(f"Iterations completed: {iteration}")
        print(f"Final Model State:")
        print(f"  Loss Metrics:")
        print(f"    Total: {running_loss['total']:.4f}")
        print(f"    Policy: {running_loss['policy']:.4f}")
        print(f"    Value: {running_loss['value']:.4f}")
        print(
            f"  Policy/Value Ratio: {running_loss['policy']/running_loss['value']:.2f}"
        )
        print(f"  Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
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
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "crunch"],
        default="train",
        help="Mode to run in: train (training), eval (evaluation), or crunch (optimize for deployment)",
    )
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
        "--load_model",
        type=str,
        default="saved_models/model_latest.pth",
        help="Path to model to load",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="optimized_models",
        help="Directory to save optimized models when using crunch mode",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with a fresh model, ignoring existing checkpoints",
    )
    parser.add_argument(
        "--stable_lr",
        action="store_true",
        help="Use stable (slower) learning rate instead of fast mode",
    )

    args = parser.parse_args()

    # Initialize model with appropriate mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ModelWrapper(
        device,
        mode=(
            "crunch"
            if args.mode == "crunch"
            else "stable" if args.stable_lr else "fast"
        ),
    )

    if not args.fresh and args.load_model and os.path.exists(args.load_model):
        model.load(args.load_model)
        print(f"Loaded model from {args.load_model}")
    else:
        if args.fresh:
            print("Starting with fresh model as requested")
        else:
            print(f"No model found at {args.load_model}, starting fresh")

    if args.mode == "train":
        train_network(
            model,
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            num_checkpoints=args.num_checkpoints,
            debug=args.debug,
        )
    elif args.mode == "eval":
        evaluate_model(model)
    elif args.mode == "crunch":
        if not os.path.exists(args.load_model):
            parser.error(
                "--load_model must point to an existing model file when using crunch mode"
            )
        os.makedirs(args.output_dir, exist_ok=True)
        model.crunch(args.load_model, args.output_dir)
