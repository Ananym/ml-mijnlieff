import os
import numpy as np
import torch
from game import GameState, PieceType, Player, GameStateRepresentation, TurnResult, Move
from typing import List, Tuple, Optional
import time
import signal
import sys
from dataclasses import dataclass
from model import ModelWrapper
from tqdm import tqdm
from opponents import RandomOpponent, StrategicOpponent
from mcts import MCTS, mcts_self_play_game
import random

# Training parameters
DEFAULT_EPISODES = 60
DEFAULT_BATCH_SIZE = 256
DEFAULT_SAVE_INTERVAL = 10
DEFAULT_NUM_CHECKPOINTS = 5
DEFAULT_MCTS_SIMS = 400
DEFAULT_MCTS_RATIO = 0.5
# Strategic ratio is the percentage of games that use the strategic opponent
# Strategic opponent is defined using algorithmic heuristics
DEFAULT_BUFFER_SIZE = 6000
DEFAULT_POLICY_WEIGHT = 1.0
DEFAULT_NUM_EPOCHS = 10
DEFAULT_LR_PATIENCE = 8
DEFAULT_STAGNATION_THRESHOLD = 0.01
DEFAULT_LR_RESET_FACTOR = 5.0

# Opponent ratio scheduling constants
INITIAL_RANDOM_OPPONENT_RATIO = 0.8  # Start with high random opponent exposure
FINAL_RANDOM_OPPONENT_RATIO = 0.2  # Phase out to minimal random exposure
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.1  # Start with low strategic opponent exposure
FINAL_STRATEGIC_OPPONENT_RATIO = 0.7  # Increase strategic opponents over time
OPPONENT_TRANSITION_ITERATIONS = 150  # Complete transition over 50 iterations

# Add to constants at top of file
BOOTSTRAP_MIN_WEIGHT = 0.1  # Much lower starting point
BOOTSTRAP_MAX_WEIGHT = 0.3  # Lower maximum
BOOTSTRAP_TRANSITION_ITERATIONS = 100  # Slower transition
DIAGNOSTIC_INTERVAL = 5  # Print detailed diagnostics every N iterations


@dataclass
class TrainingExample:
    state_rep: GameStateRepresentation
    policy: np.ndarray  # One-hot encoded actual move made
    value: float  # Game outcome from current player's perspective
    current_player: Player  # Store which player made the move
    mcts_value: float = None  # Optional MCTS value prediction


def hybrid_training_loop(
    model: ModelWrapper,
    num_episodes: int = DEFAULT_EPISODES,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    mcts_sim_count: int = DEFAULT_MCTS_SIMS,
    mcts_game_ratio: float = DEFAULT_MCTS_RATIO,
    bootstrap_weight: float = DEFAULT_BOOTSTRAP_WEIGHT,
    policy_weight: float = DEFAULT_POLICY_WEIGHT,
    initial_random_ratio: float = INITIAL_RANDOM_OPPONENT_RATIO,
    final_random_ratio: float = FINAL_RANDOM_OPPONENT_RATIO,
    initial_strategic_ratio: float = INITIAL_STRATEGIC_OPPONENT_RATIO,
    final_strategic_ratio: float = FINAL_STRATEGIC_OPPONENT_RATIO,
    transition_iterations: int = OPPONENT_TRANSITION_ITERATIONS,
    save_interval: int = DEFAULT_SAVE_INTERVAL,
    num_checkpoints: int = DEFAULT_NUM_CHECKPOINTS,
    debug: bool = False,
):
    """Training loop that combines supervised learning from expert demonstrations
    and reinforcement learning through self-play.
    """
    # Initialize random number generator
    rng = np.random.Generator(np.random.PCG64())

    replay_buffer = []
    running_loss = {
        "total": 1e-8,
        "policy": 1e-8,
        "value": 1e-8,
    }
    running_count = 0
    iteration = 0
    training_start = time.time()
    strategic_opponent = StrategicOpponent()
    random_opponent = RandomOpponent()
    latest_model_path = "saved_models/model_latest.pth"

    # Learning rate monitoring and control
    lr_history = []
    loss_history = []
    stagnation_counter = 0
    best_loss = float("inf")

    # Base stats template
    stats_template = {
        "strategic_wins_as_p1": 0,
        "strategic_wins_as_p2": 0,
        "strategic_games_as_p1": 0,
        "strategic_games_as_p2": 0,
        "random_wins_as_p1": 0,
        "random_wins_as_p2": 0,
        "random_games_as_p1": 0,
        "random_games_as_p2": 0,
        "self_play_p1_wins": 0,
        "self_play_games": 0,
        "mcts_moves": 0,
        "direct_moves": 0,
        "total_moves": 0,
        "total_games": 0,
        "win_examples": 0,
        "loss_examples": 0,
        "draw_examples": 0,
    }

    # Keep track of last N checkpoints
    checkpoint_files = []

    def get_adjusted_value(game, winner, move_count, current_player):
        """Get time-aware value that incorporates score difference"""
        # Constants
        AVG_GAME_LENGTH = 15.0
        MIN_VALUE_SCALE = 0.4  # Minimum scaling for early moves
        SCORE_WEIGHT = 0.3  # How much to weigh score difference vs win/loss

        # Calculate how far into the game we are
        progress = min(1.0, move_count / AVG_GAME_LENGTH)

        # Calculate base scale based on game progression
        value_scale = MIN_VALUE_SCALE + (1.0 - MIN_VALUE_SCALE) * progress

        # Get scores for both players
        p1_score = game.scores[Player.ONE]
        p2_score = game.scores[Player.TWO]

        # Calculate normalized score difference from current player's perspective
        if current_player == Player.ONE:
            score_diff = (p1_score - p2_score) / max(1, p1_score + p2_score)
        else:
            score_diff = (p2_score - p1_score) / max(1, p1_score + p2_score)

        # Blend win/loss signal with score difference
        if winner is None:  # Draw
            # Use pure score difference for draws
            return score_diff * value_scale
        elif winner == current_player:  # Win
            # Win plus score advantage
            outcome_value = value_scale
            return (
                1 - SCORE_WEIGHT
            ) * outcome_value + SCORE_WEIGHT * score_diff * value_scale
        else:  # Loss
            # Loss plus score disadvantage
            outcome_value = -value_scale
            return (
                1 - SCORE_WEIGHT
            ) * outcome_value + SCORE_WEIGHT * score_diff * value_scale

    def create_opponent_example(game, move, opponent_type):
        """Create a training example from an opponent's move"""
        state_rep = game.get_game_state_representation()
        legal_moves = game.get_legal_moves()

        # Create one-hot encoded policy for the opponent's move
        policy_target = np.zeros_like(legal_moves)
        policy_target[move.x, move.y, move.piece_type.value] = 1.0

        return TrainingExample(
            state_rep=state_rep,
            policy=policy_target,
            value=0.0,  # Will be filled in later
            current_player=game.current_player,
        )

    # Helper function to get model's move, either via MCTS or direct policy
    def get_model_move_for_play(game, use_mcts=False, temperature=1.0):
        """Get a move from the model for gameplay (returns the move and one-hot policy)"""
        legal_moves = game.get_legal_moves()
        state_rep = game.get_game_state_representation()

        if use_mcts:
            # Use pure rollout MCTS for move selection (no model)
            mcts = MCTS(num_simulations=mcts_sim_count)
            mcts.set_temperature(temperature)
            mcts_policy, root_node = mcts.search(game)
            policy = mcts_policy
            # No predicted value in pure rollout mode
            root_value = 0.0
        else:
            # Use direct policy from neural network
            policy, value_pred = model.predict(
                state_rep.board, state_rep.flat_values, legal_moves
            )
            policy = policy.squeeze(0)
            root_value = value_pred.squeeze(0)[0]

        # Apply mask for legal moves
        masked_policy = policy * legal_moves
        masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

        # Select move based on temperature
        if temperature > 0:
            # Sample from the probability distribution
            policy_flat = masked_policy.flatten()
            move_idx = rng.choice(len(policy_flat), p=policy_flat)
            move_coords = np.unravel_index(move_idx, masked_policy.shape)
        else:
            # Choose move deterministically
            move_coords = np.unravel_index(masked_policy.argmax(), masked_policy.shape)

        move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))

        # Create one-hot encoded policy target (for gameplay only)
        policy_target = np.zeros_like(masked_policy)
        policy_target[move_coords] = 1.0

        return move, policy_target, root_value, state_rep

    def get_temperature(move_count, iteration):
        """Dynamic temperature that encourages exploration early and exploitation later"""
        # More exploration in early moves
        if move_count < 4:
            base_temp = 1.5
        elif move_count < 8:
            base_temp = 1.2
        elif move_count < 12:
            base_temp = 1.0
        else:
            base_temp = 0.8

        # More exploration in early training iterations
        iteration_factor = max(0.5, 1.0 - (iteration / 50))
        return base_temp * iteration_factor

    def get_model_move_with_policy(game, use_mcts=True, temperature=1.0):
        """Get a move from the model along with the training policy target
        (full MCTS distribution if MCTS was used, otherwise one-hot)"""
        legal_moves = game.get_legal_moves()
        state_rep = game.get_game_state_representation()

        if use_mcts:
            # Use model-guided MCTS (AlphaZero style) instead of pure rollouts
            mcts = MCTS(
                model=model,  # Pass the model to guide search
                num_simulations=mcts_sim_count,
                c_puct=1.0,
            )
            mcts.set_temperature(temperature)
            mcts_policy, root_node = mcts.search(game)
            policy = mcts_policy

            # Use full MCTS distribution as policy target when MCTS is used
            policy_target = mcts_policy.copy()

            # Get value prediction from root node
            root_value = (
                root_node.predicted_value
                if root_node.predicted_value is not None
                else 0.0
            )
        else:
            # Use direct policy from neural network
            policy, value_pred = model.predict(
                state_rep.board, state_rep.flat_values, legal_moves
            )
            policy = policy.squeeze(0)
            root_value = value_pred.squeeze(0)[0]

            # Apply mask for legal moves
            masked_policy = policy * legal_moves
            masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

            # Select move based on temperature
            if temperature > 0:
                # Sample from the probability distribution
                policy_flat = masked_policy.flatten()
                move_idx = rng.choice(len(policy_flat), p=policy_flat)
                move_coords = np.unravel_index(move_idx, masked_policy.shape)
            else:
                # Choose move deterministically
                move_coords = np.unravel_index(
                    masked_policy.argmax(), masked_policy.shape
                )

            # For direct policy, use one-hot encoding as the target
            policy_target = np.zeros_like(masked_policy)
            policy_target[move_coords] = 1.0

        # Apply mask for legal moves to the final policy
        masked_policy = policy * legal_moves
        masked_policy = masked_policy / (masked_policy.sum() + 1e-8)

        # Select move based on temperature
        if temperature > 0:
            # Sample from the probability distribution
            policy_flat = masked_policy.flatten()
            move_idx = rng.choice(len(policy_flat), p=policy_flat)
            move_coords = np.unravel_index(move_idx, masked_policy.shape)
        else:
            # Choose move deterministically
            move_coords = np.unravel_index(masked_policy.argmax(), masked_policy.shape)

        move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))

        # Add small Dirichlet noise to root node policy for exploration
        if game.current_player == Player.ONE:  # Only add noise as P1 for consistency
            noise = np.random.dirichlet([0.3] * len(policy_flat), size=1)[0]
            policy_flat = 0.9 * policy_flat + 0.1 * noise  # 10% noise
            policy_flat /= np.sum(policy_flat)  # Renormalize

        return move, policy_target, root_value, state_rep

    # Set up signal handler for graceful interruption
    interrupt_received = False

    def handle_interrupt(signum, frame):
        nonlocal interrupt_received
        if interrupt_received:  # Second interrupt, exit immediately
            print("\nForced exit...")
            sys.exit(1)
        print("\nInterrupt received, finishing current iteration...")
        interrupt_received = True

    signal.signal(signal.SIGINT, handle_interrupt)

    # Function to update opponent ratios based on current iteration
    def get_opponent_ratios(iteration):
        progress = min(1.0, iteration / transition_iterations)

        # Linear interpolation between initial and final values
        random_ratio = initial_random_ratio + progress * (
            final_random_ratio - initial_random_ratio
        )
        strategic_ratio = initial_strategic_ratio + progress * (
            final_strategic_ratio - initial_strategic_ratio
        )

        # Ensure self-play ratio is always positive by capping total opponent ratio
        total_opponent_ratio = random_ratio + strategic_ratio
        if total_opponent_ratio > 0.9:
            # Scale down both to keep relative proportions but cap total
            scale = 0.9 / total_opponent_ratio
            random_ratio *= scale
            strategic_ratio *= scale

        return random_ratio, strategic_ratio

    # Initialize opponent ratios for first iteration
    random_opponent_ratio, strategic_opponent_ratio = get_opponent_ratios(iteration)

    # Add function to calculate bootstrap weight based on iteration
    def get_bootstrap_weight(iteration):
        """Start with lower bootstrapping and increase gradually"""
        progress = min(1.0, iteration / BOOTSTRAP_TRANSITION_ITERATIONS)
        return BOOTSTRAP_MIN_WEIGHT + progress * (
            BOOTSTRAP_MAX_WEIGHT - BOOTSTRAP_MIN_WEIGHT
        )

    print("Starting hybrid training loop with value bootstrapping")
    print("Episodes per iteration:", num_episodes)
    print("MCTS simulations per move:", mcts_sim_count)
    print("MCTS move ratio: {:.1%}".format(mcts_game_ratio))
    print("Initial opponent ratios:")
    print("  Random: {:.1%}".format(random_opponent_ratio))
    print("  Strategic: {:.1%}".format(strategic_opponent_ratio))
    print(
        "  Self-play: {:.1%}".format(
            1.0 - random_opponent_ratio - strategic_opponent_ratio
        )
    )
    print("Final opponent ratios after {} iterations:".format(transition_iterations))
    print("  Random: {:.1%}".format(final_random_ratio))
    print("  Strategic: {:.1%}".format(final_strategic_ratio))
    print(
        "  Self-play: {:.1%}".format(1.0 - final_random_ratio - final_strategic_ratio)
    )
    print("Using AlphaZero-style MCTS policy targets for better learning")
    print("Initial bootstrap weight: {:.2f}".format(get_bootstrap_weight(iteration)))
    print("Final bootstrap weight: {:.2f}".format(BOOTSTRAP_MAX_WEIGHT))
    print("Batch size:", batch_size)
    print("Replay buffer size:", buffer_size)
    print("Policy weight: {:.1f}".format(policy_weight))
    print("Press Ctrl+C to stop training gracefully")

    try:
        while not interrupt_received:  # Run until interrupted
            iteration += 1
            iteration_start = time.time()

            # Update opponent ratios based on current iteration
            random_opponent_ratio, strategic_opponent_ratio = get_opponent_ratios(
                iteration
            )

            # Update bootstrap weight based on current iteration
            bootstrap_weight = get_bootstrap_weight(iteration)

            # Reset iteration stats
            iter_stats = stats_template.copy()

            # Create progress bar for all games
            episode_pbar = tqdm(
                range(num_episodes),
                desc=f"Games (Iter {iteration})",
                leave=False,
            )

            # Generate games with various opponents
            for _ in range(num_episodes):
                # First, decide the opponent type
                game_type_roll = rng.random()

                if game_type_roll < random_opponent_ratio:
                    # Random opponent
                    opponent_type = "random"
                    opponent = random_opponent
                    # Decide if model plays as P1 or P2
                    model_plays_p1 = rng.random() < 0.5
                elif game_type_roll < random_opponent_ratio + strategic_opponent_ratio:
                    # Strategic opponent
                    opponent_type = "strategic"
                    opponent = strategic_opponent
                    # Decide if model plays as P1 or P2
                    model_plays_p1 = rng.random() < 0.5
                else:
                    # Self-play game
                    opponent_type = "self-play"
                    model_plays_p1 = True  # Doesn't matter for self-play

                # Initialize game
                game = GameState()
                examples = []
                move_count = 0

                # Play the game
                while True:
                    legal_moves = game.get_legal_moves()
                    if not np.any(legal_moves):
                        game.pass_turn()
                        continue

                    if (
                        opponent_type == "self-play"
                        or (game.current_player == Player.ONE) == model_plays_p1
                    ):
                        # Model's turn
                        # Decide whether to use MCTS for this move
                        use_mcts = rng.random() < mcts_game_ratio
                        if use_mcts:
                            iter_stats["mcts_moves"] += 1
                        else:
                            iter_stats["direct_moves"] += 1

                        # Use temperature based on move count
                        temperature = get_temperature(move_count, iteration)

                        # Get model's move with AlphaZero-style policy
                        move, policy_target, value_pred, state_rep = (
                            get_model_move_with_policy(
                                game, use_mcts=use_mcts, temperature=temperature
                            )
                        )

                        # When storing examples after MCTS-based moves
                        if use_mcts:
                            # Root node value is already available from MCTS search
                            # Store it with the example for later use
                            examples.append(
                                TrainingExample(
                                    state_rep=state_rep,
                                    policy=policy_target,
                                    value=0.0,  # Will be filled in later
                                    current_player=game.current_player,
                                    mcts_value=value_pred,  # Store the MCTS value prediction
                                )
                            )
                    else:
                        # Opponent's turn
                        move = opponent.get_move(game)
                        if move is None:
                            game.pass_turn()
                            continue

                        # Only store examples from strategic opponent
                        if opponent_type == "strategic":
                            # Create and store example from opponent's move
                            opponent_example = create_opponent_example(
                                game, move, opponent_type
                            )
                            examples.append(opponent_example)

                    # Make move
                    result = game.make_move(move)
                    move_count += 1

                    if result == TurnResult.GAME_OVER:
                        # Game is over, get the winner
                        winner = game.get_winner()

                        # Set value targets based on game outcome and bootstrapping
                        for i, example in enumerate(examples):
                            # Get the final outcome value
                            outcome_value = get_adjusted_value(
                                game, winner, move_count, example.current_player
                            )

                            # If this move used MCTS and we have an MCTS value, use improved target
                            if (
                                hasattr(example, "mcts_value")
                                and example.mcts_value is not None
                            ):
                                example.value = get_improved_value_target(
                                    outcome_value, example.mcts_value, i
                                )
                            else:
                                # For non-MCTS moves, use standard bootstrapping
                                # The next player made a move, so get their perspective's value prediction
                                next_example = examples[i + 1]
                                next_state_rep = next_example.state_rep

                                # Get a fresh value prediction for the next state
                                _, next_value = model.predict(
                                    next_state_rep.board, next_state_rep.flat_values
                                )

                                # Negate since it's from opponent's perspective
                                bootstrap_value = -float(next_value[0][0])

                                # Mix the outcome with bootstrapped value
                                example.value = (
                                    (1 - bootstrap_weight) * outcome_value
                                    + bootstrap_weight * bootstrap_value
                                )

                            # Update stats
                            if outcome_value > 0.2:
                                iter_stats["win_examples"] += 1
                            elif outcome_value < -0.2:
                                iter_stats["loss_examples"] += 1
                            else:
                                iter_stats["draw_examples"] += 1

                        # Update game statistics
                        iter_stats["total_games"] += 1
                        iter_stats["total_moves"] += move_count

                        if opponent_type == "self-play":
                            iter_stats["self_play_games"] += 1
                            if winner == Player.ONE:
                                iter_stats["self_play_p1_wins"] += 1
                        elif opponent_type == "strategic":
                            if model_plays_p1:
                                iter_stats["strategic_games_as_p1"] += 1
                                if winner == Player.ONE:
                                    iter_stats["strategic_wins_as_p1"] += 1
                            else:
                                iter_stats["strategic_games_as_p2"] += 1
                                if winner == Player.TWO:
                                    iter_stats["strategic_wins_as_p2"] += 1
                        elif opponent_type == "random":
                            if model_plays_p1:
                                iter_stats["random_games_as_p1"] += 1
                                if winner == Player.ONE:
                                    iter_stats["random_wins_as_p1"] += 1
                            else:
                                iter_stats["random_games_as_p2"] += 1
                                if winner == Player.TWO:
                                    iter_stats["random_wins_as_p2"] += 1

                        break  # Game over

                # Add examples to this iteration's buffer
                replay_buffer.extend(examples)

                # Update progress bar
                episode_pbar.update(1)
                episode_pbar.set_postfix(
                    {
                        "wins": iter_stats["win_examples"],
                        "losses": iter_stats["loss_examples"],
                        "type": opponent_type,
                    }
                )

            # Trim replay buffer if needed
            if len(replay_buffer) > buffer_size:
                indices = rng.choice(
                    len(replay_buffer), size=buffer_size, replace=False
                )
                replay_buffer = [replay_buffer[i] for i in indices]

                # Log only buffer size after trimming
                tqdm.write(f"Trimmed buffer size: {len(replay_buffer)}")

            # Training phase
            train_pbar = tqdm(range(num_epochs), desc="Training Epochs", leave=False)
            epoch_losses = {"total": 0, "policy": 0, "value": 0}

            for _ in train_pbar:
                # Sample from replay buffer
                indices = rng.choice(
                    len(replay_buffer),
                    size=min(batch_size, len(replay_buffer)),
                    replace=False,
                )
                batch = [replay_buffer[i] for i in indices]

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
                    policy_weight=policy_weight,
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
                        "running_total": "{:.4f}".format(running_loss["total"]),
                        "running_policy": "{:.4f}".format(running_loss["policy"]),
                        "running_value": "{:.4f}".format(running_loss["value"]),
                    }
                )

            # Calculate average losses for this epoch
            avg_total = epoch_losses["total"] / num_epochs
            avg_policy = epoch_losses["policy"] / num_epochs
            avg_value = epoch_losses["value"] / num_epochs

            # Step the scheduler once per iteration
            model.scheduler.step()

            # Update loss and learning rate history for monitoring
            loss_history.append(avg_total)
            current_lr = model.optimizer.param_groups[0]["lr"]
            lr_history.append(current_lr)

            # Print core training statistics (simplified)
            elapsed_time = time.time() - training_start
            hours = elapsed_time // 3600
            minutes = (elapsed_time % 3600) // 60

            # Prepare key win rate statistics
            strategic_p1_rate = (
                100
                * iter_stats["strategic_wins_as_p1"]
                / max(1, iter_stats["strategic_games_as_p1"])
            )
            strategic_p2_rate = (
                100
                * iter_stats["strategic_wins_as_p2"]
                / max(1, iter_stats["strategic_games_as_p2"])
            )
            random_p1_rate = (
                100
                * iter_stats["random_wins_as_p1"]
                / max(1, iter_stats["random_games_as_p1"])
            )
            random_p2_rate = (
                100
                * iter_stats["random_wins_as_p2"]
                / max(1, iter_stats["random_games_as_p2"])
            )
            selfplay_p1_rate = (
                100
                * iter_stats["self_play_p1_wins"]
                / max(1, iter_stats["self_play_games"])
            )

            tqdm.write(
                f"\nIteration {iteration} - Time: {int(hours)}h {int(minutes)}m\n"
                f"Avg Losses: Total: {avg_total:.4f}, Policy: {avg_policy:.4f}, Value: {avg_value:.4f}\n"
                f"Learning Rate: {model.optimizer.param_groups[0]['lr']:.8f}\n"
                "Current Ratios: Random: {:.1%}, Strategic: {:.1%}, Self-play: {:.1%}\n".format(
                    random_opponent_ratio,
                    strategic_opponent_ratio,
                    1.0 - random_opponent_ratio - strategic_opponent_ratio,
                )
                + "Win Rates: Self-play P1: {:.1f}%, ".format(selfplay_p1_rate)
                + "Strategic P1: {:.1f}%, P2: {:.1f}%, ".format(
                    strategic_p1_rate, strategic_p2_rate
                )
                + "Random P1: {:.1f}%, P2: {:.1f}%".format(
                    random_p1_rate, random_p2_rate
                )
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
                        except OSError:
                            pass

            # Concise diagnostics (every DIAGNOSTIC_INTERVAL iterations)
            if iteration % DIAGNOSTIC_INTERVAL == 0:
                # Sample recent examples for diagnostics
                diagnostic_samples = random.sample(
                    replay_buffer, min(50, len(replay_buffer))
                )

                # Get raw predictions for diagnostics
                board_inputs = np.array(
                    [ex.state_rep.board for ex in diagnostic_samples]
                )
                flat_inputs = np.array(
                    [ex.state_rep.flat_values for ex in diagnostic_samples]
                )

                value_preds = []
                with torch.no_grad():
                    for i in range(len(diagnostic_samples)):
                        _, value = model.predict(
                            board_inputs[i : i + 1],  # Add batch dimension
                            flat_inputs[i : i + 1],  # Add batch dimension
                        )
                        value_preds.append(value[0][0])  # Extract the scalar value
                value_preds = np.array(value_preds)

                # Track core metrics
                correct_value_preds = 0
                avg_top1_prob = 0

                for i, example in enumerate(diagnostic_samples):
                    # Value accuracy (sign match)
                    true_sign = np.sign(example.value)
                    pred_sign = np.sign(value_preds[i])
                    if true_sign == pred_sign or (
                        true_sign == 0 and abs(pred_sign) < 0.3
                    ):
                        correct_value_preds += 1

                    # Policy concentration
                    avg_top1_prob += np.max(example.policy)

                value_accuracy = correct_value_preds / len(diagnostic_samples) * 100
                avg_top1_prob = avg_top1_prob / len(diagnostic_samples)

                # Print core diagnostics
                tqdm.write("\nDIAGNOSTICS:")
                tqdm.write(
                    f"  Value accuracy: {value_accuracy:.1f}% (matching outcome direction)"
                )
                tqdm.write(
                    f"  Policy confidence: {avg_top1_prob:.3f} (avg top move probability)"
                )
                tqdm.write(
                    f"  Loss trend: {'Increasing' if len(loss_history) > 1 and avg_total > loss_history[-2] else 'Decreasing/Stable'}"
                )
                tqdm.write(f"  LR: {model.optimizer.param_groups[0]['lr']:.8f}")
                tqdm.write(
                    f"  Current params: bootstrap={bootstrap_weight:.2f}, random={random_opponent_ratio:.2f}, strategic={strategic_opponent_ratio:.2f}\n"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted, saving final model...")
        model.save(f"saved_models/model_iter_final.pth")
        print("Final model saved.")
    finally:
        # Calculate final elapsed time
        elapsed_time = time.time() - training_start
        hours = elapsed_time // 3600
        minutes = (elapsed_time % 3600) // 60

        print("\nTraining Summary:")
        print(f"Total training time: {int(hours)}h {int(minutes)}m")
        print(f"Iterations completed: {iteration}")
        print(
            "Final Loss: Total={:.4f}, Policy={:.4f}, Value={:.4f}".format(
                running_loss["total"], running_loss["policy"], running_loss["value"]
            )
        )
        print(
            "Final Learning Rate: {:.8f}".format(model.optimizer.param_groups[0]["lr"])
        )
        print("Model saved at: saved_models/model_iter_final.pth")


def evaluate_model(
    model: ModelWrapper,
    num_games: int = 100,
    mcts_sims: int = 20,
    use_mcts: bool = False,
    deterministic: bool = True,
    debug: bool = False,
):
    """
    Evaluate model against different opponents and report win rates.

    Args:
        model: The model to evaluate
        num_games: Number of games to play against each opponent
        mcts_sims: Number of MCTS simulations per move (if use_mcts=True)
        use_mcts: Whether to use MCTS for model moves
        deterministic: If True, use deterministic policy (argmax), else sample
        debug: Print additional debug information
    """
    # Initialize opponents
    random_opponent = RandomOpponent()
    strategic_opponent = StrategicOpponent()

    # Random number generator for move sampling
    rng = np.random.Generator(np.random.PCG64())

    # Initialize statistics
    results = {
        "random": {
            "model_as_p1": {"wins": 0, "draws": 0, "losses": 0},
            "model_as_p2": {"wins": 0, "draws": 0, "losses": 0},
        },
        "strategic": {
            "model_as_p1": {"wins": 0, "draws": 0, "losses": 0},
            "model_as_p2": {"wins": 0, "draws": 0, "losses": 0},
        },
        "self_play": {"p1_wins": 0, "draws": 0, "p2_wins": 0},
    }

    def get_model_move(game: GameState, use_temperature: bool = False):
        """Helper function to get model's move"""
        legal_moves = game.get_legal_moves()
        state_rep = game.get_game_state_representation()

        # If using MCTS, run search to get improved policy
        if use_mcts:
            mcts = MCTS(num_simulations=mcts_sims)  # Pure rollout mode
            mcts_policy, _ = mcts.search(game)
            policy = mcts_policy
        else:
            # Get policy directly from model
            policy, _ = model.predict(
                state_rep.board, state_rep.flat_values, legal_moves
            )
            policy = policy.squeeze(0)

        # Apply legal move mask
        masked_policy = policy * legal_moves
        masked_policy = masked_policy / (np.sum(masked_policy) + 1e-8)

        # Choose move based on policy
        if deterministic and not use_temperature:
            # Choose best move deterministically
            move_coords = np.unravel_index(
                np.argmax(masked_policy), masked_policy.shape
            )
        else:
            # Sample from policy distribution
            policy_flat = masked_policy.flatten()
            move_idx = rng.choice(len(policy_flat), p=policy_flat)
            move_coords = np.unravel_index(move_idx, masked_policy.shape)

        return Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))

    print(f"\n{'=' * 50}")
    print(f"Starting model evaluation ({num_games} games per opponent)")
    print(
        f"MCTS: {'Enabled' if use_mcts else 'Disabled'}, Simulations: {mcts_sims if use_mcts else 'N/A'}"
    )
    print(f"Move selection: {'Deterministic' if deterministic else 'Sampling'}")
    print(f"{'=' * 50}\n")

    # 1. Play against random opponent
    print(f"Evaluating against Random opponent...")
    for game_idx in tqdm(range(num_games * 2), desc="Random opponent"):
        # Alternate between playing as P1 and P2
        model_plays_p1 = game_idx < num_games

        key = "model_as_p1" if model_plays_p1 else "model_as_p2"
        game = GameState()
        move_count = 0

        while not game.is_over:
            is_model_turn = (game.current_player == Player.ONE) == model_plays_p1

            if is_model_turn:
                # Model's turn
                move = get_model_move(game, use_temperature=(move_count < 5))
            else:
                # Random opponent's turn
                move = random_opponent.get_move(game)
                if move is None:
                    game.pass_turn()
                    continue

            game.make_move(move)
            move_count += 1

        # Record game result
        winner = game.get_winner()
        if winner is None:
            results["random"][key]["draws"] += 1
        elif (winner == Player.ONE) == model_plays_p1:
            results["random"][key]["wins"] += 1
        else:
            results["random"][key]["losses"] += 1

    # 2. Play against strategic opponent
    print(f"Evaluating against Strategic opponent...")
    for game_idx in tqdm(range(num_games * 2), desc="Strategic opponent"):
        # Alternate between playing as P1 and P2
        model_plays_p1 = game_idx < num_games

        key = "model_as_p1" if model_plays_p1 else "model_as_p2"
        game = GameState()
        move_count = 0

        while not game.is_over:
            is_model_turn = (game.current_player == Player.ONE) == model_plays_p1

            if is_model_turn:
                # Model's turn
                move = get_model_move(game, use_temperature=(move_count < 5))
            else:
                # Strategic opponent's turn
                move = strategic_opponent.get_move(game)
                if move is None:
                    game.pass_turn()
                    continue

            game.make_move(move)
            move_count += 1

        # Record game result
        winner = game.get_winner()
        if winner is None:
            results["strategic"][key]["draws"] += 1
        elif (winner == Player.ONE) == model_plays_p1:
            results["strategic"][key]["wins"] += 1
        else:
            results["strategic"][key]["losses"] += 1

    # 3. Self-play (model vs itself)
    print(f"Evaluating Self-play (model vs itself)...")
    for game_idx in tqdm(range(num_games), desc="Self-play"):
        game = GameState()
        move_count = 0

        while not game.is_over:
            # Use model for both sides
            move = get_model_move(game, use_temperature=(move_count < 5))

            game.make_move(move)
            move_count += 1

        # Record game result
        winner = game.get_winner()
        if winner is None:
            results["self_play"]["draws"] += 1
        elif winner == Player.ONE:
            results["self_play"]["p1_wins"] += 1
        else:
            results["self_play"]["p2_wins"] += 1

    # Print results in a nice table format
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS (Games per matchup: {num_games})")
    print("=" * 70)

    def print_matchup_results(opponent_name, results_dict):
        p1_wins = results_dict["model_as_p1"]["wins"]
        p1_draws = results_dict["model_as_p1"]["draws"]
        p1_losses = results_dict["model_as_p1"]["losses"]
        p1_winrate = (p1_wins / num_games) * 100

        p2_wins = results_dict["model_as_p2"]["wins"]
        p2_draws = results_dict["model_as_p2"]["draws"]
        p2_losses = results_dict["model_as_p2"]["losses"]
        p2_winrate = (p2_wins / num_games) * 100

        print(f"\n{opponent_name} Opponent:")
        print(
            f"  Model as P1: {p1_wins} wins, {p1_draws} draws, {p1_losses} losses ({p1_winrate:.1f}% win rate)"
        )
        print(
            f"  Model as P2: {p2_wins} wins, {p2_draws} draws, {p2_losses} losses ({p2_winrate:.1f}% win rate)"
        )
        print(
            f"  Overall:     {p1_wins + p2_wins} wins, {p1_draws + p2_draws} draws, {p1_losses + p2_losses} losses ({((p1_wins + p2_wins) / (num_games * 2)) * 100:.1f}% win rate)"
        )

    # Print random opponent results
    print_matchup_results("Random", results["random"])

    # Print strategic opponent results
    print_matchup_results("Strategic", results["strategic"])

    # Print self-play results
    p1_wins = results["self_play"]["p1_wins"]
    draws = results["self_play"]["draws"]
    p2_wins = results["self_play"]["p2_wins"]
    p1_rate = (p1_wins / num_games) * 100
    p2_rate = (p2_wins / num_games) * 100

    print("\nSelf-Play:")
    print(
        f"  P1 Wins: {p1_wins} ({p1_rate:.1f}%), Draws: {draws} ({(draws / num_games) * 100:.1f}%), P2 Wins: {p2_wins} ({p2_rate:.1f}%)"
    )

    # Print overall summary
    total_wins = (
        results["random"]["model_as_p1"]["wins"]
        + results["random"]["model_as_p2"]["wins"]
        + results["strategic"]["model_as_p1"]["wins"]
        + results["strategic"]["model_as_p2"]["wins"]
    )
    total_games = num_games * 4  # 2 opponents x 2 sides
    overall_winrate = (total_wins / total_games) * 100

    print("\nOverall Win Rate vs Opponents:")
    print(f"  {total_wins}/{total_games} ({overall_winrate:.1f}%)")
    print("=" * 70 + "\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train model with MCTS integration and value bootstrapping"
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES, help="Episodes per iteration"
    )
    parser.add_argument(
        "--mcts_sims",
        type=int,
        default=DEFAULT_MCTS_SIMS,
        help="Number of MCTS simulations per move",
    )
    parser.add_argument(
        "--mcts_ratio",
        type=float,
        default=DEFAULT_MCTS_RATIO,
        help="Ratio of model moves that use MCTS",
    )
    parser.add_argument(
        "--strategic_ratio",
        type=float,
        default=DEFAULT_STRATEGIC_RATIO,
        help="Ratio of games against strategic opponent",
    )
    parser.add_argument(
        "--random_ratio",
        type=float,
        default=DEFAULT_RANDOM_RATIO,
        help="Ratio of games against random opponent",
    )
    parser.add_argument(
        "--bootstrap",
        type=float,
        default=DEFAULT_BOOTSTRAP_WEIGHT,
        help="Weight for value bootstrapping (0-1)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size"
    )
    parser.add_argument(
        "--load_model",
        type=str,
        default="saved_models/model_latest.pth",
        help="Path to model to load",
    )
    parser.add_argument(
        "--stable_lr",
        action="store_true",
        help="Use stable (slower) learning rate instead of fast mode",
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=DEFAULT_LR_PATIENCE,
        help="Patience for learning rate scheduler",
    )
    parser.add_argument(
        "--lr_reset_factor",
        type=float,
        default=DEFAULT_LR_RESET_FACTOR,
        help="Factor to increase LR when resetting",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with a fresh model, ignoring any existing saved model",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate model instead of training",
    )
    parser.add_argument(
        "--eval_games",
        type=int,
        default=100,
        help="Number of games to play per opponent in evaluation",
    )
    parser.add_argument(
        "--eval_use_mcts",
        action="store_true",
        help="Use MCTS for model moves during evaluation",
    )
    parser.add_argument(
        "--eval_sample",
        action="store_true",
        help="Sample from policy during evaluation instead of taking argmax",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=DEFAULT_SAVE_INTERVAL,
        help=f"Save checkpoints every N iterations (default: {DEFAULT_SAVE_INTERVAL})",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=DEFAULT_NUM_CHECKPOINTS,
        help=f"Number of checkpoints to keep (default: {DEFAULT_NUM_CHECKPOINTS})",
    )

    args = parser.parse_args()

    # Initialize model with appropriate mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = ModelWrapper(device, mode="stable" if args.stable_lr else "fast")
    model = ModelWrapper(device, mode="custom_lr")

    # Load existing model unless --fresh is specified
    if not args.fresh and os.path.exists(args.load_model):
        model.load(args.load_model)
        print("Loaded model from", args.load_model)
        if hasattr(model, "optimizer") and model.optimizer is not None:
            print(f"Current learning rate: {model.optimizer.param_groups[0]['lr']:.8f}")
    else:
        if args.fresh:
            print("Starting with fresh model as requested")
        else:
            print("No model found at", args.load_model, "starting fresh")

    # Evaluation mode
    if args.eval:
        print(f"Evaluating model from {args.load_model}")
        model.model.eval()  # Set model to evaluation mode
        evaluate_model(
            model,
            num_games=args.eval_games,
            mcts_sims=args.mcts_sims,
            use_mcts=args.eval_use_mcts,
            deterministic=not args.eval_sample,
            debug=args.debug,
        )
    else:
        # Training mode
        print(
            f"Training model (starting from {args.load_model if not args.fresh else 'fresh model'})"
        )
        hybrid_training_loop(
            model,
            num_episodes=args.episodes,
            batch_size=args.batch_size,
            mcts_sim_count=args.mcts_sims,
            mcts_game_ratio=args.mcts_ratio,
            initial_random_ratio=INITIAL_RANDOM_OPPONENT_RATIO,
            final_random_ratio=FINAL_RANDOM_OPPONENT_RATIO,
            initial_strategic_ratio=INITIAL_STRATEGIC_OPPONENT_RATIO,
            final_strategic_ratio=FINAL_STRATEGIC_OPPONENT_RATIO,
            transition_iterations=OPPONENT_TRANSITION_ITERATIONS,
            bootstrap_weight=args.bootstrap,
            policy_weight=DEFAULT_POLICY_WEIGHT,
            save_interval=args.save_interval,
            num_checkpoints=args.num_checkpoints,
            debug=args.debug,
        )


# Modify the bootstrap weight to increase with training
def get_bootstrap_weight(iteration):
    """Start with lower bootstrapping and increase gradually"""
    progress = min(1.0, iteration / BOOTSTRAP_TRANSITION_ITERATIONS)
    return BOOTSTRAP_MIN_WEIGHT + progress * (
        BOOTSTRAP_MAX_WEIGHT - BOOTSTRAP_MIN_WEIGHT
    )


# Add this function to improve how value targets are calculated
def get_improved_value_target(game_outcome_value, mcts_root_value, move_count):
    """Create better value targets by blending game outcome with MCTS search value"""
    # How much to mix MCTS value into outcome
    # Early in game: rely more on MCTS evaluation
    # Late in game: rely more on actual outcome
    AVG_GAME_LENGTH = 15.0
    progress = min(1.0, move_count / AVG_GAME_LENGTH)

    # Calculate mixing ratio that changes with game progression
    # Early moves: ~70% MCTS value / 30% outcome
    # Late moves: ~30% MCTS value / 70% outcome
    mcts_weight = 0.7 - (0.4 * progress)

    # Blend the values
    blended_value = (
        1.0 - mcts_weight
    ) * game_outcome_value + mcts_weight * mcts_root_value

    # Safety clamp to [-1, 1] range
    return np.clip(blended_value, -1.0, 1.0)
