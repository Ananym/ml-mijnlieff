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
from mcts import MCTS, add_dirichlet_noise
import random
import math

# Training parameters
DEFAULT_EPISODES = 100
DEFAULT_BATCH_SIZE = 512
DEFAULT_SAVE_INTERVAL = 10
DEFAULT_NUM_CHECKPOINTS = 5
DEFAULT_MCTS_SIMS = 200
DEFAULT_MCTS_RATIO = 1.0
DEFAULT_BUFFER_SIZE = 20000
DEFAULT_POLICY_WEIGHT = 1.0
DEFAULT_NUM_EPOCHS = 32

# Opponent ratio scheduling constants
INITIAL_RANDOM_OPPONENT_RATIO = 0.0
FINAL_RANDOM_OPPONENT_RATIO = 0.0
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.5
FINAL_STRATEGIC_OPPONENT_RATIO = 0.5
OPPONENT_TRANSITION_ITERATIONS = 200

# Scales the 'difficulty' of the strategic opponent by specifying a chance to make a random move
DEFAULT_INITIAL_RANDOM_CHANCE = 0.9
DEFAULT_FINAL_RANDOM_CHANCE = 0.1
DEFAULT_RANDOM_CHANCE_TRANSITION_ITERATIONS = 200

# Bootstrap constants
BOOTSTRAP_MIN_WEIGHT = 0.4
BOOTSTRAP_MAX_WEIGHT = 0.5
BOOTSTRAP_TRANSITION_ITERATIONS = 200

# Add this constant near the top with other constants
DEFAULT_CHECKPOINT_PATH = "saved_models/checkpoint_interrupted.pth"


@dataclass
class TrainingExample:
    state_rep: GameStateRepresentation
    policy: np.ndarray  # One-hot encoded actual move made
    value: float  # Game outcome from current player's perspective
    current_player: Player  # Store which player made the move
    mcts_value: float = None  # Optional MCTS value prediction


def hybrid_training_loop(
    model: ModelWrapper,
    debug: bool = False,
    resume_path: str = None,
):
    """Training loop that combines supervised learning from expert demonstrations
    and reinforcement learning through self-play.
    """
    # Initialize random number generator
    rng = np.random.Generator(np.random.PCG64())

    # Initialize training state
    replay_buffer = []
    running_loss = {
        "total": 1e-8,
        "policy": 1e-8,
        "value": 1e-8,
    }
    running_count = 0
    iteration = 0

    # If resuming from checkpoint, load the saved state
    if resume_path:
        print(f"Resuming training from checkpoint: {resume_path}")
        training_state = model.load_checkpoint(resume_path)

        if "replay_buffer" in training_state:
            replay_buffer = training_state["replay_buffer"]
            print(f"Restored replay buffer with {len(replay_buffer)} examples")

        if "running_loss" in training_state:
            running_loss = training_state["running_loss"]

        if "running_count" in training_state:
            running_count = training_state["running_count"]

        if "iteration" in training_state:
            iteration = training_state["iteration"]
            print(f"Resuming from iteration {iteration}")

        # Re-initialize random number generator with saved state if available
        if "rng_state" in training_state:
            rng = np.random.Generator(np.random.PCG64())
            rng.__setstate__(training_state["rng_state"])

    training_start = time.time()
    strategic_opponent = StrategicOpponent()
    random_opponent = RandomOpponent()
    latest_model_path = "saved_models/model_latest.pth"

    # Enhance interrupt handler to save checkpoint on exit
    interrupt_received = False
    in_training_phase = False  # track whether we're in the training phase

    def handle_interrupt(signum, frame):
        nonlocal interrupt_received
        if interrupt_received:  # Second interrupt, exit immediately
            print("\nForced exit...")
            sys.exit(1)

        print("\nInterrupt received...")

        if in_training_phase:
            print("Currently in training phase, will save after this epoch completes.")
        else:
            print("Stopping the current iteration and saving checkpoint...")

            # Save checkpoint for resuming later
            if not os.path.exists("saved_models"):
                os.makedirs("saved_models")

            # Use the constant here
            checkpoint_path = DEFAULT_CHECKPOINT_PATH
            training_state = {
                "replay_buffer": replay_buffer,
                "running_loss": running_loss,
                "running_count": running_count,
                "iteration": iteration,
                "rng_state": rng.__getstate__(),
            }
            model.save_checkpoint(checkpoint_path, training_state)
            print(f"Saved checkpoint to {checkpoint_path}")
            print(f"Resume with: --resume")  # Simplified message

        interrupt_received = True

    signal.signal(signal.SIGINT, handle_interrupt)

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
        "mcts_policy_confidence_sum": 0,  # sum of max probability in MCTS policy
        "direct_policy_confidence_sum": 0,  # sum of max probability in direct policy
        "mcts_policy_entropy_sum": 0,  # sum of MCTS policy entropy
        "direct_policy_entropy_sum": 0,  # sum of direct policy entropy
        "value_prediction_sum": 0,  # sum of value predictions
        "value_prediction_squared_sum": 0,  # sum of squared value predictions
        "value_prediction_abs_sum": 0,  # sum of absolute value predictions
        "value_prediction_count": 0,  # count of value predictions
        "value_error_sum": 0,  # sum of |actual_outcome - predicted_value|
        "extreme_value_count": 0,  # count of values near -1 or 1 (confident predictions)
    }

    # Keep track of last N checkpoints
    checkpoint_files = []

    def get_adjusted_value(game, winner, move_count, current_player):
        if winner is None:
            return 0.0
        elif winner == current_player:
            return 1.0
        else:
            return -1.0

    def create_opponent_example(game, move, opponent_type):
        """Create a training example from an opponent's move"""
        # Use subjective=True for the state representation
        state_rep = game.get_game_state_representation(subjective=True)
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

    def get_temperature(move_count, iteration, max_iterations=150):
        """Dynamic temperature with extended exploration and smoother decay.

        This implementation:
        1. Maintains the same move-based temperature structure
        2. Extends high exploration phase (through ~iteration 60)
        3. Creates smoother decay using a cosine schedule
        4. Reaches slightly lower final temperature for better exploitation
        """
        # Move-based temperature (same as before)
        if move_count < 4:
            base_temp = 1.5
        elif move_count < 8:
            base_temp = 1.2
        elif move_count < 12:
            base_temp = 1.0
        else:
            base_temp = 0.8

        # Calculate iteration progress (0 to 1)
        progress = min(1.0, iteration / max_iterations)

        # Three-phase temperature schedule:
        # 1. Maintain high exploration for first 40% of training
        # 2. Cosine decay during middle 50% of training
        # 3. Low, stable temperature for final 10% of training
        if progress < 0.4:
            # Early phase: high exploration (0.9-1.0 factor)
            iteration_factor = 1.0 - (progress * 0.25)  # Slight decay from 1.0 to 0.9
        elif progress < 0.9:
            # Middle phase: cosine decay from 0.9 to 0.4
            # Normalize progress to 0-1 within this phase
            phase_progress = (progress - 0.4) / 0.5
            # Cosine decay provides smoother transition
            iteration_factor = (
                0.9 - 0.5 * (1 - math.cos(phase_progress * math.pi)) * 0.5
            )
        else:
            # Final phase: low exploration for fine-tuning (0.4 factor)
            iteration_factor = 0.4

        return base_temp * iteration_factor

    def get_model_move_with_policy(game, use_mcts=True, temperature=1.0, iteration=0):
        """Get a move from the model along with the training policy target
        (full MCTS distribution if MCTS was used, otherwise one-hot)"""
        legal_moves = game.get_legal_moves()
        # Use subjective=True for the state representation
        state_rep = game.get_game_state_representation(subjective=True)
        move_count = game.move_count

        if use_mcts:
            # Use model-guided MCTS (AlphaZero style)
            # Noise will be applied inside the MCTS search
            mcts = MCTS(
                model=model,
                num_simulations=DEFAULT_MCTS_SIMS,
                c_puct=1.0,
            )
            mcts.set_temperature(temperature)
            mcts.set_iteration(iteration)  # Set current iteration for noise
            mcts.set_move_count(move_count)  # Set move count for noise
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

            # Select move based on the MCTS policy
            policy_flat = policy.flatten()
            if temperature > 0:
                # Sample from the probability distribution
                move_idx = rng.choice(len(policy_flat), p=policy_flat)
            else:
                # Choose move deterministically
                move_idx = policy_flat.argmax()

            # Convert flat index to 3D coordinates
            move_coords = np.unravel_index(move_idx, policy.shape)

            # Calculate MCTS policy confidence metrics
            # Max probability in the policy distribution
            mcts_policy_confidence = np.max(policy)

            # Calculate policy entropy: -sum(p_i * log(p_i))
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            mcts_policy_entropy = -np.sum(policy * np.log(policy + epsilon))

            # Update statistics
            iter_stats["mcts_policy_confidence_sum"] += mcts_policy_confidence
            iter_stats["mcts_policy_entropy_sum"] += mcts_policy_entropy

        else:
            # Use direct policy from neural network
            policy, value_pred = model.predict(
                state_rep.board, state_rep.flat_values, legal_moves
            )
            policy = policy.squeeze(0)
            root_value = value_pred.squeeze(0)[0]

            # Apply legal moves mask and add Dirichlet noise for both players
            masked_policy = policy * legal_moves
            policy_flat = masked_policy.flatten()
            legal_flat = legal_moves.flatten()

            # Apply same dynamic Dirichlet noise for direct moves as in MCTS
            # We want exploration from both player perspectives
            policy_flat = add_dirichlet_noise(
                policy_flat, legal_flat, iteration, move_count, max_iterations=150
            )

            # Renormalize after applying noise
            policy_sum = np.sum(policy_flat)
            if policy_sum > 0:
                policy_flat = policy_flat / policy_sum
            masked_policy = policy_flat.reshape(masked_policy.shape)

            # Select move based on temperature
            if temperature > 0:
                # Sample from the probability distribution
                policy_flat = masked_policy.flatten()
                if np.sum(policy_flat) > 0:  # Ensure valid distribution
                    move_idx = rng.choice(len(policy_flat), p=policy_flat)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)
                else:
                    # Fallback to random selection among legal moves
                    legal_indices = np.where(legal_flat > 0)[0]
                    move_idx = rng.choice(legal_indices)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)
            else:
                # Choose move deterministically
                move_coords = np.unravel_index(
                    masked_policy.argmax(), masked_policy.shape
                )

            # For direct policy, use one-hot encoding as the target
            policy_target = np.zeros_like(masked_policy)
            policy_target[move_coords] = 1.0

            # Calculate direct policy confidence metrics
            # Get raw policy before masking and noise for confidence calculation
            raw_policy = policy.copy()
            direct_policy_confidence = np.max(raw_policy)

            # Calculate policy entropy on the raw policy
            epsilon = 1e-10
            direct_policy_entropy = -np.sum(raw_policy * np.log(raw_policy + epsilon))

            # Update statistics
            iter_stats["direct_policy_confidence_sum"] += direct_policy_confidence
            iter_stats["direct_policy_entropy_sum"] += direct_policy_entropy

        # Create the move object
        move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))

        # Track value prediction statistics
        iter_stats["value_prediction_sum"] += root_value
        iter_stats["value_prediction_squared_sum"] += root_value * root_value
        iter_stats["value_prediction_abs_sum"] += abs(root_value)
        iter_stats["value_prediction_count"] += 1
        if abs(root_value) > 0.8:  # Track highly confident value predictions
            iter_stats["extreme_value_count"] += 1

        return move, policy_target, root_value, state_rep

    # Function to update opponent ratios based on current iteration
    def get_opponent_ratios(iteration):
        progress = min(1.0, iteration / OPPONENT_TRANSITION_ITERATIONS)

        # Linear interpolation between initial and final values
        random_ratio = INITIAL_RANDOM_OPPONENT_RATIO + progress * (
            FINAL_RANDOM_OPPONENT_RATIO - INITIAL_RANDOM_OPPONENT_RATIO
        )
        strategic_ratio = INITIAL_STRATEGIC_OPPONENT_RATIO + progress * (
            FINAL_STRATEGIC_OPPONENT_RATIO - INITIAL_STRATEGIC_OPPONENT_RATIO
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

    # Function to calculate improved value target using bootstrapping
    def get_improved_value_target(outcome_value, mcts_value, move_index):
        """Blend the final outcome with MCTS predicted value for better training signal"""
        # Early moves use more bootstrapping, later moves use more outcome
        bootstrap_factor = max(0.2, 1.0 - (move_index / 30))
        return (1 - bootstrap_factor) * outcome_value + bootstrap_factor * mcts_value

    def get_random_chance(iteration):
        """Calculate the random chance for strategic opponent based on current iteration.

        Linearly decreases from DEFAULT_INITIAL_RANDOM_CHANCE to DEFAULT_FINAL_RANDOM_CHANCE
        over DEFAULT_RANDOM_CHANCE_TRANSITION_ITERATIONS iterations.
        """
        if iteration >= DEFAULT_RANDOM_CHANCE_TRANSITION_ITERATIONS:
            return DEFAULT_FINAL_RANDOM_CHANCE

        progress = iteration / DEFAULT_RANDOM_CHANCE_TRANSITION_ITERATIONS
        return DEFAULT_INITIAL_RANDOM_CHANCE - progress * (
            DEFAULT_INITIAL_RANDOM_CHANCE - DEFAULT_FINAL_RANDOM_CHANCE
        )

    print("Starting hybrid training loop with value bootstrapping")
    print("Episodes per iteration:", DEFAULT_EPISODES)
    print("MCTS simulations per move:", DEFAULT_MCTS_SIMS)
    print("MCTS move ratio: {:.1%}".format(DEFAULT_MCTS_RATIO))
    print("Initial opponent ratios:")
    print("  Random: {:.1%}".format(random_opponent_ratio))
    print("  Strategic: {:.1%}".format(strategic_opponent_ratio))
    print(
        "  Self-play: {:.1%}".format(
            1.0 - random_opponent_ratio - strategic_opponent_ratio
        )
    )
    print("Initial bootstrap weight: {:.2f}".format(get_bootstrap_weight(iteration)))
    print("Final bootstrap weight: {:.2f}".format(BOOTSTRAP_MAX_WEIGHT))
    print("Batch size:", DEFAULT_BATCH_SIZE)
    print("Replay buffer size:", DEFAULT_BUFFER_SIZE)
    print("Policy weight: {:.1f}".format(DEFAULT_POLICY_WEIGHT))
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
                range(DEFAULT_EPISODES),
                desc=f"Games (Iter {iteration})",
                leave=False,
            )

            # Generate games with various opponents
            for _ in range(DEFAULT_EPISODES):
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
                        use_mcts = rng.random() < DEFAULT_MCTS_RATIO
                        if use_mcts:
                            iter_stats["mcts_moves"] += 1
                        else:
                            iter_stats["direct_moves"] += 1

                        # Use temperature based on move count
                        temperature = get_temperature(move_count, iteration)

                        # Get model's move with AlphaZero-style policy
                        move, policy_target, value_pred, state_rep = (
                            get_model_move_with_policy(
                                game,
                                use_mcts=use_mcts,
                                temperature=temperature,
                                iteration=iteration,
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
                        if opponent_type == "strategic":
                            # Get random chance based on current iteration
                            random_chance = get_random_chance(iteration)
                            move = opponent.get_move(game, random_chance)
                        else:
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
                                # Track value prediction error before applying blending
                                iter_stats["value_error_sum"] += abs(
                                    outcome_value - example.mcts_value
                                )

                                example.value = get_improved_value_target(
                                    outcome_value, example.mcts_value, i
                                )
                            else:
                                # For non-MCTS moves, use standard bootstrapping
                                # The next player made a move, so get their perspective's value prediction
                                if i + 1 < len(examples):
                                    next_example = examples[i + 1]
                                    next_state_rep = next_example.state_rep

                                    # Get a fresh value prediction for the next state
                                    _, next_value = model.predict(
                                        next_state_rep.board, next_state_rep.flat_values
                                    )

                                    # Negate since it's from opponent's perspective
                                    bootstrap_value = -float(next_value[0][0])

                                    # Track value prediction error for bootstrapped values too
                                    iter_stats["value_error_sum"] += abs(
                                        outcome_value - bootstrap_value
                                    )

                                    # Mix the outcome with bootstrapped value
                                    example.value = (
                                        (1 - bootstrap_weight) * outcome_value
                                        + bootstrap_weight * bootstrap_value
                                    )
                                else:
                                    # Last move - just use outcome
                                    example.value = outcome_value

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

                # Add directly to replay buffer - no conversion needed
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
            if len(replay_buffer) > DEFAULT_BUFFER_SIZE:
                indices = rng.choice(
                    len(replay_buffer), size=DEFAULT_BUFFER_SIZE, replace=False
                )
                replay_buffer = [replay_buffer[i] for i in indices]

                # Log only buffer size after trimming
                tqdm.write(f"Trimmed buffer size: {len(replay_buffer)}")

            # Training phase
            train_pbar = tqdm(
                range(DEFAULT_NUM_EPOCHS), desc="Training Epochs", leave=False
            )
            epoch_losses = {"total": 0, "policy": 0, "value": 0}

            in_training_phase = True  # Set flag to indicate we're in training

            for _ in train_pbar:
                # Sample from replay buffer
                indices = rng.choice(
                    len(replay_buffer),
                    size=min(DEFAULT_BATCH_SIZE, len(replay_buffer)),
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
                    policy_weight=DEFAULT_POLICY_WEIGHT,
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
            avg_total = epoch_losses["total"] / DEFAULT_NUM_EPOCHS
            avg_policy = epoch_losses["policy"] / DEFAULT_NUM_EPOCHS
            avg_value = epoch_losses["value"] / DEFAULT_NUM_EPOCHS

            # Step the scheduler once per iteration
            model.scheduler.step()

            in_training_phase = False  # Reset flag after training phase

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
            self_play_p1_rate = (
                100
                * iter_stats["self_play_p1_wins"]
                / max(1, iter_stats["self_play_games"])
            )

            # Calculate move statistics and policy metrics
            mcts_ratio = iter_stats["mcts_moves"] / max(1, iter_stats["total_moves"])
            direct_ratio = iter_stats["direct_moves"] / max(
                1, iter_stats["total_moves"]
            )

            # Calculate average policy metrics
            avg_mcts_confidence = iter_stats["mcts_policy_confidence_sum"] / max(
                1, iter_stats["mcts_moves"]
            )
            avg_direct_confidence = iter_stats["direct_policy_confidence_sum"] / max(
                1, iter_stats["direct_moves"]
            )
            avg_mcts_entropy = iter_stats["mcts_policy_entropy_sum"] / max(
                1, iter_stats["mcts_moves"]
            )
            avg_direct_entropy = iter_stats["direct_policy_entropy_sum"] / max(
                1, iter_stats["direct_moves"]
            )

            # Calculate value prediction statistics
            if iter_stats["value_prediction_count"] > 0:
                avg_value_prediction = (
                    iter_stats["value_prediction_sum"]
                    / iter_stats["value_prediction_count"]
                )
                avg_abs_value = (
                    iter_stats["value_prediction_abs_sum"]
                    / iter_stats["value_prediction_count"]
                )
                value_variance = (
                    iter_stats["value_prediction_squared_sum"]
                    / iter_stats["value_prediction_count"]
                ) - (avg_value_prediction * avg_value_prediction)
                value_std = math.sqrt(max(0, value_variance))
                extreme_value_ratio = (
                    iter_stats["extreme_value_count"]
                    / iter_stats["value_prediction_count"]
                )
                avg_value_error = (
                    iter_stats["value_error_sum"] / iter_stats["value_prediction_count"]
                    if iter_stats["value_error_sum"] > 0
                    else 0
                )
            else:
                avg_value_prediction = avg_abs_value = value_std = (
                    extreme_value_ratio
                ) = avg_value_error = 0

            # Print summary
            print(
                f"\nIteration {iteration} completed in {time.time() - iteration_start:.1f}s "
                f"({int(hours)}h {int(minutes)}m total) | "
                f"Buffer: {len(replay_buffer)} | "
                f"Loss: {avg_total:.4f} (p:{avg_policy:.4f}, v:{avg_value:.4f}) | "
                f"LR: {model.optimizer.param_groups[0]['lr']:.6f}"  # current learning rate
            )

            # Enhanced winrate reporting
            print(
                f"Win Rates - Strategic: P1 {iter_stats['strategic_wins_as_p1']}/{iter_stats['strategic_games_as_p1']} ({strategic_p1_rate:.1f}%), "
                f"P2 {iter_stats['strategic_wins_as_p2']}/{iter_stats['strategic_games_as_p2']} ({strategic_p2_rate:.1f}%)"
            )
            print(
                f"Win Rates - Random: P1 {iter_stats['random_wins_as_p1']}/{iter_stats['random_games_as_p1']} ({random_p1_rate:.1f}%), "
                f"P2 {iter_stats['random_wins_as_p2']}/{iter_stats['random_games_as_p2']} ({random_p2_rate:.1f}%)"
            )
            print(
                f"Win Rates - Self-play: P1 {iter_stats['self_play_p1_wins']}/{iter_stats['self_play_games']} ({self_play_p1_rate:.1f}%)"
            )

            print(
                f"Moves - MCTS: {iter_stats['mcts_moves']} ({mcts_ratio:.2f}), "
                f"Direct: {iter_stats['direct_moves']} ({direct_ratio:.2f})"
            )
            print(
                f"Policy confidence - MCTS: {avg_mcts_confidence:.3f}, Direct: {avg_direct_confidence:.3f} | "  # max prob in distribution
                f"Entropy - MCTS: {avg_mcts_entropy:.3f}, Direct: {avg_direct_entropy:.3f}"  # uncertainty in policy distribution
            )
            print(
                f"Examples - Win: {iter_stats['win_examples'] / iter_stats['total_games']:.2f}, "
                f"Loss: {iter_stats['loss_examples'] / iter_stats['total_games']:.2f}, "
                f"Draw: {iter_stats['draw_examples'] / iter_stats['total_games']:.2f}"
            )
            print(
                f"Value - Avg: {avg_value_prediction:.3f}, |Avg|: {avg_abs_value:.3f}, "
                f"Std: {value_std:.3f}, Conf: {extreme_value_ratio:.3f}, "  # ratio of confident predictions (>0.8)
                f"Error: {avg_value_error:.3f}"  # avg absolute difference between predicted & actual value
            )

            # Print opponent ratios
            print("Opponent ratios for this iteration:")
            print("  Random: {:.1%}".format(random_opponent_ratio))
            print("  Strategic: {:.1%}".format(strategic_opponent_ratio))
            print(
                "  Self-play: {:.1%}".format(
                    1.0 - random_opponent_ratio - strategic_opponent_ratio
                )
            )
            print(
                "Strategic opponent random chance: {:.1%}".format(
                    get_random_chance(iteration)
                )
            )
            print()

            # Print final random chance at the end of training
            print("Final opponent ratios:")
            print("  Random: {:.1%}".format(FINAL_RANDOM_OPPONENT_RATIO))
            print("  Strategic: {:.1%}".format(FINAL_STRATEGIC_OPPONENT_RATIO))
            print(
                "  Self-play: {:.1%}".format(
                    1.0 - FINAL_RANDOM_OPPONENT_RATIO - FINAL_STRATEGIC_OPPONENT_RATIO
                )
            )
            print(
                "Final strategic opponent random chance: {:.1%}".format(
                    DEFAULT_FINAL_RANDOM_CHANCE
                )
            )
            print()

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    print("\nTraining complete!")

    # Final save
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")
    final_path = "saved_models/model_final.pth"
    model.save_checkpoint(final_path)  # Use save_checkpoint instead of save
    print(f"Saved final model to {final_path}")

    return model


if __name__ == "__main__":
    # Parse command line arguments
    mode = "stable"  # default mode
    load_path = None
    resume_path = None
    device = None  # Don't set a default yet

    # Check for device, fast mode, and load/resume path
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--device="):
            device = arg.split("=")[1]
        elif arg == "--fast":
            mode = "fast"
        elif arg.startswith("--load="):
            load_path = arg.split("=")[1]
        elif arg.startswith("--resume="):
            resume_path = arg.split("=")[1]
        elif arg == "--resume":
            # Use default checkpoint path when just --resume is specified
            resume_path = DEFAULT_CHECKPOINT_PATH
        i += 1

    # Determine device (auto-select CUDA if available)
    if os.environ.get("FORCE_CPU") == "1":
        print("FORCE_CPU environment variable set, using CPU")
        device = "cpu"
    elif device is None:  # No explicit device specified
        try:
            import torch.cuda

            if torch.cuda.is_available():
                device = "cuda"
                print("CUDA is available, using GPU")
            else:
                device = "cpu"
                print("CUDA not available, using CPU")
        except ImportError:
            device = "cpu"
            print("torch.cuda not available, using CPU")
    # else: use the explicitly specified device

    print(f"Using device: {device}")

    # Initialize model with appropriate mode
    model = ModelWrapper(device=device, mode=mode)

    # Handle loading vs resuming
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        # Note: the model loading happens inside the training loop
    elif load_path:
        print(f"Loading model weights from {load_path}")
        model.load(load_path)

    # Start training
    hybrid_training_loop(model, resume_path=resume_path)
