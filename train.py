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
import math
from collections import defaultdict
from eval_model import policy_vs_mcts_eval
import torch.nn.functional as F

DEFAULT_EPISODES = 150  # increased from 100 for more data per iteration
DEFAULT_BATCH_SIZE = 256  # reduced from 512 - better for smaller model
DEFAULT_SAVE_INTERVAL = 5
DEFAULT_NUM_CHECKPOINTS = 5
DEFAULT_MCTS_RATIO = 1.0
DEFAULT_BUFFER_SIZE = 4000  # reduced from 5000 to match smaller model capacity
DEFAULT_POLICY_WEIGHT = 0.8  # reduced from 1.0 - balance policy and value better
DEFAULT_NUM_EPOCHS = 20  # reduced from 50 - prevent overfitting
DEFAULT_EVAL_INTERVAL = 20
DEFAULT_EVAL_GAMES = 60
DEFAULT_STRATEGIC_EVAL_GAMES = 60
DEFAULT_MIN_MCTS_SIMS = 100  # increased from 50 for better early training signal
DEFAULT_MAX_MCTS_SIMS = 200
MAX_ITERATIONS = 100

BALANCE_REPLAY_BUFFER = False

DIRICHLET_SCALE = 0.15  # Moderate exploration noise (was 0.0)
ENTROPY_BONUS_SCALE = 0.05  # Small entropy bonus to encourage exploration (was 0.0)


INITIAL_RANDOM_OPPONENT_RATIO = 0.0  # Reduced from 0.1
FINAL_RANDOM_OPPONENT_RATIO = 0.0
INITIAL_STRATEGIC_OPPONENT_RATIO = 0.0  # Reduced from 0.2
FINAL_STRATEGIC_OPPONENT_RATIO = 0.0  # Reduced from 0.3
OPPONENT_TRANSITION_ITERATIONS = MAX_ITERATIONS

DEFAULT_INITIAL_RANDOM_CHANCE = 0.0
DEFAULT_FINAL_RANDOM_CHANCE = 0.0
DEFAULT_RANDOM_CHANCE_TRANSITION_ITERATIONS = MAX_ITERATIONS

BOOTSTRAP_MIN_WEIGHT = 0.0  # Start with modest bootstrapping
BOOTSTRAP_MAX_WEIGHT = 0.0  # Increase to a moderate maximum
BOOTSTRAP_TRANSITION_ITERATIONS = MAX_ITERATIONS

DEFAULT_CHECKPOINT_PATH = "saved_models/checkpoint_interrupted.pth"


def log_iteration_report(
    iteration,
    iteration_start_time,
    training_start_time,
    model,
    replay_buffer,
    iter_stats,
    loss_stats,
    grad_stats=None,
    value_metrics=None,
):
    """
    Generate a consolidated report for the current iteration (simplified)

    Args:
        iteration: Current iteration number
        iteration_start_time: Time when iteration started
        training_start_time: Time when overall training started
        model: Model being trained
        replay_buffer: Current replay buffer
        iter_stats: Statistics collected during the iteration
        loss_stats: Loss statistics from training
        grad_stats: Gradient statistics (optional)
        value_metrics: Value prediction metrics (optional)
    """
    elapsed_time = time.time() - training_start_time
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    iteration_time = time.time() - iteration_start_time

    # extract loss statistics
    avg_total = loss_stats.get("total", 0)
    avg_policy = loss_stats.get("policy", 0)
    avg_value = loss_stats.get("value", 0)

    # extract win rate statistics
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
        100 * iter_stats["random_wins_as_p1"] / max(1, iter_stats["random_games_as_p1"])
    )
    random_p2_rate = (
        100 * iter_stats["random_wins_as_p2"] / max(1, iter_stats["random_games_as_p2"])
    )
    self_play_p1_wins = iter_stats["self_play_p1_wins"]
    self_play_games = iter_stats["self_play_games"]
    self_play_p1_rate = (
        100 * self_play_p1_wins / max(1, self_play_games) if self_play_games > 0 else 0
    )
    # calculate self-play draw rate (total - wins) / total
    self_play_draws = iter_stats.get(
        "self_play_draws", 0
    )  # add self_play_draws to stats
    self_play_draw_rate = (
        100 * self_play_draws / max(1, self_play_games) if self_play_games > 0 else 0
    )
    self_play_p2_rate = 100.0 - self_play_p1_rate - self_play_draw_rate

    # get policy confidence metric
    avg_mcts_confidence = iter_stats.get("avg_mcts_confidence", 0)

    # extract value prediction metrics
    avg_value_prediction = (
        value_metrics.get("avg_value_prediction", 0) if value_metrics else 0
    )
    avg_abs_value = value_metrics.get("avg_abs_value", 0) if value_metrics else 0
    value_std = value_metrics.get("value_std", 0) if value_metrics else 0

    # get cache performance
    total_predictions = iter_stats["cache_hits"] + iter_stats["cache_misses"]
    cache_hit_rate = (
        (iter_stats["cache_hits"] / total_predictions * 100)
        if total_predictions > 0
        else 0
    )

    # extract correlation metrics if available
    pearson_correlation = (
        value_metrics.get("pearson_correlation", 0) if value_metrics else 0
    )

    # count win/loss/draw samples in buffer
    buffer_win_samples = 0
    buffer_loss_samples = 0
    buffer_draw_samples = 0

    for example in replay_buffer:
        if example.value > 0.2:
            buffer_win_samples += 1
        elif example.value < -0.2:
            buffer_loss_samples += 1
        else:
            buffer_draw_samples += 1

    buffer_size = len(replay_buffer)
    buffer_win_pct = (buffer_win_samples / max(1, buffer_size)) * 100
    buffer_loss_pct = (buffer_loss_samples / max(1, buffer_size)) * 100
    buffer_draw_pct = (buffer_draw_samples / max(1, buffer_size)) * 100

    # Build the consolidated report
    print(f"\n{'='*35} ITERATION {iteration} REPORT {'='*35}")
    print(
        f"Time: {int(hours)}h {int(minutes)}m total | {iteration_time:.1f}s for iteration"
    )

    # Model section
    print(f"\n--- MODEL METRICS ---")
    print(f"Loss: {avg_total:.4f} (policy:{avg_policy:.4f}, value:{avg_value:.4f})")
    print(f"Learning rate: {model.optimizer.param_groups[0]['lr']:.6f}")

    # Value prediction quality section (condensed)
    print(f"\n--- VALUE PREDICTION ---")
    print(
        f"Distribution: mean={avg_value_prediction:.4f}, |mean|={avg_abs_value:.4f}, std={value_std:.4f}"
    )
    print(f"Outcome correlation (Pearson): {pearson_correlation:.4f}")

    # Performance section
    print(f"\n--- PERFORMANCE ---")
    print(
        f"Self-play: P1 win {self_play_p1_rate:.1f}%, P2 win {self_play_p2_rate:.1f}%, Draw {self_play_draw_rate:.1f}%"
    )
    print(
        f"Strategic: P1 win {strategic_p1_rate:.1f}%, P2 win {strategic_p2_rate:.1f}%"
    )
    print(f"Random: P1 win {random_p1_rate:.1f}%, P2 win {random_p2_rate:.1f}%")

    # Decision quality
    print(f"\n--- DECISION QUALITY ---")
    print(f"MCTS Confidence (avg max policy): {avg_mcts_confidence:.3f}")

    # Buffer stats
    print(f"\n--- TRAINING DATA ---")
    print(
        f"Buffer: {buffer_size} examples (win: {buffer_win_pct:.1f}%, loss: {buffer_loss_pct:.1f}%, draw: {buffer_draw_pct:.1f}%)"
    )
    print(
        f"Current iter added: {iter_stats['win_examples']} win, {iter_stats['loss_examples']} loss, {iter_stats['draw_examples']} draw"
    )
    print(f"Cache hit rate: {cache_hit_rate:.1f}%")

    # Gradient health
    if grad_stats and isinstance(grad_stats, dict) and "status" not in grad_stats:
        print(f"\n--- GRADIENT HEALTH ---")

        # For our new model structure, check value_fc4 gradients
        if (
            "value_fc3_weight" in grad_stats
        ):  # Assuming model structure has fc3 as final value layer
            fc3_weight = grad_stats.get("value_fc3_weight", {})
            fc3_weight_max_abs = max(
                abs(fc3_weight.get("min", 0)), abs(fc3_weight.get("max", 0))
            )
            fc3_weight_mean_abs = abs(fc3_weight.get("mean", 0))

            gradient_status = "HEALTHY"
            if fc3_weight_max_abs > 1.0:
                gradient_status = "⚠️ EXPLODING"
            elif fc3_weight_max_abs < 0.0001:
                gradient_status = "⚠️ VANISHING"

            print(
                f"Value head final layer - max abs grad: {fc3_weight_max_abs:.6f}, mean abs grad: {fc3_weight_mean_abs:.6f}"
            )
            print(f"Gradient status: {gradient_status}")

    print(f"\n{'='*90}")


@dataclass
class TrainingExample:
    state_rep: GameStateRepresentation
    policy: np.ndarray  # One-hot encoded actual move made
    value: float  # Game outcome from current player's perspective
    current_player: Player  # Store which player made the move
    move_count: Optional[int] = None  # Current move count when example was created
    mcts_value: Optional[float] = None  # Optional MCTS value prediction


def inspect_value_head_gradients(model):
    """Inspect gradients flowing into value head layers"""

    # gather gradients from value head layers
    gradient_stats = {}

    # check if the model has gradients yet
    if model.model.value_fc3.weight.grad is None:
        return {"status": "no gradients available yet"}

    # final value layer stats
    fc3_weight_grad = model.model.value_fc3.weight.grad
    fc3_bias_grad = model.model.value_fc3.bias.grad

    # penultimate value layer stats
    fc2_weight_grad = model.model.value_fc2.weight.grad
    fc2_bias_grad = model.model.value_fc2.bias.grad

    # compute statistics
    gradient_stats["value_fc3_weight"] = {
        "mean": float(fc3_weight_grad.mean().item()),
        "std": float(fc3_weight_grad.std().item()),
        "min": float(fc3_weight_grad.min().item()),
        "max": float(fc3_weight_grad.max().item()),
        "norm": float(fc3_weight_grad.norm().item()),
    }

    gradient_stats["value_fc3_bias"] = {
        "mean": float(fc3_bias_grad.mean().item()),
        "std": float(fc3_bias_grad.std().item()),
        "min": float(fc3_bias_grad.min().item()),
        "max": float(fc3_bias_grad.max().item()),
        "norm": float(fc3_bias_grad.norm().item()),
    }

    gradient_stats["value_fc2_weight"] = {
        "mean": float(fc2_weight_grad.mean().item()),
        "std": float(fc2_weight_grad.std().item()),
        "min": float(fc2_weight_grad.min().item()),
        "max": float(fc2_weight_grad.max().item()),
        "norm": float(fc2_weight_grad.norm().item()),
    }

    gradient_stats["value_fc2_bias"] = {
        "mean": float(fc2_bias_grad.mean().item()),
        "std": float(fc2_bias_grad.std().item()),
        "min": float(fc2_bias_grad.min().item()),
        "max": float(fc2_bias_grad.max().item()),
        "norm": float(fc2_bias_grad.norm().item()),
    }

    return gradient_stats


def training_loop(
    model: ModelWrapper,
    resume_path: str = None,
):
    rng = np.random.Generator(np.random.PCG64())

    replay_buffer = []
    running_loss = {
        "total": 1e-8,
        "policy": 1e-8,
        "value": 1e-8,
    }
    running_count = 0
    iteration = 0

    stats_template = {
        "win_rate": 0.0,
        "loss_rate": 0.0,
        "draw_rate": 0.0,
        "value_error_sum": 0.0,
        "value_error_count": 0,
    }

    win_stats = {k: {**stats_template} for k in ["self", "strategic", "random"]}
    strategic_opponent = StrategicOpponent()
    random_opponent = RandomOpponent()
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("saved_plots", exist_ok=True)

    checkpoint_path = DEFAULT_CHECKPOINT_PATH

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

    # Start the training timer
    training_start = time.time()

    # Base stats template - simplified
    stats_template = {
        # game outcome stats
        "strategic_wins_as_p1": 0,
        "strategic_wins_as_p2": 0,
        "strategic_games_as_p1": 0,
        "strategic_games_as_p2": 0,
        "random_wins_as_p1": 0,
        "random_wins_as_p2": 0,
        "random_games_as_p1": 0,
        "random_games_as_p2": 0,
        "self_play_p1_wins": 0,
        "self_play_draws": 0,  # track draws for win rate calculation
        "self_play_games": 0,
        "total_games": 0,
        # buffer content stats
        "win_examples": 0,
        "loss_examples": 0,
        "draw_examples": 0,
        # mcts/value stats
        "mcts_policy_confidence_sum": 0,
        "mcts_moves": 0,  # count mcts moves for averaging confidence
        "value_prediction_sum": 0,
        "value_prediction_squared_sum": 0,  # needed for std dev
        "value_prediction_abs_sum": 0,
        "value_prediction_count": 0,
        # correlation stats
        "value_actual_pred_products": 0,  # needed for pearson correlation
        "value_actual_squared": 0,  # needed for pearson correlation
        "value_pred_squared": 0,  # needed for pearson correlation
        # cache stats
        "cache_hits": 0,
        "cache_misses": 0,
    }

    def get_adjusted_value(game, winner, current_player):
        """Calculate training value target based purely on score differential.

        Args:
            game: Completed game state
            winner: Winner of the game (unused in this simplified version)
            current_player: Player from whose perspective we calculate value

        Returns:
            Normalized value in [-1, 1] range based on score differential
        """
        # Calculate score differential from current player's perspective
        scores = game.get_scores()
        player_score = scores[current_player]
        opponent = Player.ONE if current_player == Player.TWO else Player.TWO
        opponent_score = scores[opponent]
        score_diff = player_score - opponent_score

        # Clamp the score differential to +/- 6
        clamped_diff = max(-6.0, min(6.0, score_diff))

        # Scale to [-1, 1] range
        normalized_value = clamped_diff / 6.0

        return normalized_value

    def create_opponent_example(game, move):
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
            move_count=game.move_count,
        )

    def get_temperature(move_count, iteration, max_iterations=150):
        # too involved of a system - avoid this as a confounding factor for now
        return 1.0
        # """Dynamic temperature with extended exploration and smoother decay.

        # This implementation:
        # 1. Maintains the same move-based temperature structure
        # 2. Extends high exploration phase (through ~iteration 60)
        # 3. Creates smoother decay using a cosine schedule
        # 4. Reaches slightly lower final temperature for better exploitation
        # """
        # # Move-based temperature (same as before)
        # if move_count < 4:
        #     base_temp = 1.5
        # elif move_count < 8:
        #     base_temp = 1.2
        # elif move_count < 12:
        #     base_temp = 1.0
        # else:
        #     base_temp = 0.8

        # # Calculate iteration progress (0 to 1)
        # progress = min(1.0, iteration / max_iterations)

        # # Three-phase temperature schedule:
        # # 1. Maintain high exploration for first 40% of training
        # # 2. Cosine decay during middle 50% of training
        # # 3. Low, stable temperature for final 10% of training
        # if progress < 0.4:
        #     # Early phase: high exploration (0.9-1.0 factor)
        #     iteration_factor = 1.0 - (progress * 0.25)  # Slight decay from 1.0 to 0.9
        # elif progress < 0.9:
        #     # Middle phase: cosine decay from 0.9 to 0.4
        #     # Normalize progress to 0-1 within this phase
        #     phase_progress = (progress - 0.4) / 0.5
        #     # Cosine decay provides smoother transition
        #     iteration_factor = (
        #         0.9 - 0.5 * (1 - math.cos(phase_progress * math.pi)) * 0.5
        #     )
        # else:
        #     # Final phase: low exploration for fine-tuning (0.4 factor)
        #     iteration_factor = 0.4

        # # Apply global temperature scale to reduce exploration/noise
        # return base_temp * iteration_factor * TEMPERATURE_SCALE

    def get_mcts_sims_for_iteration(iteration):
        """Scale MCTS simulations from min to max by final iteration"""
        # Linear scaling based on iteration progress
        progress = min(1.0, iteration / MAX_ITERATIONS)
        sims = DEFAULT_MIN_MCTS_SIMS + progress * (
            DEFAULT_MAX_MCTS_SIMS - DEFAULT_MIN_MCTS_SIMS
        )
        # Return as integer
        return int(sims)

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
                num_simulations=get_mcts_sims_for_iteration(iteration),
                c_puct=1.5,  # increased from 1.0 for more exploration
            )
            mcts.set_temperature(temperature)
            mcts.set_iteration(iteration)  # Set current iteration for noise
            mcts.set_move_count(move_count)  # Set move count for noise
            mcts_policy, root_node = mcts.search(game)
            policy = mcts_policy

            # Track cache statistics
            iter_stats["cache_hits"] += mcts.cache_hits
            iter_stats["cache_misses"] += mcts.cache_misses
            # Reset MCTS cache stats for next use
            mcts.cache_hits = 0
            mcts.cache_misses = 0

            # Use full MCTS distribution as policy target when MCTS is used
            policy_target = mcts_policy.copy()

            # Get the average value derived from MCTS simulations
            root_value = root_node.get_value()

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

            # Update statistics
            iter_stats["mcts_policy_confidence_sum"] += mcts_policy_confidence

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
                    # Fall back to uniform random choice over legal moves
                    legal_flat = legal_moves.flatten()
                    legal_indices = np.nonzero(legal_flat > 0)[0]
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

        # Create the move object
        move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))

        # Track value prediction statistics (only basic ones)
        iter_stats["value_prediction_sum"] += root_value
        iter_stats["value_prediction_squared_sum"] += root_value * root_value
        iter_stats["value_prediction_abs_sum"] += abs(root_value)
        iter_stats["value_prediction_count"] += 1

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
    def get_improved_value_target(outcome_value, mcts_value, move_index, global_weight):
        """Blend the final outcome with MCTS predicted value for better training signal"""
        # Early moves use more bootstrapping, later moves use more outcome
        move_factor = max(0.2, 1.0 - (move_index / 16))
        # Combine global weight with move-specific factor
        effective_factor = move_factor * global_weight
        return (1 - effective_factor) * outcome_value + effective_factor * mcts_value

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
    print("MCTS move ratio: {:.1%}".format(DEFAULT_MCTS_RATIO))
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

            # Clear MCTS prediction cache at the start of each iteration
            MCTS.clear_cache()

            # Update opponent ratios based on current iteration
            random_opponent_ratio, strategic_opponent_ratio = get_opponent_ratios(
                iteration
            )

            # Update bootstrap weight based on current iteration
            bootstrap_weight = get_bootstrap_weight(iteration)

            # Reset iteration stats
            iter_stats = stats_template.copy()

            # Create a list to collect all examples from this iteration
            current_iteration_examples = []

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
                                    move_count=game.move_count,
                                    mcts_value=value_pred,  # Store the MCTS value prediction
                                )
                            )
                        else:
                            # also store examples for direct policy moves
                            examples.append(
                                TrainingExample(
                                    state_rep=state_rep,
                                    policy=policy_target,
                                    value=0.0,  # will be filled in later
                                    current_player=game.current_player,
                                    move_count=game.move_count,
                                    # no mcts_value for direct policy moves
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
                                game, winner, example.current_player
                            )

                            # If this move used MCTS and we have an MCTS value, use improved target
                            if (
                                hasattr(example, "mcts_value")
                                and example.mcts_value is not None
                            ):
                                # Track value prediction error before applying blending
                                # Scale outcome value by move_count/15 for better early move analysis
                                scaled_outcome_value = outcome_value * (
                                    example.move_count / 15
                                )
                                # removed detailed error calculation, keeping only correlation components
                                # error = abs(scaled_outcome_value - example.mcts_value)
                                # iter_stats["value_error_sum"] += error

                                # 4. Track for correlation calculation
                                iter_stats["value_actual_pred_products"] += (
                                    outcome_value * example.mcts_value
                                )
                                iter_stats["value_actual_squared"] += (
                                    outcome_value * outcome_value
                                )
                                iter_stats["value_pred_squared"] += (
                                    example.mcts_value * example.mcts_value
                                )

                                # removed tracking error by confidence level
                                # removed storing time series data

                                example.value = get_improved_value_target(
                                    outcome_value,
                                    example.mcts_value,
                                    i,
                                    bootstrap_weight,
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

                                    # since we use subjective=True in state_rep creation
                                    bootstrap_value = float(next_value[0][0])

                                    # removed detailed error tracking, keeping correlation components
                                    # error = abs(scaled_outcome_value - bootstrap_value)
                                    # iter_stats["value_error_sum"] += error

                                    # 4. Track for correlation calculation
                                    iter_stats["value_actual_pred_products"] += (
                                        outcome_value * bootstrap_value
                                    )
                                    iter_stats["value_actual_squared"] += (
                                        outcome_value * outcome_value
                                    )
                                    iter_stats["value_pred_squared"] += (
                                        bootstrap_value * bootstrap_value
                                    )

                                    # removed tracking error by confidence level
                                    # removed storing time series data

                                    # Track basic value statistics (already tracked in get_model_move)
                                    # iter_stats["value_prediction_sum"] += bootstrap_value
                                    # ... etc

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
                        # iter_stats["total_moves"] += move_count # removed total moves

                        if opponent_type == "self-play":
                            iter_stats["self_play_games"] += 1
                            if winner == Player.ONE:
                                iter_stats["self_play_p1_wins"] += 1
                            elif winner is None:  # track draws
                                iter_stats["self_play_draws"] += 1
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

                # Add the game examples to the iteration examples
                current_iteration_examples.extend(examples)

                # Update progress bar
                episode_pbar.update(1)

                # Calculate cache hit rate for display
                total_cache_attempts = (
                    iter_stats["cache_hits"] + iter_stats["cache_misses"]
                )
                current_hit_rate = (
                    f"{(iter_stats['cache_hits'] / total_cache_attempts * 100):.1f}%"
                    if total_cache_attempts > 0
                    else "0.0%"
                )

                # Calculate strategic opponent stats
                strategic_p1_wins = f"{iter_stats['strategic_wins_as_p1']}/{iter_stats['strategic_games_as_p1']}"
                strategic_p2_wins = f"{iter_stats['strategic_wins_as_p2']}/{iter_stats['strategic_games_as_p2']}"

                # Set postfix with expanded information
                episode_pbar.set_postfix(
                    {
                        "cache_size": len(MCTS.prediction_cache),
                        "cache": current_hit_rate,
                        "strat_p1": strategic_p1_wins,
                        "strat_p2": strategic_p2_wins,
                    }
                )

            # Balance the replay buffer using current_iteration_examples
            tqdm.write(
                f"Updating replay buffer (current size: {len(replay_buffer)}, new examples: {len(current_iteration_examples)})"
            )
            if BALANCE_REPLAY_BUFFER:
                tqdm.write("Applying balancing...")
                replay_buffer = balance_replay_buffer(
                    replay_buffer,  # Previous buffer
                    current_iteration_examples,  # New examples we just collected
                    buffer_size=DEFAULT_BUFFER_SIZE,
                )
            else:
                # if not balancing, just add new examples and trim if needed
                replay_buffer.extend(current_iteration_examples)
                if len(replay_buffer) > DEFAULT_BUFFER_SIZE:
                    # keep only the most recent examples
                    replay_buffer = replay_buffer[-DEFAULT_BUFFER_SIZE:]
                tqdm.write("Skipping balancing, buffer updated.")

            # Log buffer size and distribution after balancing
            # Count buffer categories after balancing
            buffer_win_p1 = buffer_win_p2 = buffer_loss_p1 = buffer_loss_p2 = (
                buffer_draw_p1
            ) = buffer_draw_p2 = 0
            for ex in replay_buffer:
                player = "p1" if ex.current_player == Player.ONE else "p2"
                if ex.value > 0.2:
                    if player == "p1":
                        buffer_win_p1 += 1
                    else:
                        buffer_win_p2 += 1
                elif ex.value < -0.2:
                    if player == "p1":
                        buffer_loss_p1 += 1
                    else:
                        buffer_loss_p2 += 1
                else:
                    if player == "p1":
                        buffer_draw_p1 += 1
                    else:
                        buffer_draw_p2 += 1

            buffer_size = len(replay_buffer)
            tqdm.write(
                f"Balanced buffer: size={buffer_size}, "
                f"P1 win={buffer_win_p1} ({buffer_win_p1/buffer_size:.1%}), "
                f"P2 win={buffer_win_p2} ({buffer_win_p2/buffer_size:.1%}), "
                f"P1 loss={buffer_loss_p1} ({buffer_loss_p1/buffer_size:.1%}), "
                f"P2 loss={buffer_loss_p2} ({buffer_loss_p2/buffer_size:.1%}), "
                f"P1 draw={buffer_draw_p1} ({buffer_draw_p1/buffer_size:.1%}), "
                f"P2 draw={buffer_draw_p2} ({buffer_draw_p2/buffer_size:.1%})"
            )

            # Training phase
            train_pbar = tqdm(
                range(DEFAULT_NUM_EPOCHS), desc="Training Epochs", leave=False
            )
            epoch_losses = {"total": 0, "policy": 0, "value": 0}

            in_training_phase = True  # Set flag to indicate we're in training

            for epoch_idx in train_pbar:
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
                total_loss, policy_loss, value_loss, step_grad_stats = model.train_step(
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

                # Add accuracy verification - directly compare predictions to targets we just trained on
                with torch.no_grad():
                    # Ensure proper type conversion
                    board_tensor = torch.tensor(
                        board_inputs, dtype=torch.float32, device=model.device
                    )
                    flat_tensor = torch.tensor(
                        flat_inputs, dtype=torch.float32, device=model.device
                    )
                    _, model_values = model.predict(board_tensor, flat_tensor)

                    # Check if it's already a numpy array or a tensor that needs conversion
                    if isinstance(model_values, torch.Tensor):
                        model_values = model_values.squeeze().cpu().numpy()
                    else:
                        # If it's already a numpy array, just squeeze it
                        model_values = np.squeeze(model_values)

                    # Convert targets to numpy for comparison
                    if isinstance(value_targets, torch.Tensor):
                        targets_np = value_targets.cpu().numpy()
                    else:
                        targets_np = value_targets

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

            # Run evaluation every DEFAULT_EVAL_INTERVAL iterations
            if iteration % DEFAULT_EVAL_INTERVAL == 0:
                policy_vs_mcts_eval(
                    model,
                    rng,
                    iteration=iteration,
                    num_games=DEFAULT_EVAL_GAMES,
                    strategic_games=DEFAULT_STRATEGIC_EVAL_GAMES,
                    mcts_simulations=DEFAULT_MAX_MCTS_SIMS,
                    debug=False,
                )

            # Gather all metrics for the consolidated report
            loss_stats = {"total": avg_total, "policy": avg_policy, "value": avg_value}

            # Collect gradient statistics for reporting
            grad_stats = inspect_value_head_gradients(model)

            # Calculate average MCTS confidence
            iter_stats["avg_mcts_confidence"] = (
                iter_stats["mcts_policy_confidence_sum"]
                / max(1, iter_stats["mcts_moves"])
                if iter_stats["mcts_moves"] > 0
                else 0
            )

            # Calculate value metrics (simplified)
            value_pred_count = iter_stats["value_prediction_count"]
            actual_sq = iter_stats["value_actual_squared"]
            pred_sq = iter_stats["value_pred_squared"]

            value_metrics = {
                "pearson_correlation": (
                    iter_stats["value_actual_pred_products"]
                    / (math.sqrt(actual_sq) * math.sqrt(pred_sq))
                    if actual_sq > 0 and pred_sq > 0
                    else 0
                ),
                "avg_value_prediction": (
                    iter_stats["value_prediction_sum"] / value_pred_count
                    if value_pred_count > 0
                    else 0
                ),
                "avg_abs_value": (
                    iter_stats["value_prediction_abs_sum"] / value_pred_count
                    if value_pred_count > 0
                    else 0
                ),
                "value_std": (
                    math.sqrt(
                        max(
                            0,
                            (
                                iter_stats["value_prediction_squared_sum"]
                                / value_pred_count
                            )
                            - (iter_stats["value_prediction_sum"] / value_pred_count)
                            ** 2,
                        )
                    )
                    if value_pred_count > 0
                    else 0
                ),
                # removed avg_absolute_diff, avg_relative_diff, avg_scaled_diff
                # removed extreme_value_ratio
                # removed avg_value_error
                # removed near_zero_pct
            }

            # Always use consolidated logging
            log_iteration_report(
                iteration,
                iteration_start,
                training_start,
                model,
                replay_buffer,
                iter_stats,
                loss_stats,
                grad_stats,
                value_metrics,
            )

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


def balance_replay_buffer(
    replay_buffer: List[TrainingExample],
    new_examples: List[TrainingExample],
    buffer_size: int = DEFAULT_BUFFER_SIZE,
) -> List[TrainingExample]:
    """Balance the replay buffer to achieve a targeted distribution of examples.

    The target distribution is:
    - 20% p1 perspective moves from games won by p1
    - 20% p2 perspective moves from games won by p2
    - 20% p1 perspective moves from games lost by p1
    - 20% p2 perspective moves from games lost by p2
    - 10% p1 perspective moves from drawn games
    - 10% p2 perspective moves from drawn games

    Args:
        replay_buffer: Current replay buffer
        new_examples: New examples from latest iteration (will be prioritized)
        buffer_size: Target size for the balanced buffer

    Returns:
        Balanced replay buffer
    """
    # categorize all examples into 6 buckets
    buckets = defaultdict(list)

    # first add new examples to appropriate buckets
    for ex in new_examples:
        # determine the bucket key (player_perspective, outcome)
        player = "p1" if ex.current_player == Player.ONE else "p2"

        # use value to determine outcome - using thresholds consistent with rest of codebase
        if ex.value > 0.2:
            outcome = "win"
        elif ex.value < -0.2:
            outcome = "loss"
        else:
            outcome = "draw"

        # add to appropriate bucket
        buckets[(player, outcome)].append(ex)

    # then add existing buffer examples to buckets
    # instead of checking if examples exist in new_examples (which causes issues with numpy arrays),
    # we'll just process all examples from both sources independently
    for ex in replay_buffer:
        player = "p1" if ex.current_player == Player.ONE else "p2"

        if ex.value > 0.2:
            outcome = "win"
        elif ex.value < -0.2:
            outcome = "loss"
        else:
            outcome = "draw"

        buckets[(player, outcome)].append(ex)

    # calculate target counts for each bucket
    target_counts = {
        ("p1", "win"): int(buffer_size * 0.20),
        ("p2", "win"): int(buffer_size * 0.20),
        ("p1", "loss"): int(buffer_size * 0.20),
        ("p2", "loss"): int(buffer_size * 0.20),
        ("p1", "draw"): int(buffer_size * 0.10),
        ("p2", "draw"): int(buffer_size * 0.10),
    }

    # ensure total adds up to buffer_size by adding any rounding difference to the last bucket
    total_target = sum(target_counts.values())
    if total_target < buffer_size:
        target_counts[("p2", "draw")] += buffer_size - total_target

    # create balanced buffer
    balanced_buffer = []
    rng = np.random.Generator(np.random.PCG64())

    for bucket_key, target_count in target_counts.items():
        bucket_examples = buckets[bucket_key]
        bucket_size = len(bucket_examples)

        if bucket_size == 0:
            # no examples for this category
            continue

        if bucket_size <= target_count:
            # not enough examples, use all and possibly duplicate
            balanced_buffer.extend(bucket_examples)
            # duplicate if needed to reach target count
            if bucket_size < target_count:
                # randomly duplicate examples to reach target count
                duplicates = rng.choice(
                    bucket_examples, size=target_count - bucket_size, replace=True
                )
                balanced_buffer.extend(duplicates)
        else:
            # too many examples, randomly sample without replacement
            sampled = rng.choice(bucket_examples, size=target_count, replace=False)
            balanced_buffer.extend(sampled)

    # shuffle the balanced buffer
    rng.shuffle(balanced_buffer)

    return balanced_buffer


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
    training_loop(model, resume_path=resume_path)
