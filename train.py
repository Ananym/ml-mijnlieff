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

# Import MCTS implementation
from mcts import MCTS as SerialMCTS, add_dirichlet_noise
import math
from collections import defaultdict
from eval_model import policy_vs_mcts_eval, extended_evaluation
import torch.nn.functional as F

# Simple trace logger stub (original mcts_trace_logger.py was removed)
class TraceLogger:
    def __init__(self):
        self.log_entries = []
    def enable(self): pass
    def disable(self): pass
    def log_network_prediction(self, *args, **kwargs): pass
    def log_mcts_search_start(self, *args, **kwargs): pass
    def log_mcts_search_end(self, *args, **kwargs): pass
    def log_game_end(self, *args, **kwargs): pass

def get_trace_logger():
    return TraceLogger()

# Training mode configurations
TRAINING_MODES = {
    "fast": {
        "episodes": 50,
        "epochs": 5,
        "min_mcts_sims": 200,
        "max_mcts_sims": 400,
        "max_iterations": 100,
        "eval_interval": 5,
        "eval_games": 20,
        "description": "Fast iteration for testing (50 games, 200-400 sims, 20 iters)",
    },
    "stable": {
        "episodes": 100,
        "epochs": 20,
        "min_mcts_sims": 800,
        "max_mcts_sims": 1600,
        "max_iterations": 100,  # Extended for Experiment 9
        "eval_interval": 10,
        "eval_games": 40,
        "description": "VALUE BOOTSTRAPPING (80% MCTS values, pure self-play, 800-1600 sims)",
    },
}

# Default values (will be overridden by mode)
DEFAULT_EPISODES = 100
DEFAULT_BATCH_SIZE = 256
DEFAULT_SAVE_INTERVAL = 5
DEFAULT_NUM_CHECKPOINTS = 5
DEFAULT_MCTS_RATIO = 1.0
DEFAULT_BUFFER_SIZE = 8000  # Exp 12: Increased for larger model
DEFAULT_POLICY_WEIGHT = 0.2  # Balanced - 80% value, 20% policy (was 0.05, too extreme)
DEFAULT_NUM_EPOCHS = 20
DEFAULT_EVAL_INTERVAL = 10
DEFAULT_EVAL_GAMES = 40
DEFAULT_STRATEGIC_EVAL_GAMES = 40
DEFAULT_MIN_MCTS_SIMS = 600  # REVERTED to Exp 1 baseline - faster iterations
DEFAULT_MAX_MCTS_SIMS = 1200  # REVERTED to Exp 1 baseline - faster iterations
MAX_ITERATIONS = 100  # Extended run for higher specialization while maintaining generalization

# Early stopping criteria
EARLY_STOPPING_ENABLED = True
EARLY_STOPPING_CHECK_ITERATION = 50  # Adjusted for 100-iteration run
EARLY_STOPPING_MIN_WINRATE = (
    25.0  # Minimum win rate vs Strategic opponent to continue training
)

BALANCE_REPLAY_BUFFER = False  # REVERTED to Exp 1 baseline - natural distribution

# Dirichlet noise - Experiment 12: Pure Self-Play (constant high exploration)
DIRICHLET_SCALE = 0.30  # Constant exploration throughout - higher for pure self-play diversity

ENTROPY_BONUS_SCALE = 0.07  # Encourage policy diversity for human play


def get_mcts_class(device):
    """
    Returns the SerialMCTS implementation.
    """
    print("MCTS Strategy: Using SerialMCTS (reusable instance)")
    return SerialMCTS


def get_dirichlet_scale(iteration):
    """
    Experiment 12: Pure Self-Play - constant high exploration
    Returns constant 0.30 throughout training for diversity without opponent curriculum.
    """
    return DIRICHLET_SCALE


# Experiment 12: PURE SELF-PLAY - No strategic opponent, only model vs model
INITIAL_RANDOM_OPPONENT_RATIO = 0.0
FINAL_RANDOM_OPPONENT_RATIO = 0.0

INITIAL_STRATEGIC_OPPONENT_RATIO = 0.0  # Pure self-play
PEAK_STRATEGIC_OPPONENT_RATIO = 0.0  # No strategic opponent
FINAL_STRATEGIC_OPPONENT_RATIO = 0.0  # 100% self-play throughout
PHASE_1_END = 40  # Not used in pure self-play
PHASE_2_END = 60  # Not used in pure self-play
OPPONENT_TRANSITION_ITERATIONS = 100  # Total iterations

DEFAULT_INITIAL_RANDOM_CHANCE = 0.0
DEFAULT_FINAL_RANDOM_CHANCE = 0.0
DEFAULT_RANDOM_CHANCE_TRANSITION_ITERATIONS = MAX_ITERATIONS

BOOTSTRAP_MIN_WEIGHT = 0.0  # Pure game outcomes (Exp 1 baseline) - AlphaZero's approach
BOOTSTRAP_MAX_WEIGHT = 0.0  # No bootstrapping (Exp 1 baseline)
BOOTSTRAP_TRANSITION_ITERATIONS = MAX_ITERATIONS  # No transition, keep constant

DEFAULT_CHECKPOINT_PATH = "saved_models/checkpoint_interrupted.pth"


def get_reward_values(reward_config="discrete"):
    """Get the win/loss/draw reward values for a given config.

    Returns:
        tuple: (win_value, loss_value, draw_value)
    """
    if reward_config == "discrete":
        return (1.0, -1.0, -0.1)
    elif reward_config == "discrete_mild":
        return (1.0, -1.0, -0.3)
    elif reward_config == "discrete_light":
        return (1.0, -1.0, -0.2)
    elif reward_config == "discrete_heavy":
        return (1.0, -1.0, -0.6)  # Strong draw penalty to break conservative play
    else:
        # For continuous configs, use thresholds
        return (1.0, -1.0, 0.0)


def classify_outcome(value, reward_config="discrete"):
    """Classify a value as 'win', 'loss', or 'draw'.

    Uses the actual reward values with small epsilon for floating point comparison.
    """
    win_val, loss_val, draw_val = get_reward_values(reward_config)
    epsilon = 0.01  # Small tolerance for floating point comparison

    # Check exact values (within epsilon)
    if abs(value - win_val) < epsilon:
        return "win"
    elif abs(value - loss_val) < epsilon:
        return "loss"
    elif abs(value - draw_val) < epsilon:
        return "draw"
    else:
        # For continuous configs or bootstrapped values between discrete values
        # Use midpoint thresholds
        if value > (win_val + draw_val) / 2:
            return "win"
        elif value < (loss_val + draw_val) / 2:
            return "loss"
        else:
            return "draw"


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

    # count win/loss/draw samples in buffer using shared classification logic
    buffer_win_samples = 0
    buffer_loss_samples = 0
    buffer_draw_samples = 0

    for example in replay_buffer:
        outcome = classify_outcome(example.value, reward_config)
        if outcome == "win":
            buffer_win_samples += 1
        elif outcome == "loss":
            buffer_loss_samples += 1
        else:
            buffer_draw_samples += 1

    buffer_size = len(replay_buffer)
    buffer_win_pct = (buffer_win_samples / max(1, buffer_size)) * 100
    buffer_loss_pct = (buffer_loss_samples / max(1, buffer_size)) * 100
    buffer_draw_pct = (buffer_draw_samples / max(1, buffer_size)) * 100

    # Extract gradient statistics if available
    policy_grad_norm = grad_stats.get('policy_grad_norm', 0) if grad_stats else 0
    value_grad_norm = grad_stats.get('value_grad_norm', 0) if grad_stats else 0
    grad_ratio = grad_stats.get('grad_ratio', 0) if grad_stats else 0
    adaptive_weight = grad_stats.get('adaptive_policy_weight', 0) if grad_stats else 0

    # Build the consolidated report - SIMPLIFIED for readability
    print(f"\n{'='*20} ITER {iteration} {'='*20}")
    print(f"Time: {int(hours)}h{int(minutes):02d}m | Iter: {iteration_time:.0f}s")
    print(
        f"Loss: {avg_total:.4f} (P:{avg_policy:.4f} V:{avg_value:.4f}) LR:{model.optimizer.param_groups[0]['lr']:.6f}"
    )
    if adaptive_weight > 0:
        print(
            f"Grads: P/V ratio={grad_ratio:.2f} | Adaptive weight={adaptive_weight:.3f}"
        )
    print(
        f"Value: mean={avg_value_prediction:.3f} std={value_std:.3f} corr={pearson_correlation:.3f}"
    )
    print(
        f"Self: P1={self_play_p1_rate:.0f}% P2={self_play_p2_rate:.0f}% D={self_play_draw_rate:.0f}%"
    )
    print(
        f"Buffer: {buffer_size} (W:{buffer_win_pct:.0f}% L:{buffer_loss_pct:.0f}% D:{buffer_draw_pct:.0f}%) Cache:{cache_hit_rate:.0f}%"
    )

    # Gradient health check (only show if problematic)
    if grad_stats and isinstance(grad_stats, dict) and "status" not in grad_stats:
        if "value_fc3_weight" in grad_stats:
            fc3_weight = grad_stats.get("value_fc3_weight", {})
            fc3_weight_max_abs = max(
                abs(fc3_weight.get("min", 0)), abs(fc3_weight.get("max", 0))
            )
            if fc3_weight_max_abs > 1.0:
                print(f"WARNING - EXPLODING GRADIENTS: {fc3_weight_max_abs:.6f}")
            elif fc3_weight_max_abs < 0.0001:
                print(f"WARNING - VANISHING GRADIENTS: {fc3_weight_max_abs:.6f}")

    print(f"{'='*50}")


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
    enable_early_stopping: bool = True,
    reward_config: str = "discrete",
    training_mode: str = "stable",
):
    # Select appropriate MCTS class based on device
    MCTSClass = get_mcts_class(model.device)

    # Apply training mode configuration
    global DEFAULT_EPISODES, DEFAULT_NUM_EPOCHS, DEFAULT_MIN_MCTS_SIMS, DEFAULT_MAX_MCTS_SIMS
    global MAX_ITERATIONS, DEFAULT_EVAL_INTERVAL, DEFAULT_EVAL_GAMES, DEFAULT_STRATEGIC_EVAL_GAMES

    if training_mode in TRAINING_MODES:
        mode_config = TRAINING_MODES[training_mode]
        DEFAULT_EPISODES = mode_config["episodes"]
        DEFAULT_NUM_EPOCHS = mode_config["epochs"]
        DEFAULT_MIN_MCTS_SIMS = mode_config["min_mcts_sims"]
        DEFAULT_MAX_MCTS_SIMS = mode_config["max_mcts_sims"]
        MAX_ITERATIONS = mode_config["max_iterations"]
        DEFAULT_EVAL_INTERVAL = mode_config["eval_interval"]
        DEFAULT_EVAL_GAMES = mode_config["eval_games"]
        DEFAULT_STRATEGIC_EVAL_GAMES = mode_config["eval_games"]
        print(f"\nTraining mode: {training_mode}")
        print(f"  {mode_config['description']}")
    else:
        print(f"\nWARNING: Unknown training mode '{training_mode}', using defaults")

    rng = np.random.Generator(np.random.PCG64())

    replay_buffer = []
    running_loss = {
        "total": 1e-8,
        "policy": 1e-8,
        "value": 1e-8,
    }
    running_count = 0
    iteration = 0

    # Adaptive policy weight - starts at DEFAULT but adjusts based on gradient balance
    adaptive_policy_weight = DEFAULT_POLICY_WEIGHT
    grad_ratio_history = []  # Track gradient ratio over time

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
                "adaptive_policy_weight": adaptive_policy_weight,
                "grad_ratio_history": grad_ratio_history,
            }
            model.save_checkpoint(checkpoint_path, training_state)
            print(f"Saved checkpoint to {checkpoint_path}")
            print(f"Resume with: --resume")  # Simplified message

        interrupt_received = True

    signal.signal(signal.SIGINT, handle_interrupt)

    # Start the training timer
    training_start = time.time()

    # Setup trace logging - DISABLED (stub implementation causes errors)
    trace_logger = get_trace_logger()
    trace_enabled_iteration = 999  # Never trigger
    trace_logged = False

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
        "value_actual_sum": 0,  # Σ(actual) needed for pearson correlation
        "value_pred_sum_for_corr": 0,  # Σ(predicted) for correlation only
        "value_actual_pred_products": 0,  # Σ(actual*pred) needed for pearson correlation
        "value_actual_squared": 0,  # Σ(actual²) needed for pearson correlation
        "value_pred_squared": 0,  # Σ(pred²) needed for pearson correlation
        "value_corr_count": 0,  # count of samples for correlation
        # cache stats
        "cache_hits": 0,
        "cache_misses": 0,
    }

    def get_adjusted_value(game, winner, current_player, reward_config="discrete"):
        """Calculate training value target with configurable reward schemes.

        Args:
            game: Completed game state
            winner: Winner of the game
            current_player: Player from whose perspective we calculate value
            reward_config: Reward configuration to use:
                - "discrete": Simple win/loss/draw (+1/-1/-0.1)
                - "discrete_mild": Win/loss/draw with mild penalty (+1/-1/-0.3)
                - "discrete_light": Win/loss/draw with light penalty (+1/-1/-0.15)
                - "score_diff": Original score differential (continuous)
                - "score_diff_penalty": Score diff with draw penalty

        Returns:
            Normalized value in [-1, 1] range
        """
        scores = game.get_scores()
        player_score = scores[current_player]
        opponent = Player.ONE if current_player == Player.TWO else Player.TWO
        opponent_score = scores[opponent]
        score_diff = player_score - opponent_score

        if reward_config == "discrete":
            # Win: +1, Loss: -1, Draw: -0.1 (draw worse than win, better than loss)
            if score_diff > 0:
                return 1.0
            elif score_diff < 0:
                return -1.0
            else:
                return -0.1

        elif reward_config == "discrete_mild":
            # Win: +1, Loss: -1, Draw: -0.3 (moderate draw penalty)
            if score_diff > 0:
                return 1.0
            elif score_diff < 0:
                return -1.0
            else:
                return -0.3

        elif reward_config == "discrete_light":
            # Win: +1, Loss: -1, Draw: -0.2 (light draw penalty)
            if score_diff > 0:
                return 1.0
            elif score_diff < 0:
                return -1.0
            else:
                return -0.2

        elif reward_config == "discrete_heavy":
            # Win: +1, Loss: -1, Draw: -0.6 (strong draw penalty to break conservative play)
            if score_diff > 0:
                return 1.0
            elif score_diff < 0:
                return -1.0
            else:
                return -0.6

        elif reward_config == "score_diff":
            # Original: continuous score differential
            clamped_diff = max(-6.0, min(6.0, score_diff))
            return clamped_diff / 6.0

        elif reward_config == "score_diff_penalty":
            # Score differential but with draw penalty
            if score_diff == 0:
                return -0.2  # Penalize draws
            else:
                clamped_diff = max(-6.0, min(6.0, score_diff))
                return clamped_diff / 6.0

        else:
            raise ValueError(f"Unknown reward_config: {reward_config}")

    # Store reward config at module level for access in training loop
    REWARD_CONFIG = "discrete"  # Default

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
        """Temperature schedule that lowers in endgame for more decisive play.

        Early game (moves 0-7): High exploration (T=1.0)
        Mid game (moves 8-10): Transition (T=1.0 -> 0.1)
        Late game (moves 11+): Low exploration (T=0.1)
        """
        if move_count < 8:
            return 1.0
        elif move_count < 11:
            # Linear transition from 1.0 to 0.1 over moves 8-10
            return 1.0 - 0.9 * (move_count - 8) / 3
        else:
            return 0.1
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

    def get_model_move_with_policy(
        game, use_mcts=True, temperature=1.0, iteration=0, enable_trace=False
    ):
        """Get a move from the model along with the training policy target
        (full MCTS distribution if MCTS was used, otherwise one-hot)"""
        legal_moves = game.get_legal_moves()
        # Use subjective=True for the state representation
        state_rep = game.get_game_state_representation(subjective=True)
        move_count = game.move_count

        if use_mcts:
            # Trace: Log network prediction before MCTS
            if enable_trace:
                raw_policy, raw_value = model.predict(
                    state_rep.board, state_rep.flat_values, legal_moves
                )
                raw_policy = raw_policy.squeeze(0)
                policy_flat = (raw_policy * legal_moves).flatten()
                top5_idx = policy_flat.argsort()[-5:][::-1]
                top5_moves = [
                    (np.unravel_index(idx, raw_policy.shape), policy_flat[idx])
                    for idx in top5_idx
                    if policy_flat[idx] > 0
                ]
                trace_logger.log_network_prediction(
                    f"Move {move_count}", top5_moves, raw_value.squeeze().item()
                )

            # Use model-guided MCTS (AlphaZero style)
            # Noise will be applied inside the MCTS search
            num_sims = get_mcts_sims_for_iteration(iteration)

            if enable_trace:
                trace_logger.log_mcts_search_start(num_sims)

            # Update reusable MCTS instance configuration (avoids worker startup overhead)
            reusable_mcts.num_simulations = num_sims
            reusable_mcts.dirichlet_scale = get_dirichlet_scale(iteration)
            reusable_mcts.set_temperature(temperature)
            reusable_mcts.set_iteration(iteration)  # Set current iteration for noise
            reusable_mcts.set_move_count(move_count)  # Set move count for noise
            mcts_policy, root_node = reusable_mcts.search(game)

            # Trace: Log MCTS results
            if enable_trace:
                # Get top 5 children by visit count
                children_sorted = sorted(
                    root_node.children, key=lambda c: c.visits, reverse=True
                )[:5]
                # Negate Q-values to show from parent's perspective (child values are negated in UCB)
                top5_mcts = [
                    (c.move, c.visits, -c.get_value()) for c in children_sorted
                ]
                trace_logger.log_mcts_search_end(
                    root_node.get_value(),
                    root_node.visits,
                    top5_mcts,
                    None,  # Will add chosen move later
                )
            policy = mcts_policy

            # Track cache statistics
            iter_stats["cache_hits"] += reusable_mcts.cache_hits
            iter_stats["cache_misses"] += reusable_mcts.cache_misses
            # Reset MCTS cache stats for next use
            reusable_mcts.cache_hits = 0
            reusable_mcts.cache_misses = 0

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

            # Trace: Log chosen move
            if enable_trace:
                chosen_move = Move(
                    move_coords[0], move_coords[1], PieceType(move_coords[2])
                )
                trace_logger.log_entries[-1] = trace_logger.log_entries[-1].replace(
                    "Chosen Move: None", f"Chosen Move: {chosen_move}"
                )

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
                policy_flat,
                legal_flat,
                iteration,
                move_count,
                max_iterations=150,
                dirichlet_scale=get_dirichlet_scale(iteration),
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
        """
        Experiment 11: Plateau Curriculum
        - Phase 1 (iter 1 -> 40): Ramp up quickly to 85% Strategic (fast specialization)
        - Phase 2 (iter 40 -> 60): Maintain 85% Strategic (plateau to consolidate learning)
        - Phase 3 (iter 60 -> 100): Ramp down to 40% Strategic (restore generalization)
        """
        random_ratio = INITIAL_RANDOM_OPPONENT_RATIO  # Always 0 for now

        if iteration <= PHASE_1_END:
            # Phase 1: Fast ramp up to peak (build expertise quickly)
            progress = iteration / PHASE_1_END
            strategic_ratio = INITIAL_STRATEGIC_OPPONENT_RATIO + progress * (
                PEAK_STRATEGIC_OPPONENT_RATIO - INITIAL_STRATEGIC_OPPONENT_RATIO
            )
        elif iteration <= PHASE_2_END:
            # Phase 2: Plateau at peak (consolidate learning)
            strategic_ratio = PEAK_STRATEGIC_OPPONENT_RATIO
        else:
            # Phase 3: Ramp down from peak (restore generalization)
            progress = (iteration - PHASE_2_END) / (OPPONENT_TRANSITION_ITERATIONS - PHASE_2_END)
            strategic_ratio = PEAK_STRATEGIC_OPPONENT_RATIO + progress * (
                FINAL_STRATEGIC_OPPONENT_RATIO - PEAK_STRATEGIC_OPPONENT_RATIO
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

    # Create reusable MCTS instance (configuration will be updated per move)
    mcts_kwargs = {
        "model": model,
        "num_simulations": DEFAULT_MIN_MCTS_SIMS,  # Will be updated
        "c_puct": 1.5,
        "dirichlet_scale": get_dirichlet_scale(0),  # Will be updated
        "enable_early_stopping": enable_early_stopping,
    }
    # Using SerialMCTS only - no parallel/batched implementations
    reusable_mcts = MCTSClass(**mcts_kwargs)

    try:
        while (
            not interrupt_received and iteration < MAX_ITERATIONS
        ):  # Run until interrupted or max iterations
            iteration += 1
            iteration_start = time.time()

            # Clear MCTS prediction cache at the start of each iteration
            MCTSClass.clear_cache()

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
            for game_num in range(DEFAULT_EPISODES):
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

                # Enable trace logging for first self-play game of iteration 5
                is_self_play = opponent_type == "self-play"
                enable_trace_this_game = (
                    iteration == trace_enabled_iteration
                    and not trace_logged
                    and is_self_play
                )
                if enable_trace_this_game:
                    trace_logger.enable()
                    print(
                        f"\n[TRACE] Enabling detailed MCTS trace logging for this game..."
                    )

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
                                enable_trace=enable_trace_this_game,
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

                        # Log game end if tracing
                        if enable_trace_this_game:
                            scores = game.get_scores()
                            trace_logger.log_game_end(
                                f"Player.{winner.name}" if winner else "Draw",
                                f"P1:{scores[Player.ONE]} P2:{scores[Player.TWO]}",
                                move_count,
                            )
                            trace_logger.disable()
                            trace_logged = True

                        # Set value targets based on game outcome and bootstrapping
                        for i, example in enumerate(examples):
                            # Get the final outcome value
                            outcome_value = get_adjusted_value(
                                game, winner, example.current_player, reward_config
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
                                iter_stats["value_actual_sum"] += outcome_value
                                iter_stats["value_pred_sum_for_corr"] += example.mcts_value
                                iter_stats["value_actual_pred_products"] += (
                                    outcome_value * example.mcts_value
                                )
                                iter_stats["value_actual_squared"] += (
                                    outcome_value * outcome_value
                                )
                                iter_stats["value_pred_squared"] += (
                                    example.mcts_value * example.mcts_value
                                )
                                iter_stats["value_corr_count"] += 1

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
                                    iter_stats["value_actual_sum"] += outcome_value
                                    iter_stats["value_pred_sum_for_corr"] += bootstrap_value
                                    iter_stats["value_actual_pred_products"] += (
                                        outcome_value * bootstrap_value
                                    )
                                    iter_stats["value_actual_squared"] += (
                                        outcome_value * outcome_value
                                    )
                                    iter_stats["value_pred_squared"] += (
                                        bootstrap_value * bootstrap_value
                                    )
                                    iter_stats["value_corr_count"] += 1

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

                            # Update stats using shared classification logic
                            outcome = classify_outcome(outcome_value, reward_config)
                            if outcome == "win":
                                iter_stats["win_examples"] += 1
                            elif outcome == "loss":
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
                        "cache_size": len(MCTSClass.prediction_cache),
                        "cache": current_hit_rate,
                        "strat_p1": strategic_p1_wins,
                        "strat_p2": strategic_p2_wins,
                    }
                )

            # Balance the replay buffer using current_iteration_examples (SILENCED)
            if BALANCE_REPLAY_BUFFER:
                replay_buffer = balance_replay_buffer(
                    replay_buffer,
                    current_iteration_examples,
                    buffer_size=DEFAULT_BUFFER_SIZE,
                )
            else:
                # if not balancing, just add new examples and trim if needed
                replay_buffer.extend(current_iteration_examples)
                if len(replay_buffer) > DEFAULT_BUFFER_SIZE:
                    replay_buffer = replay_buffer[-DEFAULT_BUFFER_SIZE:]

            # Training phase
            train_pbar = tqdm(
                range(DEFAULT_NUM_EPOCHS), desc="Training Epochs", leave=False
            )
            epoch_losses = {"total": 0, "policy": 0, "value": 0}
            epoch_grad_stats = {
                "policy_grad_norm": 0,
                "value_grad_norm": 0,
                "grad_ratio": 0,
                "recommended_policy_weight": 0,
                "count": 0,
            }

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

                # Train network with adaptive policy weight
                total_loss, policy_loss, value_loss, step_grad_stats = model.train_step(
                    board_inputs,
                    flat_inputs,
                    policy_targets,
                    value_targets,
                    policy_weight=adaptive_policy_weight,
                )

                # Accumulate gradient statistics for this epoch
                if 'policy_grad_norm' in step_grad_stats:
                    epoch_grad_stats['policy_grad_norm'] += step_grad_stats['policy_grad_norm']
                    epoch_grad_stats['value_grad_norm'] += step_grad_stats['value_grad_norm']
                    epoch_grad_stats['grad_ratio'] += step_grad_stats['grad_ratio']
                    epoch_grad_stats['recommended_policy_weight'] += step_grad_stats['recommended_policy_weight']
                    epoch_grad_stats['count'] += 1

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

            # Update adaptive policy weight based on gradient statistics (RE-ENABLED)
            if epoch_grad_stats['count'] > 0:
                avg_policy_grad = epoch_grad_stats['policy_grad_norm'] / epoch_grad_stats['count']
                avg_value_grad = epoch_grad_stats['value_grad_norm'] / epoch_grad_stats['count']
                avg_grad_ratio = epoch_grad_stats['grad_ratio'] / epoch_grad_stats['count']
                avg_recommended = epoch_grad_stats['recommended_policy_weight'] / epoch_grad_stats['count']

                # Smooth adaptation using exponential moving average (alpha=0.3)
                # This prevents wild swings in policy weight
                adaptive_policy_weight = 0.7 * adaptive_policy_weight + 0.3 * avg_recommended

                # Track gradient ratio history for monitoring
                grad_ratio_history.append(avg_grad_ratio)

            # Step the scheduler once per iteration
            model.scheduler.step()

            in_training_phase = False  # Reset flag after training phase

            # Run evaluation every DEFAULT_EVAL_INTERVAL iterations
            if iteration % DEFAULT_EVAL_INTERVAL == 0:
                # Exp 12: Use extended evaluation for pure self-play training
                eval_results = extended_evaluation(
                    model,
                    rng,
                    iteration=iteration,
                    num_games=DEFAULT_EVAL_GAMES // 2,  # Split across 5 eval types
                    mcts_simulations=DEFAULT_MAX_MCTS_SIMS,
                )

                # Early stopping check
                if (
                    EARLY_STOPPING_ENABLED
                    and iteration >= EARLY_STOPPING_CHECK_ITERATION
                ):
                    # Calculate combined win rate vs Strategic opponent
                    p1_games = eval_results["raw_policy_games_as_p1"]
                    p2_games = eval_results["raw_policy_games_as_p2"]
                    p1_wins = eval_results["raw_policy_wins_as_p1"]
                    p2_wins = eval_results["raw_policy_wins_as_p2"]

                    total_games = p1_games + p2_games
                    total_wins = p1_wins + p2_wins

                    if total_games > 0:
                        combined_winrate = (total_wins / total_games) * 100

                        print(f"\n=== Early Stopping Check (Iteration {iteration}) ===")
                        print(
                            f"Combined win rate vs Strategic: {combined_winrate:.1f}%"
                        )
                        print(f"Minimum required: {EARLY_STOPPING_MIN_WINRATE:.1f}%")

                        if combined_winrate < EARLY_STOPPING_MIN_WINRATE:
                            print(f"\n*** EARLY STOPPING TRIGGERED ***")
                            print(
                                f"Policy network is not learning effectively after {iteration} iterations."
                            )
                            print(
                                f"Win rate vs Strategic ({combined_winrate:.1f}%) is below threshold ({EARLY_STOPPING_MIN_WINRATE:.1f}%)."
                            )
                            print(
                                f"Consider adjusting hyperparameters and restarting training."
                            )
                            print("=" * 50)
                            interrupt_received = True  # Trigger graceful shutdown
                        else:
                            print(
                                f"Win rate vs Strategic is acceptable. Continuing training..."
                            )
                        print("=" * 50 + "\n")

            # Gather all metrics for the consolidated report
            loss_stats = {"total": avg_total, "policy": avg_policy, "value": avg_value}

            # Collect gradient statistics for reporting
            grad_stats = inspect_value_head_gradients(model)

            # Add adaptive weighting statistics to grad_stats
            if epoch_grad_stats['count'] > 0:
                grad_stats['policy_grad_norm'] = avg_policy_grad
                grad_stats['value_grad_norm'] = avg_value_grad
                grad_stats['grad_ratio'] = avg_grad_ratio
                grad_stats['adaptive_policy_weight'] = adaptive_policy_weight

            # Calculate average MCTS confidence
            iter_stats["avg_mcts_confidence"] = (
                iter_stats["mcts_policy_confidence_sum"]
                / max(1, iter_stats["mcts_moves"])
                if iter_stats["mcts_moves"] > 0
                else 0
            )

            # Calculate value metrics (simplified)
            value_pred_count = iter_stats["value_prediction_count"]

            # Pearson correlation: r = (n*Σxy - Σx*Σy) / (sqrt(n*Σx² - (Σx)²) * sqrt(n*Σy² - (Σy)²))
            n = iter_stats["value_corr_count"]
            sum_xy = iter_stats["value_actual_pred_products"]
            sum_x = iter_stats["value_actual_sum"]
            sum_y = iter_stats["value_pred_sum_for_corr"]  # Sum of predictions used in correlation
            sum_x_sq = iter_stats["value_actual_squared"]
            sum_y_sq = iter_stats["value_pred_squared"]

            if n > 1:
                numerator = n * sum_xy - sum_x * sum_y
                denom_x = n * sum_x_sq - sum_x * sum_x
                denom_y = n * sum_y_sq - sum_y * sum_y
                if denom_x > 0 and denom_y > 0:
                    pearson_corr = numerator / (math.sqrt(denom_x) * math.sqrt(denom_y))
                else:
                    pearson_corr = 0
            else:
                pearson_corr = 0

            value_metrics = {
                "pearson_correlation": pearson_corr,
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
    training_mode = "stable"  # default training mode (for hyperparameters)
    model_mode = "stable"  # default model mode (for ModelWrapper)
    load_path = None
    resume_path = None
    device = None  # Don't set a default yet
    enable_early_stopping = True  # default enabled
    reward_config = "discrete"  # default reward configuration

    # Check for device, fast mode, and load/resume path
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith("--device="):
            device = arg.split("=")[1]
        elif arg == "--fast":
            training_mode = "fast"
            model_mode = "fast"
        elif arg.startswith("--mode="):
            training_mode = arg.split("=")[1]
            model_mode = arg.split("=")[1]
        elif arg.startswith("--load="):
            load_path = arg.split("=")[1]
        elif arg.startswith("--resume="):
            resume_path = arg.split("=")[1]
        elif arg == "--resume":
            # Use default checkpoint path when just --resume is specified
            resume_path = DEFAULT_CHECKPOINT_PATH
        elif arg == "--no-early-stopping":
            enable_early_stopping = False
        elif arg.startswith("--reward="):
            reward_config = arg.split("=")[1]
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
    model = ModelWrapper(device=device, mode=model_mode)

    # Handle loading vs resuming
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")
        # Note: the model loading happens inside the training loop
    elif load_path:
        print(f"Loading model weights from {load_path}")
        model.load(load_path)

    # Print configuration
    print(f"Reward configuration: {reward_config}")
    if reward_config == "discrete":
        print(
            "  Win: +1.0, Loss: -1.0, Draw: -0.1 (draw worse than win, better than loss)"
        )
    elif reward_config == "discrete_mild":
        print("  Win: +1.0, Loss: -1.0, Draw: -0.3 (moderate draw penalty)")
    elif reward_config == "discrete_light":
        print("  Win: +1.0, Loss: -1.0, Draw: -0.15 (light draw penalty)")
    elif reward_config == "discrete_heavy":
        print("  Win: +1.0, Loss: -1.0, Draw: -0.6 (STRONG draw penalty - break conservative play)")
    elif reward_config == "score_diff":
        print("  Score differential (continuous, no draw penalty)")
    elif reward_config == "score_diff_penalty":
        print("  Score differential with draw penalty (-0.2)")

    # Start training
    training_loop(
        model,
        resume_path=resume_path,
        enable_early_stopping=enable_early_stopping,
        reward_config=reward_config,
        training_mode=training_mode,
    )
