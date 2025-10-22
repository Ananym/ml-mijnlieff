"""
Optimized parallel MCTS implementation using persistent workers with pre-loaded models.

Key optimization: Workers load the model ONCE at initialization, avoiding the
expensive model pickling overhead on every search call.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
from mcts import MCTS, clone_game_state
from game import GameState
import atexit
import os

# Global worker state (initialized once per worker process)
_worker_model = None
_worker_mcts_cache = {}

def _init_worker(model_state_dict, device, model_mode):
    """
    Initialize worker process with a pre-loaded model.

    This runs once per worker at pool creation time, not on every search.
    """
    global _worker_model

    # Lazy import to avoid issues with multiprocessing
    from model import ModelWrapper

    # Create model in this worker process
    _worker_model = ModelWrapper(device=device, mode=model_mode)

    # Load the model state dict
    if model_state_dict is not None:
        _worker_model.model.load_state_dict(model_state_dict)
        _worker_model.model.eval()


def _run_mcts_worker_v2(game_state_data, num_simulations, c_puct, temperature,
                        iteration, move_count, dirichlet_scale, enable_early_stopping,
                        worker_seed):
    """
    Worker function that runs MCTS search using pre-loaded model.

    This avoids pickling the model on every call.
    """
    global _worker_model

    # Set worker-specific random seed for diversity
    np.random.seed(worker_seed)

    # Create MCTS instance with pre-loaded model
    worker_mcts = MCTS(
        model=_worker_model,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_scale=dirichlet_scale,
        enable_early_stopping=enable_early_stopping
    )
    worker_mcts.set_temperature(temperature)
    worker_mcts.set_iteration(iteration)
    worker_mcts.set_move_count(move_count)

    # Reconstruct game state from data
    # (GameState should be lightweight to pickle)
    game_state = game_state_data

    # Run search
    probs, root = worker_mcts.search(game_state)

    # Extract visit counts from root children
    visit_counts = np.zeros((4, 4, 4), dtype=np.float32)
    for child in root.children:
        if child.move:
            x, y, piece_type = child.move.x, child.move.y, child.move.piece_type.value
            visit_counts[x, y, piece_type] = child.visits

    return visit_counts, root


class ParallelMCTSv2:
    """
    Optimized parallel MCTS using persistent workers with pre-loaded models.

    This version loads the model once per worker, eliminating the expensive
    model pickling overhead on every search.
    """

    # Class-level pool to share across instances
    _pool = None
    _pool_config = None  # (num_workers, device, model_mode)

    @classmethod
    def _ensure_pool(cls, num_workers, device, model_mode):
        """Ensure worker pool exists with correct configuration."""
        current_config = (num_workers, device, model_mode)

        if cls._pool is None or cls._pool_config != current_config:
            # Clean up old pool
            if cls._pool is not None:
                cls._pool.close()
                cls._pool.join()

            # Pool will be initialized when first model is set
            cls._pool = None
            cls._pool_config = current_config

    @classmethod
    def _create_pool(cls, model_state_dict, num_workers, device, model_mode):
        """Create worker pool with pre-loaded models."""
        if cls._pool is not None:
            return  # Already created

        # Create pool with initializer that loads the model
        cls._pool = Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(model_state_dict, device, model_mode)
        )

    @classmethod
    def shutdown_pool(cls):
        """Shutdown the worker pool."""
        if cls._pool is not None:
            cls._pool.close()
            cls._pool.join()
            cls._pool = None
            cls._pool_config = None

    # Expose the underlying MCTS class-level cache for compatibility
    prediction_cache = MCTS.prediction_cache

    @classmethod
    def clear_cache(cls):
        """Clear the MCTS prediction cache"""
        MCTS.clear_cache()
        cls.prediction_cache = MCTS.prediction_cache

    def __init__(self, model=None, num_simulations=100, c_puct=1.0,
                 dirichlet_scale=None, enable_early_stopping=True, num_workers=None):
        """
        Initialize optimized parallel MCTS.
        """
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = 1.0
        self.iteration = 0
        self.move_count = 0
        self.dirichlet_scale = dirichlet_scale
        self.enable_early_stopping = enable_early_stopping

        # Default to n-1 cores
        if num_workers is None:
            num_workers = max(1, mp.cpu_count() - 1)
        self.num_workers = num_workers

        # Get device and mode from model
        self.device = model.device if model else "cpu"
        self.model_mode = model.mode if hasattr(model, 'mode') else "stable"

        # Initialize RNG for worker seeds
        self.rng = np.random.default_rng(seed=42)

        # Cache statistics
        self.cache_hits = 0
        self.cache_misses = 0

        # Ensure pool exists with correct configuration
        self._ensure_pool(self.num_workers, self.device, self.model_mode)

        # Initialize pool with model if not already done
        if self._pool is None and model is not None:
            model_state_dict = model.model.state_dict()
            self._create_pool(model_state_dict, self.num_workers, self.device, self.model_mode)

    def search(self, game_state: GameState) -> tuple:
        """
        Perform optimized parallel MCTS search.
        """
        if self._pool is None:
            raise RuntimeError("Worker pool not initialized. Cannot search without a model.")

        # Distribute simulations across workers
        sims_per_worker = self.num_simulations // self.num_workers
        remainder = self.num_simulations % self.num_workers

        worker_sims = [sims_per_worker + (1 if i < remainder else 0)
                      for i in range(self.num_workers)]

        # Generate unique seeds for each worker
        worker_seeds = self.rng.integers(0, 2**31, size=self.num_workers)

        # Run parallel searches - only sending game state, not model!
        results = self._pool.starmap(
            _run_mcts_worker_v2,
            [
                (game_state, sims, self.c_puct, self.temperature, self.iteration,
                 self.move_count, self.dirichlet_scale, self.enable_early_stopping, seed)
                for sims, seed in zip(worker_sims, worker_seeds)
            ]
        )

        # Merge visit counts from all workers
        merged_visits = np.zeros((4, 4, 4), dtype=np.float32)
        roots = []

        for visit_counts, root in results:
            merged_visits += visit_counts
            roots.append(root)

        # Apply temperature to merged visit counts
        if abs(self.temperature) < 1e-9:
            best_idx = np.unravel_index(np.argmax(merged_visits), merged_visits.shape)
            probs = np.zeros_like(merged_visits)
            probs[best_idx] = 1.0
        else:
            visit_counts = merged_visits.copy()
            if self.temperature != 1.0:
                visit_counts = np.power(visit_counts, 1.0 / self.temperature)

            sum_visits = np.sum(visit_counts)
            if sum_visits > 0:
                probs = visit_counts / sum_visits
            else:
                legal_moves = game_state.get_legal_moves()
                probs = legal_moves / np.sum(legal_moves)

        return probs, roots[0] if roots else None

    def set_temperature(self, temperature):
        """Set temperature for move selection"""
        self.temperature = temperature

    def set_iteration(self, iteration):
        """Set current training iteration"""
        self.iteration = iteration

    def set_move_count(self, move_count):
        """Set current move count in the game"""
        self.move_count = move_count

    def print_timing_stats(self):
        """Print timing statistics"""
        print("Timing statistics not available for parallel MCTS")

    def reset_timing_stats(self):
        """Reset timing statistics"""
        pass


# Register cleanup on exit
atexit.register(ParallelMCTSv2.shutdown_pool)
