"""
Parallel MCTS implementation using root parallelization.

This module provides a drop-in replacement for MCTS that uses multiple workers
to run independent searches in parallel, then merges the results.
"""

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from mcts import MCTS, clone_game_state
from game import GameState
import atexit

# Global worker pool to avoid recreation overhead
_global_parallel = None
_global_num_workers = None

def _get_global_parallel(num_workers):
    """Get or create the global Parallel worker pool."""
    global _global_parallel, _global_num_workers

    if _global_parallel is None or _global_num_workers != num_workers:
        # Clean up old pool if it exists
        if _global_parallel is not None:
            del _global_parallel

        # Create new persistent pool
        _global_parallel = Parallel(n_jobs=num_workers, backend='loky')
        _global_num_workers = num_workers

    return _global_parallel

def _cleanup_global_parallel():
    """Clean up the global parallel pool on exit."""
    global _global_parallel
    if _global_parallel is not None:
        del _global_parallel
        _global_parallel = None

# Register cleanup on exit
atexit.register(_cleanup_global_parallel)


def _run_mcts_worker(game_state, model, num_simulations, c_puct, temperature, iteration, move_count, dirichlet_scale, enable_early_stopping, worker_seed):
    """
    Worker function that runs MCTS search.

    Args:
        game_state: The game state to search from
        model: Neural network model
        num_simulations: Number of simulations for this worker
        c_puct: Exploration constant
        temperature: Temperature for move selection
        iteration: Current training iteration
        move_count: Current move count
        dirichlet_scale: Dirichlet noise scale
        enable_early_stopping: Whether to enable early stopping
        worker_seed: Random seed for this worker

    Returns:
        Tuple of (visit_counts array, root node)
    """
    # Create independent MCTS instance for this worker
    # Each worker gets its own RNG state via the seed
    worker_mcts = MCTS(
        model=model,
        num_simulations=num_simulations,
        c_puct=c_puct,
        dirichlet_scale=dirichlet_scale,
        enable_early_stopping=enable_early_stopping
    )
    worker_mcts.set_temperature(temperature)
    worker_mcts.set_iteration(iteration)
    worker_mcts.set_move_count(move_count)

    # Set worker-specific random seed for diversity
    np.random.seed(worker_seed)

    # Run search
    probs, root = worker_mcts.search(game_state)

    # Extract visit counts from root children
    visit_counts = np.zeros((4, 4, 4), dtype=np.float32)
    for child in root.children:
        if child.move:
            x, y, piece_type = child.move.x, child.move.y, child.move.piece_type.value
            visit_counts[x, y, piece_type] = child.visits

    return visit_counts, root


class ParallelMCTS:
    """
    Parallel MCTS implementation using root parallelization.

    This class has the same interface as MCTS but runs multiple independent
    searches in parallel and merges the results.
    """

    # Expose the underlying MCTS class-level cache for compatibility
    prediction_cache = MCTS.prediction_cache

    @classmethod
    def clear_cache(cls):
        """Clear the MCTS prediction cache (delegates to underlying MCTS class)"""
        MCTS.clear_cache()
        # Update our reference to point to the cleared cache
        cls.prediction_cache = MCTS.prediction_cache

    def __init__(self, model=None, num_simulations=100, c_puct=1.0,
                 dirichlet_scale=None, enable_early_stopping=True, num_workers=None):
        """
        Initialize parallel MCTS.

        Args:
            model: Neural network model (optional)
            num_simulations: Total number of simulations across all workers
            c_puct: Exploration constant
            dirichlet_scale: Dirichlet noise scale (None uses default)
            enable_early_stopping: Whether to enable early stopping
            num_workers: Number of parallel workers (defaults to cpu_count - 1)
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
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.num_workers = num_workers

        # Initialize RNG for worker seeds
        self.rng = np.random.default_rng(seed=42)

        # Cache statistics (for compatibility with MCTS interface)
        self.cache_hits = 0
        self.cache_misses = 0

    def search(self, game_state: GameState) -> tuple:
        """
        Perform parallel MCTS search and return move probabilities.

        Args:
            game_state: Current game state

        Returns:
            Tuple of (move probabilities array, merged root node)
        """
        # Distribute simulations across workers
        sims_per_worker = self.num_simulations // self.num_workers
        remainder = self.num_simulations % self.num_workers

        # Build list of simulation counts (distribute remainder evenly)
        worker_sims = [sims_per_worker + (1 if i < remainder else 0)
                      for i in range(self.num_workers)]

        # Generate unique seeds for each worker
        worker_seeds = self.rng.integers(0, 2**31, size=self.num_workers)

        # Use global worker pool for better performance
        parallel = _get_global_parallel(self.num_workers)

        # Run parallel searches using the persistent worker pool
        results = parallel(
            delayed(_run_mcts_worker)(
                game_state,
                self.model,
                sims,
                self.c_puct,
                self.temperature,
                self.iteration,
                self.move_count,
                self.dirichlet_scale,
                self.enable_early_stopping,
                seed
            )
            for sims, seed in zip(worker_sims, worker_seeds)
        )

        # Merge visit counts from all workers
        merged_visits = np.zeros((4, 4, 4), dtype=np.float32)
        roots = []

        for visit_counts, root in results:
            merged_visits += visit_counts
            roots.append(root)

        # Apply temperature to merged visit counts
        if abs(self.temperature) < 1e-9:  # epsilon check
            # Choose most visited move deterministically
            best_idx = np.unravel_index(np.argmax(merged_visits), merged_visits.shape)
            probs = np.zeros_like(merged_visits)
            probs[best_idx] = 1.0
        else:
            # Convert visits to probabilities with temperature
            visit_counts = merged_visits.copy()
            if self.temperature != 1.0:
                visit_counts = np.power(visit_counts, 1.0 / self.temperature)

            sum_visits = np.sum(visit_counts)
            if sum_visits > 0:
                probs = visit_counts / sum_visits
            else:
                # Fallback to uniform distribution if no visits
                legal_moves = game_state.get_legal_moves()
                probs = legal_moves / np.sum(legal_moves)

        # Return first root as representative (they should all have similar structure)
        # This is mainly for compatibility with code that expects a root node
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
        """Print timing statistics (not implemented for parallel version)"""
        print("Timing statistics not available for parallel MCTS")

    def reset_timing_stats(self):
        """Reset timing statistics (not implemented for parallel version)"""
        pass
