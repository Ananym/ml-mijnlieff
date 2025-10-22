"""
Benchmark to compare ParallelMCTS v1 (with model pickling) vs v2 (pre-loaded models).
"""

import time
import numpy as np
from game import GameState, Move, PieceType
from mcts import MCTS
from mcts_parallel import ParallelMCTS
from mcts_parallel_v2 import ParallelMCTSv2
from model import ModelWrapper

def run_comparison(num_searches=10, num_simulations=400):
    """
    Compare serial, parallel v1, and parallel v2 MCTS.
    """
    print("=" * 60)
    print("PARALLEL MCTS OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print(f"Searches: {num_searches}")
    print(f"Simulations per search: {num_simulations}")
    print()

    # Initialize model
    print("Initializing model...")
    model = ModelWrapper(device="cpu", mode="stable")
    print()

    # Create test game states
    print("Creating test game states...")
    game_states = []

    # Start position
    game_states.append(GameState())

    # Early game (2 moves)
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))
    game_states.append(game)

    # Mid game (4 moves)
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))
    game.make_move(Move(2, 2, PieceType.NEAR))
    game.make_move(Move(1, 1, PieceType.ORTHO))
    game_states.append(game)

    print(f"Created {len(game_states)} test positions")
    print()

    # Benchmark Serial MCTS
    print("=" * 60)
    print("SERIAL MCTS (BASELINE)")
    print("=" * 60)

    serial_mcts = MCTS(
        model=model,
        num_simulations=num_simulations,
        c_puct=1.5,
        enable_early_stopping=False
    )

    serial_times = []
    for i in range(num_searches):
        game_state = game_states[i % len(game_states)]

        start = time.perf_counter()
        probs, root = serial_mcts.search(game_state)
        elapsed = time.perf_counter() - start
        serial_times.append(elapsed)

        print(f"Search {i+1}/{num_searches}: {elapsed:.3f}s")

    serial_avg = np.mean(serial_times)
    serial_std = np.std(serial_times)

    print()
    print(f"Average: {serial_avg:.3f}s ± {serial_std:.3f}s")
    print()

    MCTS.clear_cache()

    # Benchmark Parallel MCTS v1
    print("=" * 60)
    print("PARALLEL MCTS V1 (WITH MODEL PICKLING)")
    print("=" * 60)

    parallel_v1 = ParallelMCTS(
        model=model,
        num_simulations=num_simulations,
        c_puct=1.5,
        enable_early_stopping=False,
        num_workers=4
    )

    v1_times = []
    for i in range(num_searches):
        game_state = game_states[i % len(game_states)]

        start = time.perf_counter()
        probs, root = parallel_v1.search(game_state)
        elapsed = time.perf_counter() - start
        v1_times.append(elapsed)

        print(f"Search {i+1}/{num_searches}: {elapsed:.3f}s")

    v1_avg = np.mean(v1_times)
    v1_std = np.std(v1_times)
    v1_speedup = serial_avg / v1_avg

    print()
    print(f"Average: {v1_avg:.3f}s ± {v1_std:.3f}s")
    print(f"Speedup vs Serial: {v1_speedup:.2f}x")
    print()

    MCTS.clear_cache()

    # Benchmark Parallel MCTS v2
    print("=" * 60)
    print("PARALLEL MCTS V2 (PRE-LOADED MODELS)")
    print("=" * 60)

    parallel_v2 = ParallelMCTSv2(
        model=model,
        num_simulations=num_simulations,
        c_puct=1.5,
        enable_early_stopping=False,
        num_workers=4
    )

    v2_times = []
    for i in range(num_searches):
        game_state = game_states[i % len(game_states)]

        start = time.perf_counter()
        probs, root = parallel_v2.search(game_state)
        elapsed = time.perf_counter() - start
        v2_times.append(elapsed)

        print(f"Search {i+1}/{num_searches}: {elapsed:.3f}s")

    v2_avg = np.mean(v2_times)
    v2_std = np.std(v2_times)
    v2_speedup = serial_avg / v2_avg

    print()
    print(f"Average: {v2_avg:.3f}s ± {v2_std:.3f}s")
    print(f"Speedup vs Serial: {v2_speedup:.2f}x")
    print()

    # Cleanup
    ParallelMCTSv2.shutdown_pool()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Serial MCTS:      {serial_avg:.3f}s  (1.00x)")
    print(f"Parallel v1:      {v1_avg:.3f}s  ({v1_speedup:.2f}x)")
    print(f"Parallel v2:      {v2_avg:.3f}s  ({v2_speedup:.2f}x)")
    print()
    print(f"v2 improvement over v1: {(v1_avg / v2_avg):.2f}x")
    print(f"v2 parallel efficiency: {(v2_speedup / 4 * 100):.1f}%")
    print()

    if v2_speedup > 2.0:
        print("SUCCESS: v2 achieves >2x speedup!")
    elif v2_speedup > v1_speedup:
        print("IMPROVEMENT: v2 is faster than v1")
    else:
        print("ISSUE: v2 is not faster than v1")

    return serial_avg, v1_avg, v2_avg


if __name__ == "__main__":
    run_comparison(num_searches=10, num_simulations=400)
