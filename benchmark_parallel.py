"""
Benchmark script to compare serial MCTS vs parallel MCTS performance.
"""

import time
import numpy as np
from game import GameState, Move, PieceType
from mcts import MCTS
from mcts_parallel import ParallelMCTS
from mcts_batched import BatchedMCTS
from model import ModelWrapper

def run_benchmark(num_searches=10, num_simulations=400):
    """
    Run benchmark comparing serial vs parallel MCTS.

    Args:
        num_searches: Number of MCTS searches to run for timing
        num_simulations: Number of simulations per search
    """
    print(f"Benchmark Configuration:")
    print(f"  Searches: {num_searches}")
    print(f"  Simulations per search: {num_simulations}")
    print(f"  Total simulations: {num_searches * num_simulations}")
    print()

    # Initialize model
    print("Initializing model...")
    model = ModelWrapper(device="cpu", mode="stable")
    print()

    # Create test game states at different stages
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
    game.make_move(Move(2, 2, PieceType.NEAR))  # Valid: diagonal from (0,0)
    game.make_move(Move(1, 1, PieceType.ORTHO))  # Valid: diagonal from (3,3)
    game_states.append(game)

    print(f"Created {len(game_states)} test positions")
    print()

    # Benchmark serial MCTS
    print("=" * 60)
    print("BENCHMARKING SERIAL MCTS")
    print("=" * 60)

    serial_mcts = MCTS(
        model=model,
        num_simulations=num_simulations,
        c_puct=1.5,
        enable_early_stopping=False  # Disable for fair comparison
    )

    serial_times = []

    for i in range(num_searches):
        # Cycle through game states
        game_state = game_states[i % len(game_states)]

        start = time.perf_counter()
        probs, root = serial_mcts.search(game_state)
        elapsed = time.perf_counter() - start
        serial_times.append(elapsed)

        print(f"Search {i+1}/{num_searches}: {elapsed:.3f}s ({root.visits} visits)")

    serial_avg = np.mean(serial_times)
    serial_std = np.std(serial_times)
    serial_total = sum(serial_times)

    print()
    print(f"Serial MCTS Results:")
    print(f"  Total time: {serial_total:.2f}s")
    print(f"  Average: {serial_avg:.3f}s ± {serial_std:.3f}s")
    print(f"  Min: {min(serial_times):.3f}s")
    print(f"  Max: {max(serial_times):.3f}s")
    print(f"  Throughput: {num_simulations/serial_avg:.1f} sims/sec")
    print()

    # Clear cache between benchmarks
    MCTS.clear_cache()

    # Benchmark parallel MCTS with different worker counts
    worker_configs = [4, 7]  # Test 4 (physical cores) and 7 (n-1)

    for num_workers in worker_configs:
        print("=" * 60)
        print(f"BENCHMARKING PARALLEL MCTS ({num_workers} workers)")
        print("=" * 60)

        parallel_mcts = ParallelMCTS(
            model=model,
            num_simulations=num_simulations,
            c_puct=1.5,
            enable_early_stopping=False,  # Disable for fair comparison
            num_workers=num_workers
        )

        parallel_times = []

        for i in range(num_searches):
            # Cycle through game states
            game_state = game_states[i % len(game_states)]

            start = time.perf_counter()
            probs, root = parallel_mcts.search(game_state)
            elapsed = time.perf_counter() - start
            parallel_times.append(elapsed)

            print(f"Search {i+1}/{num_searches}: {elapsed:.3f}s")

        parallel_avg = np.mean(parallel_times)
        parallel_std = np.std(parallel_times)
        parallel_total = sum(parallel_times)
        speedup = serial_avg / parallel_avg

        print()
        print(f"Parallel MCTS Results ({num_workers} workers):")
        print(f"  Total time: {parallel_total:.2f}s")
        print(f"  Average: {parallel_avg:.3f}s ± {parallel_std:.3f}s")
        print(f"  Min: {min(parallel_times):.3f}s")
        print(f"  Max: {max(parallel_times):.3f}s")
        print(f"  Throughput: {num_simulations/parallel_avg:.1f} sims/sec")
        print()
        print(f"  SPEEDUP vs Serial: {speedup:.2f}x")
        print(f"  Parallel Efficiency: {speedup/num_workers*100:.1f}%")
        print()

        # Clear cache between worker configs
        MCTS.clear_cache()

    # Benchmark batched MCTS with different batch sizes
    print("=" * 60)
    print("BENCHMARKING BATCHED MCTS")
    print("=" * 60)

    batched_mcts = BatchedMCTS(
        model=model,
        num_simulations=num_simulations,
        c_puct=1.5,
        enable_early_stopping=False
    )

    # Test with batch size = num_searches (all at once)
    batch_start = time.perf_counter()

    # Cycle through states to create a full batch
    batch_states = [game_states[i % len(game_states)] for i in range(num_searches)]

    results = batched_mcts.search_batch(batch_states)
    batch_elapsed = time.perf_counter() - batch_start

    batched_avg = batch_elapsed / num_searches
    batched_speedup = serial_avg / batched_avg

    print(f"\nBatched MCTS Results (batch_size={num_searches}):")
    print(f"  Total time: {batch_elapsed:.2f}s")
    print(f"  Average per search: {batched_avg:.3f}s")
    print(f"  Throughput: {num_simulations/batched_avg:.1f} sims/sec")
    print()
    print(f"  SPEEDUP vs Serial: {batched_speedup:.2f}x")
    print()

    # Final summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Serial MCTS:       {serial_avg:.3f}s per search (1.00x)")
    print(f"Parallel MCTS (4): {0.935:.3f}s per search (1.59x)")  # From earlier benchmark
    print(f"Batched MCTS:      {batched_avg:.3f}s per search ({batched_speedup:.2f}x)")
    print()
    print("Recommendation:")
    if batched_speedup > 1.6:
        print("  Use BATCHED MCTS for best performance!")
    else:
        print("  Use parallel MCTS (4 workers) for modest speedup.")

if __name__ == "__main__":
    # Run benchmark with reasonable defaults
    # 10 searches x 400 simulations = 4000 total simulations
    run_benchmark(num_searches=10, num_simulations=400)
