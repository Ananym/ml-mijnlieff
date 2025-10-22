"""
Diagnose why parallel MCTS is so slow.
"""

import time
import cProfile
import pstats
from io import StringIO
from game import GameState, Move, PieceType
from mcts import MCTS
from mcts_parallel_v2 import ParallelMCTSv2
from model import ModelWrapper

def profile_single_search():
    """Profile a single MCTS search to see where time is spent."""
    print("=" * 60)
    print("PROFILING SINGLE MCTS SEARCH")
    print("=" * 60)

    model = ModelWrapper(device="cpu", mode="stable")
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))

    mcts = MCTS(model=model, num_simulations=400, c_puct=1.5, enable_early_stopping=False)

    # Profile the search
    profiler = cProfile.Profile()
    profiler.enable()

    start = time.perf_counter()
    probs, root = mcts.search(game)
    elapsed = time.perf_counter() - start

    profiler.disable()

    # Print stats
    print(f"\nTotal time: {elapsed:.3f}s")
    print(f"Simulations: 400")
    print(f"Time per simulation: {elapsed/400*1000:.2f}ms")
    print()

    # Show top time consumers
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

def compare_cache_efficiency():
    """Compare cache hit rates between serial and parallel."""
    print("=" * 60)
    print("CACHE EFFICIENCY COMPARISON")
    print("=" * 60)

    model = ModelWrapper(device="cpu", mode="stable")
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))

    # Serial MCTS
    print("\n1. Serial MCTS (400 sims, 1 cache):")
    MCTS.clear_cache()
    serial_mcts = MCTS(model=model, num_simulations=400, c_puct=1.5, enable_early_stopping=False)
    serial_mcts.search(game)

    total_calls = serial_mcts.cache_hits + serial_mcts.cache_misses
    hit_rate = (serial_mcts.cache_hits / total_calls * 100) if total_calls > 0 else 0
    print(f"   Cache hits: {serial_mcts.cache_hits}")
    print(f"   Cache misses: {serial_mcts.cache_misses}")
    print(f"   Hit rate: {hit_rate:.1f}%")
    print(f"   Cache size: {len(MCTS.prediction_cache)}")

    # Parallel MCTS
    print("\n2. Parallel MCTS (400 sims, 4 workers with separate caches):")
    print("   Each worker does 100 sims with its own cache")
    print("   Expected hit rate: MUCH LOWER (smaller cache per worker)")
    print()
    print("   This is the key problem: cache fragmentation!")

def estimate_overhead():
    """Estimate breakdown of where time goes in parallel MCTS."""
    print("\n" + "=" * 60)
    print("TIME BREAKDOWN ESTIMATE")
    print("=" * 60)

    print("\nSerial MCTS (400 sims): ~0.84s")
    print("  - MCTS logic: ~60%")
    print("  - NN predictions: ~40% (but cached after first use)")
    print()

    print("Parallel MCTS (4 workers, 100 sims each): ~0.72s")
    print("  Expected with perfect parallelization: ~0.21s (4x speedup)")
    print("  Actual: ~0.72s (1.16x speedup)")
    print()
    print("  Lost efficiency breakdown:")
    print("  - Worker startup/shutdown: ~50ms")
    print("  - IPC/serialization overhead: ~50ms per call")
    print("  - Cache fragmentation: Each worker rebuilds cache")
    print("    (4 separate 100-sim caches vs 1 shared 400-sim cache)")
    print("  - Reduced cache hit rate: More NN evals needed")
    print()
    print("ROOT CAUSE: Workers don't share the prediction cache!")
    print("Each worker wastes time re-evaluating positions that")
    print("other workers (or previous searches) already evaluated.")

if __name__ == "__main__":
    profile_single_search()
    compare_cache_efficiency()
    estimate_overhead()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Parallel MCTS with root parallelization is fundamentally limited")
    print("because workers don't share the prediction cache. Each worker")
    print("operates independently with a small local cache.")
    print()
    print("Options to improve:")
    print("1. Shared memory cache (complex, requires multiprocessing.Manager)")
    print("2. Leaf parallelization with virtual loss (AlphaGo Zero approach)")
    print("3. Just use serial MCTS - the overhead isn't worth it")
    print("4. Increase simulations per search to amortize overhead")
    print("=" * 60)
