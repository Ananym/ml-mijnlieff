"""
Benchmark Numba-optimized game functions.
"""

import time
import numpy as np
from game import GameState, Move, PieceType, Player
from game_numba import get_legal_moves_numba, calculate_score_numba, is_valid_move_numba

def benchmark_get_legal_moves(num_iterations=10000):
    """Benchmark get_legal_moves with and without Numba."""

    # Create test game states at different stages
    states = []

    # Empty board
    states.append(GameState())

    # After a few moves
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))
    states.append(game)

    # Mid game
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))
    game.make_move(Move(2, 2, PieceType.NEAR))
    game.make_move(Move(1, 1, PieceType.ORTHO))
    states.append(game)

    print("=" * 60)
    print("BENCHMARKING get_legal_moves()")
    print("=" * 60)
    print(f"Iterations: {num_iterations}")
    print()

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    for _ in range(100):
        for state in states:
            _ = get_legal_moves_numba(state)
    print()

    # Benchmark original
    print("Testing ORIGINAL implementation...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        for state in states:
            _ = state.get_legal_moves()
    original_time = time.perf_counter() - start

    print(f"  Time: {original_time:.3f}s")
    print(f"  Throughput: {num_iterations * len(states) / original_time:.0f} calls/sec")
    print()

    # Benchmark Numba
    print("Testing NUMBA implementation...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        for state in states:
            _ = get_legal_moves_numba(state)
    numba_time = time.perf_counter() - start

    print(f"  Time: {numba_time:.3f}s")
    print(f"  Throughput: {num_iterations * len(states) / numba_time:.0f} calls/sec")
    print()

    speedup = original_time / numba_time
    print(f"SPEEDUP: {speedup:.2f}x")
    print()

    # Verify correctness
    print("Verifying correctness...")
    all_match = True
    for state in states:
        original = state.get_legal_moves()
        numba = get_legal_moves_numba(state)
        if not np.array_equal(original, numba):
            all_match = False
            print(f"  MISMATCH found!")
            break

    if all_match:
        print("  [OK] All results match!")
    print()

    return speedup


def benchmark_calculate_score(num_iterations=10000):
    """Benchmark _calculate_score with and without Numba."""

    # Create test game states
    states = []

    # Just use a couple simple states
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))
    states.append(game)

    print("=" * 60)
    print("BENCHMARKING _calculate_score()")
    print("=" * 60)
    print(f"Iterations: {num_iterations}")
    print()

    # Warm up Numba JIT
    print("Warming up Numba JIT...")
    for _ in range(100):
        for state in states:
            _ = calculate_score_numba(state, Player.ONE)
    print()

    # Benchmark original
    print("Testing ORIGINAL implementation...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        for state in states:
            _ = state._calculate_score(Player.ONE)
            _ = state._calculate_score(Player.TWO)
    original_time = time.perf_counter() - start

    print(f"  Time: {original_time:.3f}s")
    print(f"  Throughput: {num_iterations * len(states) * 2 / original_time:.0f} calls/sec")
    print()

    # Benchmark Numba
    print("Testing NUMBA implementation...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        for state in states:
            _ = calculate_score_numba(state, Player.ONE)
            _ = calculate_score_numba(state, Player.TWO)
    numba_time = time.perf_counter() - start

    print(f"  Time: {numba_time:.3f}s")
    print(f"  Throughput: {num_iterations * len(states) * 2 / numba_time:.0f} calls/sec")
    print()

    speedup = original_time / numba_time
    print(f"SPEEDUP: {speedup:.2f}x")
    print()

    # Verify correctness
    print("Verifying correctness...")
    all_match = True
    for state in states:
        for player in [Player.ONE, Player.TWO]:
            original = state._calculate_score(player)
            numba = calculate_score_numba(state, player)
            if original != numba:
                all_match = False
                print(f"  MISMATCH: original={original}, numba={numba}")
                break
        if not all_match:
            break

    if all_match:
        print("  [OK] All results match!")
    print()

    return speedup


if __name__ == "__main__":
    print("\nNUMBA OPTIMIZATION BENCHMARK")
    print("=" * 60)
    print()

    # Install numba if not present
    try:
        import numba
        print(f"Numba version: {numba.__version__}")
        print()
    except ImportError:
        print("Numba not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "numba"])
        print()

    speedup_legal_moves = benchmark_get_legal_moves(num_iterations=10000)
    speedup_score = benchmark_calculate_score(num_iterations=10000)

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"get_legal_moves():  {speedup_legal_moves:.2f}x faster")
    print(f"_calculate_score(): {speedup_score:.2f}x faster")
    print()

    # Estimate impact on MCTS
    # Based on profiling: ~33% of time is in make_move/undo
    # get_legal_moves is called during that, and scoring is also called
    print("Estimated impact on MCTS:")
    print("  If 33% of MCTS time is in game logic,")
    print("  and we speed that up by ~2-3x,")
    print("  Overall MCTS speedup: ~1.2-1.3x")
