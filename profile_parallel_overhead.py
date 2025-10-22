"""
Profile the overhead in parallel MCTS to identify bottlenecks.
"""

import time
import pickle
import numpy as np
from model import ModelWrapper
from game import GameState, Move, PieceType

def measure_pickle_overhead():
    """Measure how long it takes to pickle the model and game state."""
    print("=" * 60)
    print("PICKLING OVERHEAD ANALYSIS")
    print("=" * 60)

    # Initialize model
    model = ModelWrapper(device="cpu", mode="stable")
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))
    game.make_move(Move(3, 3, PieceType.DIAG))

    # Test model pickling
    print("\n1. Model pickling:")
    start = time.perf_counter()
    for _ in range(10):
        pickled_model = pickle.dumps(model)
    model_time = (time.perf_counter() - start) / 10
    model_size = len(pickled_model) / 1024 / 1024  # MB

    print(f"   Time per pickle: {model_time*1000:.2f}ms")
    print(f"   Pickled size: {model_size:.2f}MB")

    # Test game state pickling
    print("\n2. GameState pickling:")
    start = time.perf_counter()
    for _ in range(1000):
        pickled_game = pickle.dumps(game)
    game_time = (time.perf_counter() - start) / 1000
    game_size = len(pickled_game) / 1024  # KB

    print(f"   Time per pickle: {game_time*1000:.2f}ms")
    print(f"   Pickled size: {game_size:.2f}KB")

    # Estimate overhead for 4 workers
    print("\n3. Total overhead per search (4 workers):")
    total_overhead = (model_time * 4) + (game_time * 4)
    print(f"   Model pickling: {model_time*4*1000:.2f}ms")
    print(f"   GameState pickling: {game_time*4*1000:.2f}ms")
    print(f"   Total: {total_overhead*1000:.2f}ms")

    # Estimate how this compares to search time
    print("\n4. Overhead as % of search time:")
    typical_search_time = 0.935  # seconds, from our benchmarks
    overhead_pct = (total_overhead / typical_search_time) * 100
    print(f"   Typical search time: {typical_search_time*1000:.0f}ms")
    print(f"   Overhead: {overhead_pct:.1f}%")
    print(f"   Theoretical speedup lost to overhead: {(1 / (1 - overhead_pct/100)):.2f}x")

    return model_time, game_time

if __name__ == "__main__":
    measure_pickle_overhead()

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("The model pickling overhead is the main bottleneck.")
    print("Solution: Use persistent workers that load the model once,")
    print("then only send game state on each search.")
    print("=" * 60)
