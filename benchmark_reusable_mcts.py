"""
Benchmark reusable MCTS instance vs creating new instances.
This tests the real-world scenario where we reuse the same MCTS for many searches.
"""

import time
import numpy as np
from game import GameState, Move, PieceType
from mcts import MCTS as SerialMCTS
from mcts_parallel_v2 import ParallelMCTSv2
from model import ModelWrapper

def benchmark_reusable_instance():
    print("=" * 60)
    print("BENCHMARK: Reusable MCTS Instance")
    print("=" * 60)
    print("Simulating 20 searches (similar to ~16 moves per game)")
    print()

    model = ModelWrapper(device="cpu", mode="stable")

    # Create test game states
    game_states = []
    for i in range(20):
        game = GameState()
        # Make some random moves to create variety
        for _ in range(min(i % 5, 4)):
            legal = game.get_legal_moves()
            if np.any(legal):
                moves = np.argwhere(legal > 0)
                if len(moves) > 0:
                    x, y, piece = moves[np.random.randint(len(moves))]
                    game.make_move(Move(x, y, PieceType(piece)))
        game_states.append(game)

    # Test Serial MCTS (baseline)
    print("1. Serial MCTS (Baseline):")
    print("   Creating new instance each search...")
    SerialMCTS.clear_cache()

    start = time.perf_counter()
    for game in game_states:
        mcts = SerialMCTS(model=model, num_simulations=400, c_puct=1.5)
        mcts.search(game)
    serial_new_time = time.perf_counter() - start

    print(f"   Total time: {serial_new_time:.2f}s")
    print(f"   Time per search: {serial_new_time/20*1000:.0f}ms")
    print()

    # Test Serial MCTS with reusable instance
    print("2. Serial MCTS (Reusable Instance):")
    print("   Using single instance for all searches...")
    SerialMCTS.clear_cache()

    reusable_serial = SerialMCTS(model=model, num_simulations=400, c_puct=1.5)

    start = time.perf_counter()
    for game in game_states:
        reusable_serial.search(game)
    serial_reuse_time = time.perf_counter() - start

    print(f"   Total time: {serial_reuse_time:.2f}s")
    print(f"   Time per search: {serial_reuse_time/20*1000:.0f}ms")
    print(f"   Speedup vs creating new: {serial_new_time/serial_reuse_time:.2f}x")
    print()

    # Test Parallel MCTS v2 (creating new instances)
    print("3. Parallel MCTS v2 (Creating New Instances):")
    print("   WARNING: This recreates worker pools each time!")
    SerialMCTS.clear_cache()

    start = time.perf_counter()
    for game in game_states:
        parallel = ParallelMCTSv2(model=model, num_simulations=400, c_puct=1.5, num_workers=7)
        parallel.search(game)
    ParallelMCTSv2.shutdown_pool()
    parallel_new_time = time.perf_counter() - start

    print(f"   Total time: {parallel_new_time:.2f}s")
    print(f"   Time per search: {parallel_new_time/20*1000:.0f}ms")
    print(f"   Speedup vs Serial: {serial_new_time/parallel_new_time:.2f}x (POOR!)")
    print()

    # Test Parallel MCTS v2 (reusable instance) - THE KEY TEST
    print("4. Parallel MCTS v2 (Reusable Instance, 7 Workers):")
    print("   Using single instance with persistent workers...")
    SerialMCTS.clear_cache()

    reusable_parallel = ParallelMCTSv2(model=model, num_simulations=400, c_puct=1.5, num_workers=7)

    start = time.perf_counter()
    for game in game_states:
        reusable_parallel.search(game)
    parallel_reuse_time = time.perf_counter() - start

    ParallelMCTSv2.shutdown_pool()

    print(f"   Total time: {parallel_reuse_time:.2f}s")
    print(f"   Time per search: {parallel_reuse_time/20*1000:.0f}ms")
    print(f"   Speedup vs Serial (reusable): {serial_reuse_time/parallel_reuse_time:.2f}x")
    print(f"   Speedup vs Parallel (new instances): {parallel_new_time/parallel_reuse_time:.2f}x")
    print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Serial (new instances):      {serial_new_time:.2f}s  (1.00x)")
    print(f"Serial (reusable):           {serial_reuse_time:.2f}s  ({serial_new_time/serial_reuse_time:.2f}x)")
    print(f"Parallel (new instances):    {parallel_new_time:.2f}s  ({serial_new_time/parallel_new_time:.2f}x)")
    print(f"Parallel (reusable, 7 CPU):  {parallel_reuse_time:.2f}s  ({serial_new_time/parallel_reuse_time:.2f}x) 🚀")
    print()

    if parallel_reuse_time < serial_reuse_time:
        speedup_pct = (serial_reuse_time / parallel_reuse_time - 1) * 100
        print(f"✅ SUCCESS! Parallel MCTS with reusable instance is {speedup_pct:.0f}% faster!")
        print(f"   In training: ~{speedup_pct:.0f}% faster iteration times")
    else:
        print("❌ Parallel MCTS is still slower. Cache fragmentation may be the issue.")
    print("=" * 60)

if __name__ == "__main__":
    benchmark_reusable_instance()
