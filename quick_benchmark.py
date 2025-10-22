"""
Quick benchmark: Serial vs Parallel MCTS with reusable instances.
"""

import time
import numpy as np
from game import GameState, Move, PieceType
from mcts import MCTS as SerialMCTS
from mcts_parallel_v2 import ParallelMCTSv2
from model import ModelWrapper

if __name__ == "__main__":
    print("=" * 60)
    print("QUICK BENCHMARK: Reusable MCTS Instance (5 searches)")
    print("=" * 60)

    model = ModelWrapper(device="cpu", mode="stable")

    # Create 5 test game states
    game_states = []
    for i in range(5):
        game = GameState()
        for _ in range(i % 3):
            legal = game.get_legal_moves()
            if np.any(legal):
                moves = np.argwhere(legal > 0)
                if len(moves) > 0:
                    x, y, piece = moves[np.random.randint(len(moves))]
                    game.make_move(Move(x, y, PieceType(piece)))
        game_states.append(game)

    # Serial MCTS (reusable instance)
    print("\n1. Serial MCTS (Reusable Instance):")
    SerialMCTS.clear_cache()
    reusable_serial = SerialMCTS(model=model, num_simulations=400, c_puct=1.5)

    start = time.perf_counter()
    for game in game_states:
        reusable_serial.search(game)
    serial_time = time.perf_counter() - start

    print(f"   Total: {serial_time:.2f}s | Per search: {serial_time/5*1000:.0f}ms")

    # Parallel MCTS v2 (reusable instance, 7 workers)
    print("\n2. Parallel MCTS v2 (Reusable Instance, 7 Workers):")
    SerialMCTS.clear_cache()
    reusable_parallel = ParallelMCTSv2(model=model, num_simulations=400, c_puct=1.5, num_workers=7)

    start = time.perf_counter()
    for game in game_states:
        reusable_parallel.search(game)
    parallel_time = time.perf_counter() - start

    ParallelMCTSv2.shutdown_pool()

    print(f"   Total: {parallel_time:.2f}s | Per search: {parallel_time/5*1000:.0f}ms")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Serial:   {serial_time:.2f}s (baseline)")
    print(f"Parallel: {parallel_time:.2f}s ({serial_time/parallel_time:.2f}x speedup)")

    if parallel_time < serial_time:
        speedup_pct = (serial_time / parallel_time - 1) * 100
        efficiency_pct = (serial_time / parallel_time / 7) * 100
        print(f"\n✅ Parallel is {speedup_pct:.0f}% faster!")
        print(f"   Parallel efficiency: {efficiency_pct:.0f}% (7 workers)")
        print(f"   Training iterations will be ~{speedup_pct:.0f}% faster")
    else:
        print("\n⚠️  Parallel is slower - likely cache fragmentation")
        print("   Each worker maintains separate cache (less efficient)")
    print("=" * 60)
