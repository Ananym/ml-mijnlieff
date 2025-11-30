"""Benchmark MCTS inference time at different simulation counts for production deployment"""
import time
import torch
import numpy as np
from game import GameState, Move, PieceType
from model import ModelWrapper
from mcts import MCTS

def benchmark_inference(model, sim_counts=[0, 10, 25, 50, 100, 200], num_positions=5, games_per_count=3):
    """Benchmark model inference and MCTS at different simulation counts"""

    print("=" * 60)
    print("MCTS Production Inference Benchmark")
    print("=" * 60)

    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print()

    results = {}

    for sims in sim_counts:
        times = []

        for game_idx in range(games_per_count):
            # Create fresh game
            game = GameState()

            # Make a few random legal moves to get varied positions
            for _ in range(game_idx * 2):
                if game.is_over:
                    break
                legal = game.get_legal_moves()
                if np.any(legal):
                    indices = np.argwhere(legal)
                    idx = indices[np.random.randint(len(indices))]
                    game.make_move(Move(idx[0], idx[1], PieceType(idx[2])))

            # Benchmark multiple positions per game
            for pos_idx in range(num_positions):
                if game.is_over:
                    break

                start = time.perf_counter()

                if sims == 0:
                    # Raw policy only (no MCTS)
                    state_rep = game.get_game_state_representation(subjective=True)
                    legal_moves = game.get_legal_moves()
                    policy, value = model.predict(state_rep.board, state_rep.flat_values, legal_moves)
                    probs = np.squeeze(policy, 0) if hasattr(policy, 'shape') else policy.squeeze(0).numpy()
                else:
                    # MCTS with specified simulations
                    mcts = MCTS(model=model, num_simulations=sims, c_puct=1.0, dirichlet_scale=0.0)
                    probs, _ = mcts.search(game)

                elapsed = time.perf_counter() - start
                times.append(elapsed)

                # Make a move to get to next position
                legal = game.get_legal_moves()
                if np.any(legal):
                    masked_probs = probs * legal
                    if np.sum(masked_probs) > 0:
                        masked_probs /= np.sum(masked_probs)
                    idx = np.unravel_index(np.argmax(masked_probs), masked_probs.shape)
                    game.make_move(Move(idx[0], idx[1], PieceType(idx[2])))

        avg_time = np.mean(times)
        std_time = np.std(times)
        results[sims] = {'avg': avg_time, 'std': std_time, 'samples': len(times)}

        label = "Raw Policy" if sims == 0 else f"{sims} sims"
        print(f"{label:12s}: {avg_time*1000:7.1f}ms ± {std_time*1000:5.1f}ms  ({len(times)} samples)")

    print()
    print("-" * 60)
    print("Production Recommendations:")
    print("-" * 60)

    # Calculate relative performance
    if 0 in results and 100 in results:
        raw_time = results[0]['avg']
        mcts100_time = results[100]['avg']
        slowdown = mcts100_time / raw_time
        print(f"100 sims is {slowdown:.1f}x slower than raw policy")

    print()
    print("Suggested configurations:")
    for sims, desc in [(0, "Instant response, policy only"),
                        (10, "Fast casual play"),
                        (25, "Balanced"),
                        (50, "Strong play"),
                        (100, "Maximum strength")]:
        if sims in results:
            t = results[sims]['avg']
            print(f"  {sims:3d} sims: ~{t*1000:6.1f}ms/move - {desc}")

    return results

if __name__ == "__main__":
    print("Loading model...")
    model = ModelWrapper()

    # Try to load the Exp 12 model
    import os
    model_path = "models/model_latest.pt"
    if os.path.exists(model_path):
        model.load(model_path)
        print(f"Loaded {model_path}")
    else:
        print("WARNING: No saved model found, using random weights")

    results = benchmark_inference(model)
