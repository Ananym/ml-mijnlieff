"""
Debug script to investigate P1/P2 bias in the TicTacDo AI model.

This script helps identify why the model shows significant performance differences
when playing as Player 1 vs Player 2 (e.g., 40% vs 18% win rates).

Potential causes:
1. Training data imbalance (more P1 or P2 examples)
2. Model architecture asymmetry (unlikely but possible)
3. Missing or broken data augmentation (not flipping perspectives)
4. Feature representation issues (board state encoding favors one player)
5. Natural game asymmetry (P1 has first move advantage)
"""

import torch
import numpy as np
from game import GameState, Player, PieceType, Move
from model import ModelWrapper
from mcts import MCTS
from typing import List, Tuple
import argparse
from collections import defaultdict

def play_self_play_games(model: ModelWrapper, num_games: int = 100, use_mcts: bool = False, mcts_sims: int = 100):
    """
    Play games where model plays against itself.

    Args:
        model: The model to evaluate
        num_games: Number of games to play
        use_mcts: If True, use MCTS for move selection. If False, use raw policy.
        mcts_sims: Number of MCTS simulations (if use_mcts=True)

    Returns:
        Dictionary with win/draw/loss stats from P1 and P2 perspectives
    """
    stats = {
        'p1_wins': 0,
        'p2_wins': 0,
        'draws': 0,
        'p1_scores': [],
        'p2_scores': [],
    }

    if use_mcts:
        mcts = MCTS(model=model, num_simulations=mcts_sims, c_puct=1.0)
        print(f"Playing {num_games} self-play games with MCTS ({mcts_sims} sims)...")
    else:
        print(f"Playing {num_games} self-play games with raw policy...")

    for game_idx in range(num_games):
        game = GameState()
        move_count = 0
        max_moves = 100  # Prevent infinite games

        while not game.check_game_over() and move_count < max_moves:
            if use_mcts:
                # Use MCTS for move selection
                action_probs = mcts.get_action_probs(game, temperature=0.0)
                action_probs_flat = action_probs.flatten()
                valid_moves = game.get_legal_moves()

                # Mask invalid moves
                for i, (r, c, p) in enumerate(valid_moves):
                    action_idx = r * 4 * 4 + c * 4 + p

                # Choose best action
                valid_indices = [r * 4 * 4 + c * 4 + p for r, c, p in valid_moves]
                valid_probs = [action_probs_flat[i] for i in valid_indices]
                best_idx = valid_indices[np.argmax(valid_probs)]

                row = best_idx // 16
                col = (best_idx % 16) // 4
                piece = best_idx % 4
            else:
                # Use raw policy for move selection
                state_rep = game.get_game_state_representation(subjective=True)
                legal_moves = game.get_legal_moves()
                policy, value = model.predict(state_rep.board, state_rep.flat_values, legal_moves)
                policy = policy.squeeze(0)  # Remove batch dimension

                # Get list of valid move coordinates
                legal_indices = np.argwhere(legal_moves)
                if len(legal_indices) == 0:
                    break  # No valid moves left

                # Get policy scores for valid moves
                move_probs = []
                for row, col, piece_type in legal_indices:
                    prob = policy[row, col, piece_type]
                    move_probs.append(prob)

                # Sample from policy distribution (with small temperature for diversity)
                move_probs = np.array(move_probs)
                # Normalize to create valid probability distribution
                if move_probs.sum() > 0:
                    move_probs = move_probs / move_probs.sum()
                else:
                    # Uniform if all zeros
                    move_probs = np.ones(len(move_probs)) / len(move_probs)

                # Sample move according to policy
                best_move_idx = np.random.choice(len(move_probs), p=move_probs)
                row, col, piece = legal_indices[best_move_idx]

            move = Move(row, col, PieceType(piece))
            game.make_move(move)
            move_count += 1

        # Record result
        scores = game.get_scores()
        p1_score = scores[Player.ONE]
        p2_score = scores[Player.TWO]

        stats['p1_scores'].append(p1_score)
        stats['p2_scores'].append(p2_score)

        if p1_score > p2_score:
            stats['p1_wins'] += 1
        elif p2_score > p1_score:
            stats['p2_wins'] += 1
        else:
            stats['draws'] += 1

    return stats


def check_position_symmetry(model: ModelWrapper, num_positions: int = 50):
    """
    Check if the model makes symmetric predictions when the same position
    is viewed from P1 vs P2 perspective.

    This tests if the model has learned any inherent bias in how it evaluates
    positions depending on which player's turn it is.
    """
    print(f"\nChecking position symmetry across {num_positions} random positions...")

    value_diffs = []
    policy_diffs = []

    for _ in range(num_positions):
        # Create a random mid-game position
        game = GameState()
        num_moves = np.random.randint(5, 15)

        for _ in range(num_moves):
            if game.check_game_over():
                break
            legal_moves = game.get_legal_moves()
            legal_indices = np.argwhere(legal_moves)
            if len(legal_indices) == 0:
                break
            move_idx = np.random.randint(len(legal_indices))
            row, col, piece = legal_indices[move_idx]
            move = Move(row, col, PieceType(piece))
            game.make_move(move)

        if game.check_game_over():
            continue

        # Get prediction from current player perspective
        state_rep = game.get_game_state_representation(subjective=True)
        legal_moves = game.get_legal_moves()
        policy1, value1 = model.predict(state_rep.board, state_rep.flat_values, legal_moves)

        # Flip to other player's perspective (hypothetically - we'd need to implement this)
        # For now, we'll just check consistency by making a move and undoing it
        # This is a simplified check

        # Store for analysis
        # In a full implementation, you'd flip the board representation and compare

    print("Note: Full symmetry check requires board flipping implementation")
    return


def analyze_training_data(replay_buffer_path: str = None):
    """
    Analyze the balance of training examples from P1 vs P2 perspective.

    This checks if the replay buffer has equal representation of both players.
    """
    print("\nAnalyzing training data balance...")
    print("Note: This requires access to the replay buffer from a training run")

    # If we had access to replay buffer, we would:
    # 1. Count examples from P1 perspective vs P2 perspective
    # 2. Check win/loss/draw distribution for each
    # 3. Verify data augmentation is working

    return


def test_feature_representation(model: ModelWrapper):
    """
    Test if the board state encoding is symmetric for both players.

    This verifies that the way we represent the game state doesn't
    inadvertently favor one player.
    """
    print("\nTesting feature representation...")

    # Create a simple position
    game = GameState()
    game.make_move(Move(0, 0, PieceType.DIAG))  # P1 plays
    game.make_move(Move(0, 1, PieceType.ORTHO))  # P2 plays

    # Get model's view
    state_rep = game.get_game_state_representation(subjective=True)
    legal_moves = game.get_legal_moves()
    policy, value = model.predict(state_rep.board, state_rep.flat_values, valid_moves)

    print(f"After 2 moves (P1's turn):")
    print(f"  Value prediction: {value.item():.3f}")
    print(f"  Current player: {game.current_player}")

    # Make another move
    game.make_move(Move(1, 0, PieceType.NEAR))  # P1 plays

    state_rep2 = game.get_game_state_representation(subjective=True)
    legal_moves2 = game.get_legal_moves()
    policy2, value2 = model.predict(state_rep2.board, state_rep2.flat_values, legal_moves2)
    print(f"\nAfter 3 moves (P2's turn):")
    print(f"  Value prediction: {value2.item():.3f}")
    print(f"  Current player: {game.current_player}")

    return


def analyze_policy_distribution(model: ModelWrapper, num_positions: int = 100):
    """
    Analyze if policy distributions differ systematically between P1 and P2 turns.
    """
    print(f"\nAnalyzing policy distributions across {num_positions} positions...")

    p1_entropies = []
    p2_entropies = []
    p1_max_probs = []
    p2_max_probs = []

    for _ in range(num_positions):
        game = GameState()

        # Make random moves until we have a mid-game position
        num_moves = np.random.randint(3, 12)
        for _ in range(num_moves):
            if game.check_game_over():
                break
            legal_moves = game.get_legal_moves()
            legal_indices = np.argwhere(legal_moves)
            if len(legal_indices) == 0:
                break
            move_idx = np.random.randint(len(legal_indices))
            row, col, piece = legal_indices[move_idx]
            move = Move(row, col, PieceType(piece))
            game.make_move(move)

        if game.check_game_over():
            continue

        state_rep = game.get_game_state_representation(subjective=True)
        legal_moves = game.get_legal_moves()
        policy, value = model.predict(state_rep.board, state_rep.flat_values, legal_moves)
        policy = policy.squeeze(0)  # Remove batch dimension

        # Calculate policy entropy
        policy_flat = policy.flatten()
        policy_flat = policy_flat[policy_flat > 0]  # Remove zeros
        entropy = -np.sum(policy_flat * np.log(policy_flat + 1e-8))
        max_prob = np.max(policy_flat)

        if game.current_player == Player.ONE:
            p1_entropies.append(entropy)
            p1_max_probs.append(max_prob)
        else:
            p2_entropies.append(entropy)
            p2_max_probs.append(max_prob)

    print(f"\nP1 Policy Stats (n={len(p1_entropies)}):")
    print(f"  Mean entropy: {np.mean(p1_entropies):.3f} ± {np.std(p1_entropies):.3f}")
    print(f"  Mean max prob: {np.mean(p1_max_probs):.3f} ± {np.std(p1_max_probs):.3f}")

    print(f"\nP2 Policy Stats (n={len(p2_entropies)}):")
    print(f"  Mean entropy: {np.mean(p2_entropies):.3f} ± {np.std(p2_entropies):.3f}")
    print(f"  Mean max prob: {np.mean(p2_max_probs):.3f} ± {np.std(p2_max_probs):.3f}")

    # Check for significant differences
    if abs(np.mean(p1_entropies) - np.mean(p2_entropies)) > 0.5:
        print("\n[WARNING] Significant difference in policy confidence between players!")

    return


def main():
    parser = argparse.ArgumentParser(description='Debug P1/P2 player bias in TicTacDo AI')
    parser.add_argument('--model', type=str, default='saved_models/model_final.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--games', type=int, default=200,
                      help='Number of self-play games to run')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                      help='Device to run on')
    parser.add_argument('--use-mcts', action='store_true',
                      help='Use MCTS for move selection (slower but more accurate)')
    parser.add_argument('--mcts-sims', type=int, default=100,
                      help='Number of MCTS simulations if --use-mcts is enabled')

    args = parser.parse_args()

    print("=" * 60)
    print("TicTacDo P1/P2 Bias Debugger")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {args.model}...")
    try:
        model = ModelWrapper(device=args.device, mode='crunch')
        model.load_checkpoint(args.model)
        print("[OK] Model loaded successfully")
    except FileNotFoundError:
        print(f"[ERROR] Model not found at {args.model}")
        print("  Please train a model first or specify correct path with --model")
        return

    # Run self-play test
    print("\n" + "=" * 60)
    print("TEST 1: Self-Play Win Rates")
    print("=" * 60)
    stats = play_self_play_games(model, num_games=args.games,
                                  use_mcts=args.use_mcts,
                                  mcts_sims=args.mcts_sims)

    total_games = stats['p1_wins'] + stats['p2_wins'] + stats['draws']
    p1_win_pct = 100 * stats['p1_wins'] / total_games
    p2_win_pct = 100 * stats['p2_wins'] / total_games
    draw_pct = 100 * stats['draws'] / total_games

    print(f"\nResults from {total_games} games:")
    print(f"  P1 wins: {stats['p1_wins']:3d} ({p1_win_pct:5.1f}%)")
    print(f"  P2 wins: {stats['p2_wins']:3d} ({p2_win_pct:5.1f}%)")
    print(f"  Draws:   {stats['draws']:3d} ({draw_pct:5.1f}%)")

    avg_p1_score = np.mean(stats['p1_scores'])
    avg_p2_score = np.mean(stats['p2_scores'])
    print(f"\nAverage scores:")
    print(f"  P1: {avg_p1_score:.2f}")
    print(f"  P2: {avg_p2_score:.2f}")
    print(f"  Difference: {avg_p1_score - avg_p2_score:.2f}")

    # Analyze the bias
    win_gap = abs(p1_win_pct - p2_win_pct)
    if win_gap > 15:
        print(f"\n[SEVERE BIAS DETECTED] {win_gap:.1f}% win rate gap")
    elif win_gap > 8:
        print(f"\n[MODERATE BIAS] {win_gap:.1f}% win rate gap")
    elif win_gap > 3:
        print(f"\n[SLIGHT BIAS] {win_gap:.1f}% win rate gap (may be natural P1 advantage)")
    else:
        print(f"\n[OK] Balanced play: {win_gap:.1f}% win rate gap")

    # Feature representation test
    # print("\n" + "=" * 60)
    # print("TEST 2: Feature Representation")
    # print("=" * 60)
    # test_feature_representation(model)

    # Policy distribution analysis
    print("\n" + "=" * 60)
    print("TEST 2: Policy Distribution Analysis")
    print("=" * 60)
    analyze_policy_distribution(model, num_positions=100)

    # Position symmetry check
    print("\n" + "=" * 60)
    print("TEST 3: Position Symmetry")
    print("=" * 60)
    check_position_symmetry(model, num_positions=50)

    print("\n" + "=" * 60)
    print("Summary & Recommendations")
    print("=" * 60)

    if win_gap > 10:
        print("\nRecommended actions:")
        print("  1. Check if training data augmentation is working")
        print("  2. Verify both players are equally represented in replay buffer")
        print("  3. Examine if feature encoding is truly symmetric")
        print("  4. Consider adding explicit symmetry loss during training")
    else:
        print("\n[OK] Bias appears to be within acceptable range for this game")
        print("  (Some P1 advantage is expected due to first-move advantage)")

    print()


if __name__ == '__main__':
    main()
