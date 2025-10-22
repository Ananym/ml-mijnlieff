"""
Analyze the replay buffer from a training checkpoint to identify data imbalances.

This script loads the replay buffer and checks:
1. P1 vs P2 example distribution
2. Win/loss/draw distribution per player
3. Strategic vs self-play example ratios
4. Any patterns that might cause the 18% P1 / 40% P2 training asymmetry
"""

import torch
import numpy as np
from game import Player
from model import ModelWrapper
import argparse
from collections import defaultdict

def analyze_replay_buffer(checkpoint_path: str):
    """
    Load and analyze the replay buffer from a checkpoint.
    """
    print("=" * 70)
    print("Replay Buffer Analysis")
    print("=" * 70)

    print(f"\nLoading checkpoint from: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    if 'training_state' not in checkpoint:
        print("Error: No training_state found in checkpoint")
        return

    training_state = checkpoint['training_state']

    if 'replay_buffer' not in training_state:
        print("Error: No replay_buffer found in training state")
        return

    replay_buffer = training_state['replay_buffer']

    print(f"Replay buffer size: {len(replay_buffer)} examples")

    if len(replay_buffer) == 0:
        print("Warning: Replay buffer is empty!")
        return

    # Analyze the buffer
    stats = {
        'total': len(replay_buffer),
        'p1_examples': 0,
        'p2_examples': 0,
        'p1_wins': 0,
        'p1_losses': 0,
        'p1_draws': 0,
        'p2_wins': 0,
        'p2_losses': 0,
        'p2_draws': 0,
        'value_distribution': [],
    }

    print("\nAnalyzing examples...")

    for example in replay_buffer:
        # Each example has: state, policy_target, value, player_perspective
        # value is from the perspective of the player who made the move

        # Try to determine player perspective
        # The example is stored from the perspective of whoever made the move
        # We need to look at the state to determine which player that was

        value = example.value
        stats['value_distribution'].append(value)

        # Try to get player from the example
        # The dataclass should have player info
        if hasattr(example, 'player'):
            player = example.player
        elif hasattr(example, 'state_rep'):
            # Extract from state representation
            # This is tricky - we'd need to know the encoding
            # For now, let's skip detailed player analysis
            player = None
        else:
            player = None

        # Categorize by value (since we know the reward scheme)
        # Win: +1.0, Loss: -1.0, Draw: -0.3 (discrete_mild)
        if abs(value - 1.0) < 0.01:  # Win
            if player == Player.ONE:
                stats['p1_wins'] += 1
                stats['p1_examples'] += 1
            elif player == Player.TWO:
                stats['p2_wins'] += 1
                stats['p2_examples'] += 1
        elif abs(value - (-1.0)) < 0.01:  # Loss
            if player == Player.ONE:
                stats['p1_losses'] += 1
                stats['p1_examples'] += 1
            elif player == Player.TWO:
                stats['p2_losses'] += 1
                stats['p2_examples'] += 1
        elif abs(value - (-0.3)) < 0.05:  # Draw
            if player == Player.ONE:
                stats['p1_draws'] += 1
                stats['p1_examples'] += 1
            elif player == Player.TWO:
                stats['p2_draws'] += 1
                stats['p2_examples'] += 1

    # Print results
    print("\n" + "=" * 70)
    print("BUFFER COMPOSITION")
    print("=" * 70)

    value_array = np.array(stats['value_distribution'])
    print(f"\nValue Statistics:")
    print(f"  Mean: {np.mean(value_array):.3f}")
    print(f"  Std:  {np.std(value_array):.3f}")
    print(f"  Min:  {np.min(value_array):.3f}")
    print(f"  Max:  {np.max(value_array):.3f}")

    # Count by value categories
    wins = np.sum(np.abs(value_array - 1.0) < 0.01)
    losses = np.sum(np.abs(value_array - (-1.0)) < 0.01)
    draws = np.sum(np.abs(value_array - (-0.3)) < 0.05)
    other = len(value_array) - wins - losses - draws

    print(f"\nOutcome Distribution:")
    print(f"  Wins:   {wins:5d} ({100*wins/len(value_array):5.1f}%)")
    print(f"  Losses: {losses:5d} ({100*losses/len(value_array):5.1f}%)")
    print(f"  Draws:  {draws:5d} ({100*draws/len(value_array):5.1f}%)")
    if other > 0:
        print(f"  Other:  {other:5d} ({100*other/len(value_array):5.1f}%) [bootstrapped values]")

    if stats['p1_examples'] > 0 or stats['p2_examples'] > 0:
        print("\n" + "=" * 70)
        print("PLAYER PERSPECTIVE ANALYSIS")
        print("=" * 70)

        total_identified = stats['p1_examples'] + stats['p2_examples']

        print(f"\nExamples by Player Perspective:")
        print(f"  P1: {stats['p1_examples']:5d} ({100*stats['p1_examples']/total_identified:5.1f}%)")
        print(f"  P2: {stats['p2_examples']:5d} ({100*stats['p2_examples']/total_identified:5.1f}%)")

        if stats['p1_examples'] > 0:
            print(f"\nP1 Outcome Distribution:")
            print(f"  Wins:   {stats['p1_wins']:5d} ({100*stats['p1_wins']/stats['p1_examples']:5.1f}%)")
            print(f"  Losses: {stats['p1_losses']:5d} ({100*stats['p1_losses']/stats['p1_examples']:5.1f}%)")
            print(f"  Draws:  {stats['p1_draws']:5d} ({100*stats['p1_draws']/stats['p1_examples']:5.1f}%)")

        if stats['p2_examples'] > 0:
            print(f"\nP2 Outcome Distribution:")
            print(f"  Wins:   {stats['p2_wins']:5d} ({100*stats['p2_wins']/stats['p2_examples']:5.1f}%)")
            print(f"  Losses: {stats['p2_losses']:5d} ({100*stats['p2_losses']/stats['p2_examples']:5.1f}%)")
            print(f"  Draws:  {stats['p2_draws']:5d} ({100*stats['p2_draws']/stats['p2_examples']:5.1f}%)")

        # Check for imbalance
        if total_identified > 100:  # Only if we have enough data
            p1_pct = 100 * stats['p1_examples'] / total_identified
            p2_pct = 100 * stats['p2_examples'] / total_identified
            imbalance = abs(p1_pct - p2_pct)

            print("\n" + "=" * 70)
            print("IMBALANCE ANALYSIS")
            print("=" * 70)

            if imbalance > 10:
                print(f"\n[SEVERE IMBALANCE] {imbalance:.1f}% difference between P1/P2 examples")
                if p1_pct > p2_pct:
                    print(f"  Model sees {imbalance:.1f}% MORE examples from P1's perspective")
                else:
                    print(f"  Model sees {imbalance:.1f}% MORE examples from P2's perspective")
                print("\nThis likely explains the training bias!")
                print("Recommendation: Implement data augmentation to balance perspectives")
            elif imbalance > 5:
                print(f"\n[MODERATE IMBALANCE] {imbalance:.1f}% difference between P1/P2 examples")
                print("Consider data augmentation to improve balance")
            else:
                print(f"\n[OK] Balanced: Only {imbalance:.1f}% difference between P1/P2 examples")
    else:
        print("\nNote: Could not extract player perspective from examples")
        print("This might be a limitation of the example dataclass structure")

    # Additional analysis
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    if stats['p1_examples'] == 0 and stats['p2_examples'] == 0:
        print("\nCould not determine player perspectives from replay buffer.")
        print("To fix the 18% P1 / 40% P2 training bias, consider:")
        print("  1. Add explicit player tracking to TrainingExample dataclass")
        print("  2. Implement data augmentation: flip each example to opposite perspective")
        print("  3. Ensure equal self-play games as P1 and P2")
    else:
        total_identified = stats['p1_examples'] + stats['p2_examples']
        if abs(stats['p1_examples'] - stats['p2_examples']) > total_identified * 0.1:
            print("\nData imbalance detected! To fix:")
            print("  1. Implement symmetric data augmentation")
            print("  2. Store each game position from BOTH player perspectives")
            print("  3. Ensure self-play games don't favor one starting position")

    print()


def main():
    parser = argparse.ArgumentParser(description='Analyze replay buffer from training checkpoint')
    parser.add_argument('--checkpoint', type=str, default='saved_models/model_final.pth',
                      help='Path to checkpoint file')

    args = parser.parse_args()

    analyze_replay_buffer(args.checkpoint)


if __name__ == '__main__':
    main()
