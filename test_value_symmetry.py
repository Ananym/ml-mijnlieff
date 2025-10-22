"""
Test if the model's value predictions are symmetric for P1 vs P2.

This checks if there's a systemic value head bias.
"""

import numpy as np
from game import GameState, Move, PieceType, Player
from model import ModelWrapper

def test_value_symmetry():
    """
    Test if model gives consistent value predictions from P1 and P2 perspectives.
    """
    print("Testing Value Head Symmetry")
    print("=" * 60)

    model = ModelWrapper(device='cpu', mode='crunch')
    model.load_checkpoint('saved_models/model_final.pth')

    # Create several random mid-game positions
    num_tests = 50
    value_diffs = []

    print(f"\nTesting {num_tests} random positions...")
    print("If value head is symmetric, P1 and P2 should get opposite-sign")
    print("but equal-magnitude predictions for the same board state.\n")

    for test_idx in range(num_tests):
        game = GameState()

        # Play 5-10 random moves
        num_moves = np.random.randint(5, 11)
        for _ in range(num_moves):
            if game.check_game_over():
                break
            legal_moves = game.get_legal_moves()
            legal_indices = np.argwhere(legal_moves)
            if len(legal_indices) == 0:
                break
            r, c, p = legal_indices[np.random.randint(len(legal_indices))]
            game.make_move(Move(r, c, PieceType(p)))

        if game.check_game_over():
            continue

        # Get value from current player's perspective (P1 or P2)
        current_player = game.current_player
        state_rep = game.get_game_state_representation(subjective=True)
        legal_moves = game.get_legal_moves()
        _, value = model.predict(state_rep.board, state_rep.flat_values, legal_moves)
        value = value.item()

        # Make one more move to switch perspective
        legal_indices = np.argwhere(legal_moves)
        if len(legal_indices) > 0:
            r, c, p = legal_indices[0]
            game.make_move(Move(r, c, PieceType(p)))

            # Get value from opposite player's perspective
            state_rep2 = game.get_game_state_representation(subjective=True)
            legal_moves2 = game.get_legal_moves()
            _, value2 = model.predict(state_rep2.board, state_rep2.flat_values, legal_moves2)
            value2 = value2.item()

            # The relationship between value and value2 depends on whether we made
            # a move or not. After a move, the position changed, so this isn't
            # a perfect symmetry test.
            #
            # Better test: Check average value by player across many positions
            if current_player == Player.ONE:
                value_diffs.append(('P1', value))
            else:
                value_diffs.append(('P2', value))

    # Analyze results
    p1_values = [v for player, v in value_diffs if player == 'P1']
    p2_values = [v for player, v in value_diffs if player == 'P2']

    print(f"\nResults from {len(value_diffs)} position evaluations:")
    print(f"\nP1's turn (n={len(p1_values)}):")
    print(f"  Mean value: {np.mean(p1_values):+.3f}")
    print(f"  Std dev:    {np.std(p1_values):.3f}")

    print(f"\nP2's turn (n={len(p2_values)}):")
    print(f"  Mean value: {np.mean(p2_values):+.3f}")
    print(f"  Std dev:    {np.std(p2_values):.3f}")

    diff = abs(np.mean(p1_values) - np.mean(p2_values))
    print(f"\nAbsolute difference in mean values: {diff:.3f}")

    if diff > 0.1:
        print("\n[SEVERE VALUE BIAS] Model has different value predictions")
        print("for P1 vs P2, even with subjective representation!")
        print("\nThis explains the training asymmetry:")
        if np.mean(p1_values) > np.mean(p2_values):
            print("  - Model is MORE OPTIMISTIC when it's P1's turn")
            print("  - Model is MORE PESSIMISTIC when it's P2's turn")
        else:
            print("  - Model is MORE PESSIMISTIC when it's P1's turn")
            print("  - Model is MORE OPTIMISTIC when it's P2's turn")
    elif diff > 0.05:
        print("\n[MODERATE VALUE BIAS] Some asymmetry detected")
    else:
        print("\n[OK] Value predictions are symmetric")

if __name__ == '__main__':
    test_value_symmetry()
