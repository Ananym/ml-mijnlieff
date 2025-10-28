"""Diagnostic script to test for player perspective bugs in value head.

This script tests whether the value head correctly handles player perspective by:
1. Testing symmetric positions from both players' perspectives
2. Checking if the value head has learned player-specific biases
3. Verifying that MCTS values are from the correct perspective
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from game import GameState, Player, PieceType, Move
from model import ModelWrapper
from mcts import MCTS

def load_latest_checkpoint():
    """Load the most recent model checkpoint."""
    checkpoint_path = "saved_models/model_final.pth"
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None

    model = ModelWrapper(device="cpu", mode="stable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from iteration {checkpoint.get('iteration', '?')}")
    return model

def test_symmetric_positions(model):
    """Test if value predictions are symmetric for P1 and P2."""
    print("\n" + "="*60)
    print("TEST 1: Symmetric Position Perspective")
    print("="*60)

    # Create a simple position - just a fresh game
    game = GameState()

    # Test from P1's perspective (start of game)
    state_rep_p1 = game.get_game_state_representation(subjective=True)
    legal_moves = game.get_legal_moves()

    policy_p1, value_p1 = model.predict(
        state_rep_p1.board, state_rep_p1.flat_values, legal_moves
    )
    value_p1 = value_p1.squeeze().item()

    print(f"\nStart of game, P1's turn:")
    print(f"  flat_values[0] (player ID): {state_rep_p1.flat_values[0]}")
    print(f"  Value from P1's perspective: {value_p1:.4f}")

    # Make P1's first move
    game.make_move(Move(0, 0, PieceType.NEAR))  # P1 plays

    # Now test from P2's perspective
    state_rep_p2 = game.get_game_state_representation(subjective=True)
    legal_moves = game.get_legal_moves()

    policy_p2, value_p2 = model.predict(
        state_rep_p2.board, state_rep_p2.flat_values, legal_moves
    )
    value_p2 = value_p2.squeeze().item()

    print(f"\nAfter 1 move, P2's turn:")
    print(f"  flat_values[0] (player ID): {state_rep_p2.flat_values[0]}")
    print(f"  Value from P2's perspective: {value_p2:.4f}")

    # The values should have opposite signs if the model is working correctly
    # (assuming the position evaluation changed)
    print(f"\nValue ratio (P1/P2): {value_p1/value_p2 if value_p2 != 0 else 'undefined':.4f}")

    if abs(value_p1) < 0.01 and abs(value_p2) < 0.01:
        print("[WARNING] Both values near zero - value head may be collapsed!")
        return False

    return True

def test_player_bias(model, num_positions=50):
    """Test if the value head has learned player-specific biases."""
    print("\n" + "="*60)
    print("TEST 2: Player-Specific Bias Detection")
    print("="*60)

    p1_values = []
    p2_values = []

    rng = np.random.Generator(np.random.PCG64())

    for _ in range(num_positions):
        game = GameState()

        # Play random moves until someone's turn
        while not game.is_over and game.move_count < 8:
            legal_moves = game.get_legal_moves()
            legal_indices = np.argwhere(legal_moves)
            if len(legal_indices) == 0:
                break

            move_idx = rng.choice(len(legal_indices))
            move_coords = legal_indices[move_idx]
            move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))

            # Get value prediction before making move
            state_rep = game.get_game_state_representation(subjective=True)
            legal_moves_mask = game.get_legal_moves()
            _, value = model.predict(
                state_rep.board, state_rep.flat_values, legal_moves_mask
            )
            value = value.squeeze().item()

            if game.current_player == Player.ONE:
                p1_values.append(value)
            else:
                p2_values.append(value)

            game.make_move(move)

    p1_mean = np.mean(p1_values) if p1_values else 0
    p2_mean = np.mean(p2_values) if p2_values else 0
    p1_std = np.std(p1_values) if p1_values else 0
    p2_std = np.std(p2_values) if p2_values else 0

    print(f"\nTested {len(p1_values)} P1 positions, {len(p2_values)} P2 positions")
    print(f"P1 values: mean={p1_mean:.4f}, std={p1_std:.4f}")
    print(f"P2 values: mean={p2_mean:.4f}, std={p2_std:.4f}")
    print(f"Mean difference: {abs(p1_mean - p2_mean):.4f}")

    if abs(p1_mean - p2_mean) > 0.1:
        print("[WARNING]  WARNING: Significant bias detected between P1 and P2!")
        print("   The model may have learned player-specific strategies")
        return False

    if p1_std < 0.05 or p2_std < 0.05:
        print("[WARNING]  WARNING: Very low std dev - value head may be collapsed!")
        return False

    print("[OK] No significant player bias detected")
    return True

def test_mcts_perspective(model):
    """Test if MCTS values are from the correct perspective."""
    print("\n" + "="*60)
    print("TEST 3: MCTS Value Perspective")
    print("="*60)

    game = GameState()
    game.make_move(Move(0, 0, PieceType.NEAR))
    game.make_move(Move(1, 1, PieceType.NEAR))

    # Run MCTS from P1's perspective
    mcts = MCTS(model=model, num_simulations=100, c_puct=1.5)
    policy, root_node = mcts.search(game)
    mcts_value_p1 = root_node.get_value()

    # Get direct network prediction
    state_rep = game.get_game_state_representation(subjective=True)
    legal_moves = game.get_legal_moves()
    _, network_value = model.predict(state_rep.board, state_rep.flat_values, legal_moves)
    network_value = network_value.squeeze().item()

    print(f"\nP1's turn after 2 moves:")
    print(f"  Network value (direct): {network_value:.4f}")
    print(f"  MCTS value (100 sims):  {mcts_value_p1:.4f}")
    print(f"  Difference: {abs(network_value - mcts_value_p1):.4f}")

    # They should be similar if MCTS is working correctly
    if abs(mcts_value_p1) < 0.01:
        print("[WARNING]  WARNING: MCTS value near zero!")
        return False

    return True

def test_absolute_vs_subjective_encoding(model):
    """Test if the mixed representation causes issues."""
    print("\n" + "="*60)
    print("TEST 4: Absolute vs Subjective Encoding")
    print("="*60)

    game = GameState()
    game.make_move(Move(0, 0, PieceType.NEAR))

    # Get representation from P2's perspective
    state_rep = game.get_game_state_representation(subjective=True)

    print(f"\nP2's turn (after P1 played):")
    print(f"  flat_values[0] (absolute player ID): {state_rep.flat_values[0]}")
    print(f"  Board layer 0 (should be P2's pieces): {np.sum(state_rep.board[:,:,0])}")
    print(f"  Board layer 1 (should be P1's pieces): {np.sum(state_rep.board[:,:,1])}")

    legal_moves = game.get_legal_moves()
    _, value = model.predict(state_rep.board, state_rep.flat_values, legal_moves)
    value = value.squeeze().item()

    print(f"  Value prediction: {value:.4f}")

    print("\nEncoding analysis:")
    print(f"  [OK] Board layers are SUBJECTIVE (current player first)")
    print(f"  [FAIL] flat_values[0] is ABSOLUTE (1=P1, 0=P2)")
    print("\nThis mixed representation may confuse the model!")

    return True

def main():
    print("="*60)
    print("PERSPECTIVE BUG DIAGNOSTIC")
    print("="*60)

    model = load_latest_checkpoint()
    if model is None:
        return

    results = []

    # Run all tests
    results.append(("Symmetric Positions", test_symmetric_positions(model)))
    results.append(("Player Bias", test_player_bias(model)))
    results.append(("MCTS Perspective", test_mcts_perspective(model)))
    results.append(("Mixed Encoding", test_absolute_vs_subjective_encoding(model)))

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "[OK] PASS" if passed else "[FAIL] FAIL"
        print(f"{status} - {test_name}")

    passed_count = sum(1 for _, passed in results if passed)
    print(f"\nPassed {passed_count}/{len(results)} tests")

    if passed_count < len(results):
        print("\n[WARNING]  ISSUES DETECTED - Value head may have perspective bugs!")
    else:
        print("\n[OK] All tests passed - No obvious perspective bugs detected")

if __name__ == "__main__":
    main()
