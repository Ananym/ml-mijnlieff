"""Test MCTS performance vs Strategic opponent at different simulation counts.

This script runs MCTS (without neural network) against the Strategic opponent
to determine:
1. If MCTS is working correctly
2. How many simulations are needed for MCTS to beat Strategic
3. Whether there are any bugs in MCTS implementation
"""

import numpy as np
from game import GameState, Move, PieceType, Player, TurnResult
from opponents import StrategicOpponent
from mcts import MCTS
from tqdm import tqdm


def play_game(mcts_sims: int, mcts_as_p1: bool, verbose: bool = False) -> str:
    """Play one game between MCTS and Strategic opponent.

    Args:
        mcts_sims: Number of MCTS simulations to use
        mcts_as_p1: If True, MCTS plays as Player 1, else Player 2
        verbose: If True, print move-by-move details

    Returns:
        'mcts_win', 'strategic_win', or 'draw'
    """
    game = GameState()
    mcts = MCTS(model=None, num_simulations=mcts_sims, c_puct=1.0)
    strategic = StrategicOpponent()

    move_count = 0
    max_moves = 100  # Safety limit

    while not game.is_over and move_count < max_moves:
        legal_moves = game.get_legal_moves()

        # Check if current player has no legal moves
        if not np.any(legal_moves):
            game.pass_turn()
            continue

        # Determine whose turn it is
        is_mcts_turn = (game.current_player == Player.ONE and mcts_as_p1) or (
            game.current_player == Player.TWO and not mcts_as_p1
        )

        if is_mcts_turn:
            # MCTS move
            mcts.set_temperature(0.0)  # Deterministic
            policy, _ = mcts.search(game)
            move_coords = np.unravel_index(policy.argmax(), policy.shape)
            move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))

            if verbose:
                print(f"Move {move_count}: MCTS plays {move}")
        else:
            # Strategic opponent move
            move = strategic.get_move(game, random_chance=0.0)
            if move is None:
                game.pass_turn()
                continue

            if verbose:
                print(f"Move {move_count}: Strategic plays {move}")

        result = game.make_move(move)
        move_count += 1

        if result == TurnResult.GAME_OVER:
            break

    # Determine winner
    if move_count >= max_moves:
        return "draw"

    winner = game.get_winner()
    if winner is None:
        return "draw"
    elif (winner == Player.ONE and mcts_as_p1) or (
        winner == Player.TWO and not mcts_as_p1
    ):
        return "mcts_win"
    else:
        return "strategic_win"


def test_mcts_strength(sim_counts: list, games_per_count: int = 50):
    """Test MCTS vs Strategic opponent at different simulation counts.

    Args:
        sim_counts: List of simulation counts to test
        games_per_count: Number of games to play at each simulation count
    """
    print("=" * 70)
    print("MCTS vs Strategic Opponent - Strength Test")
    print("=" * 70)
    print(f"Games per simulation count: {games_per_count}")
    print(f"MCTS plays {games_per_count//2} games as P1, {games_per_count//2} as P2\n")

    results = {}

    for sims in sim_counts:
        print(f"\n--- Testing {sims} simulations ---")

        wins_as_p1 = 0
        wins_as_p2 = 0
        draws = 0
        losses_as_p1 = 0
        losses_as_p2 = 0

        # Games as Player 1
        print(f"MCTS as Player 1...")
        for _ in tqdm(
            range(games_per_count // 2), desc=f"{sims} sims (P1)", leave=False
        ):
            result = play_game(sims, mcts_as_p1=True)
            if result == "mcts_win":
                wins_as_p1 += 1
            elif result == "strategic_win":
                losses_as_p1 += 1
            else:
                draws += 1

        # Games as Player 2
        print(f"MCTS as Player 2...")
        for _ in tqdm(
            range(games_per_count // 2), desc=f"{sims} sims (P2)", leave=False
        ):
            result = play_game(sims, mcts_as_p1=False)
            if result == "mcts_win":
                wins_as_p2 += 1
            elif result == "strategic_win":
                losses_as_p2 += 1
            else:
                draws += 1

        total_wins = wins_as_p1 + wins_as_p2
        total_losses = losses_as_p1 + losses_as_p2
        total_games = games_per_count

        win_rate = (total_wins / total_games) * 100

        results[sims] = {
            "wins_p1": wins_as_p1,
            "wins_p2": wins_as_p2,
            "losses_p1": losses_as_p1,
            "losses_p2": losses_as_p2,
            "draws": draws,
            "win_rate": win_rate,
        }

        print(f"\nResults for {sims} simulations:")
        print(f"  As P1: {wins_as_p1} wins, {losses_as_p1} losses")
        print(f"  As P2: {wins_as_p2} wins, {losses_as_p2} losses")
        print(f"  Draws: {draws}")
        print(f"  Overall: {total_wins}/{total_games} wins = {win_rate:.1f}% win rate")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Sims':<10} {'Win Rate':<12} {'Wins':<8} {'Losses':<8} {'Draws':<8}")
    print("-" * 70)

    for sims in sim_counts:
        r = results[sims]
        total_wins = r["wins_p1"] + r["wins_p2"]
        total_losses = r["losses_p1"] + r["losses_p2"]
        print(
            f"{sims:<10} {r['win_rate']:>6.1f}%     {total_wins:<8} {total_losses:<8} {r['draws']:<8}"
        )

    print("=" * 70)

    # Find minimum sims for >50% win rate
    for sims in sorted(sim_counts):
        if results[sims]["win_rate"] > 50.0:
            print(
                f"\n✅ MCTS needs ~{sims} simulations to beat Strategic opponent (>{results[sims]['win_rate']:.1f}% win rate)"
            )
            break
    else:
        print(f"\n❌ MCTS did not achieve >50% win rate at any tested simulation count")
        print(
            f"   Highest: {max(results.values(), key=lambda x: x['win_rate'])['win_rate']:.1f}% at {max(results.keys(), key=lambda k: results[k]['win_rate'])} sims"
        )


if __name__ == "__main__":
    # Test at increasing simulation counts
    sim_counts = [4000, 8000, 12000]

    print("Testing MCTS (no neural network) vs Strategic opponent")
    print("This will help determine:")
    print("  1. If MCTS is working correctly")
    print("  2. Minimum simulations needed to beat Strategic")
    print("  3. Whether there are bugs in MCTS\n")

    test_mcts_strength(sim_counts, games_per_count=50)
