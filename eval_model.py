import torch
import numpy as np
from tqdm import tqdm
from game import GameState, PieceType, Player, GameStateRepresentation, TurnResult, Move
from model import ModelWrapper
from mcts import MCTS
from opponents import StrategicOpponent
import random


def policy_vs_mcts_eval(
    model: ModelWrapper,
    rng: np.random.Generator,
    iteration: int = 0,
    num_games: int = 100,
    strategic_games: int = 100,
    mcts_simulations: int = 25,
    debug: bool = False,
):
    """Run evaluation games between raw policy and MCTS to track progress.

    This function pits the model's raw policy against MCTS-guided play
    to measure how the policy network is improving over time.

    Args:
        model: The neural network model to evaluate
        rng: Random number generator
        iteration: Current training iteration
        num_games: Number of evaluation games to play against MCTS
        strategic_games: Number of evaluation games to play against strategic opponent
        mcts_simulations: Number of MCTS simulations to use
        debug: Whether to print debug information

    Returns:
        Dictionary of evaluation statistics
    """
    print(f"\nRunning evaluation at iteration {iteration}...")
    stats = {
        "raw_policy_wins_as_p1": 0,
        "raw_policy_wins_as_p2": 0,
        "raw_policy_games_as_p1": 0,
        "raw_policy_games_as_p2": 0,
        "draws": 0,
        "total_games": 0,
    }

    # Create progress bar
    eval_pbar = tqdm(range(num_games), desc="Evaluation vs MCTS", leave=False)

    for _ in eval_pbar:
        # Initialize game
        game = GameState()

        # Decide randomly which player uses raw policy
        raw_policy_plays_p1 = random.random() < 0.5

        if raw_policy_plays_p1:
            stats["raw_policy_games_as_p1"] += 1
        else:
            stats["raw_policy_games_as_p2"] += 1

        move_count = 0

        # Play the game
        while True:
            legal_moves = game.get_legal_moves()
            if not np.any(legal_moves):
                game.pass_turn()
                continue

            # Determine which player's turn it is
            is_p1_turn = game.current_player == Player.ONE

            # Use raw policy if it's this player's turn
            use_raw_policy = (is_p1_turn and raw_policy_plays_p1) or (
                not is_p1_turn and not raw_policy_plays_p1
            )

            # Get state representation
            state_rep = game.get_game_state_representation(subjective=True)

            if use_raw_policy:
                # Use direct policy from neural network with softmax sampling
                policy, _ = model.predict(
                    state_rep.board, state_rep.flat_values, legal_moves
                )
                policy = policy.squeeze(0)

                # Apply legal moves mask
                masked_policy = policy * legal_moves

                # Renormalize
                policy_sum = np.sum(masked_policy)
                if policy_sum > 0:
                    masked_policy = masked_policy / policy_sum

                # Use softmax sampling (temperature=1.0)
                policy_flat = masked_policy.flatten()
                if np.sum(policy_flat) > 0:
                    move_idx = rng.choice(len(policy_flat), p=policy_flat)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)
                else:
                    # Fallback to random selection among legal moves
                    legal_flat = legal_moves.flatten()
                    legal_indices = np.nonzero(legal_flat > 0)[0]
                    move_idx = rng.choice(legal_indices)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)
            else:
                # Use model-guided MCTS with more simulations for evaluation
                mcts = MCTS(
                    model=model,
                    num_simulations=mcts_simulations,
                    c_puct=1.0,
                )
                mcts.set_temperature(0.0)  # Use deterministic play during evaluation
                mcts_policy, _ = mcts.search(game)

                # Select best move according to MCTS policy
                move_coords = np.unravel_index(mcts_policy.argmax(), mcts_policy.shape)

            # Create and make the move
            move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
            result = game.make_move(move)
            move_count += 1

            # Update progress bar
            current_p1_wins = (
                f"{stats['raw_policy_wins_as_p1']}/{stats['raw_policy_games_as_p1']}"
            )
            current_p2_wins = (
                f"{stats['raw_policy_wins_as_p2']}/{stats['raw_policy_games_as_p2']}"
            )
            eval_pbar.set_postfix(
                {
                    "P1": current_p1_wins,
                    "P2": current_p2_wins,
                    "draws": stats["draws"],
                }
            )

            if result == TurnResult.GAME_OVER:
                # Game is over, get the winner
                winner = game.get_winner()
                stats["total_games"] += 1

                if winner is None:
                    stats["draws"] += 1
                elif (winner == Player.ONE and raw_policy_plays_p1) or (
                    winner == Player.TWO and not raw_policy_plays_p1
                ):
                    # Raw policy player won
                    if raw_policy_plays_p1:
                        stats["raw_policy_wins_as_p1"] += 1
                    else:
                        stats["raw_policy_wins_as_p2"] += 1

                break  # Game over

            # Safety check for extremely long games
            if move_count > 100:
                stats["draws"] += 1
                stats["total_games"] += 1
                break

    # Now run evaluation against strategic opponent
    strategic_stats = {
        "raw_policy_wins_as_p1": 0,
        "raw_policy_losses_as_p1": 0,
        "raw_policy_draws_as_p1": 0,
        "raw_policy_wins_as_p2": 0,
        "raw_policy_losses_as_p2": 0,
        "raw_policy_draws_as_p2": 0,
        "raw_policy_games_as_p1": 0,
        "raw_policy_games_as_p2": 0,
        "total_games": 0,
    }

    strategic_opponent = StrategicOpponent()
    eval_pbar = tqdm(
        range(strategic_games),
        desc="Evaluation vs Strategic",
        leave=False,
    )

    for _ in eval_pbar:
        # Initialize game
        game = GameState()

        # Decide randomly which player uses raw policy
        raw_policy_plays_p1 = random.random() < 0.5

        if raw_policy_plays_p1:
            strategic_stats["raw_policy_games_as_p1"] += 1
        else:
            strategic_stats["raw_policy_games_as_p2"] += 1

        move_count = 0

        # Play the game
        while True:
            legal_moves = game.get_legal_moves()
            if not np.any(legal_moves):
                game.pass_turn()
                continue

            # Determine which player's turn it is
            is_p1_turn = game.current_player == Player.ONE

            # Determine if it's raw policy's turn or strategic opponent's turn
            if (is_p1_turn and raw_policy_plays_p1) or (
                not is_p1_turn and not raw_policy_plays_p1
            ):
                # Raw policy's turn
                state_rep = game.get_game_state_representation(subjective=True)

                # Use direct policy from neural network with softmax sampling
                policy, _ = model.predict(
                    state_rep.board, state_rep.flat_values, legal_moves
                )
                policy = policy.squeeze(0)

                # Apply legal moves mask
                masked_policy = policy * legal_moves

                # Renormalize
                policy_sum = np.sum(masked_policy)
                if policy_sum > 0:
                    masked_policy = masked_policy / policy_sum

                # Use softmax sampling (temperature=1.0)
                policy_flat = masked_policy.flatten()
                if np.sum(policy_flat) > 0:
                    move_idx = rng.choice(len(policy_flat), p=policy_flat)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)
                else:
                    # Fallback to random selection among legal moves
                    legal_flat = legal_moves.flatten()
                    legal_indices = np.nonzero(legal_flat > 0)[0]
                    move_idx = rng.choice(legal_indices)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)

                # Create move
                move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
            else:
                # Strategic opponent's turn
                # No random chance during evaluation
                move = strategic_opponent.get_move(game, random_chance=0.0)
                if move is None:
                    game.pass_turn()
                    continue

            # Make the move
            result = game.make_move(move)
            move_count += 1

            # Update progress bar
            p1_stats = f"P1: {strategic_stats['raw_policy_wins_as_p1']}-{strategic_stats['raw_policy_draws_as_p1']}-{strategic_stats['raw_policy_losses_as_p1']}"
            p2_stats = f"P2: {strategic_stats['raw_policy_wins_as_p2']}-{strategic_stats['raw_policy_draws_as_p2']}-{strategic_stats['raw_policy_losses_as_p2']}"
            eval_pbar.set_postfix({"P1": p1_stats, "P2": p2_stats})

            if result == TurnResult.GAME_OVER:
                # Game is over, get the winner
                winner = game.get_winner()
                strategic_stats["total_games"] += 1

                if winner is None:
                    # Draw
                    if raw_policy_plays_p1:
                        strategic_stats["raw_policy_draws_as_p1"] += 1
                    else:
                        strategic_stats["raw_policy_draws_as_p2"] += 1
                elif (winner == Player.ONE and raw_policy_plays_p1) or (
                    winner == Player.TWO and not raw_policy_plays_p1
                ):
                    # Raw policy won
                    if raw_policy_plays_p1:
                        strategic_stats["raw_policy_wins_as_p1"] += 1
                    else:
                        strategic_stats["raw_policy_wins_as_p2"] += 1
                else:
                    # Strategic opponent won
                    if raw_policy_plays_p1:
                        strategic_stats["raw_policy_losses_as_p1"] += 1
                    else:
                        strategic_stats["raw_policy_losses_as_p2"] += 1

                break  # Game over

            # Safety check for extremely long games
            if move_count > 100:
                # Count as draw
                if raw_policy_plays_p1:
                    strategic_stats["raw_policy_draws_as_p1"] += 1
                else:
                    strategic_stats["raw_policy_draws_as_p2"] += 1
                strategic_stats["total_games"] += 1
                break

    # Calculate win rates for raw policy vs MCTS
    p1_winrate = (
        stats["raw_policy_wins_as_p1"] / max(1, stats["raw_policy_games_as_p1"])
    ) * 100
    p2_winrate = (
        stats["raw_policy_wins_as_p2"] / max(1, stats["raw_policy_games_as_p2"])
    ) * 100
    draw_rate = (stats["draws"] / max(1, stats["total_games"])) * 100

    # Calculate P1 loss rate
    p1_lossrate = (
        100
        - p1_winrate
        - (stats["draws"] / max(1, stats["raw_policy_games_as_p1"]) * 100)
    )

    # Calculate P2 loss rate
    p2_lossrate = (
        100
        - p2_winrate
        - (stats["draws"] / max(1, stats["raw_policy_games_as_p2"]) * 100)
    )

    # Calculate strategic opponent stats
    p1_strategic_winrate = (
        strategic_stats["raw_policy_wins_as_p1"]
        / max(1, strategic_stats["raw_policy_games_as_p1"])
    ) * 100
    p1_strategic_drawrate = (
        strategic_stats["raw_policy_draws_as_p1"]
        / max(1, strategic_stats["raw_policy_games_as_p1"])
    ) * 100
    p1_strategic_lossrate = (
        strategic_stats["raw_policy_losses_as_p1"]
        / max(1, strategic_stats["raw_policy_games_as_p1"])
    ) * 100

    p2_strategic_winrate = (
        strategic_stats["raw_policy_wins_as_p2"]
        / max(1, strategic_stats["raw_policy_games_as_p2"])
    ) * 100
    p2_strategic_drawrate = (
        strategic_stats["raw_policy_draws_as_p2"]
        / max(1, strategic_stats["raw_policy_games_as_p2"])
    ) * 100
    p2_strategic_lossrate = (
        strategic_stats["raw_policy_losses_as_p2"]
        / max(1, strategic_stats["raw_policy_games_as_p2"])
    ) * 100

    # Print results
    print("\n=== Evaluation Results ===")
    print(f"Raw Policy vs MCTS ({num_games} games)")

    # Calculate counts of wins, draws, losses for P1
    p1_wins = stats["raw_policy_wins_as_p1"]
    p1_draws = stats["draws"] // 2
    p1_losses = stats["raw_policy_games_as_p1"] - p1_wins - p1_draws

    # Calculate counts of wins, draws, losses for P2
    p2_wins = stats["raw_policy_wins_as_p2"]
    p2_draws = stats["draws"] // 2
    p2_losses = stats["raw_policy_games_as_p2"] - p2_wins - p2_draws

    # Format display with explicit labels
    print(
        f"Raw Policy as P1: W:{p1_wins} D:{p1_draws} L:{p1_losses} ({p1_winrate:.1f}% - {draw_rate/2:.1f}% - {p1_lossrate:.1f}%)"
    )
    print(
        f"Raw Policy as P2: W:{p2_wins} D:{p2_draws} L:{p2_losses} ({p2_winrate:.1f}% - {draw_rate/2:.1f}% - {p2_lossrate:.1f}%)"
    )
    print(f"Total Draws: {stats['draws']} ({draw_rate:.1f}%)")

    print(f"\nRaw Policy vs Strategic ({strategic_games} games)")

    # Strategic opponent results (already have explicit win/draw/loss counts)
    print(
        f"Raw Policy as P1: W:{strategic_stats['raw_policy_wins_as_p1']} D:{strategic_stats['raw_policy_draws_as_p1']} L:{strategic_stats['raw_policy_losses_as_p1']} ({p1_strategic_winrate:.1f}% - {p1_strategic_drawrate:.1f}% - {p1_strategic_lossrate:.1f}%)"
    )
    print(
        f"Raw Policy as P2: W:{strategic_stats['raw_policy_wins_as_p2']} D:{strategic_stats['raw_policy_draws_as_p2']} L:{strategic_stats['raw_policy_losses_as_p2']} ({p2_strategic_winrate:.1f}% - {p2_strategic_drawrate:.1f}% - {p2_strategic_lossrate:.1f}%)"
    )
    print("=========================\n")

    # Merge stats for return
    combined_stats = {**stats, **strategic_stats}
    return combined_stats
