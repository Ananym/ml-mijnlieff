import torch
import numpy as np
from tqdm import tqdm
from game import GameState, PieceType, Player, GameStateRepresentation, TurnResult, Move
from model import ModelWrapper
from mcts import MCTS
import random


def policy_vs_mcts_eval(
    model: ModelWrapper,
    num_games: int = 50,
    mcts_simulations: int = 400,
    debug: bool = False,
):
    """Run evaluation games between raw policy and MCTS to track progress.

    This function pits the model's raw policy against MCTS-guided play
    to measure how the policy network is improving over time.

    Args:
        model: The neural network model to evaluate
        num_games: Number of evaluation games to play
        mcts_simulations: Number of MCTS simulations to use
        debug: Whether to print debug information

    Returns:
        Dictionary of evaluation statistics
    """
    print(f"\nRunning evaluation ({num_games} games)...")
    stats = {
        "raw_policy_wins_as_p1": 0,
        "raw_policy_wins_as_p2": 0,
        "raw_policy_games_as_p1": 0,
        "raw_policy_games_as_p2": 0,
        "draws": 0,
        "total_games": 0,
    }

    # Create progress bar
    eval_pbar = tqdm(range(num_games), desc="Evaluation", leave=False)

    for _ in eval_pbar:
        # Initialize game
        game = GameState()

        # decide randomly which player uses raw policy
        raw_policy_plays_p1 = random.random() < 0.5

        if raw_policy_plays_p1:
            stats["raw_policy_games_as_p1"] += 1
        else:
            stats["raw_policy_games_as_p2"] += 1

        move_count = 0

        # play the game
        while True:
            legal_moves = game.get_legal_moves()
            if not np.any(legal_moves):
                game.pass_turn()
                continue

            # determine which player's turn it is
            is_p1_turn = game.current_player == Player.ONE

            # use raw policy if it's this player's turn
            use_raw_policy = (is_p1_turn and raw_policy_plays_p1) or (
                not is_p1_turn and not raw_policy_plays_p1
            )

            # get state representation
            state_rep = game.get_game_state_representation(subjective=True)

            if use_raw_policy:
                # use direct policy from neural network
                policy, value_pred = model.predict(
                    state_rep.board, state_rep.flat_values, legal_moves
                )
                policy = policy.squeeze(0)

                # apply legal moves mask
                masked_policy = policy * legal_moves

                # renormalize
                policy_sum = np.sum(masked_policy)
                if policy_sum > 0:
                    masked_policy = masked_policy / policy_sum

                # choose best move deterministically during evaluation
                move_coords = np.unravel_index(
                    masked_policy.argmax(), masked_policy.shape
                )
            else:
                # use model-guided MCTS with more simulations for evaluation
                mcts = MCTS(
                    model=model,
                    num_simulations=mcts_simulations,
                    c_puct=1.0,
                )
                mcts.set_temperature(0.0)  # use deterministic play during evaluation
                mcts_policy, _ = mcts.search(game)

                # select best move according to MCTS policy
                move_coords = np.unravel_index(mcts_policy.argmax(), mcts_policy.shape)

            # create and make the move
            move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
            result = game.make_move(move)
            move_count += 1

            # update progress bar
            current_p1_wins = (
                f"{stats['raw_policy_wins_as_p1']}/{stats['raw_policy_games_as_p1']}"
            )
            current_p2_wins = (
                f"{stats['raw_policy_wins_as_p2']}/{stats['raw_policy_games_as_p2']}"
            )
            eval_pbar.set_postfix(
                {"P1": current_p1_wins, "P2": current_p2_wins, "draws": stats["draws"]}
            )

            if result == TurnResult.GAME_OVER:
                # game is over, get the winner
                winner = game.get_winner()
                stats["total_games"] += 1

                if winner is None:
                    stats["draws"] += 1
                elif (winner == Player.ONE and raw_policy_plays_p1) or (
                    winner == Player.TWO and not raw_policy_plays_p1
                ):
                    # raw policy player won
                    if raw_policy_plays_p1:
                        stats["raw_policy_wins_as_p1"] += 1
                    else:
                        stats["raw_policy_wins_as_p2"] += 1

                break  # game over

            # safety check for extremely long games
            if move_count > 100:
                stats["draws"] += 1
                stats["total_games"] += 1
                break

    # calculate win rates
    p1_winrate = (
        stats["raw_policy_wins_as_p1"] / max(1, stats["raw_policy_games_as_p1"])
    ) * 100
    p2_winrate = (
        stats["raw_policy_wins_as_p2"] / max(1, stats["raw_policy_games_as_p2"])
    ) * 100
    draw_rate = (stats["draws"] / max(1, stats["total_games"])) * 100

    # print results
    print("\n=== Evaluation Results ===")
    print(f"Raw Policy vs MCTS ({num_games} games)")
    print(
        f"Raw Policy as P1: {stats['raw_policy_wins_as_p1']}/{stats['raw_policy_games_as_p1']} ({p1_winrate:.1f}%)"
    )
    print(
        f"Raw Policy as P2: {stats['raw_policy_wins_as_p2']}/{stats['raw_policy_games_as_p2']} ({p2_winrate:.1f}%)"
    )
    print(f"Draws: {stats['draws']} ({draw_rate:.1f}%)")
    print("=========================\n")

    return stats
