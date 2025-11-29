import torch
import numpy as np
from tqdm import tqdm
from game import GameState, PieceType, Player, GameStateRepresentation, TurnResult, Move
from model import ModelWrapper
from mcts import MCTS
from opponents import StrategicOpponent
import random


def self_play_balance_eval(
    model: ModelWrapper,
    rng: np.random.Generator,
    num_games: int = 20,
    eval_temperature: float = 0.4,
):
    """Evaluate model balance by having it play against itself (no MCTS).

    Both P1 and P2 use the raw policy network to measure if the model
    has learned balanced play or has a bias toward one side.

    Returns:
        Dictionary with p1_wins, p2_wins, draws counts
    """
    stats = {
        "p1_wins": 0,
        "p2_wins": 0,
        "draws": 0,
        "total_games": 0,
    }

    eval_pbar = tqdm(range(num_games), desc="Self-play balance", leave=False)

    for _ in eval_pbar:
        game = GameState()
        move_count = 0

        while True:
            legal_moves = game.get_legal_moves()
            if not np.any(legal_moves):
                game.pass_turn()
                continue

            # Get state representation (subjective to current player)
            state_rep = game.get_game_state_representation(subjective=True)

            # Use direct policy from neural network
            policy, _ = model.predict(
                state_rep.board, state_rep.flat_values, legal_moves
            )
            policy = policy.squeeze(0)

            # Apply legal moves mask
            masked_policy = policy * legal_moves
            policy_sum = np.sum(masked_policy)
            if policy_sum > 0:
                masked_policy = masked_policy / policy_sum

            # Apply temperature
            policy_flat = masked_policy.flatten()
            if eval_temperature != 1.0 and np.sum(policy_flat) > 0:
                log_probs = np.log(policy_flat + 1e-10)
                scaled_logits = log_probs / eval_temperature
                scaled_logits = scaled_logits - np.max(scaled_logits)
                exp_logits = np.exp(scaled_logits)
                policy_flat = exp_logits / np.sum(exp_logits)

            if np.sum(policy_flat) > 0:
                move_idx = rng.choice(len(policy_flat), p=policy_flat)
                move_coords = np.unravel_index(move_idx, masked_policy.shape)
            else:
                legal_flat = legal_moves.flatten()
                legal_indices = np.nonzero(legal_flat > 0)[0]
                move_idx = rng.choice(legal_indices)
                move_coords = np.unravel_index(move_idx, masked_policy.shape)

            move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
            result = game.make_move(move)
            move_count += 1

            eval_pbar.set_postfix({
                "P1": stats["p1_wins"],
                "P2": stats["p2_wins"],
                "D": stats["draws"],
            })

            if result == TurnResult.GAME_OVER:
                winner = game.get_winner()
                stats["total_games"] += 1

                if winner is None:
                    stats["draws"] += 1
                elif winner == Player.ONE:
                    stats["p1_wins"] += 1
                else:
                    stats["p2_wins"] += 1
                break

            if move_count > 100:
                stats["draws"] += 1
                stats["total_games"] += 1
                break

    return stats


def mcts_contribution_eval(
    model: ModelWrapper,
    rng: np.random.Generator,
    num_games: int = 20,
    mcts_simulations: int = 100,
    mcts_as_p1: bool = True,
    eval_temperature: float = 0.4,
):
    """Evaluate how much MCTS improves over raw policy.

    Pits raw policy against MCTS-guided play to measure MCTS contribution.

    Args:
        mcts_as_p1: If True, MCTS plays as P1, raw policy as P2
                    If False, raw policy plays as P1, MCTS as P2

    Returns:
        Dictionary with mcts_wins, policy_wins, draws counts
    """
    stats = {
        "mcts_wins": 0,
        "policy_wins": 0,
        "draws": 0,
        "total_games": 0,
    }

    side_label = "P1" if mcts_as_p1 else "P2"
    eval_pbar = tqdm(range(num_games), desc=f"MCTS as {side_label}", leave=False)

    for _ in eval_pbar:
        game = GameState()
        move_count = 0

        while True:
            legal_moves = game.get_legal_moves()
            if not np.any(legal_moves):
                game.pass_turn()
                continue

            is_p1_turn = game.current_player == Player.ONE
            use_mcts = (is_p1_turn and mcts_as_p1) or (not is_p1_turn and not mcts_as_p1)

            state_rep = game.get_game_state_representation(subjective=True)

            if use_mcts:
                # MCTS-guided play
                mcts = MCTS(
                    model=model,
                    num_simulations=mcts_simulations,
                    c_puct=1.0,
                )
                mcts.set_temperature(0.0)  # Deterministic
                mcts_policy, _ = mcts.search(game)
                move_coords = np.unravel_index(mcts_policy.argmax(), mcts_policy.shape)
            else:
                # Raw policy
                policy, _ = model.predict(
                    state_rep.board, state_rep.flat_values, legal_moves
                )
                policy = policy.squeeze(0)

                masked_policy = policy * legal_moves
                policy_sum = np.sum(masked_policy)
                if policy_sum > 0:
                    masked_policy = masked_policy / policy_sum

                policy_flat = masked_policy.flatten()
                if eval_temperature != 1.0 and np.sum(policy_flat) > 0:
                    log_probs = np.log(policy_flat + 1e-10)
                    scaled_logits = log_probs / eval_temperature
                    scaled_logits = scaled_logits - np.max(scaled_logits)
                    exp_logits = np.exp(scaled_logits)
                    policy_flat = exp_logits / np.sum(exp_logits)

                if np.sum(policy_flat) > 0:
                    move_idx = rng.choice(len(policy_flat), p=policy_flat)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)
                else:
                    legal_flat = legal_moves.flatten()
                    legal_indices = np.nonzero(legal_flat > 0)[0]
                    move_idx = rng.choice(legal_indices)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)

            move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
            result = game.make_move(move)
            move_count += 1

            eval_pbar.set_postfix({
                "MCTS": stats["mcts_wins"],
                "Policy": stats["policy_wins"],
                "D": stats["draws"],
            })

            if result == TurnResult.GAME_OVER:
                winner = game.get_winner()
                stats["total_games"] += 1

                if winner is None:
                    stats["draws"] += 1
                elif (winner == Player.ONE and mcts_as_p1) or (winner == Player.TWO and not mcts_as_p1):
                    stats["mcts_wins"] += 1
                else:
                    stats["policy_wins"] += 1
                break

            if move_count > 100:
                stats["draws"] += 1
                stats["total_games"] += 1
                break

    return stats


def vs_strategic_eval(
    model: ModelWrapper,
    rng: np.random.Generator,
    num_games: int = 20,
    model_as_p1: bool = True,
    eval_temperature: float = 0.4,
):
    """Evaluate raw policy against strategic opponent.

    Args:
        model_as_p1: If True, model plays as P1, Strategic as P2
                    If False, Strategic plays as P1, model as P2

    Returns:
        Dictionary with model_wins, strategic_wins, draws counts
    """
    stats = {
        "model_wins": 0,
        "strategic_wins": 0,
        "draws": 0,
        "total_games": 0,
    }

    strategic_opponent = StrategicOpponent()
    side_label = "P1" if model_as_p1 else "P2"
    eval_pbar = tqdm(range(num_games), desc=f"vs Strategic as {side_label}", leave=False)

    for _ in eval_pbar:
        game = GameState()
        move_count = 0

        while True:
            legal_moves = game.get_legal_moves()
            if not np.any(legal_moves):
                game.pass_turn()
                continue

            is_p1_turn = game.current_player == Player.ONE
            is_model_turn = (is_p1_turn and model_as_p1) or (not is_p1_turn and not model_as_p1)

            if is_model_turn:
                state_rep = game.get_game_state_representation(subjective=True)
                policy, _ = model.predict(
                    state_rep.board, state_rep.flat_values, legal_moves
                )
                policy = policy.squeeze(0)

                masked_policy = policy * legal_moves
                policy_sum = np.sum(masked_policy)
                if policy_sum > 0:
                    masked_policy = masked_policy / policy_sum

                policy_flat = masked_policy.flatten()
                if eval_temperature != 1.0 and np.sum(policy_flat) > 0:
                    log_probs = np.log(policy_flat + 1e-10)
                    scaled_logits = log_probs / eval_temperature
                    scaled_logits = scaled_logits - np.max(scaled_logits)
                    exp_logits = np.exp(scaled_logits)
                    policy_flat = exp_logits / np.sum(exp_logits)

                if np.sum(policy_flat) > 0:
                    move_idx = rng.choice(len(policy_flat), p=policy_flat)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)
                else:
                    legal_flat = legal_moves.flatten()
                    legal_indices = np.nonzero(legal_flat > 0)[0]
                    move_idx = rng.choice(legal_indices)
                    move_coords = np.unravel_index(move_idx, masked_policy.shape)

                move = Move(move_coords[0], move_coords[1], PieceType(move_coords[2]))
            else:
                move = strategic_opponent.get_move(game, random_chance=0.0)
                if move is None:
                    game.pass_turn()
                    continue

            result = game.make_move(move)
            move_count += 1

            eval_pbar.set_postfix({
                "Model": stats["model_wins"],
                "Strat": stats["strategic_wins"],
                "D": stats["draws"],
            })

            if result == TurnResult.GAME_OVER:
                winner = game.get_winner()
                stats["total_games"] += 1

                if winner is None:
                    stats["draws"] += 1
                elif (winner == Player.ONE and model_as_p1) or (winner == Player.TWO and not model_as_p1):
                    stats["model_wins"] += 1
                else:
                    stats["strategic_wins"] += 1
                break

            if move_count > 100:
                stats["draws"] += 1
                stats["total_games"] += 1
                break

    return stats


def extended_evaluation(
    model: ModelWrapper,
    rng: np.random.Generator,
    iteration: int = 0,
    num_games: int = 20,
    mcts_simulations: int = 100,
):
    """Comprehensive evaluation suite for pure self-play training.

    Runs 5 evaluation types:
    1. Self-play balance (no MCTS) - P1 vs P2 win rates
    2. MCTS contribution as P1 - MCTS wins over raw policy
    3. MCTS contribution as P2 - MCTS wins over raw policy
    4. vs Strategic as P1 - real-world performance proxy
    5. vs Strategic as P2 - real-world performance proxy

    Returns:
        Dictionary with all evaluation results
    """
    print(f"\n=== Extended Evaluation (Iteration {iteration}) ===")

    results = {}

    # 1. Self-play balance test
    print("\n1. Self-play balance (no MCTS):")
    balance = self_play_balance_eval(model, rng, num_games)
    results["self_play"] = balance
    p1_rate = balance["p1_wins"] / max(1, balance["total_games"]) * 100
    p2_rate = balance["p2_wins"] / max(1, balance["total_games"]) * 100
    draw_rate = balance["draws"] / max(1, balance["total_games"]) * 100
    print(f"   P1 wins: {balance['p1_wins']} ({p1_rate:.1f}%)")
    print(f"   P2 wins: {balance['p2_wins']} ({p2_rate:.1f}%)")
    print(f"   Draws: {balance['draws']} ({draw_rate:.1f}%)")

    # 2. MCTS contribution as P1
    print("\n2. MCTS vs Policy (MCTS as P1):")
    mcts_p1 = mcts_contribution_eval(model, rng, num_games, mcts_simulations, mcts_as_p1=True)
    results["mcts_as_p1"] = mcts_p1
    mcts_rate = mcts_p1["mcts_wins"] / max(1, mcts_p1["total_games"]) * 100
    print(f"   MCTS wins: {mcts_p1['mcts_wins']} ({mcts_rate:.1f}%)")
    print(f"   Policy wins: {mcts_p1['policy_wins']}")
    print(f"   Draws: {mcts_p1['draws']}")

    # 3. MCTS contribution as P2
    print("\n3. MCTS vs Policy (MCTS as P2):")
    mcts_p2 = mcts_contribution_eval(model, rng, num_games, mcts_simulations, mcts_as_p1=False)
    results["mcts_as_p2"] = mcts_p2
    mcts_rate = mcts_p2["mcts_wins"] / max(1, mcts_p2["total_games"]) * 100
    print(f"   MCTS wins: {mcts_p2['mcts_wins']} ({mcts_rate:.1f}%)")
    print(f"   Policy wins: {mcts_p2['policy_wins']}")
    print(f"   Draws: {mcts_p2['draws']}")

    # 4. vs Strategic as P1
    print("\n4. Policy vs Strategic (model as P1):")
    strat_p1 = vs_strategic_eval(model, rng, num_games, model_as_p1=True)
    results["vs_strategic_as_p1"] = strat_p1
    win_rate = strat_p1["model_wins"] / max(1, strat_p1["total_games"]) * 100
    print(f"   Model wins: {strat_p1['model_wins']} ({win_rate:.1f}%)")
    print(f"   Strategic wins: {strat_p1['strategic_wins']}")
    print(f"   Draws: {strat_p1['draws']}")

    # 5. vs Strategic as P2
    print("\n5. Policy vs Strategic (model as P2):")
    strat_p2 = vs_strategic_eval(model, rng, num_games, model_as_p1=False)
    results["vs_strategic_as_p2"] = strat_p2
    win_rate = strat_p2["model_wins"] / max(1, strat_p2["total_games"]) * 100
    print(f"   Model wins: {strat_p2['model_wins']} ({win_rate:.1f}%)")
    print(f"   Strategic wins: {strat_p2['strategic_wins']}")
    print(f"   Draws: {strat_p2['draws']}")

    # Summary
    total_strat_games = strat_p1["total_games"] + strat_p2["total_games"]
    total_strat_wins = strat_p1["model_wins"] + strat_p2["model_wins"]
    combined_strat_winrate = total_strat_wins / max(1, total_strat_games) * 100

    print(f"\n=== Summary ===")
    print(f"Self-play balance: P1 {p1_rate:.1f}% / P2 {p2_rate:.1f}%")
    print(f"MCTS contribution: {(mcts_p1['mcts_wins'] + mcts_p2['mcts_wins'])}/{mcts_p1['total_games'] + mcts_p2['total_games']} games")
    print(f"vs Strategic combined: {combined_strat_winrate:.1f}% ({total_strat_wins}/{total_strat_games})")
    print("=" * 40 + "\n")

    # Add summary stats for compatibility with early stopping
    results["combined_strategic_winrate"] = combined_strat_winrate
    results["raw_policy_wins_as_p1"] = strat_p1["model_wins"]
    results["raw_policy_wins_as_p2"] = strat_p2["model_wins"]
    results["raw_policy_games_as_p1"] = strat_p1["total_games"]
    results["raw_policy_games_as_p2"] = strat_p2["total_games"]

    return results


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

                # Apply temperature for human-like play (lower = more deterministic)
                eval_temperature = 0.4
                policy_flat = masked_policy.flatten()

                # Apply temperature scaling
                if eval_temperature != 1.0 and np.sum(policy_flat) > 0:
                    log_probs = np.log(policy_flat + 1e-10)
                    scaled_logits = log_probs / eval_temperature
                    # Numerically stable softmax
                    scaled_logits = scaled_logits - np.max(scaled_logits)
                    exp_logits = np.exp(scaled_logits)
                    policy_flat = exp_logits / np.sum(exp_logits)

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

                # Apply temperature for human-like play (lower = more deterministic)
                eval_temperature = 0.4
                policy_flat = masked_policy.flatten()

                # Apply temperature scaling
                if eval_temperature != 1.0 and np.sum(policy_flat) > 0:
                    log_probs = np.log(policy_flat + 1e-10)
                    scaled_logits = log_probs / eval_temperature
                    # Numerically stable softmax
                    scaled_logits = scaled_logits - np.max(scaled_logits)
                    exp_logits = np.exp(scaled_logits)
                    policy_flat = exp_logits / np.sum(exp_logits)

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
