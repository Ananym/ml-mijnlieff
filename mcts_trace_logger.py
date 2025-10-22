"""
One-time MCTS trace logger for debugging and sanity checking.
Logs detailed MCTS behavior and network interaction to a separate file.
"""
import json
from typing import Optional
from pathlib import Path

class MCTSTraceLogger:
    def __init__(self, filepath="mcts_trace.log"):
        self.filepath = filepath
        self.enabled = False
        self.log_entries = []
        self.move_count = 0

    def enable(self):
        """Enable logging for one game"""
        self.enabled = True
        self.log_entries = []
        self.move_count = 0

    def disable(self):
        """Disable logging and write to file"""
        if self.enabled and self.log_entries:
            with open(self.filepath, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MCTS TRACE LOG - Network & Search Interaction\n")
                f.write("=" * 80 + "\n\n")
                for entry in self.log_entries:
                    f.write(entry + "\n")
            print(f"\n[TRACE] Wrote {len(self.log_entries)} log entries to {self.filepath}")
        self.enabled = False
        self.log_entries = []

    def log_move_start(self, move_num, current_player, game_state_summary):
        """Log start of a move"""
        if not self.enabled:
            return
        self.move_count = move_num
        self.log_entries.append(f"\n{'='*60}")
        self.log_entries.append(f"MOVE {move_num} - {current_player}")
        self.log_entries.append(f"{'='*60}")
        self.log_entries.append(f"Game State: {game_state_summary}")

    def log_network_prediction(self, state_summary, policy_top5, value_pred):
        """Log network's initial prediction"""
        if not self.enabled:
            return
        self.log_entries.append(f"\n--- Network Prediction ---")
        self.log_entries.append(f"State: {state_summary}")
        self.log_entries.append(f"Value Prediction: {value_pred:.4f}")
        self.log_entries.append(f"Top-5 Policy Moves:")
        for move, prob in policy_top5:
            self.log_entries.append(f"  {move}: {prob:.4f}")

    def log_mcts_search_start(self, num_sims):
        """Log MCTS search initialization"""
        if not self.enabled:
            return
        self.log_entries.append(f"\n--- MCTS Search ({num_sims} simulations) ---")

    def log_mcts_search_end(self, root_value, root_visits, policy_top5, chosen_move):
        """Log MCTS search results"""
        if not self.enabled:
            return
        self.log_entries.append(f"\n--- MCTS Results ---")
        self.log_entries.append(f"Root Value (avg): {root_value:.4f}")
        self.log_entries.append(f"Root Visits: {root_visits}")
        self.log_entries.append(f"Final Policy (top-5 by visits):")
        for move, visits, q_val in policy_top5:
            self.log_entries.append(f"  {move}: visits={visits}, Q={q_val:.4f}")
        self.log_entries.append(f"Chosen Move: {chosen_move}")

    def log_node_expansion(self, node_info, network_value, children_count):
        """Log node expansion details"""
        if not self.enabled:
            return
        self.log_entries.append(f"\n  Expanded: {node_info}")
        self.log_entries.append(f"    Network Value: {network_value:.4f}")
        self.log_entries.append(f"    Children: {children_count}")

    def log_value_backprop(self, path_length, values):
        """Log value backpropagation"""
        if not self.enabled:
            return
        self.log_entries.append(f"\n  Backprop: {path_length} nodes")
        self.log_entries.append(f"    Values: {' -> '.join([f'{v:.3f}' for v in values[:5]])}...")

    def log_game_end(self, winner, final_scores, move_count):
        """Log game completion"""
        if not self.enabled:
            return
        self.log_entries.append(f"\n{'='*60}")
        self.log_entries.append(f"GAME END")
        self.log_entries.append(f"{'='*60}")
        self.log_entries.append(f"Winner: {winner}")
        self.log_entries.append(f"Scores: {final_scores}")
        self.log_entries.append(f"Total Moves: {move_count}")

# Global singleton
_trace_logger = None

def get_trace_logger():
    global _trace_logger
    if _trace_logger is None:
        _trace_logger = MCTSTraceLogger()
    return _trace_logger
