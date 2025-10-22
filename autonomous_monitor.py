#!/usr/bin/env python3
"""Autonomous training monitor - runs in background and logs progress"""
import time
import re
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("training_output.log")
MONITOR_LOG = Path("monitor_log.md")
CHECK_INTERVAL = 1800  # 30 minutes

def extract_latest_metrics(log_content):
    """Extract latest training metrics from log"""
    metrics = {}

    # Find latest iteration
    iter_matches = re.findall(r"ITER (\d+)", log_content)
    if iter_matches:
        metrics["latest_iter"] = int(iter_matches[-1])

    # Find latest loss
    loss_matches = re.findall(r"Loss: ([\d.]+)", log_content)
    if loss_matches:
        metrics["latest_loss"] = float(loss_matches[-1])

    # Find evaluation results
    eval_matches = re.findall(r"Combined win rate vs MCTS: ([\d.]+)%", log_content)
    if eval_matches:
        metrics["latest_winrate"] = float(eval_matches[-1])
        metrics["eval_iter"] = metrics.get("latest_iter", "?")

    # Check for early stopping
    if "EARLY STOPPING TRIGGERED" in log_content:
        metrics["early_stopped"] = True
        metrics["status"] = "FAILED - Early stop"
    elif "Training complete" in log_content:
        metrics["status"] = "COMPLETE"
    elif "Error during training" in log_content:
        # Extract error
        error_match = re.search(r"Error during training: (.+)", log_content)
        if error_match:
            metrics["status"] = f"CRASHED - {error_match.group(1)[:100]}"
    else:
        metrics["status"] = "RUNNING"

    return metrics

def log_status(metrics, check_num):
    """Log current status to monitor log"""
    with open(MONITOR_LOG, "a") as f:
        f.write(f"\n## Check #{check_num} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"- **Status:** {metrics.get('status', 'UNKNOWN')}\n")

        if "latest_iter" in metrics:
            f.write(f"- **Latest Iteration:** {metrics['latest_iter']}\n")

        if "latest_loss" in metrics:
            f.write(f"- **Latest Loss:** {metrics['latest_loss']:.4f}\n")

        if "latest_winrate" in metrics:
            f.write(f"- **Win Rate at Iter {metrics['eval_iter']}:** {metrics['latest_winrate']:.1f}%\n")

        if metrics.get("early_stopped"):
            f.write(f"\n**EARLY STOPPING TRIGGERED - Training failed to reach 40% win rate**\n")

        f.write("\n")

def main():
    """Main monitoring loop"""
    print(f"Starting autonomous monitor - logging to {MONITOR_LOG}")
    print(f"Checking every {CHECK_INTERVAL/60:.0f} minutes")

    # Initialize monitor log
    with open(MONITOR_LOG, "w") as f:
        f.write(f"# Autonomous Training Monitor Log\n\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Check interval: {CHECK_INTERVAL/60:.0f} minutes\n\n")

    check_num = 0

    while True:
        check_num += 1

        try:
            if LOG_FILE.exists():
                log_content = LOG_FILE.read_text(encoding='utf-8', errors='ignore')
                metrics = extract_latest_metrics(log_content)
                log_status(metrics, check_num)

                print(f"Check #{check_num}: {metrics.get('status', 'UNKNOWN')} - Iter {metrics.get('latest_iter', '?')}")

                # Stop monitoring if training finished or crashed
                if metrics.get("status") in ["COMPLETE", "FAILED - Early stop"] or "CRASHED" in metrics.get("status", ""):
                    print(f"Training finished: {metrics['status']}")
                    log_status({"status": f"Monitor stopped - {metrics['status']}"}, check_num + 1)
                    break
            else:
                print(f"Check #{check_num}: Waiting for training log file...")
                log_status({"status": "Waiting for log file"}, check_num)

        except Exception as e:
            print(f"Error during check #{check_num}: {e}")
            with open(MONITOR_LOG, "a") as f:
                f.write(f"\n**ERROR in check #{check_num}:** {str(e)}\n\n")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
