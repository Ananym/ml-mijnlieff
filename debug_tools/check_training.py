#!/usr/bin/env python3
"""Quick script to check training progress"""
import subprocess
import sys

# Get the bash output
result = subprocess.run(
    ["tail", "-100", "training_output.txt"],
    capture_output=True,
    text=True,
    cwd=r"C:\Users\Sam Tope\Documents\GitHub\tictacdo_ai"
)

if result.returncode != 0:
    print("Could not read training output")
    sys.exit(1)

output = result.stdout

# Extract key metrics
import re

# Find latest iteration report
iter_matches = re.findall(r"ITER (\d+)", output)
if iter_matches:
    latest_iter = iter_matches[-1]
    print(f"Latest iteration: {latest_iter}")

# Find eval results
eval_matches = re.findall(r"Combined win rate vs MCTS: ([\d.]+)%", output)
if eval_matches:
    latest_winrate = eval_matches[-1]
    print(f"Latest win rate: {latest_winrate}%")

# Find early stopping
if "EARLY STOPPING TRIGGERED" in output:
    print("WARNING - Early stopping triggered - training failed")
elif "Training complete" in output:
    print("Training complete!")
else:
    print("Training in progress...")
