# First cell - Setup and Imports
!pip install torch numpy dataclasses typing

# Mount Google Drive to save checkpoints
from google.colab import drive
drive.mount('/content/drive')

# Clone your repository if it's on GitHub, or upload files
# If uploading manually, skip this
!git clone [your-repo-url]

# Set up checkpoint directory
!mkdir -p '/content/drive/MyDrive/tictacdo_checkpoints'

# Check GPU availability
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Second cell - Training
import os
import sys

# Update these paths based on your setup
CHECKPOINT_DIR = '/content/drive/MyDrive/tictacdo_checkpoints'
MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'latest_model.pth')

# Add your code directory to path
sys.path.append('/content/tictacdo_ai')  # Update this path

from train import train_network
from model import DualNetworkWrapper

# Initialize network
device = "cuda" if torch.cuda.is_available() else "cpu"
network_wrapper = DualNetworkWrapper(device)

# Load previous checkpoint if exists
if os.path.exists(MODEL_PATH):
    print(f"Loading previous model from {MODEL_PATH}")
    network_wrapper.load(MODEL_PATH)

# Training configuration
config = {
    'training_minutes': 1380,  # 23 hours
    'save_interval_minutes': 30,
    'num_simulations': 200,
    'batch_size': 128,
    'buffer_size': 30000,
}

# Custom save callback
def save_callback(model, path):
    # Save to both specified path and latest_model
    model.save(path)
    model.save(MODEL_PATH)
    print(f"Model saved to {path} and {MODEL_PATH}")

# Start training
train_network(
    network_wrapper=network_wrapper,
    **config,
    save_callback=save_callback  # You'll need to modify your train_network to accept this
)