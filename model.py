import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import random
from torch.quantization import quantize_dynamic
import torch.quantization
import os


class ResBlock(nn.Module):
    """Residual block with two convolutions and a skip connection"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class PolicyValueNet(nn.Module):
    """Network with residual blocks and balanced capacity"""

    def __init__(self):
        super().__init__()
        # Initial convolution with 256 channels
        self.conv_in = nn.Conv2d(10, 256, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(256)

        # Middle section with residual blocks at 256 channels
        self.res_blocks = nn.ModuleList(
            [ResBlock(256) for _ in range(8)]  # 8 residual blocks at 256 channels
        )

        # Policy head (outputs 4x4x4 move probabilities)
        self.policy_conv1 = nn.Conv2d(256, 128, 1)
        self.policy_bn = nn.BatchNorm2d(128)
        self.policy_conv2 = nn.Conv2d(128, 4, 1)

        # Value head
        self.value_conv1 = nn.Conv2d(256, 128, 1)
        self.value_bn = nn.BatchNorm2d(128)
        self.value_fc1 = nn.Linear(128 * 4 * 4 + 10, 256)  # Reduced FC layer size
        self.value_fc2 = nn.Linear(256, 128)
        self.value_fc3 = nn.Linear(128, 1)

    def forward(self, board_state, flat_state):
        # Initial convolution
        x = F.relu(self.bn_in(self.conv_in(board_state)))

        # Middle section
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv1(x)))
        policy = self.policy_conv2(policy)
        policy = policy.permute(0, 2, 3, 1)  # NCHW -> NHWC for 4x4x4 output

        # Value head
        value = F.relu(self.value_bn(self.value_conv1(x)))
        value = value.flatten(1)  # Flatten all dims except batch
        value = torch.cat([value, flat_state], dim=1)
        value = F.relu(self.value_fc1(value))
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))

        return policy, value


class ModelWrapper:
    # Simplify to just top-k selection
    DIFFICULTY_SETTINGS = {
        0: {"name": "Grandmaster", "top_k": (1, 1)},  # Always best move
        1: {"name": "Expert", "top_k": (1, 2)},  # 1-2 best moves
        2: {"name": "Intermediate", "top_k": (1, 3)},  # Up to 3 worst moves
        3: {"name": "Beginner", "top_k": (1, 6)},  # Up to 6 worst moves
    }

    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mode: str = "stable",
    ):
        """Initialize the model wrapper.

        Args:
            device: Device to run the model on ('cuda' or 'cpu')
            mode: One of:
                - 'fast': Faster learning rate for quick experiments
                - 'stable': Slower, more stable learning rate for final training
                - 'crunch': Load and optimize model for deployment (no training)
        """
        self.device = device
        self.model = PolicyValueNet().to(device)
        self.mode = mode.lower()

        if self.mode == "crunch":
            print("Using crunch mode - for model optimization and deployment")
            return

        # Configure learning rate based on mode
        if self.mode == "fast":
            self.lr = 0.0001
            print("Using fast training mode (lr=0.0001) - good for quick experiments")
        elif self.mode == "stable":
            self.lr = 0.000005
            print("Using stable training mode (lr=0.000005) - good for final training")
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be one of: fast, stable, crunch"
            )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )

        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

    def get_lr(self) -> float:
        """Return current learning rate"""
        return self.lr

    def set_lr(self, new_lr: float):
        """Manually set learning rate"""
        self.lr = new_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"Learning rate set to: {new_lr}")

    def predict(
        self,
        board_state,
        flat_state,
        legal_moves=None,
        difficulty: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get move probabilities and value estimate with optional difficulty setting"""
        self.model.eval()
        with torch.no_grad():
            board_state = torch.FloatTensor(board_state).to(self.device)
            flat_state = torch.FloatTensor(flat_state).to(self.device)
            if len(board_state.shape) == 3:
                board_state = board_state.unsqueeze(0)
                flat_state = flat_state.unsqueeze(0)
            board_state = board_state.permute(0, 3, 1, 2)

            policy_logits, value = self.model(board_state, flat_state)

            # Ensure consistent shapes throughout
            batch_size = policy_logits.shape[0]
            policy_flat = policy_logits.view(
                batch_size, -1
            )  # Use view instead of reshape

            # Handle legal moves first
            if legal_moves is not None:
                legal_moves = torch.FloatTensor(legal_moves).to(self.device)
                if len(legal_moves.shape) == 3:
                    legal_moves = legal_moves.unsqueeze(0)
                legal_moves_flat = legal_moves.view(
                    batch_size, -1
                )  # Match policy_flat shape

                # Get legal move indices
                legal_indices = torch.where(legal_moves_flat[0] > 0)[0]
                num_legal = len(legal_indices)

                # Apply difficulty settings if specified
                if difficulty is not None and difficulty in self.DIFFICULTY_SETTINGS:
                    settings = self.DIFFICULTY_SETTINGS[difficulty]
                    min_k, max_k = settings["top_k"]

                    # Always use at least 1 move, and no more than available
                    k = min(max(1, random.randint(1, min(max_k, num_legal))), num_legal)

                    # Get logits for legal moves
                    legal_logits = policy_flat[0, legal_indices]

                    if difficulty <= 1:  # Grandmaster and Expert: best moves
                        _, selected_indices = torch.topk(
                            legal_logits, k=k, largest=True
                        )
                    else:  # Intermediate and Beginner: worst moves
                        _, selected_indices = torch.topk(
                            -legal_logits, k=k, largest=True
                        )

                    # Create mask for selected moves
                    selected_legal_indices = legal_indices[selected_indices]
                    mask = torch.zeros_like(policy_flat)
                    mask[0, selected_legal_indices] = 1.0

                    # Apply mask to logits before softmax
                    policy_flat = policy_flat.masked_fill(mask == 0, float("-inf"))

                # Mask illegal moves
                policy_flat = policy_flat.masked_fill(
                    legal_moves_flat == 0, float("-inf")
                )

            # Convert to probabilities and normalize
            policy_flat = F.softmax(policy_flat, dim=1)
            policy = policy_flat.view(policy_logits.shape)  # Restore original shape

            return policy.cpu().numpy(), value.cpu().numpy()

    def train_step(
        self,
        board_states,
        flat_states,
        target_policies,
        target_values,
        legal_moves=None,
        policy_weight: float = 2.0,
    ):
        """Train on a batch of examples"""
        if self.mode == "crunch":
            raise ValueError(
                "Cannot train in crunch mode - use fast or stable mode for training"
            )

        self.model.train()
        board_states = (
            torch.FloatTensor(board_states).to(self.device).permute(0, 3, 1, 2)
        )
        flat_states = torch.FloatTensor(flat_states).to(self.device)
        target_policies = torch.FloatTensor(target_policies).to(self.device)
        target_values = torch.FloatTensor(target_values).to(self.device)

        self.optimizer.zero_grad()
        policy_logits, value_pred = self.model(board_states, flat_states)

        # Apply legal moves mask if provided
        if legal_moves is not None:
            legal_moves = torch.FloatTensor(legal_moves).to(self.device)
            policy_logits = policy_logits.masked_fill(legal_moves == 0, float("-inf"))

        # Policy loss (cross entropy over legal moves)
        policy_pred = F.softmax(
            policy_logits.reshape(policy_logits.shape[0], -1), dim=1
        )
        policy_pred = policy_pred.reshape(policy_logits.shape)

        # Increase policy weight for opening moves
        move_counts = (board_states != 0).sum(axis=(1, 2, 3))
        opening_moves_mask = (move_counts < 2).float()  # Focus on first move
        policy_weights = policy_weight * (1 + opening_moves_mask)

        # Calculate losses
        policy_loss = -torch.mean(
            policy_weights
            * torch.sum(target_policies * torch.log(policy_pred + 1e-8), dim=(1, 2, 3))
        )
        value_loss = F.mse_loss(value_pred.squeeze(), target_values)

        # Total loss with stronger policy emphasis
        total_loss = 2.0 * policy_loss + value_loss  # Fixed 2:1 ratio

        total_loss.backward()
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        """Load a model from path, automatically detecting if it's quantized.

        Args:
            path: Path to either a regular or quantized model checkpoint
        """
        checkpoint = torch.load(path)

        # Check if this is a quantized model (just the state dict) or regular checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            # Regular checkpoint with optimizer state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if hasattr(
                self, "optimizer"
            ):  # Only load optimizer if we're not in crunch mode
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            # Quantized model - need to quantize the model first
            print("Detected quantized model, preparing architecture...")
            self.model = quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d},
                dtype=torch.qint8,
            )
            self.model.load_state_dict(checkpoint)

    def crunch(self, input_path: str, output_dir: str):
        """Optimize model for minimal file size while keeping PyTorch compatibility.

        Args:
            input_path: Path to input PyTorch model
            output_dir: Directory to save optimized model
        """
        print(f"Loading model from {input_path}")
        self.load(input_path)
        self.model.eval()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Quantize the model to reduce size
        print("Quantizing model...")
        quantized_model = quantize_dynamic(
            self.model,
            {
                torch.nn.Linear,
                torch.nn.Conv2d,
                torch.nn.BatchNorm2d,
            },  # Quantize more layer types
            dtype=torch.qint8,
        )

        # Save in a more compressed format
        output_path = os.path.join(output_dir, "model_compressed.pth")
        torch.save(
            quantized_model.state_dict(),
            output_path,
            _use_new_zipfile_serialization=True,  # Use more efficient storage format
        )

        # Print size comparison
        original_size = os.path.getsize(input_path) / (1024 * 1024)  # MB
        compressed_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        reduction = (1 - compressed_size / original_size) * 100

        print(f"\nSize comparison:")
        print(f"Original model:    {original_size:.1f} MB")
        print(f"Compressed model:  {compressed_size:.1f} MB")
        print(f"Size reduction:    {reduction:.1f}%")
        print(f"\nOptimized model saved to: {output_path}")
