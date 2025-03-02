import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import random
import torch.package
import time  # Add time import


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
        device=None,
        mode: str = "stable",
    ):
        t0 = time.time()
        # Only use CPU if forced or if no device specified
        self.device = "cpu" if os.getenv("FORCE_CPU") or not device else device
        self.model = PolicyValueNet().to(self.device)
        t1 = time.time()
        print(f"Model creation and device setup: {t1 - t0:.2f}s")

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

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

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
            policy_flat = policy_logits.view(batch_size, -1)

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

                    # If there's only one legal move, just use that
                    if num_legal == 1:
                        k = 1
                    else:
                        # Ensure min_k and max_k are valid
                        max_k = min(
                            max_k, num_legal
                        )  # Don't exceed number of legal moves
                        min_k = min(min_k, max_k)  # Don't let min exceed max

                        # If min_k equals max_k, just use that value
                        if min_k == max_k:
                            k = min_k
                        else:
                            k = random.randint(min_k, max_k)

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
        board_inputs: np.ndarray,
        flat_inputs: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        policy_weight: float = 1.0,
    ) -> Tuple[float, float, float]:
        """Perform a single training step on a batch of data.

        Args:
            board_inputs: Batch of board states (N, H, W, C)
            flat_inputs: Batch of flat state values (N, F)
            policy_targets: Batch of policy targets (N, H, W, 4)
            value_targets: Batch of value targets (N,)
            policy_weight: Weight for policy loss relative to value loss

        Returns:
            Tuple of (total_loss, policy_loss, value_loss)
        """
        self.model.train()

        # Convert inputs to tensors and move to device
        board_inputs = torch.FloatTensor(board_inputs).to(self.device)
        flat_inputs = torch.FloatTensor(flat_inputs).to(self.device)
        policy_targets = torch.FloatTensor(policy_targets).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)

        # Ensure correct shapes
        board_inputs = board_inputs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Forward pass
        policy_logits, value_pred = self.model(board_inputs, flat_inputs)

        # Calculate policy loss (cross entropy on move probabilities)
        batch_size = policy_logits.shape[0]
        policy_logits_flat = policy_logits.reshape(
            batch_size, -1
        )  # Flatten to (batch_size, 64)
        policy_targets_flat = policy_targets.reshape(batch_size, -1)  # Same shape
        policy_indices = torch.argmax(
            policy_targets_flat, dim=1
        )  # Get indices of 1s in one-hot vectors
        policy_loss = F.cross_entropy(policy_logits_flat, policy_indices)

        # Calculate value loss (MSE on game outcome predictions)
        value_loss = F.mse_loss(value_pred.squeeze(-1), value_targets)

        # Combine losses with weighting
        total_loss = policy_weight * policy_loss + value_loss

        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # Update learning rate
        self.scheduler.step(total_loss)

        return (total_loss.item(), policy_loss.item(), value_loss.item())

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path):
        """Load a model from path, automatically detecting if it's a package or state dict.

        Args:
            path: Path to either a regular checkpoint or package
        """
        start_time = time.time()

        if path.endswith(".pt"):  # Package format
            print("Loading from torch package...")
            load_start = time.time()
            importer = torch.package.PackageImporter(path)
            self.model = importer.load_pickle("model", "model.pkl")
            print(f"Package loading took {time.time() - load_start:.2f} seconds")
        else:  # Regular checkpoint
            load_start = time.time()
            checkpoint = torch.load(path)
            print(f"Model loading took {time.time() - load_start:.2f} seconds")

            # Check if this is a quantized model by looking for quantization parameters
            is_quantized = (
                any("_packed_params" in key for key in checkpoint.keys())
                if isinstance(checkpoint, dict)
                else False
            )

            if is_quantized:
                print("Loading quantized model...")
                quant_start = time.time()
                from torch.quantization import quantize_dynamic

                self.model = quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
                )
                print(f"Quantization took {time.time() - quant_start:.2f} seconds")

            state_dict_start = time.time()
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                if hasattr(self, "optimizer"):
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            print(
                f"State dict loading took {time.time() - state_dict_start:.2f} seconds"
            )

            print(
                f"Model loaded successfully, device: {next(self.model.parameters()).device}"
            )
            print(f"Total loading time: {time.time() - start_time:.2f} seconds")

    def reset_optimizer(self, new_lr: float = 0.001):
        """Reset optimizer with new learning rate to escape local minimum"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=new_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )
