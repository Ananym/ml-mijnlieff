import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Enhanced network for 4x4 grid game with movement constraints"""

    def __init__(self):
        super().__init__()
        # Match existing state representation
        in_channels = 10  # Current state representation channels
        hidden_channels = 128  # increased from 96 for more capacity

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        # Add residual blocks for better feature extraction
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(hidden_channels),
                ResBlock(hidden_channels),
                ResBlock(hidden_channels),  # Added one more ResBlock
            ]
        )

        # Policy head - enhanced for more accurate move prediction
        # Multiple stages of refinement with attention to the placement constraints
        self.policy_conv1 = nn.Conv2d(hidden_channels, 64, kernel_size=1)
        self.policy_bn1 = nn.BatchNorm2d(64)

        # Second policy layer with larger capacity
        self.policy_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.policy_bn2 = nn.BatchNorm2d(32)

        # Final policy output layer - one channel per piece type (4 types)
        self.policy_conv3 = nn.Conv2d(32, 4, kernel_size=1)

        # Value head - correctly sized for your inputs
        self.value_conv = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 4 * 4 + 10, 64)  # +10 for your flat state
        self.value_fc2 = nn.Linear(64, 32)
        self.value_fc3 = nn.Linear(32, 1)

        # Initialize weights for better training stability
        self._initialize_weights()

    def _initialize_weights(self):
        # Kaiming initialization for ReLU networks
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, board_state, flat_state):
        # Initial feature extraction
        x = F.relu(self.bn1(self.conv1(board_state)))

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy head - outputs move probabilities (4x4x4 - one for each piece type)
        policy = F.relu(self.policy_bn1(self.policy_conv1(x)))
        policy = F.relu(self.policy_bn2(self.policy_conv2(policy)))
        policy = self.policy_conv3(policy)
        policy = policy.permute(0, 2, 3, 1)  # NCHW -> NHWC for 4x4x4 output

        # Value head - outputs single value prediction
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.flatten(1)
        value = torch.cat([value, flat_state], dim=1)
        value = F.relu(self.value_fc1(value))
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))

        return policy, value

    def count_parameters(self):
        """Count the number of trainable parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Enhanced wrapper with improved training dynamics
class ModelWrapper:
    DIFFICULTY_SETTINGS = {
        0: {"name": "Grandmaster", "top_k": (1, 1)},  # Always best move
        1: {"name": "Expert", "top_k": (1, 2)},  # 1-2 best moves
        2: {"name": "Intermediate", "top_k": (1, 3)},  # Up to 3 best moves
        3: {"name": "Beginner", "top_k": (2, 6)},  # Up to 6 moves, avoiding best
    }

    def __init__(self, device=None, mode: str = "stable"):
        # Initialize with the same interface as before
        self.device = "cpu" if device is None else device
        self.model = PolicyValueNet().to(self.device)

        # Print model size
        param_count = self.model.count_parameters()
        print(f"Model created with {param_count:,} trainable parameters")

        self.mode = mode.lower()

        if self.mode == "crunch":
            print("Using crunch mode - for model optimization and deployment")
            return

        # Simplified learning rate configuration with sensible defaults
        if self.mode == "fast":
            self.lr = 0.001  # Higher but reasonable learning rate
            print(f"Using fast training mode (lr={self.lr:.6f})")
        elif self.mode == "stable":
            self.lr = 0.0003  # Moderate learning rate
            print(f"Using stable training mode (lr={self.lr:.6f})")
        elif self.mode == "custom_lr":
            self.lr = 0.0001  # More standard custom rate
            print(f"Using custom learning rate: {self.lr:.6f}")
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Must be one of: fast, stable, crunch, custom_lr"
            )

        # Standard AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # More sensible one-cycle scheduler with moderate values
        # Total training iterations (adjust based on your expected training length)
        max_iterations = 140

        # One-cycle LR with moderate parameters
        self.scheduler = SafeOneCycleLR(
            self.optimizer,
            max_lr=self.lr * 6,
            total_steps=max_iterations,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=30,  # Much smaller reduction (was 100)
        )

        # Print actual initial learning rate
        print(f"Initial learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Will peak at: {self.lr * 6:.6f}")
        print(f"Will finish at: {self.lr * 6 / 1000:.6f}")

    def predict(
        self, board_state, flat_state, legal_moves=None, difficulty: int = None
    ):
        """Get move probabilities and value estimate with optional difficulty setting"""
        self.model.eval()
        with torch.no_grad():
            board_state = torch.FloatTensor(board_state).to(self.device)
            flat_state = torch.FloatTensor(flat_state).to(self.device)
            if len(board_state.shape) == 3:
                board_state = board_state.unsqueeze(0)
                flat_state = flat_state.unsqueeze(0)
            board_state = board_state.permute(0, 3, 1, 2)  # NHWC -> NCHW

            policy_logits, value = self.model(board_state, flat_state)

            # Ensure consistent shapes
            batch_size = policy_logits.shape[0]
            policy_flat = policy_logits.view(batch_size, -1)  # Flatten to B x (H*W*4)

            # Apply legal moves mask if provided
            if legal_moves is not None:
                legal_moves = torch.FloatTensor(legal_moves).to(self.device)
                if len(legal_moves.shape) == 3:
                    legal_moves = legal_moves.unsqueeze(0)
                legal_moves_flat = legal_moves.view(batch_size, -1)

                # Get legal move indices for selection
                legal_indices = torch.where(legal_moves_flat[0] > 0)[0]
                num_legal = len(legal_indices)

                # Apply difficulty settings if specified
                if difficulty is not None and difficulty in self.DIFFICULTY_SETTINGS:
                    settings = self.DIFFICULTY_SETTINGS[difficulty]
                    min_k, max_k = settings["top_k"]

                    # If only one legal move, just use that
                    if num_legal == 1:
                        k = 1
                    else:
                        # Ensure k values are valid for available moves
                        max_k = min(max_k, num_legal)
                        min_k = min(min_k, max_k)

                        # Select random k value within range
                        k = torch.randint(min_k, max_k + 1, (1,)).item()

                    # Get logits for legal moves only
                    legal_logits = policy_flat[0, legal_indices]

                    # For Grandmaster and Expert: select from best moves
                    # For others: avoid the very best moves
                    if difficulty <= 1:
                        _, selected_indices = torch.topk(
                            legal_logits, k=k, largest=True
                        )
                    else:
                        # For weaker difficulties, exclude top moves
                        num_to_exclude = 1 if difficulty == 2 else 2
                        if num_legal > num_to_exclude + 1:
                            # Get indices of top moves to exclude
                            _, top_indices = torch.topk(
                                legal_logits, k=num_to_exclude, largest=True
                            )

                            # Create mask excluding top moves
                            exclude_mask = torch.ones_like(legal_logits)
                            exclude_mask[top_indices] = 0

                            # Apply mask and get remaining moves
                            masked_logits = legal_logits * exclude_mask
                            masked_logits[top_indices] = float("-inf")

                            # Select k from remaining moves
                            _, selected_indices = torch.topk(
                                masked_logits, k=k, largest=True
                            )
                        else:
                            # Not enough moves to exclude, just select randomly
                            selected_indices = torch.randperm(num_legal)[:k]

                    # Create mask for final selected moves
                    selected_legal_indices = legal_indices[selected_indices]
                    mask = torch.zeros_like(policy_flat)
                    mask[0, selected_legal_indices] = 1.0

                    # Apply selected moves mask
                    policy_flat = policy_flat.masked_fill(mask == 0, float("-inf"))

                # Always mask illegal moves regardless of difficulty
                policy_flat = policy_flat.masked_fill(
                    legal_moves_flat == 0, float("-inf")
                )

            # Convert to probabilities
            policy_flat = F.softmax(policy_flat, dim=1)
            policy = policy_flat.view(
                policy_logits.shape
            )  # Reshape to original dimensions

            return policy.cpu().numpy(), value.cpu().numpy()

    def train_step(
        self,
        board_inputs,
        flat_inputs,
        policy_targets,
        value_targets,
        policy_weight=1.5,  # Increased policy weight to focus more on move selection
    ):
        """Perform a single training step with improved loss functions"""
        self.model.train()

        # Convert to tensors and move to device
        board_inputs = torch.FloatTensor(board_inputs).to(self.device)
        flat_inputs = torch.FloatTensor(flat_inputs).to(self.device)
        policy_targets = torch.FloatTensor(policy_targets).to(self.device)
        value_targets = torch.FloatTensor(value_targets).to(self.device)

        # Correct shapes for the network
        board_inputs = board_inputs.permute(0, 3, 1, 2)  # NHWC -> NCHW

        # Forward pass
        policy_logits, value_pred = self.model(board_inputs, flat_inputs)

        # Calculate policy loss
        batch_size = policy_logits.shape[0]
        policy_logits_flat = policy_logits.reshape(batch_size, -1)
        policy_targets_flat = policy_targets.reshape(batch_size, -1)

        # Determine if targets are one-hot or distributions
        is_distribution = (policy_targets_flat.sum(dim=1) > 1.01).any() or (
            policy_targets_flat.max(dim=1)[0] < 0.99
        ).any()

        if is_distribution:
            # KL divergence for distribution targets (from MCTS)
            policy_probs = F.softmax(policy_logits_flat, dim=1)
            log_policy = F.log_softmax(policy_logits_flat, dim=1)
            policy_loss = torch.mean(
                -torch.sum(policy_targets_flat * log_policy, dim=1)
            )

            # Add entropy bonus to encourage exploration (smaller for distributions)
            entropy = -torch.sum(policy_probs * log_policy, dim=1).mean()
            policy_loss = policy_loss - 0.01 * entropy
        else:
            # Cross-entropy for one-hot targets (supervised learning)
            policy_indices = torch.argmax(policy_targets_flat, dim=1)
            policy_loss = F.cross_entropy(policy_logits_flat, policy_indices)

            # Stronger entropy bonus for one-hot targets
            log_policy = F.log_softmax(policy_logits_flat, dim=1)
            policy_probs = F.softmax(policy_logits_flat, dim=1)
            entropy = -torch.sum(policy_probs * log_policy, dim=1).mean()
            policy_loss = policy_loss - 0.03 * entropy

        # Value loss with label smoothing to prevent overconfidence
        # This is crucial for games with complex scoring systems
        smoothed_targets = value_targets * 0.93  # Scale targets toward zero
        value_loss = F.mse_loss(value_pred.squeeze(-1), smoothed_targets)

        # Small L2 regularization for value predictions
        value_l2_reg = 0.0001 * torch.mean(torch.square(value_pred))
        value_loss = value_loss + value_l2_reg

        # Combine losses with policy_weight emphasis
        total_loss = policy_weight * policy_loss + value_loss

        # Optimization step with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients to prevent training instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return (total_loss.item(), policy_loss.item(), value_loss.item())

    def save(self, path):
        """Save model and optimizer state"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if hasattr(self, "scheduler") else None
                ),
            },
            path,
        )

    def load(self, path):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer if available and we're not in crunch mode
            if hasattr(self, "optimizer") and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # Check if learning rate is too low and reset if needed
                current_lr = self.optimizer.param_groups[0]["lr"]
                min_acceptable_lr = 0.00001

                if current_lr < min_acceptable_lr:
                    print(
                        f"Warning: Loaded learning rate is too low ({current_lr:.8f})"
                    )
                    new_lr = self.lr  # Reset to initialization value
                    print(f"Resetting learning rate to {new_lr:.8f}")
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr

            # Load scheduler if available
            if hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)

        print(f"Model loaded successfully on {self.device}")
        return self.model.count_parameters()

    def reset_optimizer(self, new_lr=None):
        """Reset optimizer with optional new learning rate"""
        if new_lr is None:
            new_lr = self.lr  # Use default from initialization

        print(f"Resetting optimizer with learning rate: {new_lr:.6f}")

        # Create fresh optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=new_lr,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # One-cycle LR with moderate parameters

        # Reset scheduler too with new optimizer
        max_iterations = 140  # Estimate of total training iterations
        remaining_steps = max_iterations  # Could be refined if we track iteration
        self.scheduler = SafeOneCycleLR(
            self.optimizer,
            max_lr=self.lr * 6,
            total_steps=remaining_steps,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=30,  # Much smaller reduction (was 100)
        )

        # Print actual initial learning rate from optimizer
        print(f"Initial learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"Will peak at: {new_lr * 6:.6f}")
        print(f"Will finish at: {new_lr * 6 / 1000:.6f}")

        return new_lr


class SafeOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """OneCycleLR that safely handles extra steps beyond total_steps"""

    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)
        self.total_steps_completed = False
        self.min_lr = min(self.get_lr())  # Store minimum learning rate

    def step(self, epoch=None):
        if self.total_steps_completed:
            # Already reached end of schedule, maintain minimum rate
            for param_group, lr in zip(
                self.optimizer.param_groups,
                [self.min_lr] * len(self.optimizer.param_groups),
            ):
                param_group["lr"] = lr
            return

        try:
            # Try normal step
            super().step(epoch)
        except ValueError as e:
            # Scheduler reached the end, set all learning rates to minimum
            self.total_steps_completed = True
            for param_group, lr in zip(
                self.optimizer.param_groups,
                [self.min_lr] * len(self.optimizer.param_groups),
            ):
                param_group["lr"] = lr
            print(
                f"OneCycleLR completed, continuing with minimum learning rate: {self.min_lr:.6f}"
            )
