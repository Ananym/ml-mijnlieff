import torch
import torch.nn as nn
import torch.nn.functional as F

# Scale factor for policy entropy bonus (lower = less exploration)
# This should be synced with ENTROPY_BONUS_SCALE in train.py
ENTROPY_BONUS_SCALE = 0.07  # INCREASED from 0.05 for better generalization vs human players


class ResBlock(nn.Module):
    """Simple residual block with two convolutions and a skip connection"""

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
    """Streamlined network for 4x4 grid game with movement constraints"""

    def __init__(self):
        super().__init__()
        # input channels: 6 channels for board state
        in_channels = 6
        # Experiment 8: Reverted to baseline (Exp 7 showed capacity wasn't bottleneck)
        hidden_channels = 128  # ~1.2M params baseline

        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        # 3 residual blocks - RESTORED for better capacity
        self.res_blocks = nn.ModuleList(
            [
                ResBlock(hidden_channels),
                ResBlock(hidden_channels),
                ResBlock(hidden_channels),
            ]
        )

        # Policy head - RESTORED to original capacity
        policy_channels = 64  # RESTORED from 32
        self.policy_conv = nn.Conv2d(hidden_channels, policy_channels, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_out = nn.Conv2d(policy_channels, 4, kernel_size=1)  # 4 piece types

        # Value head - RESTORED to original capacity
        value_channels = 64  # RESTORED from 32
        self.value_conv = nn.Conv2d(hidden_channels, value_channels, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_channels)

        # Spatial features processing
        # 64 channels on 4x4 board = 1024 features (was 512 with 32 channels)
        self.value_fc1 = nn.Linear(value_channels * 4 * 4, 256)  # INCREASED from 128
        self.value_bn2 = nn.BatchNorm1d(256)  # UPDATED to match

        # Flat features processing
        flat_feature_size = 12  # assuming 12 flat features
        self.flat_fc = nn.Linear(flat_feature_size, 64)  # INCREASED from 32
        self.flat_bn = nn.BatchNorm1d(64)  # UPDATED to match

        # Combined processing with restored capacity
        self.value_fc2 = nn.Linear(256 + 64, 128)  # spatial + flat (INCREASED)
        self.value_bn3 = nn.BatchNorm1d(128)  # UPDATED to match
        self.value_fc3 = nn.Linear(128, 64)  # INCREASED from (64, 32)
        self.value_bn4 = nn.BatchNorm1d(64)  # UPDATED to match
        self.value_fc4 = nn.Linear(64, 1)  # INCREASED from (32, 1)

        # Apply initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming initialization for conv layers
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                # standard init for batch norm
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # xavier for linear layers
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # special init for value head final layer to prevent vanishing gradients
        nn.init.uniform_(self.value_fc4.weight, -0.03, 0.03)
        nn.init.constant_(self.value_fc4.bias, 0)

    def forward(self, board_state, flat_state):
        # Initial feature extraction
        x = F.relu(self.bn1(self.conv1(board_state)))

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy head - outputs move probabilities (4x4x4 - one for each piece type)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_out(policy)
        policy = policy.permute(0, 2, 3, 1)  # NCHW -> NHWC for 4x4x4 output

        # Value head - predicts score differential
        value = F.relu(self.value_bn(self.value_conv(x)))
        value_flat = value.flatten(1)  # flatten spatial dimensions

        # Process flat features
        flat_features = F.relu(self.flat_bn(self.flat_fc(flat_state)))

        # Process spatial features
        value = F.relu(self.value_bn2(self.value_fc1(value_flat)))

        # Combine spatial and flat features
        value_combined = torch.cat([value, flat_features], dim=1)

        # Final value processing with modest dropout
        value = F.dropout(value_combined, p=0.1, training=self.training)
        value = F.relu(self.value_bn3(self.value_fc2(value)))
        value = F.dropout(value, p=0.1, training=self.training)
        value = F.relu(self.value_bn4(self.value_fc3(value)))

        # Final output with tanh activation for [-1, 1] range
        value = torch.tanh(self.value_fc4(value))

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

        # Learning rate parameters
        self.div_factor = 2.5  # Faster warmup (was 3)
        self.final_div_factor = 4  # Less aggressive decay (was 5)
        self.max_iterations = 50  # Keep synchronized with train.py MAX_ITERATIONS

        # Learning rates - increased for smaller model
        if self.mode == "fast":
            self.max_lr = 0.015  # Increased from 0.01
        elif self.mode == "stable":
            self.max_lr = 0.005  # Increased from 0.003 for faster convergence

        # Optimizer parameters
        weight_decay = 2e-5  # Reduced from 3e-5 for smaller model
        betas = (
            0.9,
            0.999,
        )  # Standard betas - more stable for smaller model

        # AdamW optimizer with tuned parameters
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.max_lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=1e-8,
        )

        # Create scheduler with default max iterations
        self.scheduler = self._create_scheduler(self.optimizer, self.max_iterations)

        # Print learning rate info
        print(f"Initial learning rate: {self.max_lr:.6f}")
        print(f"Will peak at: {self.max_lr:.6f}")
        print(
            f"Will finish at: {self.max_lr / (self.div_factor * self.final_div_factor):.6f}"
        )

    def _create_scheduler(self, optimizer, remaining_iterations):
        """Helper function to create a scheduler with consistent parameters"""
        return SafeOneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=remaining_iterations,
            pct_start=0.3,  # longer warmup period
            anneal_strategy="cos",
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
        )

    def predict(
        self, board_state, flat_state, legal_moves=None, difficulty: int = None
    ):
        """Get move probabilities and value estimate with optional difficulty setting"""
        self.model.eval()
        with torch.no_grad():
            # check if inputs are already tensors and handle device conversion
            if not isinstance(board_state, torch.Tensor):
                board_state = torch.FloatTensor(board_state).to(self.device)
            else:
                board_state = board_state.to(self.device)

            if not isinstance(flat_state, torch.Tensor):
                flat_state = torch.FloatTensor(flat_state).to(self.device)
            else:
                flat_state = flat_state.to(self.device)

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
                if not isinstance(legal_moves, torch.Tensor):
                    legal_moves = torch.FloatTensor(legal_moves).to(self.device)
                else:
                    legal_moves = legal_moves.to(self.device)

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
        policy_weight,  # balanced weighting (was 0.3)
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

        # No value target smoothing - learn exact values
        smoothed_targets = value_targets

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

            # Add entropy bonus
            entropy = -torch.sum(policy_probs * log_policy, dim=1).mean()
            policy_loss = policy_loss - ENTROPY_BONUS_SCALE * entropy
        else:
            # Cross-entropy for one-hot targets (supervised learning)
            policy_indices = torch.argmax(policy_targets_flat, dim=1)
            policy_loss = F.cross_entropy(policy_logits_flat, policy_indices)

            # Add entropy bonus
            log_policy = F.log_softmax(policy_logits_flat, dim=1)
            policy_probs = F.softmax(policy_logits_flat, dim=1)
            entropy = -torch.sum(policy_probs * log_policy, dim=1).mean()
            policy_loss = policy_loss - ENTROPY_BONUS_SCALE * entropy

        # Use Huber loss for value - better for RL scenarios
        value_loss = F.smooth_l1_loss(value_pred.squeeze(-1), smoothed_targets)

        # Combine losses with policy weight
        total_loss = policy_weight * policy_loss + (1.0 - policy_weight) * value_loss

        # Optimization step with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()

        # Capture gradient statistics for monitoring AND adaptive weighting
        grad_stats = {}

        # Compute gradient norms for policy and value heads separately
        policy_grad_norm = 0.0
        value_grad_norm = 0.0

        # Policy head parameters
        for param in [self.model.policy_conv.weight, self.model.policy_conv.bias,
                      self.model.policy_out.weight, self.model.policy_out.bias]:
            if param.grad is not None:
                policy_grad_norm += param.grad.norm().item() ** 2
        policy_grad_norm = policy_grad_norm ** 0.5

        # Value head parameters
        for param in [self.model.value_conv.weight, self.model.value_conv.bias,
                      self.model.value_fc1.weight, self.model.value_fc1.bias,
                      self.model.flat_fc.weight, self.model.flat_fc.bias,
                      self.model.value_fc2.weight, self.model.value_fc2.bias,
                      self.model.value_fc3.weight, self.model.value_fc3.bias,
                      self.model.value_fc4.weight, self.model.value_fc4.bias]:
            if param.grad is not None:
                value_grad_norm += param.grad.norm().item() ** 2
        value_grad_norm = value_grad_norm ** 0.5

        # Calculate adaptive policy weight for next iteration
        # Target: equal gradient magnitudes (ratio = 1.0)
        # policy_weight * policy_grad = (1 - policy_weight) * value_grad
        # Solve for policy_weight: policy_weight = value_grad / (policy_grad + value_grad)
        if policy_grad_norm > 0 and value_grad_norm > 0:
            recommended_policy_weight = value_grad_norm / (policy_grad_norm + value_grad_norm)
            # Clamp to reasonable range [0.1, 0.9]
            recommended_policy_weight = max(0.1, min(0.9, recommended_policy_weight))
        else:
            recommended_policy_weight = policy_weight

        grad_stats['policy_grad_norm'] = policy_grad_norm
        grad_stats['value_grad_norm'] = value_grad_norm
        grad_stats['grad_ratio'] = policy_grad_norm / (value_grad_norm + 1e-8)
        grad_stats['recommended_policy_weight'] = recommended_policy_weight

        # Get value head gradients for debugging (UPDATE not reassign)
        if self.model.value_fc4.weight.grad is not None:
            fc4_w_grad = self.model.value_fc4.weight.grad
            fc4_b_grad = self.model.value_fc4.bias.grad
            fc3_w_grad = self.model.value_fc3.weight.grad
            fc3_b_grad = self.model.value_fc3.bias.grad

            grad_stats.update({
                "value_fc4_weight": {
                    "mean": float(fc4_w_grad.mean().item()),
                    "std": float(fc4_w_grad.std().item()),
                    "min": float(fc4_w_grad.min().item()),
                    "max": float(fc4_w_grad.max().item()),
                    "norm": float(fc4_w_grad.norm().item()),
                },
                "value_fc4_bias": {
                    "mean": float(fc4_b_grad.mean().item()),
                    "std": float(fc4_b_grad.std().item()),
                    "min": float(fc4_b_grad.min().item()),
                    "max": float(fc4_b_grad.max().item()),
                    "norm": float(fc4_b_grad.norm().item()),
                },
                "value_fc3_weight": {
                    "mean": float(fc3_w_grad.mean().item()),
                    "std": float(fc3_w_grad.std().item()),
                    "min": float(fc3_w_grad.min().item()),
                    "max": float(fc3_w_grad.max().item()),
                    "norm": float(fc3_w_grad.norm().item()),
                },
                "value_fc3_bias": {
                    "mean": float(fc3_b_grad.mean().item()),
                    "std": float(fc3_b_grad.std().item()),
                    "min": float(fc3_b_grad.min().item()),
                    "max": float(fc3_b_grad.max().item()),
                    "norm": float(fc3_b_grad.norm().item()),
                },
            })

        # Apply gradient clipping to prevent instability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return (total_loss.item(), policy_loss.item(), value_loss.item(), grad_stats)

    def save_checkpoint(self, path, training_state=None):
        """Save model checkpoint with optional training state"""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if hasattr(self, "scheduler") else None
            ),
        }

        # Add training state if provided
        if training_state is not None:
            checkpoint.update(training_state)

        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """Load full checkpoint including training state"""
        checkpoint = torch.load(path, map_location=self.device)

        # Load model state
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer if available and we're not in crunch mode
            if hasattr(self, "optimizer") and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler if available
            if hasattr(self, "scheduler") and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # Return training state information (excluding model components)
            training_state = {
                k: v
                for k, v in checkpoint.items()
                if k
                not in [
                    "model_state_dict",
                    "optimizer_state_dict",
                    "scheduler_state_dict",
                ]
            }
            return training_state
        else:
            # Legacy loading (just weights)
            self.model.load_state_dict(checkpoint)
            print(f"Model weights loaded successfully on {self.device}")
            return {}

    def reset_optimizer(self, new_max_lr=None):
        """Reset optimizer with optional new learning rate"""
        if new_max_lr is None:
            new_max_lr = self.max_lr  # Use default from initialization

        print(f"Resetting optimizer with max learning rate: {new_max_lr:.6f}")

        # Update max_lr
        self.max_lr = new_max_lr

        # Create fresh optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.max_lr,
            weight_decay=3e-5,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        # Reset scheduler with new optimizer
        remaining_steps = self.max_iterations  # use class variable
        self.scheduler = self._create_scheduler(self.optimizer, remaining_steps)

        # Print learning rate info
        print(f"Initial learning rate: {self.max_lr:.6f}")
        print(f"Will peak at: {self.max_lr:.6f}")
        print(
            f"Will finish at: {self.max_lr / (self.div_factor * self.final_div_factor):.6f}"
        )

        return self.max_lr


class SafeOneCycleLR(torch.optim.lr_scheduler.OneCycleLR):
    """OneCycleLR that safely handles extra steps beyond total_steps"""

    def __init__(self, optimizer, **kwargs):
        # Set attributes before calling super().__init__
        self.total_steps_completed = False

        # Call parent init
        super().__init__(optimizer, **kwargs)

        # Store minimum learning rate after parent init
        self.min_lr = min(self.get_lr())

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
        except ValueError:
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
