import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fast_mode: bool = False,
    ):
        self.device = device
        self.model = PolicyValueNet().to(device)

        # Two different learning rate configurations
        if fast_mode:
            # Faster learning rate for quick experiments
            self.lr = 0.0001
            print("Using fast training mode (lr=0.0001) - good for quick experiments")
        else:
            # Slower, more stable learning rate for production training
            self.lr = 0.000005
            print("Using stable training mode (lr=0.000005) - good for final training")

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

    def predict(self, board_state, flat_state, legal_moves=None):
        """Get move probabilities and value estimate"""
        self.model.eval()
        with torch.no_grad():
            board_state = torch.FloatTensor(board_state).to(self.device)
            flat_state = torch.FloatTensor(flat_state).to(self.device)
            if len(board_state.shape) == 3:
                board_state = board_state.unsqueeze(0)
                flat_state = flat_state.unsqueeze(0)
            board_state = board_state.permute(
                0, 3, 1, 2
            )  # NHWC -> NCHW for convolutions

            policy_logits, value = self.model(board_state, flat_state)

            # Apply legal moves mask if provided
            if legal_moves is not None:
                legal_moves = torch.FloatTensor(legal_moves).to(self.device)
                policy_logits = policy_logits.masked_fill(
                    legal_moves == 0, float("-inf")
                )

            # Convert to probabilities
            policy = F.softmax(policy_logits.reshape(policy_logits.shape[0], -1), dim=1)
            policy = policy.reshape(policy_logits.shape)

            # Ensure illegal moves have zero probability
            if legal_moves is not None:
                policy = policy * legal_moves
                # Renormalize if needed
                policy = policy / (policy.sum() + 1e-8)

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
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
