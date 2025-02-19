import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyValueNet(nn.Module):
    """Simple network that directly predicts moves and game outcome"""

    def __init__(self):
        super().__init__()
        # Shared layers
        self.conv1 = nn.Conv2d(10, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)

        # Policy head (outputs 4x4x4 move probabilities)
        self.policy_conv = nn.Conv2d(32, 4, 1)

        # Value head (outputs win probability)
        self.value_conv = nn.Conv2d(32, 16, 1)  # Reduce channels before flattening
        self.value_fc = nn.Linear(16 * 4 * 4 + 10, 1)  # 4x4 spatial dims

    def forward(self, board_state, flat_state):
        # Main trunk
        x = F.relu(self.conv1(board_state))
        x = F.relu(self.conv2(x))

        # Policy head
        policy = self.policy_conv(x)
        policy = policy.permute(0, 2, 3, 1)  # NCHW -> NHWC for 4x4x4 output

        # Value head
        value = F.relu(self.value_conv(x))
        value = value.flatten(1)  # Flatten all dims except batch
        value = torch.cat([value, flat_state], dim=1)
        value = torch.tanh(self.value_fc(value))

        return policy, value


class ModelWrapper:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = PolicyValueNet().to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.999)
        )

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
        policy_loss = -torch.mean(
            torch.sum(target_policies * torch.log(policy_pred + 1e-8), dim=(1, 2, 3))
        )

        # Value loss (MSE)
        value_loss = F.mse_loss(value_pred.squeeze(), target_values)

        # Total loss with L2 regularization (already included in optimizer)
        total_loss = policy_loss + value_loss

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
