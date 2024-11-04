import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(0.1)  # Added dropout to residual blocks

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class DualNetworkModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared layers
        self.conv1 = nn.Conv2d(10, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        # Residual blocks in shared trunk
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(128), ResidualBlock(128), ResidualBlock(128)]
        )

        # Process flat input separately
        self.flat_fc = nn.Linear(10, 32)
        self.flat_bn = nn.BatchNorm1d(32)

        # Policy head - outputs 4x4x4 grid of move probabilities
        self.policy_conv = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(64)
        self.policy_conv2 = nn.Conv2d(64, 4, kernel_size=1)
        self.policy_dropout = nn.Dropout(0.1)  # Added dropout to policy head

        # Value head with additional layers and dropout
        self.value_conv = nn.Conv2d(128, 32, kernel_size=3, padding=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_dropout1 = nn.Dropout(0.3)  # Increased dropout for value head
        self.value_fc1 = nn.Linear(32 * 16 + 32, 256)
        self.value_bn2 = nn.BatchNorm1d(256)
        self.value_dropout2 = nn.Dropout(0.3)
        # Added intermediate layer
        self.value_fc_intermediate = nn.Linear(256, 128)
        self.value_bn3 = nn.BatchNorm1d(128)
        self.value_dropout3 = nn.Dropout(0.3)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, grid_input, flat_input):
        batch_size = grid_input.size(0)

        # Process shared layers
        x = F.relu(self.bn1(self.conv1(grid_input)))

        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Process flat input
        flat = F.relu(self.flat_bn(self.flat_fc(flat_input)))

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_dropout(policy)
        policy = self.policy_conv2(policy)
        policy = policy.permute(0, 2, 3, 1)

        # Apply softmax over all possible moves
        policy_flat = policy.reshape(batch_size, -1)
        policy_probs = F.softmax(policy_flat, dim=1)
        policy = policy_probs.reshape(batch_size, 4, 4, 4)

        # Value head with additional processing
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = self.value_dropout1(value)
        value = value.reshape(batch_size, -1)
        value = torch.cat((value, flat), dim=1)

        value = F.relu(self.value_bn2(self.value_fc1(value)))
        value = self.value_dropout2(value)

        value = F.relu(self.value_bn3(self.value_fc_intermediate(value)))
        value = self.value_dropout3(value)

        value = torch.sigmoid(self.value_fc2(value)).squeeze(-1)

        # Clip value predictions to avoid extremes
        value = torch.clamp(value, min=0.1, max=0.9)

        # Calculate value prediction regularization
        value_entropy_reg = 0.1 * (
            torch.mean(value * torch.log(value + 1e-8))
            + torch.mean((1 - value) * torch.log(1 - value + 1e-8))
        )

        return policy, value, value_entropy_reg


class DualNetworkWrapper:
    def __init__(self, device):
        self.device = device
        self.model = DualNetworkModel().to(device)

        # Modified optimizer with increased weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0002,  # Reduced learning rate
            weight_decay=0.02,  # Increased from 0.01
            betas=(0.9, 0.999),
        )

        # More aggressive learning rate scheduling
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.3,  # More aggressive reduction
            patience=3,  # Reduced patience
            verbose=True,  # Print when learning rate changes
            min_lr=1e-5,  # Add minimum learning rate
        )

    def predict(self, grid_input, flat_input):
        self.model.eval()
        with torch.no_grad():
            grid_input = torch.tensor(grid_input, dtype=torch.float32).to(self.device)
            grid_input = grid_input.permute(0, 3, 1, 2)
            flat_input = torch.tensor(flat_input, dtype=torch.float32).to(self.device)
            policy, value, _ = self.model(grid_input, flat_input)
            return policy.cpu().numpy(), value.cpu().numpy()

    def train(self, grid_inputs, flat_inputs, policy_targets, value_targets):
        self.model.train()

        grid_inputs = torch.tensor(grid_inputs, dtype=torch.float32).to(self.device)
        grid_inputs = grid_inputs.permute(0, 3, 1, 2)
        flat_inputs = torch.tensor(flat_inputs, dtype=torch.float32).to(self.device)
        policy_targets = torch.tensor(policy_targets, dtype=torch.float32).to(
            self.device
        )
        value_targets = torch.tensor(value_targets, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()

        policy_pred, value_pred, value_entropy_reg = self.model(
            grid_inputs, flat_inputs
        )

        # Calculate losses with regularization
        policy_loss = -torch.mean(
            torch.sum(policy_targets * torch.log(policy_pred + 1e-8), dim=[1, 2, 3])
        )

        value_loss = F.binary_cross_entropy(value_pred, value_targets)

        # Add L2 regularization for value predictions
        value_l2_reg = 0.01 * torch.mean(value_pred * value_pred)

        # Combine all value-related losses
        total_value_loss = value_loss + value_l2_reg + value_entropy_reg

        # Adjust the weighting of losses (reduced value loss weight)
        total_loss = policy_loss + 0.5 * total_value_loss

        total_loss.backward()

        # Stronger gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        self.optimizer.step()
        self.scheduler.step(total_loss)

        return (total_loss.item(), policy_loss.item(), total_value_loss.item())

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
