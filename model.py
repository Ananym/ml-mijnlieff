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
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class DualNetworkModel(nn.Module):
    def __init__(self, base_channels=128):
        super().__init__()
        # Shared layers
        self.conv1 = nn.Conv2d(10, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)

        # 4 residual blocks
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(base_channels) for _ in range(4)]
        )

        # Process flat input
        self.flat_fc = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Policy head
        self.policy_conv = nn.Conv2d(base_channels, 32, kernel_size=3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_conv2 = nn.Conv2d(32, 4, kernel_size=1)

        # Value head
        self.value_conv = nn.Conv2d(base_channels, 32, kernel_size=3, padding=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 16 + 64, 256)
        self.value_bn2 = nn.BatchNorm1d(256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_bn3 = nn.BatchNorm1d(128)
        self.value_fc3 = nn.Linear(128, 1)

    def forward(self, grid_input, flat_input):
        batch_size = grid_input.size(0)

        # Process flat input
        flat = self.flat_fc(flat_input)

        # Process shared layers
        x = F.relu(self.bn1(self.conv1(grid_input)))

        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = self.policy_conv2(policy)
        policy = policy.permute(0, 2, 3, 1)

        policy_flat = policy.reshape(batch_size, -1)
        policy_probs = F.softmax(policy_flat, dim=1)
        policy = policy_probs.reshape(batch_size, 4, 4, 4)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(batch_size, -1)
        value = torch.cat((value, flat), dim=1)

        value = F.relu(self.value_bn2(self.value_fc1(value)))
        value = F.relu(self.value_bn3(self.value_fc2(value)))
        value = torch.sigmoid(self.value_fc3(value)).squeeze(-1)

        return policy, value


class DualNetworkWrapper:
    def __init__(self, device):
        self.device = device
        self.model = DualNetworkModel().to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.0002,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True,
            min_lr=1e-5,
        )

    def predict(self, grid_input, flat_input):
        self.model.eval()
        with torch.no_grad():
            grid_input = torch.tensor(grid_input, dtype=torch.float32).to(self.device)
            grid_input = grid_input.permute(0, 3, 1, 2)
            flat_input = torch.tensor(flat_input, dtype=torch.float32).to(self.device)
            policy, value = self.model(grid_input, flat_input)
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

        policy_pred, value_pred = self.model(grid_inputs, flat_inputs)

        policy_loss = -torch.mean(
            torch.sum(policy_targets * torch.log(policy_pred + 1e-8), dim=[1, 2, 3])
        )

        value_loss = F.binary_cross_entropy(value_pred, value_targets)

        total_loss = policy_loss + value_loss

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

        self.optimizer.step()
        self.scheduler.step(total_loss.item())

        return total_loss.item(), policy_loss.item(), value_loss.item()

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
