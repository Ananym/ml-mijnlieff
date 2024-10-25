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

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ValueNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Increase initial channels for better feature extraction
        self.conv1 = nn.Conv2d(10, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Add residual blocks for deeper feature extraction
        self.res_blocks = nn.ModuleList([ResidualBlock(64), ResidualBlock(64)])

        # Process flat input separately
        self.flat_fc = nn.Linear(10, 32)
        self.flat_bn = nn.BatchNorm1d(32)

        # Combine processed grid and flat inputs
        self.fc1 = nn.Linear(64 * 4 * 4 + 32, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)

        # Value head with multiple outputs for auxiliary tasks
        self.value_head = nn.Linear(128, 1)  # Main value output
        self.line_count_head = nn.Linear(128, 2)  # Predict line count for each player
        self.piece_count_head = nn.Linear(128, 2)  # Predict remaining pieces

    def forward(self, grid_input, flat_input):
        # Process grid input
        x = F.relu(self.bn1(self.conv1(grid_input)))

        # Apply residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)

        x = x.reshape(-1, 64 * 4 * 4)

        # Process flat input separately
        flat = F.relu(self.flat_bn(self.flat_fc(flat_input)))

        # Combine processed inputs
        combined = torch.cat((x, flat), dim=1)

        # Fully connected layers with dropout
        x = F.relu(self.bn3(self.fc1(combined)))
        x = self.dropout1(x)

        x = F.relu(self.bn4(self.fc2(x)))
        x = self.dropout2(x)

        # Multiple output heads
        value = torch.sigmoid(self.value_head(x)).squeeze(-1)
        lines = self.line_count_head(x)
        pieces = self.piece_count_head(x)

        return value, lines, pieces


class ValueNetworkWrapper:
    def __init__(self, device):
        self.device = device
        self.model = ValueNetwork().to(device)

        # Use AdamW optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

    def predict(self, grid_input, flat_input):
        self.model.eval()
        with torch.no_grad():
            grid_input = torch.tensor(grid_input, dtype=torch.float32).to(self.device)
            grid_input = grid_input.permute(0, 3, 1, 2)
            flat_input = torch.tensor(flat_input, dtype=torch.float32).to(self.device)
            value, _, _ = self.model(grid_input, flat_input)
            return value.cpu().numpy()

    def train(
        self, grid_inputs, flat_inputs, targets, line_counts=None, piece_counts=None
    ):
        self.model.train()

        grid_inputs = torch.tensor(grid_inputs, dtype=torch.float32).to(self.device)
        grid_inputs = grid_inputs.permute(0, 3, 1, 2)
        flat_inputs = torch.tensor(flat_inputs, dtype=torch.float32).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

        # Default auxiliary targets if not provided
        if line_counts is None:
            line_counts = torch.zeros((targets.shape[0], 2), dtype=torch.float32).to(
                self.device
            )
        if piece_counts is None:
            piece_counts = torch.zeros((targets.shape[0], 2), dtype=torch.float32).to(
                self.device
            )

        self.optimizer.zero_grad()

        value_pred, lines_pred, pieces_pred = self.model(grid_inputs, flat_inputs)

        # Focal loss component without requiring gradients
        with torch.no_grad():
            focal_weight = (1 - value_pred).pow(2) * targets + value_pred.pow(2) * (
                1 - targets
            )

        # Main value loss
        value_loss = F.binary_cross_entropy(value_pred, targets, weight=focal_weight)

        # Auxiliary losses (if targets provided)
        lines_loss = (
            F.mse_loss(lines_pred, line_counts) if line_counts is not None else 0
        )
        pieces_loss = (
            F.mse_loss(pieces_pred, piece_counts) if piece_counts is not None else 0
        )

        # Combined loss
        total_loss = value_loss
        if line_counts is not None:
            total_loss += 0.2 * lines_loss
        if piece_counts is not None:
            total_loss += 0.2 * pieces_loss

        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step(total_loss)

        return total_loss.item()

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


#
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ValueNetwork(nn.Module):

#     def __init__(self):
#         super(ValueNetwork, self).__init__()
#         # Input: 4x4x10 grid + 10 flat values
#         self.conv1 = nn.Conv2d(10, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.fc1 = nn.Linear(64 * 4 * 4 + 10, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 64)
#         self.bn4 = nn.BatchNorm1d(64)
#         self.fc3 = nn.Linear(64, 1)

#     def forward(self, grid_input, flat_input):
#         x = F.relu(self.bn1(self.conv1(grid_input)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = x.reshape(-1, 64 * 4 * 4)  # Use reshape instead of view
#         x = torch.cat((x, flat_input), dim=1)
#         x = F.relu(self.bn3(self.fc1(x)))
#         x = F.relu(self.bn4(self.fc2(x)))
#         x = self.fc3(x)
#         return torch.sigmoid(x).squeeze(-1)


# class ValueNetworkWrapper:

#     def __init__(self, device):
#         self.device = device
#         self.model = ValueNetwork().to(device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

#     # def predict(self, grid_input, flat_input):
#     #     self.model.eval()
#     #     with torch.no_grad():
#     #         grid_input = torch.tensor(grid_input,
#     #                                   dtype=torch.float32).to(self.device)
#     #         flat_input = torch.tensor(flat_input,
#     #                                   dtype=torch.float32).to(self.device)
#     #         return self.model(grid_input, flat_input).cpu().numpy()
#     def predict(self, grid_input, flat_input):
#         self.model.eval()
#         with torch.no_grad():
#             grid_input = torch.tensor(grid_input, dtype=torch.float32).to(self.device)
#             grid_input = grid_input.permute(
#                 0, 3, 1, 2
#             )  # Shape becomes [batch_size, 10, 4, 4]

#             flat_input = torch.tensor(flat_input, dtype=torch.float32).to(self.device)
#             return self.model(grid_input, flat_input).cpu().numpy()

#     def train(self, grid_inputs, flat_inputs, targets):
#         self.model.train()
#         grid_inputs = torch.tensor(grid_inputs, dtype=torch.float32).to(self.device)
#         grid_inputs = grid_inputs.permute(
#             0, 3, 1, 2
#         )  # Shape becomes [batch_size, 10, 4, 4]

#         flat_inputs = torch.tensor(flat_inputs, dtype=torch.float32).to(self.device)
#         targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

#         self.optimizer.zero_grad()
#         outputs = self.model(grid_inputs, flat_inputs)

#         loss = F.binary_cross_entropy(outputs, targets)
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()

#     def save(self, path):
#         torch.save(self.model.state_dict(), path)

#     def load(self, path):
#         self.model.load_state_dict(torch.load(path))
