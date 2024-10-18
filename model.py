import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueNetwork(nn.Module):

    def __init__(self):
        super(ValueNetwork, self).__init__()
        # Input: 4x4x6 grid + 8 flat values
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4 + 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4 * 4 * 4)  # Output: 4x4x4 grid

    def forward(self, grid_input, flat_input):
        x = F.relu(self.conv1(grid_input))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.cat((x, flat_input), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, 4, 4, 4)
        return F.softmax(x.view(-1, 64), dim=1).view(-1, 4, 4, 4)


class ValueNetworkWrapper:

    def __init__(self, device):
        self.device = device
        self.model = ValueNetwork().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def predict(self, grid_input, flat_input):
        self.model.eval()
        with torch.no_grad():
            grid_input = torch.tensor(grid_input,
                                      dtype=torch.float32).to(self.device)
            flat_input = torch.tensor(flat_input,
                                      dtype=torch.float32).to(self.device)
            return self.model(grid_input, flat_input).cpu().numpy()

    def train(self, grid_inputs, flat_inputs, targets):
        self.model.train()
        grid_inputs = torch.tensor(grid_inputs,
                                   dtype=torch.float32).to(self.device)
        flat_inputs = torch.tensor(flat_inputs,
                                   dtype=torch.float32).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(grid_inputs, flat_inputs)

        # Use binary cross-entropy loss
        loss = F.binary_cross_entropy(outputs, (targets + 1) / 2)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
