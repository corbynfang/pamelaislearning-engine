import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
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

class ChessModel(nn.Module):
    def __init__(self, input_channels=19, num_classes=1):
        super(ChessModel, self).__init__()

        # Initial convolution to increase channels
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        # Residual blocks for deep pattern recognition
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(10)])

        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)

        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 128)
        self.value_fc3 = nn.Linear(128, num_classes)

        # Initialize weights using improved methods
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial convolution
        x = self.initial_conv(x)

        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 32 * 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = F.relu(self.value_fc2(value))
        value = self.value_fc3(value)

        # Scale output using tanh to keep within bounds
        # Using tanh ensures output is between -1 and 1
        value = torch.tanh(value)

        return value
