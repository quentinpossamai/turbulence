import torch
import torch.nn as nn
from torch.nn.functional import relu


class CNNResidual(nn.Module):
    def __init__(self, motor_size: int, past: int, unit_size: int):
        super(CNNResidual, self).__init__()
        self.motor_size = motor_size
        self.unit_size = unit_size

        self.time_size = past
        self.feature_size = 19

        # In shape (batch, channel=1, self.time_size, self.feature_size)
        self.conv2d0 = nn.Conv2d(in_channels=1, out_channels=unit_size, kernel_size=(1, self.feature_size),
                                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # In/Out shape (batch, channel=50, self.time_size, 1)
        self.conv2d1 = nn.Conv2d(in_channels=unit_size, out_channels=6, kernel_size=(self.time_size, 1),
                                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # Out shape (batch, channel=6, 1, 1)

    def forward(self, motor, state, dt):
        """
        Estimate the residual = GT - model
        :param motor: (batch, step, values=6)
        :param state: (batch, step, values=13)
        :param dt: (batch, step, values=1)
        :return: (batch, value=6)
        """
        batch_size = state.shape[0]
        state_motor = torch.cat([state, motor], dim=2).view(batch_size, 1, self.time_size, self.feature_size)
        out = relu(self.conv2d0(state_motor))
        out = self.conv2d1(out).squeeze()
        return out  # residual
