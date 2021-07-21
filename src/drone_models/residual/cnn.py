import torch
import torch.nn as nn
from torch.nn.functional import relu


class MLPResidual(nn.Module):
    def __init__(self, motor_size: int, n_past: int, unit_size: int):
        super(MLPResidual, self).__init__()
        self.motor_size = motor_size

        self.time_size = n_past
        self.feature_size = 20

        # In shape (batch, channel=1, self.time_size, self.feature_size)
        self.conv2d0 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(1, self.feature_size),
                                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # In/Out shape (batch, channel=50, self.time_size, 1)
        self.conv2d1 = nn.Conv2d(in_channels=50, out_channels=6, kernel_size=(self.time_size, 1),
                                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros")
        # Out shape (batch, channel=6, 1, 1)

    def forward(self, state, motor, dt):
        """
        Estimate the residual = GT - model
        :param state: (batch, step, values=13)
        :param motor: (batch, step, values=6)
        :param dt: (batch, step, values=1)
        :return: (batch, value=6)
        """
        batch_size = state.shape[0]
        state_motor = torch.cat([state, motor, dt], dim=2).view(batch_size, 1, self.time_size, self.feature_size)
        out = relu(self.conv2d0(state_motor))
        out = self.conv2d1(out)
        print()

        return out  # residual
