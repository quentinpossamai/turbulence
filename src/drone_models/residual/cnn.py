import torch
import torch.nn as nn
from torch.nn.functional import relu


class MLPResidual(nn.Module):
    def __init__(self, motor_size: int, n_past: int, unit_size: int):
        super(MLPResidual, self).__init__()
        self.motor_size = motor_size

        self.time_size = n_past
        self.feature_size = 20

        self.conv2d = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(1, self.feature_size),
                                )


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


        return out  # residual
