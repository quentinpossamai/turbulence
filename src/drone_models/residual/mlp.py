import torch
import torch.nn as nn
from torch.nn.functional import relu


class MLPResidual(nn.Module):
    def __init__(self, motor_size: int, n_past: int, unit_size: int):
        super(MLPResidual, self).__init__()
        self.motor_size = motor_size
        self.n_past = n_past
        self.unit_size = unit_size

        self.batch_norm0 = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(in_features=self.n_past * (1 + 13 + self.motor_size), out_features=self.unit_size,
                                 bias=True)
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.linear2 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm2 = nn.BatchNorm2d(1)
        self.linear3 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm3 = nn.BatchNorm2d(1)
        self.linear4 = nn.Linear(in_features=self.unit_size, out_features=6, bias=True)

    def forward(self, state, motor, dt):
        """
        Estimate the residual = GT - model
        :param state: (batch, step, values=13)
        :param motor: (batch, step, values=6)
        :param dt: (batch, step, values=1)
        :return: (batch, value=6)
        """
        batch_size = state.shape[0]
        state_motor = torch.cat([state.view((batch_size, -1)), motor.view((batch_size, -1)),
                                 dt.view((batch_size, -1))], dim=1)
        input_size = state_motor.shape[1]

        out = state_motor
        out = self.batch_norm0(out.view((batch_size, 1, input_size, 1))).view(batch_size, input_size)
        out = relu(self.linear1(out))
        out = self.batch_norm1(out.view((batch_size, 1, self.unit_size, 1))).view((batch_size, self.unit_size))
        out = relu(self.linear2(out))
        out = self.batch_norm2(out.view((batch_size, 1, self.unit_size, 1))).view((batch_size, self.unit_size))
        out = relu(self.linear3(out))
        out = self.batch_norm3(out.view((batch_size, 1, self.unit_size, 1))).view((batch_size, self.unit_size))
        out = self.linear4(out)
        return out  # residual
