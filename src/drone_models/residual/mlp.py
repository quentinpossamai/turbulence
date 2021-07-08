import torch
import torch.nn as nn
from torch.nn.functional import relu

from src.utils.quaternion import qrot, qmul


class MLPResidual(nn.Module):
    def __init__(self, motor_size: int, n_past):
        super(MLPResidual, self).__init__()
        self.motor_size = motor_size
        self.n_past = n_past
        self.linear1 = nn.Linear(in_features=n_past * (1 + 13 + motor_size), out_features=256, bias=True)
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.linear3 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.linear4 = nn.Linear(in_features=256, out_features=6, bias=True)

    def forward(self, state, motor, dt):
        """
        Estimate the residual = GT - model
        :param state: (batch * step * values=13)
        :param motor: (batch * step * values=6)
        :param dt: (batch * step * value=1)
        :return: (batch, value=6)
        """
        batch_size = state.shape[0]
        state_motor = torch.cat([state.view((batch_size, -1)), motor.view((batch_size, -1)),
                                 dt.view((batch_size, -1))], dim=1)
        out = relu(self.linear1(state_motor))
        out = relu(self.linear2(out))
        out = relu(self.linear3(out))
        return self.linear4(out)  # residual
