import torch
import torch.nn as nn
from torch.nn.functional import relu

from src.utils.quaternion import qrot, qmul


class MLPResidual(nn.Module):
    def __init__(self, motor_size: int):
        super(MLPResidual, self).__init__()
        self.motor_size = motor_size
        self.linear1 = nn.Linear(in_features=14 + motor_size, out_features=256, bias=True)
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.linear3 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.linear4 = nn.Linear(in_features=256, out_features=6, bias=True)

    def forward(self, state, motor, dt):
        state_motor = torch.cat([state, motor, dt], dim=1)
        out = relu(self.linear1(state_motor))
        out = relu(self.linear2(out))
        out = relu(self.linear3(out))
        return self.linear4(out)  # fa
