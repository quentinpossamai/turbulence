import torch
import torch.nn as nn
from torch.nn.functional import relu

from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel


class GroundTruth(nn.Module):
    def __init__(self):
        super(GroundTruth, self).__init__()
        self.model = DynamicalModel(paper=False)
        self.prms = nn.Parameter(torch.zeros(1))

    def forward(self, state, motor, dt, next_state):
        """
        Estimate the residual = GT - model with the ground truth information to check code.
        :param state: (batch, step, values=13)
        :param motor: (batch, step, values=6)
        :param dt: (batch, step, values=1)
        :param next_state: (batch, step, values=13)
        :return: (batch, value=6)
        """
        state = state[:, -1]
        motor = motor[:, -1]
        dt = dt[:, -1]
        next_state = next_state.squeeze(1)
        out = self.model.residual_dynamic(motor, state, next_state, dt)
        return out  # residual
