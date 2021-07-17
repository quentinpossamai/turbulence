import torch
import torch.nn as nn
from torch.nn.functional import relu


class GRUResidual(nn.Module):
    def __init__(self, motor_size: int, n_past: int, unit_size: int):
        super(GRUResidual, self).__init__()
        self.motor_size = motor_size
        self.n_past = n_past
        self.unit_size = unit_size
        self.num_layers = 4

        self.batch_norm_gru0 = nn.BatchNorm1d(self.n_past)
        self.gru0 = nn.GRU(input_size=1 + 13 + self.motor_size,
                           hidden_size=self.unit_size,
                           num_layers=self.num_layers,
                           bias=True)

        self.batch_norm0 = nn.BatchNorm1d(self.unit_size)
        self.linear1 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm1 = nn.BatchNorm1d(self.unit_size)
        self.linear2 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm2 = nn.BatchNorm1d(self.unit_size)
        self.linear3 = nn.Linear(in_features=self.unit_size, out_features=self.unit_size, bias=True)
        self.batch_norm3 = nn.BatchNorm1d(self.unit_size)
        self.linear4 = nn.Linear(in_features=self.unit_size, out_features=6, bias=True)

    def forward(self, state, motor, dt):
        """
        Estimate the residual = GT - model
        :param state: (batch, step, values=13)
        :param motor: (batch, step, values=6)
        :param dt: (batch, step, values=1)
        :return: (batch, value=6)
        """
        state_motor = torch.cat([state, motor, dt], dim=2)

        out = state_motor  # (batch, step=self.n_past, values=20)
        out = self.batch_norm_gru0(out)
        # out: (batch, step, values=20)
        out, _ = self.gru0(out.transpose(0, 1))  # gru takes input with shape: (step, batch, values)
        out = out.transpose(0, 1)  # gru output with shape: (step, batch, values)
        batch_size = out.shape[0]

        out = out.reshape(batch_size * self.n_past, self.unit_size)

        out = self.batch_norm0(out)
        out = relu(self.linear1(out))
        out = self.batch_norm1(out)
        out = relu(self.linear2(out))
        out = self.batch_norm2(out)
        out = relu(self.linear3(out))
        out = self.batch_norm3(out)
        # out = self.batch_norm3(out.view((batch_size, 1, self.unit_size))).view((batch_size, self.unit_size))
        out = self.linear4(out)
        return out.view(batch_size, self.n_past, 6)  # residual
