import torch
import torch.nn as nn
from torch.nn.functional import relu

from src.utils.quaternion import qrot, qmul


class MLP(nn.Module):
    def __init__(self, motor_size: int, n_past):
        super(MLP, self).__init__()
        self.motor_size = motor_size
        self.n_past = n_past

        self.batch_norm0 = nn.BatchNorm2d(1)
        self.linear1 = nn.Linear(in_features=self.n_past * (1 + 13 + self.motor_size), out_features=256, bias=True)
        self.batch_norm1 = nn.BatchNorm2d(1)
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.batch_norm2 = nn.BatchNorm2d(1)
        self.linear3 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.batch_norm3 = nn.BatchNorm2d(1)
        self.linear4 = nn.Linear(in_features=256, out_features=6, bias=True)

    def forward(self, motor, state, dt):
        """
        Estimate the residual = GT - model
        :param state: (batch * step * values=13)
        :param motor: (batch * step * values=6)
        :param dt: (batch * step * value=1)
        :return: (batch, value=6)
        """
        position = state[:, -1, :3]
        quat = state[:, -1, 3:7]
        v = state[:, -1, 7:10]
        wb = state[:, -1, 10:]
        last_state = state[:, -1]
        last_dt = dt[:, -1]

        batch_size = state.shape[0]
        past = state.shape[1]
        state_motor = torch.cat([state.view((batch_size, -1)), motor.view((batch_size, -1)), dt.expand((batch_size, past))], dim=1)
        input_size = state_motor.shape[1]

        out = state_motor
        out = self.batch_norm0(out.view((batch_size, 1, input_size, 1))).view(batch_size, input_size)
        out = relu(self.linear1(out))
        out = self.batch_norm1(out.view((batch_size, 1, 256, 1))).view((batch_size, 256))
        out = relu(self.linear2(out))
        out = self.batch_norm2(out.view((batch_size, 1, 256, 1))).view((batch_size, 256))
        out = relu(self.linear3(out))
        out = self.batch_norm3(out.view((batch_size, 1, 256, 1))).view((batch_size, 256))
        out = self.linear4(out)

        dot_v = out[:, :3]  # Global linear speed
        dot_wb = out[:, 3:]  # Body's angular velocity
        dot_pos = dot_v * last_dt + v
        dot_quat = 1 / 2 * qmul(quat, torch.cat([dot_wb * last_dt + wb, torch.zeros(len(state), 1)], dim=1))
        dstate = torch.cat([dot_pos, dot_quat, dot_v, dot_wb], dim=1)
        return dstate * last_dt + last_state


def _test():
    n = 1000
    dt = torch.randn(n, 1, dtype=float)
    tmp = torch.randn(n, 4, dtype=float)
    trans = torch.randn(n, 3, dtype=float)
    quat = (tmp / torch.linalg.norm(tmp, axis=1).unsqueeze(1)).clone().detach()
    linear_speed = torch.randn(n, 3, dtype=float)
    angular_velocity = torch.randn(n, 3, dtype=float)

    state = torch.cat([trans, quat, linear_speed, angular_velocity], dim=1)

    motor = torch.randn(n, 4, dtype=float, requires_grad=True)

    model = MLP()
    output = model(state, motor, dt)
    print("Test done")


if __name__ == '__main__':
    _test()
