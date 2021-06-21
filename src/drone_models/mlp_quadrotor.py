import torch
import torch.nn as nn
from torch.nn.functional import relu

from src.utils.quaternion import qrot, qmul


class MLPQuadrotor(nn.Module):
    def __init__(self):
        super(MLPQuadrotor, self).__init__()
        self.linear1 = nn.Linear(in_features=17, out_features=256, bias=True).double()
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True).double()
        self.linear3 = nn.Linear(in_features=256, out_features=256, bias=True).double()
        self.linear4 = nn.Linear(in_features=256, out_features=6, bias=True).double()

    def forward(self, state, motor, dt):
        position = state[:, :3]
        quat = state[:, 3:7]
        v = state[:, 7:10]
        wb = state[:, 10:]

        state_motor = torch.cat([state, motor, dt], dim=1)
        out = relu(self.linear1(state_motor))
        out = relu(self.linear2(out))
        out = relu(self.linear3(out))
        out = self.linear4(out)

        dot_v = out[:, :3]  # Global linear speed
        dot_wb = out[:, 3:]  # Body's angular velocity
        dot_pos = dot_v * dt + v
        dot_quat = 1 / 2 * qmul(quat, torch.cat([dot_wb * dt + wb, torch.zeros(len(state), 1)], dim=1))
        return torch.cat([dot_pos, dot_quat, dot_v, dot_wb], dim=1)


def _test():
    n = 1000
    tmp = torch.randn(n, 4, dtype=float)
    trans = torch.randn(n, 3, dtype=float)
    quat = (tmp / torch.linalg.norm(tmp, axis=1).unsqueeze(1)).clone().detach()
    linear_speed = torch.randn(n, 3, dtype=float)
    angular_velocity = torch.randn(n, 3, dtype=float)

    state = torch.cat([trans, quat, linear_speed, angular_velocity], dim=1)

    motor = torch.randn(n, 4, dtype=float, requires_grad=True)

    model = MLPQuadrotor()
    output = model(state, motor)


if __name__ == '__main__':
    _test()
