import torch
import torch.nn as nn
from torch.nn.functional import relu


class MLPQuadrotor(nn.Module):
    def __init__(self):
        super(MLPQuadrotor, self).__init__()
        self.linear1 = nn.Linear(in_features=16, out_features=256, bias=True).double()
        self.linear2 = nn.Linear(in_features=256, out_features=256, bias=True).double()
        self.linear3 = nn.Linear(in_features=256, out_features=256, bias=True).double()
        self.linear4 = nn.Linear(in_features=256, out_features=6, bias=True).double()

    def forward(self, state, motor):
        position = state[:, :3]
        quat = state[:, 3:7]
        linear_speed = state[:, 7:10]
        angular_velocity = state[:, 10:]

        state_motor = torch.cat([state, motor], dim=1)
        out = relu(self.linear1(state_motor))
        out = relu(self.linear2(out))
        out = relu(self.linear3(out))
        out = self.linear4(out)

        dot_pos = quat dot_vb quat_bar
        dot_quat = 1/2 q rond angular_velocity
        dot_v = out[:, :3]  # Global linear speed
        dot_wb = out[:, 3:]  # Body's angular velocity

        return torch.cat([dot_pos, dot_quat, dot_v, dot_wb])
