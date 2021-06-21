import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation


class DynamicalModel(nn.Module):
    def __init__(self):
        super(DynamicalModel, self).__init__()
        self.theta = torch.tensor([[2.0574e-06],
                                   [3.6350e-06],
                                   [1.9637e-06],
                                   [1.5637e-03]])
        self.permutation = [1, 0, 3, 2]
        self.DT = 1 / 360

    def forward(self, state, command):
        pos = state[:, :3]
        quat = state[:, 3:7]
        vel = state[:, 7:10]
        rpy = state[:, 10:]

        P = torch.FloatTensor([[+1, +1, +1, +1],
                               [-1, +1, +1, -1],
                               [-1, -1, +1, +1],
                               [-1, +1, -1, +1]]).view(1, 4, 4)

        rpm = command[:, self.permutation]
        Omega = rpm ** 2
        motor_forces = P @ Omega.unsqueeze(-1)  # [fa, tx, ty, tz]
        B = state.shape[0]
        # Linear PFD
        gravity = torch.FloatTensor([0, 0, 9.81]).view(1, 3)
        rotation_matrix = torch.FloatTensor(Rotation.from_quat(quat).as_matrix())
        thrust = torch.cat([torch.zeros(B, 1), torch.zeros(B, 1), motor_forces[:, 0]], dim=1)
        thrust_world_frame = rotation_matrix @ thrust.unsqueeze(-1)

        next_vel = vel + self.DT * (-gravity + self.theta[0] * thrust_world_frame.squeeze(-1))
        next_pos = pos + self.DT * next_vel

        # Angular PFD
        wyz = (rpy[:, 1] * rpy[:, 2]).unsqueeze(-1)
        wxz = (rpy[:, 0] * rpy[:, 2]).unsqueeze(-1)

        next_wx = rpy[:, 0] + self.DT * (self.theta[1] * motor_forces[:, 1] - wyz * (self.theta[3] - 1)).squeeze()
        next_wy = rpy[:, 0] + self.DT * (self.theta[1] * motor_forces[:, 2] - wxz * (1 - self.theta[3])).squeeze()
        next_wz = rpy[:, 0] + self.DT * (self.theta[2] * motor_forces[:, 3]).squeeze()

        next_rpy = torch.stack([next_wx, next_wy, next_wz], dim=-1)
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        Q_1 = torch.stack([qw, -qz, qy], dim=-1)
        Q_2 = torch.stack([qz, qw, -qx], dim=-1)
        Q_3 = torch.stack([-qy, qx, qw], dim=-1)
        Q_4 = torch.stack([-qx, -qy, qz], dim=-1)
        Q = torch.stack([Q_1, Q_2, Q_3, Q_4], dim=1)
        next_quat = quat + (self.DT * 0.5 * Q @ next_rpy.unsqueeze(-1)).squeeze(-1)
        next_quat = next_quat / torch.sqrt(((next_quat) ** 2).sum(-1, keepdim=True))

        return torch.cat([next_pos, next_quat.squeeze(-1), next_vel, next_rpy], dim=-1)


if __name__ == '__main__':
    from metrics import MSE_error

    f = DynamicalModel()
    print(MSE_error(f, 1, "test"))
