import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation


class DynamicalModel(nn.Module):
    def __init__(self):
        super(DynamicalModel, self).__init__()
        self.theta = torch.tensor([[2.1019e-06],  # kt / m
                                   [4.2478e-06],  # kt l / Ix
                                   [2.1783e-06],  # km / Iz
                                   [2.3617e-01]])  # Iz / Ix MSE results

        # self.theta = torch.tensor([[2.08e-6],
        #                            [3.58e-6],
        #                            [3.77e-8],
        #                            [1.40]])  # Parameter from blackbird paper
        self.permutation = [1, 0, 3, 2]
        self.dt = 1 / 360

    def forward(self, state, command, dt=None):
        assert command.shape[-1] == 4, "Motor len is not 4"
        pos = state[:, :3]
        quat = state[:, 3:7]
        vel = state[:, 7:10]
        rpy = state[:, 10:]

        motor_matrix = torch.FloatTensor([[+1, +1, +1, +1],
                                          [-1, +1, +1, -1],
                                          [-1, -1, +1, +1],
                                          [-1, +1, -1, +1]]).view(1, 4, 4)

        rpm = command[:, self.permutation]
        omega_squared = rpm ** 2
        motor_torsor = motor_matrix @ omega_squared.unsqueeze(-1)  # [fa, tx, ty, tz]
        batch_size = state.shape[0]
        # Linear PFD
        gravity = torch.FloatTensor([0, 0, 9.81]).view(1, 3)
        rotation_matrix = torch.FloatTensor(Rotation.from_quat(quat).as_matrix())
        thrust = torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), motor_torsor[:, 0]], dim=1)
        thrust_world_frame = rotation_matrix @ thrust.unsqueeze(-1)

        next_vel = vel + self.dt * (-gravity + self.theta[0] * thrust_world_frame.squeeze(-1))
        next_pos = pos + self.dt * next_vel

        # Angular PFD
        wyz = (rpy[:, 1] * rpy[:, 2]).unsqueeze(-1)
        wxz = (rpy[:, 0] * rpy[:, 2]).unsqueeze(-1)

        next_wx = rpy[:, 0] + self.dt * (self.theta[1] * motor_torsor[:, 1] + wyz * (1 - self.theta[3])).squeeze()
        next_wy = rpy[:, 1] + self.dt * (self.theta[1] * motor_torsor[:, 2] + wxz * (self.theta[3] - 1)).squeeze()
        next_wz = rpy[:, 2] + self.dt * (self.theta[2] * motor_torsor[:, 3]).squeeze()

        next_rpy = torch.stack([next_wx, next_wy, next_wz], dim=-1)
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        q_1 = torch.stack([qw, -qz, qy], dim=-1)
        q_2 = torch.stack([qz, qw, -qx], dim=-1)
        q_3 = torch.stack([-qy, qx, qw], dim=-1)
        q_4 = torch.stack([-qx, -qy, qz], dim=-1)
        q = torch.stack([q_1, q_2, q_3, q_4], dim=1)
        next_quat = quat + (self.dt * 0.5 * q @ next_rpy.unsqueeze(-1)).squeeze(-1)
        next_quat = next_quat / torch.sqrt((next_quat ** 2).sum(-1, keepdim=True))

        return torch.cat([next_pos, next_quat.squeeze(-1), next_vel, next_rpy], dim=-1)


if __name__ == '__main__':
    from src.metrics import mse_error, visualise_metrics

    f = DynamicalModel()
    for i in [9, 180, 360]:
        print(f"Metric {i} on test: {mse_error(f, i, 'blackbird', 'test'):.4E}")
    # visualise_metrics(f, 1000, mode="test", plot_range=[15000, 16000])
