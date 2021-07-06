import pickle

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

import src.data.euroc_mav.loading as euroc
import src.utils.quaternion as quaternion


class DynamicalModel(nn.Module):
    def __init__(self):
        super(DynamicalModel, self).__init__()
        # Point: 6
        self.permutation = [4, 5, 0, 1, 2, 3]
        self.theta = torch.tensor([[1.1559688e-04],
                                   [1.4442127e-03],
                                   [-6.8990281e-05],
                                   [1.8237591e+01]])

        # self.identifications[0]
        # ((3, 4, 5, 2, 1, 0), tensor([[ 1.1560e-04],
        #         [ 2.3920e-03],
        #         [ 6.8990e-05],
        #         [-1.0757e+01]]), tensor(7.0828e-06))
        # self.identifications = pickle.load(open(euroc.ANALYSIS_SECTION_PATH + "identifications.pkl", "rb"))
        # self.theta = self.identifications[identification_nb][1]
        # self.permutation = self.identifications[identification_nb][0]

        # # Paper parameters from
        # # Achtelik, M. et al. (2012) ‘Design of a Multi Rotor MAV with regard to Efficiency, Dynamics and Redundancy’
        # m = 0.64
        # # From Akkinapalli et al. ‘Attitude Control of a Multicopter Using L1 Augmented Quaternion Based Backstepping’
        # ix = 10.007e-3  # [kg.m2]
        # iy = 10.2335e-3  # [kg.m2]  # Hypothesis ix == iy nearly confirmed
        # iz = 8.1e-3  # [kg.m2]
        # kt = 6.546e-6  # [N.s2.rad-1]
        # km = 1.2864e-7  # [N.m2.rad-1]
        # l = 0.215  # [m]
        # theta = torch.tensor([[kt / m], [kt * l / ix], [km / iz], [iz / ix]])
        # self.theta = torch.tensor([[1.0228e-05],
        #                            [1.4064e-04],
        #                            [1.5881e-05],
        #                            [8.0943e-01]])

        # self.theta = torch.tensor([[2.08e-6],
        #                            [2 ** 0.5 * 3.58e-6],
        #                            [3.77e-8],
        #                            [1.40]])  # Parameter from euroc mav paper
        self.dt = 1 / 200

    def forward(self, state, command, dt=None):
        assert command.shape[-1] == 6, "Motor len is not 6"
        pos = state[:, :3]
        quat = state[:, 3:7]
        vel = state[:, 7:10]
        rpy = state[:, 10:]

        motor_torsor = self.motor_dynamic(command)

        batch_size = state.shape[0]
        # Linear PFD
        gravity = torch.FloatTensor([0, 0, 9.81]).view(1, 3)
        thrust = torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), motor_torsor[:, 3]], dim=1)
        thrust_world_frame = quaternion.qrot(quat, thrust)

        next_vel = vel + self.dt * (-gravity + self.theta[0] * thrust_world_frame)
        next_pos = pos + self.dt * next_vel

        # Angular PFD
        wyz = (rpy[:, 1] * rpy[:, 2]).unsqueeze(-1)
        wxz = (rpy[:, 0] * rpy[:, 2]).unsqueeze(-1)

        next_wx = rpy[:, 0] + self.dt * (wyz * (1 - self.theta[3]) + self.theta[1] * motor_torsor[:, 0]).squeeze()
        next_wy = rpy[:, 1] + self.dt * (wxz * (self.theta[3] - 1) + self.theta[1] * motor_torsor[:, 1]).squeeze()
        next_wz = rpy[:, 2] + self.dt * (self.theta[2] * motor_torsor[:, 2]).squeeze()

        next_rpy = torch.stack([next_wx, next_wy, next_wz], dim=-1)
        next_quat = quat + self.dt * 0.5 * quaternion.qmul(quat, torch.cat([next_rpy, torch.zeros(len(next_rpy), 1)],
                                                                           dim=1))
        next_quat = next_quat / torch.sqrt((next_quat ** 2).sum(-1, keepdim=True))

        return torch.cat([next_pos, next_quat.squeeze(-1), next_vel, next_rpy], dim=-1)

    def motor_dynamic(self, command):
        a = 0.5
        b = 3 ** 0.5 / 2
        motor_matrix = torch.FloatTensor([[-a, +a, +1, +a, -a, -1],
                                          [-b, -b, +0, +b, +b, +0],
                                          [-1, +1, -1, +1, -1, +1],
                                          [+1, +1, +1, +1, +1, +1]]).view(1, 4, 6)

        motor = command[:, self.permutation]
        omega_squared = motor ** 2
        motor_torsor = motor_matrix @ omega_squared.unsqueeze(-1)  # [tx, ty, tz, T]
        return motor_torsor.squeeze()

    def forces(self, motor, state, next_state, dt):
        quat = next_state[:, 3:7]
        ang_vel = next_state[:, 10:]
        lin_acc = (next_state[:, 7:10] - state[:, 7:10]) / dt
        ang_acc = (next_state[:, 10:] - state[:, 10:]) / dt
        motor_torsor = self.motor_dynamic(motor)

        quat_bar = quat * torch.tensor([-1, -1, -1, 1]).view(1, 4)
        batch_size = len(quat)
        g = torch.tensor([0, 0, 9.81]).view(1, 3)
        fu = torch.cat([torch.zeros((batch_size, 2)), self.theta[0] * motor_torsor[:, 3].view(-1, 1)], dim=1)
        kfa = quaternion.qrot(quat_bar, lin_acc) - fu + g

        ktaua = (ang_acc - torch.cross(torch.tensor([1, 1, self.theta[3]]) * ang_vel, ang_vel)
                 - torch.tensor([self.theta[1], self.theta[1], self.theta[2]]).view(1, 3) - motor_torsor[:, :3])
        return torch.cat([kfa, ktaua], dim=1)


if __name__ == '__main__':
    from src.metrics import mse_error, visualise_metrics

    res = np.zeros(3)
    f = DynamicalModel()
    for i, e_i in enumerate([5, 100, 200]):
        res[i] = (mse_error(f, e_i, 'euroc_mav', 'test').item())
    np.set_printoptions(precision=4)
    print(res)

    # f = DynamicalModel(k)
    # for i in [5, 100, 200]:
    #     print(f"Metric {i} on test: {mse_error(f, i, 'euroc_mav', 'validation'):.4E}")
    # visualise_metrics(f, 1000, mode="test", plot_range=[15000, 16000])
