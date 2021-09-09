import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import src.utils.quaternion as quaternion


class DynamicalModel(nn.Module):
    def __init__(self, paper, dtype):
        """
        :param paper: True, load paper parameters. False, load mse parameters
        :param dtype: Datatype of the model.
        """
        super(DynamicalModel, self).__init__()
        assert dtype in [torch.float32, torch.float64]
        self.dtype = dtype
        # Point: 6
        self.permutation = [5, 0, 1, 2, 3, 4]
        if paper:
            # # Paper parameters from
            # # Achtelik, M. et al. (2012) ‘Design of a Multi Rotor MAV with regard to Efficiency, Dynamics and
            # Redundancy’
            # m = 0.64
            # # From Akkinapalli et al. ‘Attitude Control of a Multicopter Using L1 Augmented Quaternion Based
            # Backstepping’
            # ix = 10.007e-3  # [kg.m2]
            # iy = 10.2335e-3  # [kg.m2]  # Hypothesis ix == iy nearly confirmed
            # iz = 8.1e-3  # [kg.m2]
            # kt = 6.546e-6  # [N.s2.rad-1]
            # km = 1.2864e-7  # [N.m2.rad-1]
            # l = 0.215  # [m]
            # theta = torch.tensor([[kt / m], [kt * l / ix], [km / iz], [iz / ix]])
            self.theta = torch.tensor([[1.0228e-05],
                                       [1.4064e-04],
                                       [1.5881e-05],
                                       [8.0943e-01]], dtype=self.dtype)
        else:
            self.theta = torch.tensor([[1.1566750e-04],
                                       [6.9880742e-04],
                                       [-5.1250136e-05],
                                       [2.6935432e+00]], dtype=self.dtype)  # MSE

    def forward(self, motor, state, dt, residual=None):
        assert motor.shape[-1] == 6, "Motor len is not 6"
        pos = state[:, :3]
        quat = state[:, 3:7]
        lin_vel = state[:, 7:10]
        ang_vel = state[:, 10:]
        if residual is None:
            kfa = torch.zeros(len(state), 3)
            ktaua = torch.zeros(len(state), 3)
        else:
            kfa = residual[:, :3]
            ktaua = residual[:, 3:]

        motor_torsor = self.motor_dynamic(motor)

        batch_size = state.shape[0]
        # Linear PFD
        g = torch.tensor([0, 0, 9.81], dtype=self.dtype).view(1, 3)
        thrust = torch.cat([torch.zeros(batch_size, 2), motor_torsor[:, 3:]], dim=1)
        next_lin_vel = lin_vel + dt * (quaternion.qrot(quat, self.theta[0] * thrust + kfa) - g)

        # Position
        next_pos = pos + dt * next_lin_vel

        # Angular PFD
        t3_tensor = torch.tensor([1, 1, self.theta[3]], dtype=self.dtype).view(1, 3)
        t12_tensor = torch.tensor([self.theta[1], self.theta[1], self.theta[2]], dtype=self.dtype).view(1, 3)
        next_ang_vel = ang_vel + dt * (torch.cross(t3_tensor * ang_vel, ang_vel)
                                       + t12_tensor * motor_torsor[:, :3]
                                       + ktaua)

        # Quat
        next_quat = quat + dt * 0.5 * quaternion.qmul(quat, torch.cat([next_ang_vel, torch.zeros((len(next_ang_vel), 1),
                                                                                                 dtype=torch.float32)],
                                                                      dim=1))
        next_quat = next_quat / torch.sqrt((next_quat ** 2).sum(-1, keepdim=True))

        return torch.cat([next_pos, next_quat.squeeze(-1), next_lin_vel, next_ang_vel], dim=-1)

    def motor_dynamic(self, command):
        a = 0.5
        b = 3 ** 0.5 / 2
        motor_matrix = torch.tensor([[-a, +a, +1, +a, -a, -1],
                                     [-b, -b, +0, +b, +b, +0],
                                     [-1, +1, -1, +1, -1, +1],
                                     [+1, +1, +1, +1, +1, +1]], dtype=self.dtype).view(1, 4, 6)

        motor = command[:, self.permutation]
        omega_squared = motor ** 2
        motor_torsor = motor_matrix @ omega_squared.unsqueeze(-1)  # [tx, ty, tz, T]
        return motor_torsor.squeeze(-1)

    def residual_dynamic(self, motor, state, dt, next_state):
        quat = state[:, 3:7]
        ang_vel = state[:, 10:]
        lin_acc = (next_state[:, 7:10] - state[:, 7:10]) / dt
        ang_acc = (next_state[:, 10:] - state[:, 10:]) / dt
        motor_torsor = self.motor_dynamic(motor)

        quat_bar = quat * torch.tensor([-1, -1, -1, 1], dtype=self.dtype).view(1, 4)
        batch_size = len(quat)
        g = torch.tensor([0, 0, 9.81], dtype=self.dtype).view(1, 3)

        thrust = torch.cat([torch.zeros((batch_size, 2)), self.theta[0] * motor_torsor[:, 3].view(-1, 1)], dim=1)
        kfa = quaternion.qrot(quat_bar, lin_acc + g) - thrust

        t3_tensor = torch.tensor([1, 1, self.theta[3]], dtype=self.dtype).view(1, 3)
        t12_tensor = torch.tensor([self.theta[1], self.theta[1], self.theta[2]], dtype=self.dtype).view(1, 3)
        ktaua = (ang_acc - torch.cross(t3_tensor * ang_vel, ang_vel, dim=-1)
                 - t12_tensor * motor_torsor[:, :3])
        return torch.cat([kfa, ktaua], dim=1)
