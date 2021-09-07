import torch
import torch.nn as nn

import src.utils.quaternion as quaternion


# Connected to pydev debugger (build 211.7628.24)
# 100%|██████████| 35/35 [00:01<00:00, 24.62it/s]
# Metric 9 on test: 3.6545E-04
# 100%|██████████| 35/35 [00:04<00:00,  7.44it/s]
# Metric 180 on test: 7.5337E-02
# 100%|██████████| 35/35 [00:09<00:00,  3.74it/s]
# Metric 360 on test: 3.9234E-01


class DynamicalModel(nn.Module):
    def __init__(self, paper: bool, dtype):
        """
        :param paper: True, load paper parameters. False, load mse parameters
        """
        super(DynamicalModel, self).__init__()
        assert dtype in [torch.float32, torch.float64]
        self.dtype = dtype
        if paper:
            self.theta = torch.tensor([[2.08e-6],
                                       [2 ** 0.5 * 3.58e-6],
                                       [3.77e-8],
                                       [1.40]], dtype=self.dtype)  # Parameter from blackbird paper
        else:
            self.theta = torch.tensor([[2.1019e-06],  # kt / m
                                       [2 ** 0.5 * 4.2478e-06],  # kt l / Ix
                                       [2.1783e-06],  # km / Iz
                                       [2.3617e-01]], dtype=self.dtype)  # Iz / Ix MSE results

        self.permutation = [1, 0, 3, 2]
        self.dt = 1 / 360

    def forward(self, motor, state, dt, residual=None):
        assert motor.shape[-1] == 4, "Motor len is not 4"
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
        a = 1 / torch.sqrt(2)
        motor_matrix = torch.tensor([[-a, +a, +a, -a],
                                     [-a, -a, +a, +a],
                                     [-1, +1, -1, +1],
                                     [+1, +1, +1, +1]], dtype=self.dtype).view(1, 4, 4)

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


if __name__ == '__main__':
    pass
