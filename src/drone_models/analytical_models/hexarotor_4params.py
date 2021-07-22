import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import src.data.euroc_mav.loading as euroc
import src.utils.quaternion as quaternion


class DynamicalModel(nn.Module):
    def __init__(self, paper):
        """
        :param paper: True, load paper parameters. False, load mse parameters
        """
        super(DynamicalModel, self).__init__()
        # Point: 6
        self.permutation = [4, 5, 0, 1, 2, 3]
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
                                       [8.0943e-01]], dtype=float)
        else:
            self.theta = torch.tensor([[1.1559688e-04],
                                       [1.4442127e-03],
                                       [-6.8990281e-05],
                                       [1.8237591e+01]], dtype=float)  # MSE

    def forward(self, state, command, dt, residual=None):
        assert command.shape[-1] == 6, "Motor len is not 6"
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

        motor_torsor = self.motor_dynamic(command)

        batch_size = state.shape[0]
        # Linear PFD
        g = torch.DoubleTensor([0, 0, 9.81]).view(1, 3)
        thrust = torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), motor_torsor[:, 3:]], dim=1)
        next_lin_vel = lin_vel + dt * (quaternion.qrot(quat, self.theta[0] * thrust + kfa) - g)

        # Position
        next_pos = pos + dt * next_lin_vel

        # Angular PFD
        t3_tensor = torch.DoubleTensor([1, 1, self.theta[3]]).view(1, 3)
        t12_tensor = torch.DoubleTensor([self.theta[1], self.theta[1], self.theta[2]]).view(1, 3)
        next_ang_vel = ang_vel + dt * (torch.cross(t3_tensor * ang_vel, ang_vel)
                                       + t12_tensor * motor_torsor[:, :3]
                                       + ktaua)

        # Quat
        next_quat = quat + dt * 0.5 * quaternion.qmul(quat, torch.cat([next_ang_vel, torch.zeros((len(next_ang_vel), 1),
                                                                                                 dtype=float)],
                                                                      dim=1))
        next_quat = next_quat / torch.sqrt((next_quat ** 2).sum(-1, keepdim=True))

        return torch.cat([next_pos, next_quat.squeeze(-1), next_lin_vel, next_ang_vel], dim=-1)

    def motor_dynamic(self, command):
        a = 0.5
        b = 3 ** 0.5 / 2
        motor_matrix = torch.DoubleTensor([[-a, +a, +1, +a, -a, -1],
                                          [-b, -b, +0, +b, +b, +0],
                                          [-1, +1, -1, +1, -1, +1],
                                          [+1, +1, +1, +1, +1, +1]]).view(1, 4, 6)

        motor = command[:, self.permutation]
        omega_squared = motor ** 2
        motor_torsor = motor_matrix @ omega_squared.unsqueeze(-1)  # [tx, ty, tz, T]
        return motor_torsor.squeeze(-1)

    def residual_dynamic(self, motor, state, next_state, dt):
        quat = state[:, 3:7]
        ang_vel = state[:, 10:]
        lin_acc = (next_state[:, 7:10] - state[:, 7:10]) / dt
        ang_acc = (next_state[:, 10:] - state[:, 10:]) / dt
        motor_torsor = self.motor_dynamic(motor)

        quat_bar = quat * torch.DoubleTensor([-1, -1, -1, 1]).view(1, 4)
        batch_size = len(quat)
        g = torch.DoubleTensor([0, 0, 9.81]).view(1, 3)

        thrust = torch.cat([torch.zeros((batch_size, 2)), self.theta[0] * motor_torsor[:, 3].view(-1, 1)], dim=1)
        kfa = quaternion.qrot(quat_bar, lin_acc + g) - thrust

        t3_tensor = torch.DoubleTensor([1, 1, self.theta[3]]).view(1, 3)
        t12_tensor = torch.DoubleTensor([self.theta[1], self.theta[1], self.theta[2]]).view(1, 3)
        ktaua = (ang_acc - torch.cross(t3_tensor * ang_vel, ang_vel)
                 - t12_tensor * motor_torsor[:, :3])

        # discontinuities [2850, 2875]
        # start, stop = 0, len(quat)
        # plt.plot(kfa[start:  stop])  # Problem
        # plt.show()
        # plt.plot(dt[start:stop])  #
        # plt.plot(quat_bar[start:stop])  #
        # plt.plot(state[start:stop, 7:10])  #
        # plt.plot(ang_vel[start:stop])  #
        # plt.plot(lin_acc[start:stop, 0])  #
        # plt.plot(ang_acc[start:stop])
        # plt.plot(motor_torsor[start:stop])  #
        # plt.plot((next_state[:, 7:10] - state[:, 7:10]) / dt)

        return torch.cat([kfa, ktaua], dim=1)


def main():
    model = DynamicalModel(paper=False)
    # Sanity check
    def sanity_check(data):
        analysis1 = torch.zeros(len(data), 13)
        analysis2 = torch.zeros(len(data), 13)
        analysis3 = torch.zeros(len(data), 6)
        for i in range(len(data)):
            state = data[i]["state"][-2].view(1, -1)
            motor = data[i]["motor"][-2].view(1, -1)
            residual_gt = data[i]["residual"][-1].view(1, -1)
            next_state_gt = data[i]["state"][-1].view(1, -1)
            dt = data[i]["t"][-1].view(1, -1) - data[i]["t"][-2].view(1, -1)

            residual_hat = model.residual_dynamic(motor, state, next_state_gt, dt)
            next_state_hat = model(state, motor, dt, residual_gt)

            analysis1[i, :] = (next_state_gt - next_state_hat) ** 2
            # analysis2[i, :] = next_state_hat
            analysis3[i, :] = (residual_hat - residual_gt) ** 2
        print(torch.mean(analysis1, dim=0))
        # assert (torch.max(test, 0)[0] < 2.5e-7).all()

    data = euroc.EurocMAVDatasetHorizon("train", traj_len=21)
    sanity_check(data)
    data = euroc.EurocMAVDatasetHorizon("validation", traj_len=21)
    sanity_check(data)
    data = euroc.EurocMAVDatasetHorizon("test", traj_len=21)
    sanity_check(data)


if __name__ == '__main__':
    main()
