import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from utils.quaternion import qmul, qrot
from torch.autograd.functional import jacobian
import os
from random import randint

plt.style.use("seaborn")
parser = argparse.ArgumentParser()
parser.add_argument('--path_dataset', default='/Volumes/Samsung_T5/BlackBird/blackbird/intermediate', type=str)
args = parser.parse_args()

PATH = args.path_dataset
FLIGHT_LIST = []
for flight in os.listdir(PATH):
    for orientation in os.listdir(PATH + "/" + flight):
        for speed in os.listdir(PATH + "/" + flight + "/" + orientation):
            FLIGHT_LIST.append(flight + "/" + orientation + "/" + speed + "/data.pkl")


def load_flight(flight_name):
    data = pickle.load(open(f"{PATH}/{flight_name}", 'rb'))
    return data


class DynamicalModel(nn.Module):
    def __init__(self):
        """
        :param paper: True, load paper parameters. False, load mse parameters
        """
        super(DynamicalModel, self).__init__()
        self.dtype = torch.float32
        self.theta = torch.tensor([[2.1019e-06],  # kt / m
                                   [2 ** 0.5 * 4.2478e-06],  # kt l / Ix
                                   [2.1783e-06],  # km / Iz
                                   [2.3617e-01]], dtype=self.dtype)  # Iz / Ix MSE results

        self.permutation = [1, 0, 3, 2]
        self.dt = 1 / 360

    def forward(self, state, motor, dt):
        pos = state[:, :3]
        quat = state[:, 3:7]
        lin_vel = state[:, 7:10]
        ang_vel = state[:, 10:13]

        motor_torsor = self.motor_dynamic(motor)

        batch_size = state.shape[0]
        # Linear PFD
        g = torch.tensor([0, 0, 9.81], dtype=self.dtype).view(1, 3)
        thrust = torch.cat([torch.zeros(batch_size, 2), motor_torsor[:, 3:]], dim=1)

        lin_acc = (qrot(quat, self.theta[0] * thrust) - g)
        next_lin_vel = lin_vel + dt * lin_acc

        # Position
        next_pos = pos + dt * next_lin_vel

        # Angular PFD
        t3_tensor = torch.tensor([1, 1, self.theta[3]], dtype=self.dtype).view(1, 3)
        t12_tensor = torch.tensor([self.theta[1], self.theta[1], self.theta[2]], dtype=self.dtype).view(1, 3)
        next_ang_vel = ang_vel + dt * (torch.cross(t3_tensor * ang_vel, ang_vel)
                                       + t12_tensor * motor_torsor[:, :3])

        # Quat
        next_quat = quat + dt * 0.5 * qmul(quat, torch.cat([next_ang_vel, torch.zeros((len(next_ang_vel), 1),
                                                                                      dtype=torch.float32)],
                                                           dim=1))
        next_quat = next_quat / torch.sqrt((next_quat ** 2).sum(-1, keepdim=True))

        return torch.cat([next_pos, next_quat.squeeze(-1), next_lin_vel, next_ang_vel, lin_acc], dim=-1)

    def motor_dynamic(self, command):
        a = 1 / 2 ** 0.5
        motor_matrix = torch.tensor([[-a, +a, +a, -a],
                                     [-a, -a, +a, +a],
                                     [-1, +1, -1, +1],
                                     [+1, +1, +1, +1]], dtype=self.dtype).view(1, 4, 4)

        motor = command[:, self.permutation]
        omega_squared = motor ** 2
        motor_torsor = motor_matrix @ omega_squared.unsqueeze(-1)  # [tx, ty, tz, T]
        return motor_torsor.squeeze(-1)


def h(state):
    quat = state[:, 3:7]
    ang_vel = state[:, 10:13]
    lin_acc = state[:, 13:] - torch.FloatTensor([[0, 0, -9.81]])
    acc = qrot(quat * torch.FloatTensor([[-1, -1, -1, 1]]), lin_acc)

    observation = torch.cat([ang_vel, acc], dim=-1)
    return observation


class KalmanFiltering(nn.Module):
    def __init__(self):
        super(KalmanFiltering, self).__init__()
        self.sigmaQ = nn.Parameter(torch.rand(1), requires_grad=True)
        self.sigmaR = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x0, imu, motor, dt):
        states = [x0.unsqueeze(0)]
        model = DynamicalModel()

        P = torch.diag(torch.FloatTensor([2.1214, 2.0971, 0.2545, 0.1070, 0.1114, 0.4629, 0.6530, 1.5294, 1.5736,
                                          0.1791, 0.5588, 0.5862, 0.7417, 2.3706, 2.4067, 0.4138]))
        Q = torch.FloatTensor([[1.1849e-10, -7.4684e-12, -3.4853e-12, 3.3164e-12, 2.8155e-11,
                                5.6941e-13, -7.4316e-12, 1.1183e-08, -4.2373e-09, -7.5792e-10,
                                5.1765e-09, 3.5563e-08, 1.1771e-09, 4.0230e-06, -1.5246e-06,
                                -2.7257e-07],
                               [-7.4684e-12, 1.1681e-10, 2.1872e-12, -2.3719e-11, 3.4583e-12,
                                5.8579e-13, 1.1961e-11, 3.8468e-09, 9.3854e-09, 2.4828e-10,
                                -2.8962e-08, 5.3225e-09, 3.0067e-10, 1.3841e-06, 3.3769e-06,
                                8.9331e-08],
                               [-3.4853e-12, 2.1872e-12, 5.0077e-12, -1.2812e-12, -1.5489e-12,
                                -7.9360e-13, 2.5667e-12, -2.9300e-10, 1.1322e-10, 8.6031e-10,
                                -9.0008e-10, -2.9641e-09, -2.3623e-10, -1.0538e-07, 4.0731e-08,
                                3.0950e-07],
                               [3.3164e-12, -2.3719e-11, -1.2812e-12, 1.0737e-09, -9.3383e-13,
                                -3.1667e-10, -6.6294e-10, -4.5104e-10, -2.3859e-09, -1.3426e-10,
                                6.1558e-09, 3.0815e-09, -5.4149e-10, -1.6228e-07, -8.5813e-07,
                                -4.8284e-08],
                               [2.8155e-11, 3.4583e-12, -1.5489e-12, -9.3383e-13, 1.0248e-09,
                                -1.7032e-10, 1.3346e-09, 5.9667e-09, -2.1974e-10, -4.5732e-10,
                                -4.3465e-09, 2.7857e-08, 8.3611e-10, 2.1461e-06, -7.9059e-08,
                                -1.6453e-07],
                               [5.6941e-13, 5.8579e-13, -7.9360e-13, -3.1667e-10, -1.7032e-10,
                                5.9689e-09, 6.3462e-10, 2.5195e-10, 5.2123e-12, -2.1457e-10,
                                -6.8314e-10, 2.8094e-09, -2.0356e-09, 9.0467e-08, 1.8915e-09,
                                -7.7208e-08],
                               [-7.4316e-12, 1.1961e-11, 2.5667e-12, -6.6294e-10, 1.3346e-09,
                                6.3462e-10, 3.1218e-08, -2.3738e-09, 3.4156e-09, 3.6041e-10,
                                -1.4306e-08, -9.8934e-09, 1.0064e-11, -8.5452e-07, 1.2294e-06,
                                1.2967e-07],
                               [1.1183e-08, 3.8468e-09, -2.9300e-10, -4.5104e-10, 5.9667e-09,
                                2.5195e-10, -2.3738e-09, 3.8061e-06, -1.2422e-07, -1.4134e-07,
                                -1.4162e-07, 5.4787e-06, 3.8084e-07, 1.3692e-03, -4.4684e-05,
                                -5.0812e-05],
                               [-4.2373e-09, 9.3854e-09, 1.1322e-10, -2.3859e-09, -2.1974e-10,
                                5.2123e-12, 3.4156e-09, -1.2422e-07, 3.2132e-06, 2.0508e-08,
                                -2.7676e-06, 2.5040e-07, -2.2263e-07, -4.4678e-05, 1.1561e-03,
                                7.3901e-06],
                               [-7.5792e-10, 2.4828e-10, 8.6031e-10, -1.3426e-10, -4.5732e-10,
                                -2.1457e-10, 3.6041e-10, -1.4134e-07, 2.0508e-08, 3.0975e-07,
                                -1.9151e-07, -7.9446e-07, -1.3602e-07, -5.0838e-05, 7.3742e-06,
                                1.1143e-04],
                               [5.1765e-09, -2.8962e-08, -9.0008e-10, 6.1558e-09, -4.3465e-09,
                                -6.8314e-10, -1.4306e-08, -1.4162e-07, -2.7676e-06, -1.9151e-07,
                                7.4307e-05, -1.6757e-06, 1.2516e-06, -5.0962e-05, -9.9556e-04,
                                -6.8889e-05],
                               [3.5563e-08, 5.3225e-09, -2.9641e-09, 3.0815e-09, 2.7857e-08,
                                2.8094e-09, -9.8934e-09, 5.4787e-06, 2.5040e-07, -7.9446e-07,
                                -1.6757e-06, 9.0490e-05, 1.0571e-06, 1.9707e-03, 9.0061e-05,
                                -2.8575e-04],
                               [1.1771e-09, 3.0067e-10, -2.3623e-10, -5.4149e-10, 8.3611e-10,
                                -2.0356e-09, 1.0064e-11, 3.8084e-07, -2.2263e-07, -1.3602e-07,
                                1.2516e-06, 1.0571e-06, 9.9117e-06, 1.3694e-04, -8.0085e-05,
                                -4.8981e-05],
                               [4.0230e-06, 1.3841e-06, -1.0538e-07, -1.6228e-07, 2.1461e-06,
                                9.0467e-08, -8.5452e-07, 1.3692e-03, -4.4678e-05, -5.0838e-05,
                                -5.0962e-05, 1.9707e-03, 1.3694e-04, 4.9259e-01, -1.6072e-02,
                                -1.8277e-02],
                               [-1.5246e-06, 3.3769e-06, 4.0731e-08, -8.5813e-07, -7.9059e-08,
                                1.8915e-09, 1.2294e-06, -4.4684e-05, 1.1561e-03, 7.3742e-06,
                                -9.9556e-04, 9.0061e-05, -8.0085e-05, -1.6072e-02, 4.1599e-01,
                                2.6573e-03],
                               [-2.7257e-07, 8.9331e-08, 3.0950e-07, -4.8284e-08, -1.6453e-07,
                                -7.7208e-08, 1.2967e-07, -5.0812e-05, 7.3901e-06, 1.1143e-04,
                                -6.8889e-05, -2.8575e-04, -4.8981e-05, -1.8277e-02, 2.6573e-03,
                                4.0088e-02]])
        R = torch.FloatTensor([[8.6052e-03, -4.3617e-04, 3.9815e-05, 2.1549e-04, 7.8445e-06, -6.7628e-04],
                               [-4.3617e-04, 6.6174e-03, 1.4289e-04, -1.8307e-03, 6.2787e-04, 1.1476e-03],
                               [3.9815e-05, 1.4289e-04, 5.9638e-04, 5.8608e-04, 2.1002e-04, -3.1699e-04],
                               [2.1549e-04, -1.8307e-03, 5.8608e-04, 8.6795e-02, -1.3805e-02, -3.8084e-04],
                               [7.8445e-06, 6.2787e-04, 2.1002e-04, -1.3805e-02, 6.9136e-02, 1.7426e-03],
                               [-6.7628e-04, 1.1476e-03, -3.1699e-04, -3.8084e-04, 1.7426e-03, 8.8171e-02]])
        for t in tqdm(range(1, imu.shape[0])):
            x = model(states[-1], motor[t].unsqueeze(0), dt)
            F, _, _ = jacobian(model, (states[-1], motor[t].unsqueeze(0), dt))
            F = F[0, :, 0, :]

            P = F @ P @ F.transpose(0, 1) + Q

            y = imu[t] - h(x)
            H = jacobian(h, x)
            H = H[0, :, 0, :]

            S = H @ P @ H.transpose(0, 1) + R
            K = P @ H.transpose(0, 1) @ torch.inverse(S)

            x = x + (K @ y[0].unsqueeze(-1)).reshape(1, 16)
            P = (torch.eye(16, 16) - K @ H) @ P

            x[3:7] = x[3:7] / torch.sqrt((x[3:7] ** 2).sum())
            states.append(x)
        states = torch.cat(states, dim=0)
        return states


if __name__ == '__main__':

    model = KalmanFiltering()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    flights = [load_flight(FLIGHT_LIST[k]) for k in range(len(FLIGHT_LIST))][4:]

    # f = DynamicalModel()
    # e = []
    # for k in range(4, len(FLIGHT_LIST)):
    #     data = load_flight(FLIGHT_LIST[k])
    #     state = torch.FloatTensor(data["state"])
    #     time = torch.FloatTensor(data["t"])
    #     dt = (time[1:] - time[:-1]).mean()
    #
    #     acc = torch.cat([torch.zeros(1, 3), (state[1:, 7:10] - state[:-1, 7:10]) / dt], dim=0)
    #     state = torch.cat([state, acc], dim=-1)
    #
    #     motor = torch.FloatTensor(data["motor"])
    #     imu = data["imu_ang_acc"]
    #
    #     pi_rotation = Rotation.from_euler("zyx", np.array([np.pi / 2, 0, np.pi]), degrees=False)
    #     imu_rpy = pi_rotation.apply(imu[:, :3])
    #     imu_acc = pi_rotation.apply(imu[:, 3:])
    #     imu = torch.FloatTensor(np.concatenate([imu_rpy, imu_acc], axis=-1))
    #
    #     pred_next_state, ang_acc = f(state[:-1], motor[1:], dt)
    #     # e.append(state)
    #     fig, ax = plt.subplots(3, 1)
    #     for i in range(3):
    #         ax[i].plot(state[1001:1111, 13 + i], marker=".", label="Mocap")
    #         ax[i].plot(pred_next_state[1001:1111, i + 13], marker=".", label="Model")
    #         ax[i].legend()
    #         ax[i].set_title("Acceleration " + str(i))
    #
    #     fig, ax = plt.subplots(3, 1)
    #     s = torch.cat([torch.zeros(1, 3), (state[1:, 10:13] - state[:-1, 10:13]) / dt], dim=0)
    #     for i in range(3):
    #         ax[i].plot(s[1001:1111, i], marker=".", label="Mocap")
    #         ax[i].plot(ang_acc[1001:1111, i], marker=".", label="Model")
    #         ax[i].legend()
    #         ax[i].set_title("Acc Angle " + str(i))
    #     plt.show()

    # e.append((state[1:] - pred_next_state).detach().numpy())

    for k in range(4, len(FLIGHT_LIST)):
        data = load_flight(FLIGHT_LIST[k])
        START = 10_000
        LENGTH = 5_000
        time = torch.FloatTensor(data["t"][START:START + LENGTH])
        dt = (time[1:] - time[:-1]).mean()
        state = torch.FloatTensor(data["state"])[START:START + LENGTH]

        acc = torch.cat([torch.zeros(1, 3), (state[1:, 7:10] - state[:-1, 7:10]) / dt], dim=0)
        state = torch.cat([state, acc], dim=-1)

        motor = torch.FloatTensor(data["motor"])[START:START + LENGTH]
        imu = data["imu_ang_acc"][START:START + LENGTH]

        pi_rotation = Rotation.from_euler("zyx", np.array([np.pi / 2, 0, np.pi]), degrees=False)
        imu_rpy = pi_rotation.apply(imu[:, :3])
        imu_acc = pi_rotation.apply(imu[:, 3:])
        imu = torch.FloatTensor(np.concatenate([imu_rpy, imu_acc], axis=-1))

        pred = model(state[0], imu, motor, dt)

        fig, ax = plt.subplots(4, 4)
        label = ["X", "Y", "Z"]
        for k in range(3):
            ax[0][k].plot(time, state[:, k], label="gt")
            ax[0][k].plot(time, pred[:, k], label="pred")
            ax[0][k].set_title(label[k])
            ax[0][k].legend()

        label = ["Qx", "Qy", "Qz", "Qw"]
        for k in range(4):
            ax[1][k].plot(time, state[:, 3 + k], label="gt")
            ax[1][k].plot(time, pred[:, 3 + k], label="pred")
            ax[1][k].set_title(label[k])
            ax[1][k].legend()

        label = ["Vx", "Vy", "Vz"]
        for k in range(3):
            ax[2][k].plot(time, state[:, k + 7], label="gt")
            ax[2][k].plot(time, pred[:, k + 7], label="pred")
            ax[2][k].set_title(label[k])
            ax[2][k].legend()

        label = ["Wx", "Wy", "Wz"]
        for k in range(3):
            ax[3][k].plot(time, state[:, k + 10], label="gt")
            ax[3][k].plot(time, pred[:, k + 10], label="pred")
            ax[3][k].set_title(label[k])
            ax[3][k].legend()
        fig.tight_layout()
        plt.show()
