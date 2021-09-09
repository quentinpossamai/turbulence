import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
import utils.quaternion as quaternion_utils
from torch.autograd.functional import jacobian

plt.style.use("seaborn")

parser = argparse.ArgumentParser()
parser.add_argument('--path_dataset', default='/Volumes/Samsung_T5/BlackBird/euroc_mav/intermediate', type=str)
parser.add_argument('--flight_name', default='V1_01_easy', type=str)
args = parser.parse_args()


def load_flight():
    data = pickle.load(open(f"{args.path_dataset}/{args.flight_name}/data.pkl", 'rb'))
    return data


class DynamicalModel(nn.Module):
    def __init__(self):
        """
        :param paper: True, load paper parameters. False, load mse parameters
        :param dtype: Datatype of the model.
        """
        super(DynamicalModel, self).__init__()
        self.permutation = [5, 0, 1, 2, 3, 4]
        self.theta = torch.tensor([[1.1566750e-04],
                                   [6.9880742e-04],
                                   [-5.1250136e-05],
                                   [2.6935432e+00]])  # MSE

    def forward(self, state, motor, dt):
        pos = state[:, :3]
        quat = state[:, 3:7]
        lin_vel = state[:, 7:10]
        ang_vel = state[:, 10:13]

        motor_torsor = self.motor_dynamic(motor)

        batch_size = state.shape[0]
        # Linear PFD
        g = torch.tensor([0, 0, 9.81]).view(1, 3)
        thrust = torch.cat([torch.zeros(batch_size, 2), motor_torsor[:, 3:]], dim=1)
        next_lin_acc = (quaternion_utils.qrot(quat, self.theta[0] * thrust) - g)

        next_lin_vel = lin_vel + dt * next_lin_acc

        # Position
        next_pos = pos + dt * next_lin_vel

        # Angular PFD
        t3_tensor = torch.tensor([1, 1, self.theta[3]]).view(1, 3)
        t12_tensor = torch.tensor([self.theta[1], self.theta[1], self.theta[2]]).view(1, 3)
        next_ang_vel = ang_vel + dt * (torch.cross(t3_tensor * ang_vel, ang_vel)
                                       + t12_tensor * motor_torsor[:, :3])

        # Quat
        next_quat = quat + dt * 0.5 * quaternion_utils.qmul(quat,
                                                            torch.cat(
                                                                [next_ang_vel, torch.zeros((len(next_ang_vel), 1))],
                                                                dim=1))
        next_quat = next_quat / torch.sqrt((next_quat ** 2).sum(-1, keepdim=True))

        return torch.cat([next_pos, next_quat.squeeze(-1), next_lin_vel, next_ang_vel, next_lin_acc], dim=-1)

    def motor_dynamic(self, command):
        a = 0.5
        b = 3 ** 0.5 / 2
        motor_matrix = torch.tensor([[-a, +a, +1, +a, -a, -1],
                                     [-b, -b, +0, +b, +b, +0],
                                     [-1, +1, -1, +1, -1, +1],
                                     [+1, +1, +1, +1, +1, +1]]).view(1, 4, 6)
        motor = command[:, self.permutation]
        omega_squared = motor ** 2
        motor_torsor = motor_matrix @ omega_squared.unsqueeze(-1)  # [tx, ty, tz, T]
        return motor_torsor.squeeze(-1)


def h(state):
    pos = state[:, :3]
    quat = state[:, 3:7]
    lin_vel = state[:, 7:10]
    ang_vel = state[:, 10:13]
    lin_acc = state[:, 13:]

    temp = quaternion_utils.qmul(quat * torch.FloatTensor([-1, -1, -1, 1]),
                                 torch.cat([lin_acc, torch.zeros(1, 1)], dim=-1))
    acc = quaternion_utils.qmul(temp, quat)[:, :3]

    observation = torch.cat([ang_vel, acc], dim=-1)
    return observation


def kalman(x0, imu, motor, dt):
    states = [x0.unsqueeze(0)]
    model = DynamicalModel()

    P = torch.eye(13, 13)
    Q = torch.eye(13, 13)
    R = torch.eye(6, 6)
    for t in tqdm(range(1, imu.shape[0])):
        x = model(states[-1], motor[t].unsqueeze(0), dt)
        F, _, _ = jacobian(model, (states[-1], motor[t].unsqueeze(0), dt))
        F = F[0, :13, 0, :]

        P = F @ P @ F.transpose(0, 1) + Q

        y = imu[t] - h(x)
        H = jacobian(h, x)
        H = H[0, :, 0, :13]  # TODO : Rajouter l'acceleration partout dans l'état, pour avoir un état de taille 16

        S = H @ P @ H.transpose(0, 1) + R
        K = P @ H.transpose(0, 1) @ torch.inverse(S)

        x = x[:, :13] + (K @ y[0].unsqueeze(-1)).reshape(1, 13)
        P = (torch.eye(13, 13) - K @ H) @ P

        states.append(x)
    states = torch.cat(states, dim=0)
    return states


if __name__ == '__main__':
    state = torch.randn(13)
    motor = torch.randn(4)
    data = load_flight()

    model = DynamicalModel()

    START = 10_000
    LENGTH = 100
    time = torch.FloatTensor(data["t"][START:START + LENGTH])
    dt = (time[1:] - time[:-1]).mean()
    state = torch.FloatTensor(data["state"])[START:START + LENGTH]
    motor = torch.FloatTensor(data["motor"])[START:START + LENGTH]
    imu = data["imu_ang_acc"][START:START + LENGTH]

    pi_rotation = Rotation.from_euler("zyx", np.array([np.pi / 2, 0, np.pi]), degrees=False)
    imu_rpy = pi_rotation.apply(imu[:, :3])
    imu_acc = pi_rotation.apply(imu[:, 3:])
    imu = torch.FloatTensor(np.concatenate([imu_rpy, imu_acc], axis=-1))

    pred = kalman(state[0], imu, motor, dt)
    fig, ax = plt.subplots(3, 1)
    for k in range(3):
        ax[k].plot(state[:, k])
        ax[k].plot(pred[:, k])
    plt.show()
