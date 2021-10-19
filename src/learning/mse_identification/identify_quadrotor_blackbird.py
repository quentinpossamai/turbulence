""" Identify the non-physical parameters of the model:
theta1 = Kt / m
theta2 = L Kt / Ixy
theta3 = Km / Iz
theta4 = Iz / Ixy"""

import itertools
import matplotlib.pyplot as plt
import pickle
import torch
from tqdm import tqdm
import itertools
from src.metrics import load_blackbird_flights
from scipy.spatial.transform import Rotation

DT = 1 / 360


def identify_model():
    """Solve the Ax = y problem where x is the parameters of the model"""

    flights = load_blackbird_flights("train")

    # Load flights in a big vector column
    states, future_states, command = [], [], []
    for f in flights.keys():
        s = torch.FloatTensor(flights[f]["state"])
        motor = torch.FloatTensor(flights[f]["motor"])
        states.append(s[:-1])
        future_states.append(s[1:])
        command.append(motor[:-1])
    states = torch.cat(states, dim=0)
    future_states = torch.cat(future_states, dim=0)
    command = torch.cat(command, dim=0)
    batch_size = states.shape[0]

    best_permutation, best_theta, best_error = None, None, 1000
    permutations = list(itertools.permutations([1, 2, 3, 4]))

    linear_acceleration = (future_states[:, 7:10] - states[:, 7:10]) / DT
    angular_aceleration = (future_states[:, 10:] - states[:, 10:]) / DT

    plt.subplot(131)
    plt.plot(linear_acceleration[10000:20000, 0])
    plt.subplot(132)
    plt.plot(linear_acceleration[10000:20000, 1])
    plt.subplot(133)
    plt.plot(linear_acceleration[10000:20000, 2])

    plt.figure()
    plt.subplot(131)
    plt.plot(command[10000:20000, 0])
    plt.subplot(132)
    plt.plot(command[10000:20000, 1])
    plt.subplot(133)
    plt.plot(command[10000:20000, 2])
    plt.show()

    for p in tqdm(permutations):
        omega = command[:, p] ** 2
        motor_matrix = torch.FloatTensor([[+1, +1, +1, +1],
                                          [-1, +1, +1, -1],
                                          [-1, -1, +1, +1],
                                          [-1, +1, -1, +1]]).view(1, 4, 4)
        motor_forces = motor_matrix @ omega.unsqueeze(-1)  # [fa, tx, ty, tz]

        # Linear PFD
        gravity = torch.FloatTensor([0, 0, 9.81]).view(1, 3, 1)
        rotation_matrix = torch.FloatTensor(Rotation.from_quat(states[:, 3:7]).as_matrix())
        thrust = DT * torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), motor_forces[:, 0]], dim=1)
        thrust_world_frame = rotation_matrix @ thrust.unsqueeze(-1)
        A_speed = torch.cat([thrust_world_frame, torch.zeros(batch_size, 3, 3)], dim=-1)
        b_speed = states[:, 7:10].unsqueeze(-1) - gravity * DT

        # Angular PFD
        wyz = (states[:, 11] * states[:, 12]).unsqueeze(-1)
        wxz = (states[:, 10] * states[:, 12]).unsqueeze(-1)
        theta2_component = DT * torch.cat([motor_forces[:, 1], motor_forces[:, 2], torch.zeros(batch_size, 1)], dim=-1)
        theta3_component = DT * torch.cat([torch.zeros(batch_size, 2), motor_forces[:, 3]], dim=-1)
        theta4_component = DT * torch.cat([-wyz, wxz, torch.zeros(batch_size, 1)], dim=-1)
        A_angular = torch.stack([torch.zeros(batch_size, 3),
                                 theta2_component,
                                 theta3_component,
                                 theta4_component], dim=-1)
        b_angular = states[:, 10:] + DT * torch.cat([wyz, -wxz, torch.zeros(batch_size, 1)], dim=-1)

        # Solve the linear problem
        A = torch.cat([A_speed, A_angular], dim=1).view(-1, 4)
        b = torch.cat([b_speed.squeeze(-1), b_angular], dim=-1).view(-1, 1)
        y = future_states[:, 7:].reshape(-1, 1)
        theta = torch.pinverse(A) @ (y - b)

        error = ((y - (A @ theta + b)) ** 2).mean()
        if error < best_error:  # and theta[0] > 0 and theta[1] > 0 and theta[2] > 0 and theta[3] > 0:
            best_error = error
            best_permutation = p
            best_theta = theta.clone()
            print(best_permutation, best_theta, best_error)
        # print(theta, error, p)


if __name__ == '__main__':
    identify_model()
