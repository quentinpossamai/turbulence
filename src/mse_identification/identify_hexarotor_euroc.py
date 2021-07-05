""" Identify the non-physical parameters of the model:
theta1 = KT/m
theta2 = LKT/[Ixy]
theta3 = Km/Iz
theta4 = Iz/Ixy"""

import itertools
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.utils.quaternion import qrot
import src.data.euroc_mav.loading as euroc

DT = 1 / 360


def identify_model():
    """Solve the Ax = y problem where x is the parameters of the model"""

    flights = euroc.EurocMAVDatasetHorizon("train", horizon=1)

    # Load flights in a big vector column
    states, future_states, command = [], [], []
    for i in range(len(flights)):
        s = torch.FloatTensor(flights[i]["state"])
        motor = torch.FloatTensor(flights[i]["motor"])
        states.append(s[:-1])
        future_states.append(s[1:])
        command.append(motor[:-1])
    states = torch.cat(states, dim=0)
    future_states = torch.cat(future_states, dim=0)
    command = torch.cat(command, dim=0)
    batch_size = states.shape[0]

    # Create A matrix
    # Compute forces and torques:
    best_permutation, best_theta, best_error = None, None, 1000
    permutations = list(itertools.permutations([0, 1, 2, 3, 4, 5]))
    save_identifications = []
    for p in tqdm(permutations):
        omega = command[:, p] ** 2

        a = 0.5
        b = 3 ** 0.5 / 2
        motor_matrix = torch.FloatTensor([[-a, +a, +1, +a, -a, -1],
                                          [-b, -b, +0, +b, +b, +0],
                                          [-1, +1, -1, +1, -1, +1],
                                          [+1, +1, +1, +1, +1, +1]]).view(1, 4, 6)
        motor_forces = motor_matrix @ omega.unsqueeze(-1)  # [tx, ty, tz, T]

        # Linear PFD
        gravity = torch.FloatTensor([0, 0, 9.81]).view(1, 3, 1)
        thrust = DT * torch.cat([torch.zeros(batch_size, 1), torch.zeros(batch_size, 1), motor_forces[:, 3]], dim=1)
        thrust_world_frame = qrot(states[:, 3:7], thrust).unsqueeze(-1)
        a_speed = torch.cat([thrust_world_frame, torch.zeros(batch_size, 3, 3)], dim=-1)
        b_speed = states[:, 7:10].unsqueeze(-1) - gravity * DT

        # Angular PFD
        wyz = (states[:, 11] * states[:, 12]).unsqueeze(-1)
        wxz = (states[:, 10] * states[:, 12]).unsqueeze(-1)
        theta1_component = DT * torch.cat([motor_forces[:, 0], motor_forces[:, 1], torch.zeros(batch_size, 1)], dim=-1)
        theta2_component = DT * torch.cat([torch.zeros(batch_size, 2), motor_forces[:, 2]], dim=-1)
        theta3_component = DT * torch.cat([-wyz, wxz, torch.zeros(batch_size, 1)], dim=-1)
        a_angular = torch.stack([torch.zeros(batch_size, 3),
                                 theta1_component,
                                 theta2_component,
                                 theta3_component], dim=-1)
        b_angular = states[:, 10:] + DT * torch.cat([wyz, -wxz, torch.zeros(batch_size, 1)], dim=-1)

        # Solve the linear problem
        a = torch.cat([a_speed, a_angular], dim=1).view(-1, 4)
        b = torch.cat([b_speed.squeeze(-1), b_angular], dim=-1).view(-1, 1)
        y = future_states[:, 7:].reshape(-1, 1)
        theta = torch.pinverse(a) @ (y - b)

        error = ((y - (a @ theta + b)) ** 2).mean()
        save_identifications.append((p, theta.clone(), error))
        if error < best_error and theta[0] > 0 and theta[1] > 0 and theta[3] > 0:
            best_error = error
            best_permutation = p
            best_theta = theta.clone()
            # print(best_permutation, best_theta, best_error)
        print(theta, error, p)
    # Best permutation all: (3, 4, 5, 2, 1, 0) tensor([[ 1.1560e-04],
    #         [ 2.3920e-03],
    #         [ 6.8990e-05],
    #         [-1.0757e+01]]) tensor(7.0828e-06)
    # Best permutation verify criteria: (5, 4, 1, 0, 3, 2) tensor([[1.1560e-04],
    #         [2.6741e-03],
    #         [6.8990e-05],
    #         [1.3661e+01]]) tensor(7.6091e-06)
    print(f"Best permutation:", best_permutation, best_theta, best_error)
    plt.plot(sorted([e[2] for e in save_identifications]))
    plt.show()


if __name__ == '__main__':
    identify_model()
