""" Identify the non-physical parameters of the model:
theta1 = KT/m
theta2 = LKT/[sqrt(2)*Ixy]
theta3 = Km/Iz
theta4 = Iz/Ixy"""

import torch
from tqdm import tqdm
import itertools
from src.metrics import load_flights
from scipy.spatial.transform import Rotation

DT = 1 / 360


def identify_model():
    """Solve the Ax = y problem where x is the parameters of the model"""

    flights = load_flights("test")

    ## Load flights in a big vector column
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
    B = states.shape[0]

    ## Create A matrix
    # Compute forces and torques:
    best_p, best_theta, best_error = None, None, 1000
    permutations = list(itertools.permutations([1, 0, 3, 2]))
    for p in tqdm(permutations):
        Omega = command[:, p] ** 2
        P = torch.FloatTensor([[+1, +1, +1, +1],
                               [-1, +1, +1, -1],
                               [-1, -1, +1, +1],
                               [-1, +1, -1, +1]]).view(1, 4, 4)
        motor_forces = P @ Omega.unsqueeze(-1)  # [fa, tx, ty, tz]

        # Linear PFD
        gravity = torch.FloatTensor([0, 0, 9.81]).view(1, 3, 1)
        rotation_matrix = torch.FloatTensor(Rotation.from_quat(states[:, 3:7]).as_matrix())
        thrust = DT * torch.cat([torch.zeros(B, 1), torch.zeros(B, 1), motor_forces[:, 0]], dim=1)
        thrust_world_frame = rotation_matrix @ thrust.unsqueeze(-1)
        A_speed = torch.cat([thrust_world_frame, torch.zeros(B, 3, 3)], dim=-1)
        b_speed = states[:, 7:10].unsqueeze(-1) - gravity * DT

        # Angular PFD
        wyz = (states[:, 11] * states[:, 12]).unsqueeze(-1)
        wxz = (states[:, 10] * states[:, 12]).unsqueeze(-1)
        theta2_component = DT * torch.cat([motor_forces[:, 1], motor_forces[:, 2], torch.zeros(B, 1)], dim=-1)
        theta3_component = DT * torch.cat([torch.zeros(B, 2), motor_forces[:, 3]], dim=-1)
        theta4_component = DT * torch.cat([-wyz, wxz, torch.zeros(B, 1)], dim=-1)
        A_angular = torch.stack([torch.zeros(B, 3),
                                 theta2_component,
                                 theta3_component,
                                 theta4_component], dim=-1)
        b_angular = states[:, 10:] + DT * torch.cat([wyz, -wxz, torch.zeros(B, 1)], dim=-1)

        ## Solve the linear problem
        A = torch.cat([A_speed, A_angular], dim=1).view(-1, 4)
        b = torch.cat([b_speed.squeeze(-1), b_angular], dim=-1).view(-1, 1)
        y = future_states[:, 7:].reshape(-1, 1)
        theta = torch.pinverse(A) @ (y - b)

        error = ((y - (A @ theta + b)) ** 2).mean()
        if error < best_error:  # and theta[0] > 0 and theta[1] > 0 and theta[2] > 0 and theta[3] > 0:
            best_error = error
            best_p = p
            best_theta = theta.clone()
            print(best_p, best_theta, best_error)
        # print(theta, error, p)


if __name__ == '__main__':
    identify_model()
