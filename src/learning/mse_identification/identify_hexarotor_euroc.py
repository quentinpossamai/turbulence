""" Identify the non-physical parameters of the model:
theta1 = KT/m
theta2 = LKT/[Ixy]
theta3 = Km/Iz
theta4 = Iz/Ixy"""

import itertools
import matplotlib.pyplot as plt
import pickle
import torch
from tqdm import tqdm

from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
import src.data.euroc_mav.loading as euroc
from src.utils.quaternion import qrot

DT = 1 / 200

a = 0.5
b = 3 ** 0.5 / 2
MOTOR_MATRIX = torch.FloatTensor([[-a, +a, +1, +a, -a, -1],
                                  [-b, -b, +0, +b, +b, +0],
                                  [-1, +1, -1, +1, -1, +1],
                                  [+1, +1, +1, +1, +1, +1]]).view(1, 4, 6)


def identify_model():
    """Solve the Ax = y problem where x is the parameters of the model"""

    flights = euroc.EurocMAVDatasetHorizon("train", traj_len=2, dtype=torch.float64, batch_dim=False)

    # Load flights in a big vector column
    state, future_state, motor = [], [], []
    for i in range(len(flights)):
        s = torch.FloatTensor(flights[i]["state"])
        m = torch.FloatTensor(flights[i]["motor"])
        state.append(s[:-1])
        future_state.append(s[1:])
        motor.append(m[:1])
    state = torch.cat(state, dim=0)
    future_state = torch.cat(future_state, dim=0)
    motor = torch.cat(motor, dim=0)
    batch_size = state.shape[0]

    # Create A matrix
    # Compute forces and torques:
    best_permutation, best_theta, best_error = None, None, 1000
    permutations = list(itertools.permutations([0, 1, 2, 3, 4, 5]))
    save_identifications = []
    for p in tqdm(permutations):
        omega = motor[:, p] ** 2
        motor_forces = MOTOR_MATRIX @ omega.unsqueeze(-1)  # [tx, ty, tz, T]

        # Linear PFD
        gravity = torch.FloatTensor([0, 0, 9.81]).view(1, 3, 1)
        thrust_body = torch.cat([torch.zeros(batch_size, 2), motor_forces[:, 3]], dim=1)
        thrust_world = qrot(state[:, 3:7], thrust_body).view(batch_size, 3, 1)
        a_speed = torch.cat([DT * thrust_world, torch.zeros(batch_size, 3, 3)], dim=-1)
        b_speed = state[:, 7:10].unsqueeze(-1) - gravity * DT

        # Angular PFD
        wyz = (state[:, 11] * state[:, 12]).unsqueeze(-1)
        wxz = (state[:, 10] * state[:, 12]).unsqueeze(-1)
        theta1_component = DT * torch.cat([motor_forces[:, 0], motor_forces[:, 1], torch.zeros(batch_size, 1)], dim=-1)
        theta2_component = DT * torch.cat([torch.zeros(batch_size, 2), motor_forces[:, 2]], dim=-1)
        theta3_component = DT * torch.cat([-wyz, wxz, torch.zeros(batch_size, 1)], dim=-1)
        a_angular = torch.stack([torch.zeros(batch_size, 3),
                                 theta1_component,
                                 theta2_component,
                                 theta3_component], dim=-1)
        b_angular = state[:, 10:] + DT * torch.cat([wyz, -wxz, torch.zeros(batch_size, 1)], dim=-1)

        # Solve the linear problem
        a = torch.cat([a_speed, a_angular], dim=1).view(-1, 4)
        b = torch.cat([b_speed.squeeze(-1), b_angular], dim=-1).view(-1, 1)
        y = future_state[:, 7:].reshape(-1, 1)
        theta = torch.pinverse(a) @ (y - b)

        error = ((y - (a @ theta + b)) ** 2).mean()

        horizon = 100
        rollout_error, _ = DynamicalModel(param={"permutation": p,
                                                 "theta": theta}, dtype=torch.float64).rollout(
            motor=motor[:horizon + 1],
            state=state[:horizon + 1],
            horizon=horizon)
        save_identifications.append((p, theta.clone(), error, rollout_error))

        if error < best_error and theta[0] > 0 and theta[1] > 0 and theta[3] > 0:
            best_error = error
            best_permutation = p
            best_theta = theta.clone()
            # print(best_permutation, best_theta, best_error)

    # Best permutation all: (3, 4, 5, 2, 1, 0) tensor([[ 1.1560e-04],
    #         [ 2.3920e-03],
    #         [ 6.8990e-05],
    #         [-1.0757e+01]]) tensor(7.0828e-06)
    # Best permutation verify criteria: (5, 4, 1, 0, 3, 2) tensor([[1.1560e-04],
    #         [2.6741e-03],
    #         [6.8990e-05],
    #         [1.3661e+01]]) tensor(7.6091e-06)
    # print(f"Best permutation:", best_permutation, best_theta, best_error)
    save_identifications = sorted(save_identifications, key=lambda e: e[2])

    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.plot([e[2] for e in save_identifications], marker="x")
    for i, e in enumerate(save_identifications):
        ax.text(i, e[2], f"{i}")
        if e[1][0] > 0 and e[1][1] > 0 and e[1][3] > 0:
            ax.scatter(i, e[2], color="red", zorder=3)
    ax.grid()
    ax.set_xlabel("Configuration number")
    ax.set_ylabel("Error on train set")

    def print_point(idx):
        e = save_identifications[idx]
        print(f"Point: {idx}")
        print(f"Permutation: {list(e[0])}")
        print(f"Theta: {e[1].numpy()}")
        print(f"Error: {e[2].item():0.4e}")

    for i in range(10):
        print_point(i)
    pickle.dump(save_identifications, open(euroc.ANALYSIS_SECTION_PATH + "identifications.pkl", "wb"))

    # TODO ROLLOUT
    print()
    plt.show()


if __name__ == '__main__':
    identify_model()
