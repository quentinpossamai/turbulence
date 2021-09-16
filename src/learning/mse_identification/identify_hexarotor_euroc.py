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
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = f"{GENERAL_ANALYSIS_PATH}euroc_identification/"
mkdir_if_not_exist([ANALYSIS_PATH])

DT = 1 / 200
DTYPE = torch.float64

coef1 = 0.5
coef2 = 3 ** 0.5 / 2
MOTOR_MATRIX = torch.tensor([[-coef1, +coef1, +1, +coef1, -coef1, -1],
                             [-coef2, -coef2, +0, +coef2, +coef2, +0],
                             [-1, +1, -1, +1, -1, +1],
                             [+1, +1, +1, +1, +1, +1]], dtype=DTYPE).view(1, 4, 6)


def identify_model():
    """Solve the Ax = y problem where x is the parameters of the model"""
    # Data identification
    flights = euroc.EurocMAVDatasetHorizon("train", traj_len=2, dtype=DTYPE, batch_dim=False)

    # Load flights in a big vector column
    state, future_state, motor = [], [], []
    for i in range(len(flights)):
        s = flights[i]["state"]
        m = flights[i]["motor"]
        state.append(s[:-1])
        future_state.append(s[1:])
        motor.append(m[:1])
    state = torch.cat(state, dim=0)
    future_state = torch.cat(future_state, dim=0)
    motor = torch.cat(motor, dim=0)
    batch_size = state.shape[0]

    # Create A matrix
    # Compute forces and torques:
    permutations = list(itertools.permutations([0, 1, 2, 3, 4, 5]))
    save_identifications = []
    for p in tqdm(permutations):
        omega = motor[:, p] ** 2
        motor_forces = MOTOR_MATRIX @ omega.unsqueeze(-1)  # [tx, ty, tz, T]

        # Linear PFD
        gravity = torch.tensor([0, 0, 9.81], dtype=DTYPE).view(1, 3, 1)
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
        save_identifications.append({"permutation": p, "theta": theta.clone(), "mse": error})
        pickle.dump(save_identifications, open(f"{ANALYSIS_PATH}identification.pkl", "wb"))


def rollout():
    param_range = pickle.load(open(f"{ANALYSIS_PATH}identification.pkl", "rb"))
    max_horizon = 200
    horizon_range = range(1, max_horizon)
    long_traj = euroc.EurocMAVDatasetHorizon("train", traj_len=max_horizon + 1, dtype=DTYPE, batch_dim=False)
    batch = long_traj[0]
    motor_roll = batch["motor"]
    state_roll = batch["state"]

    for i in range(len(param_range)):
        param_range[i]["rank"] = []

    for i, param in enumerate(tqdm(param_range)):
        permutation, theta = param["permutation"], param["theta"]
        model = DynamicalModel(param={"permutation": permutation, "theta": theta}, dtype=torch.float64)
        _, rollout_error, _ = model.rollout(motor=motor_roll, state=state_roll, horizon=max_horizon)
        param_range[i]["rollout_error"] = rollout_error

    for j, horizon in enumerate(tqdm(horizon_range)):
        order = torch.argsort(torch.tensor([e["rollout_error"][j] for e in param_range], dtype=DTYPE))
        for i, _ in enumerate(param_range):
            param_range[i]["rank"].append(order[i])
    pickle.dump(param_range, open(f"{ANALYSIS_PATH}ranking.pkl", "wb"))


def plot_identification():
    def print_point(idx):
        e = save_identifications[idx]
        print(f"Point: {idx}")
        print(f"Permutation: {list(e[0])}")
        print(f"Theta: {e[1].numpy()}")
        print(f"Error: {e[2].item():0.4e}")

    # Plot
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})
    error_id = 3
    save_identifications = sorted(save_identifications, key=lambda e: e[error_id])
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.plot([e[error_id] for e in save_identifications], marker="x")
    for i, e in enumerate(save_identifications):
        ax.text(i, e[error_id], f"{i}")
        # ax.text(i + 10, e[2], f"{e[3]}")

        if e[1][0] > 0 and e[1][1] > 0 and e[1][3] > 0:
            ax.scatter(i, e[error_id], color="red", zorder=3)
    ax.grid()
    ax.set_xlabel("Configuration number")
    ax.set_ylabel("Error on train set")

    save_identifications = sorted(save_identifications, key=lambda e: e[3])
    for i in range(10):
        print_point(i)

    pickle.dump(save_identifications, open(euroc.ANALYSIS_SECTION_PATH + "identifications.pkl", "wb"))

    print()
    plt.show()


def plot_rollout():
    param_range = pickle.load(open(f"{ANALYSIS_PATH}ranking.pkl", "rb"))
    error_tab = torch.zeros(len(param_range), len(param_range[0]["rank"]))
    rank_tab = torch.zeros(len(param_range), len(param_range[0]["rank"]))
    for i, param in enumerate(param_range):
        error_tab[i, :] = param["rollout_error"]
        rank_tab[i, :] = torch.tensor(param["rank"])
    # TODO solve shape error
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    for i, is_at_some_point_top in enumerate((rank_tab < 1).any(dim=1)):
        if is_at_some_point_top:
            ax.plot(error_tab[i], label=f"{i}")
    # for i, param in enumerate(param_range):
    #     ax.plot(error_tab[i], label=f"{i}")
    # ax.plot(error_tab[68], label=f"{68}")
    ax.grid()
    ax.set_xlabel("Rollout length")
    ax.set_ylabel("Rollout loss on train set")
    ax.legend()
    # plt.show()


rollout100 = """Point: 0
Permutation: [4, 5, 0, 3, 2, 1]
Theta: [[1.14061868e-04]
 [3.20589610e-04]
 [2.90530320e-05]
 [2.86682918e+00]]
Error: 2.9689e-05
Point: 1
Permutation: [3, 2, 1, 4, 5, 0]
Theta: [[ 1.14061868e-04]
 [-3.20589610e-04]
 [-2.90530320e-05]
 [ 2.86682918e+00]]
Error: 2.9689e-05
Point: 2
Permutation: [1, 2, 3, 0, 5, 4]
Theta: [[ 1.14061868e-04]
 [-3.17937380e-04]
 [-2.90530320e-05]
 [ 2.86055981e+00]]
Error: 2.9704e-05
Point: 3
Permutation: [0, 5, 4, 1, 2, 3]
Theta: [[1.14061868e-04]
 [3.17937380e-04]
 [2.90530320e-05]
 [2.86055981e+00]]
Error: 2.9704e-05
Point: 4
Permutation: [2, 3, 1, 4, 5, 0]
Theta: [[ 1.14061868e-04]
 [-8.39880442e-05]
 [-8.23386633e-05]
 [ 2.78678989e+00]]
Error: 3.0272e-05
Point: 5
Permutation: [4, 5, 0, 2, 3, 1]
Theta: [[1.14061868e-04]
 [8.39880442e-05]
 [8.23386633e-05]
 [2.78678989e+00]]
Error: 3.0272e-05
Point: 6
Permutation: [1, 2, 3, 5, 0, 4]
Theta: [[ 1.14061868e-04]
 [-8.25252357e-05]
 [-8.08946405e-05]
 [ 2.77702406e+00]]
Error: 3.0283e-05
Point: 7
Permutation: [5, 0, 4, 1, 2, 3]
Theta: [[1.14061868e-04]
 [8.25252357e-05]
 [8.08946405e-05]
 [2.77702406e+00]]
Error: 3.0283e-05
Point: 8
Permutation: [4, 2, 3, 5, 0, 1]
Theta: [[ 1.14061868e-04]
 [-9.36372650e-05]
 [ 8.23386633e-05]
 [ 2.79934546e+00]]
Error: 3.0183e-05
Point: 9
Permutation: [5, 0, 1, 4, 2, 3]
Theta: [[ 1.14061868e-04]
 [ 9.36372650e-05]
 [-8.23386633e-05]
 [ 2.79934546e+00]]
Error: 3.0183e-05"""

rollout50 = """Point: 0
Permutation: [0, 1, 2, 3, 4, 5]
Theta: [[1.14061868e-04]
 [2.35845582e-04]
 [2.90530320e-05]
 [2.52134583e+00]]
Error: 2.9714e-05
Point: 1
Permutation: [0, 1, 2, 3, 5, 4]
Theta: [[1.14061868e-04]
 [4.11977480e-05]
 [7.31840187e-05]
 [2.79562970e+00]]
Error: 3.0595e-05
Point: 2
Permutation: [0, 1, 2, 4, 3, 5]
Theta: [[1.14061868e-04]
 [6.47773152e-05]
 [8.81164757e-05]
 [2.72455292e+00]]
Error: 3.0353e-05
Point: 3
Permutation: [0, 1, 2, 4, 5, 3]
Theta: [[1.14061868e-04]
 [1.98385753e-05]
 [7.31840187e-05]
 [2.80218741e+00]]
Error: 3.0672e-05
Point: 4
Permutation: [0, 1, 2, 5, 3, 4]
Theta: [[1.14061868e-04]
 [3.79399584e-05]
 [8.81164757e-05]
 [2.83067897e+00]]
Error: 3.0518e-05
Point: 5
Permutation: [0, 1, 2, 5, 4, 3]
Theta: [[1.14061868e-04]
 [1.13843429e-04]
 [2.90530320e-05]
 [2.85253488e+00]]
Error: 3.0531e-05
Point: 6
Permutation: [0, 1, 3, 2, 4, 5]
Theta: [[1.14061868e-04]
 [6.13697678e-05]
 [8.23386633e-05]
 [2.74297219e+00]]
Error: 3.0442e-05
Point: 7
Permutation: [0, 1, 3, 2, 5, 4]
Theta: [[ 1.14061868e-04]
 [ 3.75309916e-05]
 [-8.39432842e-05]
 [ 2.79937491e+00]]
Error: 3.0533e-05
Point: 8
Permutation: [0, 1, 3, 4, 2, 5]
Theta: [[1.14061868e-04]
 [3.21659044e-05]
 [8.81164757e-05]
 [2.78911325e+00]]
Error: 3.0539e-05
Point: 9
Permutation: [0, 1, 3, 4, 5, 2]
Theta: [[ 1.14061868e-04]
 [-4.12313327e-05]
 [-8.39432842e-05]
 [ 2.80487504e+00]]
Error: 3.0517e-05"""

rollout200 = """Point: 0
Permutation: [0, 1, 2, 3, 4, 5]
Theta: [[1.14061868e-04]
 [2.35845582e-04]
 [2.90530320e-05]
 [2.52134583e+00]]
Error: 2.9714e-05
Point: 1
Permutation: [0, 1, 2, 3, 5, 4]
Theta: [[1.14061868e-04]
 [4.11977480e-05]
 [7.31840187e-05]
 [2.79562970e+00]]
Error: 3.0595e-05
Point: 2
Permutation: [0, 1, 2, 4, 3, 5]
Theta: [[1.14061868e-04]
 [6.47773152e-05]
 [8.81164757e-05]
 [2.72455292e+00]]
Error: 3.0353e-05
Point: 3
Permutation: [0, 1, 2, 4, 5, 3]
Theta: [[1.14061868e-04]
 [1.98385753e-05]
 [7.31840187e-05]
 [2.80218741e+00]]
Error: 3.0672e-05
Point: 4
Permutation: [0, 1, 2, 5, 3, 4]
Theta: [[1.14061868e-04]
 [3.79399584e-05]
 [8.81164757e-05]
 [2.83067897e+00]]
Error: 3.0518e-05
Point: 5
Permutation: [0, 1, 2, 5, 4, 3]
Theta: [[1.14061868e-04]
 [1.13843429e-04]
 [2.90530320e-05]
 [2.85253488e+00]]
Error: 3.0531e-05
Point: 6
Permutation: [0, 1, 3, 2, 4, 5]
Theta: [[1.14061868e-04]
 [6.13697678e-05]
 [8.23386633e-05]
 [2.74297219e+00]]
Error: 3.0442e-05
Point: 7
Permutation: [0, 1, 3, 2, 5, 4]
Theta: [[ 1.14061868e-04]
 [ 3.75309916e-05]
 [-8.39432842e-05]
 [ 2.79937491e+00]]
Error: 3.0533e-05
Point: 8
Permutation: [0, 1, 3, 4, 2, 5]
Theta: [[1.14061868e-04]
 [3.21659044e-05]
 [8.81164757e-05]
 [2.78911325e+00]]
Error: 3.0539e-05
Point: 9
Permutation: [0, 1, 3, 4, 5, 2]
Theta: [[ 1.14061868e-04]
 [-4.12313327e-05]
 [-8.39432842e-05]
 [ 2.80487504e+00]]
Error: 3.0517e-05
"""


def main():
    # identify_model()
    # plot_identification()
    # rollout()
    plot_rollout()


if __name__ == '__main__':
    main()
