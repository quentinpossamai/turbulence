""" Identify the non-physical parameters of the model:
theta1 = Kt / m
theta2 = L Kt / Ixy
theta3 = Km / Iz
theta4 = Iz / Ixy"""

import itertools
import matplotlib.pyplot as plt
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
import src.data.euroc_mav.loading as euroc
from src.utils.quaternion import qrot
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = f"{GENERAL_ANALYSIS_PATH}euroc_identification/"
mkdir_if_not_exist([ANALYSIS_PATH])

DTYPE = torch.float64
DT = 1 / 200
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
    dt = torch.tensor([[DT]], dtype=DTYPE)

    # Load flights in a big vector column
    state, next_state, motor = [], [], []
    for i in range(len(flights)):
        s = flights[i]["state"]
        m = flights[i]["motor"]
        state.append(s[:-1])
        next_state.append(s[1:])
        motor.append(m[1:])
    state = torch.cat(state, dim=0)
    next_state = torch.cat(next_state, dim=0)
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
        a_speed = torch.cat([dt * thrust_world, torch.zeros(batch_size, 3, 3)], dim=-1)
        b_speed = state[:, 7:10].unsqueeze(-1) - gravity * dt

        # Angular PFD
        wyz = (state[:, 11] * state[:, 12]).unsqueeze(-1)
        wxz = (state[:, 10] * state[:, 12]).unsqueeze(-1)
        theta1_component = dt * torch.cat([motor_forces[:, 0], motor_forces[:, 1], torch.zeros(batch_size, 1)], dim=-1)
        theta2_component = dt * torch.cat([torch.zeros(batch_size, 2), motor_forces[:, 2]], dim=-1)
        theta3_component = dt * torch.cat([-wyz, wxz, torch.zeros(batch_size, 1)], dim=-1)
        a_angular = torch.stack([torch.zeros(batch_size, 3),
                                 theta1_component,
                                 theta2_component,
                                 theta3_component], dim=-1)
        b_angular = state[:, 10:] + dt * torch.cat([wyz, -wxz, torch.zeros(batch_size, 1)], dim=-1)

        # Solve the linear problem
        a = torch.cat([a_speed, a_angular], dim=1).view(-1, 4)
        b = torch.cat([b_speed.squeeze(-1), b_angular], dim=-1).view(-1, 1)
        y = next_state[:, 7:].reshape(-1, 1)
        theta = torch.pinverse(a) @ (y - b)

        error = ((y - (a @ theta + b)) ** 2).mean()
        save_identifications.append({"permutation": p, "theta": theta.clone(), "mse": error})
        pickle.dump(save_identifications, open(f"{ANALYSIS_PATH}identification.pkl", "wb"))


def rollout():
    param_range = pickle.load(open(f"{ANALYSIS_PATH}identification.pkl", "rb"))
    max_horizon = 200
    dataset = euroc.EurocMAVDatasetHorizon("validation", traj_len=max_horizon + 1, dtype=DTYPE, batch_dim=False)
    long_traj = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    for i in range(len(param_range)):
        param_range[i]["rank"] = []

    batch = next(iter(long_traj))
    motor_roll = batch["motor"]
    state_roll = batch["state"]
    dt_shape = list(state_roll.shape)
    dt_shape[-1] = 1
    dt = torch.tensor(DT, dtype=DTYPE).repeat(dt_shape)

    for i, param in enumerate(tqdm(param_range)):
        permutation, theta = param["permutation"], param["theta"]
        model = DynamicalModel(param={"permutation": permutation, "theta": theta}, dtype=torch.float64)
        _, rollout_error, _ = model.rollout(motor=motor_roll, state=state_roll, horizon=max_horizon, dt=dt)
        param_range[i]["rollout_error"] = rollout_error.mean(dim=(0, 2))

    for j, horizon in enumerate(tqdm(range(max_horizon))):
        order = torch.argsort(torch.tensor([e["rollout_error"][j] for e in param_range], dtype=DTYPE))
        for rank, i in enumerate(order):
            param_range[i]["rank"].append(rank)
    pickle.dump(param_range, open(f"{ANALYSIS_PATH}ranking.pkl", "wb"))


def plot_identification():
    param_range = pickle.load(open(f"{ANALYSIS_PATH}identification.pkl", "rb"))
    param_range = sorted(param_range, key=lambda e: e["mse"])

    def print_point(idx):
        e = param_range[idx]
        print(f"Point: {idx}")
        print(f"Permutation: {list(e['permutation'])}")
        print(f"Theta: {e['theta'].numpy()}")
        print(f"Error: {e['mse'].item():0.4e}")

    # Plot
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})

    for i in range(len(param_range)):
        if i == 10:
            break
        print_point(i)

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.plot([e["mse"] for e in param_range], marker="x")
    for i, e in enumerate(param_range):
        ax.text(i, e["mse"], f"{i}")
        # ax.text(i + 10, e[2], f"{e[3]}")

        if e['theta'][0] > 0 and e['theta'][1] > 0 and e['theta'][3] > 0:
            ax.scatter(i, e["mse"], color="red", zorder=3)
    ax.grid()
    ax.set_xlabel("Configuration number")
    ax.set_ylabel("Error on train set")
    plt.show()


def plot_rollout():
    param_range = pickle.load(open(f"{ANALYSIS_PATH}ranking.pkl", "rb"))
    error_tab = torch.zeros(len(param_range), len(param_range[0]["rank"]))
    rank_tab = torch.zeros(len(param_range), len(param_range[0]["rank"]), dtype=int)
    for i, param in enumerate(param_range):
        error_tab[i, :] = param["rollout_error"]
        rank_tab[i, :] = torch.tensor(param["rank"], dtype=torch.int)

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    for i, is_at_some_point_top in enumerate((rank_tab < 5).any(dim=1)):
        if is_at_some_point_top:
            ax.plot(error_tab[i], label=f"{i}")
            # ax.plot(rank_tab[i], label=f"{i}")
    ax.grid()
    ax.set_xlabel("Rollout length")
    ax.set_ylabel("Rollout loss on train set")
    # ax.set_ylabel("Rollout ranking")
    ax.legend()
    # plt.show()

    # Best
    winner = torch.unique(torch.argmin(error_tab, dim=0)[0]).item()  # 431 with pair 506 from plot
    assert (torch.argmin(error_tab, dim=0) == winner).all()
    pair1 = param_range[winner]
    pair2 = param_range[506]
    print(f"pair1 | Permutation {pair1['permutation']} | Theta {pair1['theta']}")
    print(f"pair2 | Permutation {pair2['permutation']} | Theta {pair2['theta']}")


def main():
    # identify_model()
    plot_identification()
    # rollout()
    # plot_rollout()


if __name__ == '__main__':
    main()
