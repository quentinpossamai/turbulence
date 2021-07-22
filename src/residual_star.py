import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch

import src.data.euroc_mav.loading as euroc
from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel


def plot_residual_torsor(time, residual_torsor, title, xlim: tuple):
    plt.rcParams.update({"text.usetex": False,
                         "font.family": "serif",
                         "font.sans-serif": ["Computer Modern Roman"]})
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    start = np.where(time >= xlim[0])[0][0]
    stop = np.where(time <= xlim[1])[0][-1]

    [ax.plot(time[start:stop], residual_torsor[start:stop, i], label=name) for i, name in enumerate(["x", "y", "z"])]
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("fa [m/s2]")
    ax.set_xlim(xlim)
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    [ax.plot(time[start:stop], residual_torsor[start:stop, 3 + i],
             label=name) for i, name in enumerate(["x", "y", "z"])]
    ax.grid()
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("tau_a [rad/s2]")
    ax.set_xlim(xlim)
    ax.legend()
    plt.show()
    plt.rcParams.update({"text.usetex": False,
                         "font.family": ['sans-serif'],
                         "font.sans-serif": ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif',
                                             'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica',
                                             'Avant Garde', 'sans-serif']})


def euroc_mav_residual():
    model = DynamicalModel(paper=False)
    for flight in euroc.FLIGHT_LIST:
        # Importing data
        data = euroc.load_flight(flight)
        time = torch.from_numpy(data["t"][:-1])
        motor = torch.from_numpy(data["motor"][:-1])
        dt = torch.tensor([1 / 200], dtype=torch.float64).view((1, 1))
        state = torch.from_numpy(data["state"][:-1])
        next_state = torch.from_numpy(data["state"][1:])

        # Test-1 Quaternion norm is 1 and standard deviation is 1e-18
        # state[:, 3:7] = state[:, 3:7] / torch.norm(state[:, 3:7], dim=1).view(-1, 1)
        # next_state[:, 3:7] = next_state[:, 3:7] / torch.norm(next_state[:, 3:7], dim=1).view(-1, 1)
        norm = torch.norm(state[:, 3:7], dim=1)
        print("Quaternion's norm mean: ", torch.mean(norm))
        print("Quaternion's norm std: ", torch.std(norm))

        # Test-2 integrating perfect derivative lead to perfect pose
        assert abs((state[1:, :3] - state[:-1, :3]) / dt - state[1:, 7:10]).sum() == 0

        # Computing residual
        residual_torsor = model.residual_dynamic(motor, state, next_state, dt)
        data["residual"] = torch.cat([torch.zeros((1, 6)), residual_torsor], dim=0).numpy()

        # plot_residual_torsor(time, data["residual"], flight, xlim=(0, time[-1]))

        # Saving
        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
        pickle.dump(data, open(intermediate_flight_path, "wb"))
    euroc.splitting()


def test_euroc():
    model = DynamicalModel(paper=False).double()
    for flight in euroc.FLIGHT_LIST:
        # Importing data
        data = euroc.load_flight(flight)
        time = torch.from_numpy(data["t"][:-1])
        motor = torch.from_numpy(data["motor"][:-1])
        # dt = torch.from_numpy(data["t"][1:] - data["t"][:-1]).view(-1, 1)
        dt = torch.tensor([1 / 200], dtype=torch.float64).view((1, 1))
        state = torch.from_numpy(data["state"][:-1])
        next_state = torch.from_numpy(data["state"][1:])
        residual_torsor = torch.from_numpy(data["residual"][1:])

        next_state_hat = model(state, motor, dt, residual_torsor)
        print(torch.abs(next_state - next_state_hat).mean(dim=0))
        print(next_state.allclose(next_state_hat))

        traj = [state[0].reshape((1, -1))]
        horizon = 10000
        for i in range(horizon - 1):
            traj.append(model(traj[-1], motor[i].reshape((1, -1)),
                              dt, residual_torsor[i].reshape((1, -1))).reshape((1, -1)))
        traj = np.concatenate(traj, axis=0)
        plt.plot(state[:horizon, 3:7])
        plt.plot(traj[:, 3:7])
        abs(state[:horizon, 3:7] - traj[:, 3:7]).mean()


if __name__ == '__main__':
    euroc_mav_residual()
    test_euroc()
