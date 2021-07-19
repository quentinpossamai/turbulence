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
        data = euroc.load_flight(flight)
        time = torch.from_numpy(data["t"][:-1]).float()
        motor = torch.from_numpy(data["motor"][:-1]).float()
        state = torch.from_numpy(data["state"][:-1]).float()
        dt = torch.from_numpy(data["t"][1:] - data["t"][:-1]).float().view(-1, 1)
        next_state = torch.from_numpy(data["state"][1:]).float()
        residual_torsor = model.residual_dynamic(motor, state, next_state, dt)
        print(torch.min(residual_torsor, 0))
        print(torch.max(residual_torsor, 0))

        plot_residual_torsor(time, residual_torsor, flight, xlim=(50, 60))

        data["residual"] = torch.cat([torch.zeros((1, 6)), residual_torsor], dim=0).numpy()

        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
        pickle.dump(data, open(intermediate_flight_path, "wb"))
    euroc.splitting()


if __name__ == '__main__':
    euroc_mav_residual()
