import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader

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
    dtype = torch.float64
    model = DynamicalModel(paper=False, dtype=dtype)
    for flight in euroc.FLIGHT_LIST:
        # Importing data
        data = euroc.load_flight(flight)
        time = torch.from_numpy(data["t"][:-1])
        motor = torch.from_numpy(data["motor"][:-1])
        dt = torch.tensor([1 / 200], dtype=dtype).view((1, 1))
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
        residual_torsor = model.residual_dynamic(motor, state, dt, next_state)
        data["residual"] = torch.cat([torch.zeros((1, 6)), residual_torsor], dim=0).numpy()

        # plot_residual_torsor(time, data["residual"], flight, xlim=(0, time[-1]))

        # Saving
        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
        pickle.dump(data, open(intermediate_flight_path, "wb"))
        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/c-ideal.pkl"
        pickle.dump({"butterworth": data["residual"]}, open(intermediate_flight_path, "wb"))
    euroc.splitting()


def _test_euroc():
    dtype = torch.float64
    model = DynamicalModel(paper=False, dtype=dtype)
    for flight in euroc.FLIGHT_LIST:
        # Importing data
        data = euroc.load_flight(flight)
        time = torch.from_numpy(data["t"][:-1]).type(dtype)
        motor = torch.from_numpy(data["motor"][:-1]).type(dtype)
        # dt = torch.from_numpy(data["t"][1:] - data["t"][:-1]).view(-1, 1)
        dt = torch.tensor([[1 / 200]], dtype=dtype)
        state = torch.from_numpy(data["state"][:-1]).type(dtype)
        next_state = torch.from_numpy(data["state"][1:]).type(dtype)
        residual_torsor = torch.from_numpy(data["residual"][1:]).type(dtype)

        next_state_hat = model(motor, state, dt, residual_torsor)
        print(torch.abs(next_state - next_state_hat).mean(dim=0))
        print(next_state.allclose(next_state_hat))

        traj = [state[0].reshape((1, -1))]
        horizon = 10000
        for i in range(horizon - 1):
            traj.append(model(motor[i].reshape((1, -1)), traj[-1],
                              dt, residual_torsor[i].reshape((1, -1))).reshape((1, -1)))
        traj = np.concatenate(traj, axis=0)
        # plt.plot(state[:horizon, 3:7])
        # plt.plot(traj[:, 3:7])
        print(abs(state[:horizon, 3:7] - traj[:, 3:7]).mean())

        # tensor([1.6281e-08, 3.0245e-08, 4.2788e-09, 6.6299e-10, 6.4368e-10, 1.7825e-08,
        #         2.9932e-08, 4.7618e-09, 6.2739e-09, 4.1462e-09, 5.8048e-09, 4.8914e-09,
        #         4.8063e-09])
        # False
        # tensor(1.4273e-06)
        # tensor([2.0686e-08, 2.3317e-08, 5.0654e-09, 1.9938e-09, 2.3139e-09, 2.2066e-08,
        #         2.7499e-08, 1.0400e-08, 1.4242e-08, 5.5348e-09, 7.0967e-09, 6.4880e-09,
        #         8.8543e-09])
        # False
        # tensor(4.8385e-06)
        # tensor([1.8088e-08, 1.9594e-08, 6.1421e-09, 3.7639e-09, 3.9524e-09, 2.1350e-08,
        #         2.8061e-08, 8.4490e-09, 1.0455e-08, 6.2868e-09, 6.6865e-09, 1.0682e-08,
        #         6.6712e-09])
        # False
        # tensor(6.8227e-06)


def _test_euroc_torch_dataset():
    for name in ["train", "validation", "test"]:
        dtype = torch.float64
        dataset = DataLoader(euroc.EurocMAVDatasetHorizon(train_validation_test=name, traj_len=21, dtype=dtype,
                                                          for_loader=True),
                             batch_size=32, shuffle=True)
        model = DynamicalModel(paper=False, dtype=dtype)
        dt = torch.tensor([[1 / 200]], dtype=dtype)
        next_state_hat, next_state_gt = [], []
        residual_hat, residual_gt = [], []
        n = len(euroc.EurocMAVDatasetHorizon(train_validation_test=name, traj_len=21, dtype=dtype, for_loader=True))
        print("len: ", n)
        print()
        for e in dataset:
            # next_state_hat = model(state, motor, dt, residual_torsor)
            next_state_hat.append(model(e["motor"][:, -2], e["state"][:, -2], dt, e["residual"][:, -1]))
            next_state_gt.append(e["state"][:, -1])

            residual_hat.append(model.residual_dynamic(e["motor"][:, -2], e["state"][:, -2], dt, e["state"][:, -1]))
            residual_gt.append(e["residual"][:, -1])

        next_state_hat = torch.cat(next_state_hat, dim=0)
        next_state_gt = torch.cat(next_state_gt, dim=0)
        residual_hat = torch.cat(residual_hat, dim=0)
        residual_gt = torch.cat(residual_gt, dim=0)
        assert len(residual_hat) == n

        # print("mean: ", abs(next_state_hat - next_state_gt).mean())
        # print("std: ", abs(next_state_hat - next_state_gt).std())
        # print()

        # mean:  tensor(1.1272e-08)
        # std:  tensor(2.7786e-08)
        # mean:  tensor(1.1240e-08)
        # std:  tensor(2.6749e-08)
        # mean:  tensor(9.9072e-09)
        # std:  tensor(2.3456e-08)

        print("mean: ", abs(residual_hat - residual_gt).mean())
        print("std: ", abs(residual_hat - residual_gt).std())
        print()

        # For torch.float32 :
        # mean:  tensor(1.7927e-06)
        # std:  tensor(2.3017e-06)
        # mean:  tensor(1.9369e-06)
        # std:  tensor(2.5113e-06)
        # mean:  tensor(1.6900e-06)
        # std:  tensor(2.2114e-06)

        # For torch.float64 0 for each !


def compare_denoising():
    for flight in euroc.FLIGHT_LIST:
        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
        current = pickle.load(open(intermediate_flight_path, "rb"))["residual"]
        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/c-raw.pkl"
        raw = pickle.load(open(intermediate_flight_path, "rb"))["butterworth"]
        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/c-ideal.pkl"
        ideal = pickle.load(open(intermediate_flight_path, "rb"))["butterworth"]
        intermediate_flight_path = euroc.INTERMEDIATE_DATASET_PATH + f"{flight}/c-butterworth.pkl"
        butterworth = pickle.load(open(intermediate_flight_path, "rb"))["butterworth"]

        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        i = 3
        ax.plot(current[:, i], label="Current")
        ax.plot(raw[:, i], label="Raw")
        ax.plot(ideal[:, i], label="Ideal")
        ax.plot(butterworth[:, i], label="Butterworth")
        ax.legend()
        ax.grid()
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Fa [m/s2]")


if __name__ == '__main__':
    # euroc_mav_residual()
    # _test_euroc()
    # _test_euroc_torch_dataset()
    compare_denoising()
