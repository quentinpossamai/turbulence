import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.fft import fft, fftfreq, ifft
import torch

from src.data.blackbird.loading import BlackbirdDataset, load_flight, TORCH_DATASET_PATH
from src.drone_models.learned_sabatino import Quadrotor
from src.drone_models.mlp_quadrotor import MLPQuadrotor
from src.trajectory import compare_traj
from src.utils.quaternion import qmul
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
mkdir_if_not_exist([ANALYSIS_PATH])


def test_learned_quadrotor_on_training():
    weights = torch.load(ANALYSIS_PATH + "training/20210615_111629_batch_size_64000/weights.pt")
    print(weights)
    model = Quadrotor()
    model.load_state_dict(weights)
    model.eval()

    # OrderedDict([('a', tensor([0.0024, 0.0081, 0.0015])),
    #              ('b', tensor([[-0.0016,  0.0002,  0.0026, -0.0023],
    #                            [ 0.0009, -0.0010,  0.0002, -0.0027],
    #                            [-0.0004, -0.0020,  0.0037,  0.0024],
    #                            [-0.0025, -0.0006, -0.0015,  0.0012]]))])
    # OrderedDict([('a', tensor([-0.0027, -0.0171,  0.0093])),
    # ('b', tensor([[ 1.2916e-05, -1.3483e-05, -1.3511e-05,  1.4097e-05],
    #         [ 3.9398e-06,  1.8162e-06, -4.1185e-06, -2.6684e-06],
    #         [-1.8498e-06,  1.2445e-06, -3.2462e-06,  3.5668e-06],
    #         [ 6.7813e-06,  7.5055e-06, -2.7151e-06, -5.9875e-06]]))])

    # Training flight
    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    flight = f"{flight_name}/{flight_yaw}/{flight_speed}/"
    data = load_flight(flight_name, flight_yaw, flight_speed)
    splitting_config = pickle.load(open(SPLITTING_CONFIG_PATH, "rb"))  # train_flights, validation_flights, test_flights
    assert flight in splitting_config["train_flights"]
    print("Testing on training")

    # return {"t": data[0:1], "state": data[1:14], "motor": data[14:18], "imu": data[18:24], "next_t": data[24:25],
    #         "dstate": data[25:38], "next_state": data[38:]}
    state = torch.from_numpy(data[:, 1:14])
    motor = torch.from_numpy(data[:, 14:18])
    time = torch.from_numpy(data[:, 24])
    dt = torch.from_numpy(data[:, 24]) - torch.from_numpy(data[:, 0])
    next_state = torch.from_numpy(data[:, 38:])
    with torch.no_grad():
        hat_dstate = model(state, motor)  # predict dstate
        hat_ns = state + dt.reshape((-1, 1)) * hat_dstate

    traj = [state[0, :]]
    with torch.no_grad():
        for i in range(len(state) - 1):
            traj.append(dt[i] * model(traj[-1].reshape((1, -1)),
                                      motor[i, :].reshape((1, -1))).squeeze() + traj[-1])  # predict dstate
    hat_ns = torch.stack(traj)

    start = 0
    stop = 100
    fig, axs = plt.subplots(4, 3, figsize=(19.2, 10.8), dpi=100)
    for i, name in zip(range(4), ["orientation [rad]", "angular_velocity [rad/s]", "linear_speed [m/s]",
                                  "position [m]"]):
        for j, axe in zip(range(3), ["x", "y", "z"]):
            ax = axs[i, j]
            ax.plot(time[start:stop], state[start:stop, i * 3 + j], label="GT")
            ax.plot(time[start:stop], hat_ns[start:stop, i * 3 + j], label="Model")
            ax.set_ylabel(f"{name} {axe}")
            ax.set_xlabel(f"Time [s]")
            ax.grid()
            ax.legend()
    plt.tight_layout()
    plt.show()

    # # State orientation
    # # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    # for i, name in zip(range(3), ("phi", "theta", "psi")):
    #     ax.plot(range(len(data)), data[:, 0 + i], label=name)
    # ax.set_xlabel("Sample number")
    # ax.set_ylabel("State orientation [rad]")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # # plt.savefig(fig_path + "state_orientation.png")
    # plt.show()
    #
    # # State angular velocity
    # # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    # fig, ax = plt.subplots(1, 1)
    # ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    # for i, name in zip(range(3), ("phi", "theta", "psi")):
    #     ax.plot(range(len(data)), data[:, 3 + i], label=name)
    # ax.set_xlabel("Sample number")
    # ax.set_ylabel("State orientation [rad]")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # plt.show()


def test_mlp():
    # train_flights, validation_flights, test_flights
    flight_train_list = list(e[:-2] for e in open(TORCH_DATASET_PATH + "train.txt", "r"))
    flight_validation_list = list(e[:-2] for e in open(TORCH_DATASET_PATH + "validation.txt", "r"))

    weights = torch.load(ANALYSIS_PATH + "training/20210618_171105_MLPQuadrotor_bs_64000/5336_weights.pt")
    # print(weights)
    model = MLPQuadrotor()
    model.load_state_dict(weights)
    model.eval()

    # Training flight
    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    flight = f"{flight_name}/{flight_yaw}/{flight_speed}"
    data = load_flight(flight_name, flight_yaw, flight_speed)
    assert flight in flight_train_list
    print("Testing on training")

    # return {"t": data[0:1], "state": data[1:14], "motor": data[14:18], "imu": data[18:24], "next_t": data[24:25],
    # "next_state": data[25:]}
    state = torch.from_numpy(data["state"][:-1])
    motor = torch.from_numpy(data["motor"][:-1])
    time = torch.from_numpy(data["t"][:-1])
    next_time = torch.from_numpy(data["t"][1:])
    next_state = torch.from_numpy(data["state"][1:])
    dt = (next_time - time)

    # Test MLP formula
    angular_velocity = state[:, 10:]
    quat = state[:, 3:7]
    dquat_hat = 1 / 2 * qmul(quat, torch.cat([angular_velocity, torch.zeros(len(state), 1)], dim=1))
    quat_hat = torch.cumsum(dquat_hat * dt.view((-1, 1)), dim=0) + quat[:1, :]

    # Without integration
    with torch.no_grad():
        dstate_hat = model(state, motor)
    state_hat = state + dstate_hat * dt.unsqueeze(-1)  # Integrating with Euler

    # # Integrating with Fourier
    # period = torch.mean(dt).item()
    # n = len(dstate_hat)
    # frequencies = fftfreq(n, d=period) + 0.0000001
    # fourier_dstate_hat = fft(dstate_hat.numpy(), axis=0)
    # fourier_state_hat = fourier_dstate_hat / (2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)))
    # state_hat = np.real(ifft(fourier_state_hat, axis=0)) + state[0, :].numpy().reshape((1, 13))

    # compare_traj(next_state[1:], state_hat, next_time, 0, 360)

    # With integration
    first_state_idx = 4000
    traj = [state[first_state_idx]]
    with torch.no_grad():
        for i in range(len(state) - 1 - first_state_idx):
            traj.append(model(traj[-1].unsqueeze(0), motor[i, :].unsqueeze(0)).squeeze() * dt[i] + traj[-1])
    traj_stack = torch.stack(traj, dim=0)

    start = 0
    compare_traj(next_state[first_state_idx:], traj_stack, next_time[first_state_idx:], start, start + 360)

    print()

    # # Testing on validation flight
    # flight = splitting_config["validation_flights"][0]
    # flight_name, flight_yaw, flight_speed = flight.split("/")[:-1]
    # data = load_flight(flight_name, flight_yaw, flight_speed)
    # print("Testing on validation")
    # Training flight
    flight = flight_validation_list[0]
    flight_name, flight_yaw, flight_speed = flight.split("/")
    data = load_flight(flight_name, flight_yaw, flight_speed)
    print("Testing on Validation")

    # return {"t": data[0:1], "state": data[1:14], "motor": data[14:18], "imu": data[18:24], "next_t": data[24:25],
    # "next_state": data[25:]}
    state = torch.from_numpy(data["state"][:-1])
    motor = torch.from_numpy(data["motor"][:-1])
    time = torch.from_numpy(data["t"][:-1])
    next_time = torch.from_numpy(data["t"][1:])
    next_state = torch.from_numpy(data["state"][1:])
    dt = (next_time - time)

    # # Without integration
    # with torch.no_grad():
    #     dstate_hat = model(state, motor)
    # state_velocities_only_hat = state + dstate_hat * dt

    # With integration
    first_state_idx = 4000
    traj = [state[first_state_idx]]
    with torch.no_grad():
        for i in range(len(state) - 1 - first_state_idx):
            traj.append(model(traj[-1].unsqueeze(0), motor[i, :].unsqueeze(0)).squeeze() * dt[i] + traj[-1])
    traj_stack = torch.stack(traj, dim=0)

    start = 0
    compare_traj(next_state[first_state_idx:], traj_stack, next_time[first_state_idx:], start, start + 360)


if __name__ == '__main__':
    test_mlp()
