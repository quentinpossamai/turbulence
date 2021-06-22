import torch
from torch.nn import MSELoss

from src.data.blackbird.loading import BlackbirdDatasetHorizon, load_flight, DATASET_PATH
from src.drone_models.mlp_quadrotor import MLPQuadrotor
from src.trajectory import compare_traj
from src.utils.quaternion import qmul
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist
from src.metrics import mse_error, visualise_metrics

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
mkdir_if_not_exist([ANALYSIS_PATH])


def test_mlp():
    # train_flights, validation_flights, test_flights
    flight_train_list = list(e[:-2] for e in open(DATASET_PATH + "train.txt", "r"))
    flight_validation_list = list(e[:-2] for e in open(DATASET_PATH + "validation.txt", "r"))

    weights = torch.load(ANALYSIS_PATH + "training/20210621_145132_MLPQuadrotor_bs_64000/11136_weights.pt")
    # print(weights)
    model = MLPQuadrotor()
    model.load_state_dict(weights)
    model.eval()

    # # Training flight
    # flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    # flight = f"{flight_name}/{flight_yaw}/{flight_speed}"
    # data = load_flight(flight_name, flight_yaw, flight_speed)
    # assert flight in flight_train_list
    # print("Testing on training")
    #
    # # return {"t": data[0:1], "state": data[1:14], "motor": data[14:18], "imu": data[18:24], "next_t": data[24:25],
    # # "next_state": data[25:]}
    # state = torch.from_numpy(data["state"][:-1])
    # motor = torch.from_numpy(data["motor"][:-1])
    # time = torch.from_numpy(data["t"][:-1])
    # next_time = torch.from_numpy(data["t"][1:])
    # next_state = torch.from_numpy(data["state"][1:])
    # dt = (next_time - time)
    # dt = 1 / 360. * torch.ones(dt)
    #
    # # Test MLP formula
    # angular_velocity = state[:, 10:]
    # quat = state[:, 3:7]
    # dquat_hat = 1 / 2 * qmul(quat, torch.cat([angular_velocity, torch.zeros(len(state), 1)], dim=1))
    # quat_hat = torch.cumsum(dquat_hat * dt.view((-1, 1)), dim=0) + quat[:1, :]
    #
    # # Without integration
    # with torch.no_grad():
    #     dstate_hat = model(state, motor)
    # state_hat = state + dstate_hat * dt.unsqueeze(-1)  # Integrating with Euler

    # # Integrating with Fourier
    # period = torch.mean(dt).item()
    # n = len(dstate_hat)
    # frequencies = fftfreq(n, d=period) + 0.0000001
    # fourier_dstate_hat = fft(dstate_hat.numpy(), axis=0)
    # fourier_state_hat = fourier_dstate_hat / (2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)))
    # state_hat = np.real(ifft(fourier_state_hat, axis=0)) + state[0, :].numpy().reshape((1, 13))

    # compare_traj(next_state[1:], state_hat, next_time, 0, 360)
    #
    # # With integration
    # first_state_idx = 4000
    # traj = [state[first_state_idx]]
    # with torch.no_grad():
    #     for i in range(len(state) - 1 - first_state_idx):
    #         traj.append(model(traj[-1].unsqueeze(0), motor[i, :].unsqueeze(0)).squeeze() * dt[i] + traj[-1])
    # traj_stack = torch.stack(traj, dim=0)
    #
    # start = 0
    # compare_traj(next_state[first_state_idx:], traj_stack, next_time[first_state_idx:], start, start + 360)
    #
    # print()

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
    # dt = 1 / 360. * torch.ones_like(dt)

    # Without integration
    with torch.no_grad():
        next_state_hat = model(state, motor, dt.reshape((-1, 1)))

    # With integration
    traj = [state[0]]
    with torch.no_grad():
        for i in range(len(state) - 1):
            traj.append(model(traj[-1].unsqueeze(0), motor[i, :].unsqueeze(0), dt[i].reshape((1, 1))).squeeze())
    traj_stack = torch.stack(traj, dim=0)

    start = 0
    compare_traj(next_state, traj_stack, next_time, start, start + 360)
    compare_traj(next_state, next_state_hat, next_time, start, start + 360)

    print(mse_error(model, 100, "train"))



def test_mlp_metrics():
    weights = torch.load(ANALYSIS_PATH + "training/20210618_171105_MLPQuadrotor_bs_64000/5336_weights.pt")
    model = MLPQuadrotor()
    model.load_state_dict(weights)
    model.eval()
    print(mse_error(model, 100, "train"))
    # visualise_metrics(f, 1000, mode="test", plotrange=[15000, 16000])


if __name__ == '__main__':
    test_mlp()
    # test_mlp_metrics()
