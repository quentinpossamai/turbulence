import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data.blackbird.blackbird_loading import BlackbirdDataset, load_flight, SPLITTING_CONFIG_PATH
from src.drone_models.learned_quadrotor import Quadrotor
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist

ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
mkdir_if_not_exist([ANALYSIS_PATH])
use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def loss(state, dstate_hat, next_state_gt, dt):
    mse_loss = nn.MSELoss()
    # Without the angles
    next_state_hat = state + dstate_hat * dt

    # Todo use cos sin
    angle_loss = (torch.mean(torch.fmod(next_state_hat[:, 0] - next_state_gt[:, 0], np.pi) ** 2) +
                  torch.mean(torch.fmod(next_state_hat[:, 1] - next_state_gt[:, 1], np.pi / 2) ** 2) +
                  torch.mean(torch.fmod(next_state_hat[:, 2] - next_state_gt[:, 2], np.pi) ** 2))

    angular_velocity_loss = mse_loss(next_state_hat[:, 3:6], next_state_gt[:, 3:6])
    linear_speed_loss = mse_loss(next_state_hat[:, 6:9], next_state_gt[:, 6:9])
    position_loss = mse_loss(next_state_hat[:, 9:], next_state_gt[:, 9:])
    return angle_loss + angular_velocity_loss + linear_speed_loss + position_loss
    # return 10. * angle_loss + angular_velocity_loss + linear_speed_loss + 1/10. * position_loss


def validate(model, dataloader, global_iteration, epoch):
    model.eval()
    loss_log = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation: {epoch}"):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            next_state = batch["next_state"].to(device)
            dt = batch["next_t"].to(device) - batch["t"].to(device)
            output = model(state, motor)  # dstate
            validation_loss = loss(state=state, dstate_hat=output, next_state_gt=next_state, dt=dt)
            loss_log.append(validation_loss)
    return torch.mean(torch.tensor(loss_log).float())


def si_blackbird_quadrotor():
    # Training parameters
    batch_size = 64000
    epochs = 100
    memory = 100000

    # Tensorboard
    # /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/tensorboard/
    # tensorboard --logdir /Users/quentin/phd/turbulence/analysis/blackbird_learned_quadrotor_training/training/
    # --load_fast=false --samples_per_plugin scalars=999999999
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = now + f"_batch_size_{batch_size}"
    training_path = ANALYSIS_PATH + f"training/{name}/"
    print("Tensorboard logdir: ", training_path)
    writer = SummaryWriter(training_path)

    model = Quadrotor().to(device)
    train_data = DataLoader(BlackbirdDataset("train"), batch_size=batch_size, shuffle=True, num_workers=1,
                            pin_memory=True)
    validation_data = DataLoader(BlackbirdDataset("validation"), batch_size=batch_size, shuffle=True, num_workers=1,
                                 pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    global_iteration = 0
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(tqdm(train_data, desc=f"Epoch: {epoch}")):
            state = batch["state"].to(device)
            motor = batch["motor"].to(device)
            next_state = batch["next_state"].to(device)
            dt = batch["next_t"].to(device) - batch["t"].to(device)
            output = model(state, motor)  # dstate
            training_loss = loss(state=state, dstate_hat=output, next_state_gt=next_state, dt=dt)

            writer.add_scalars("Loss", {"Training loss": training_loss.item()}, global_step=global_iteration)

            optimizer.zero_grad()
            training_loss.backward()
            optimizer.step()
            global_iteration = global_iteration + 1

        validation_loss = validate(model, validation_data, global_iteration, epoch)
        writer.add_scalars("Loss", {"Validation loss": validation_loss.item()}, global_step=global_iteration)
        if validation_loss < memory:
            memory = validation_loss
            torch.save(model.state_dict(), training_path + f"{global_iteration}_weights.pt")


def test_model_on_training():
    weights = torch.load(ANALYSIS_PATH + "training/20210615_114637_batch_size_64000/weights.pt")
    print(weights)
    model = Quadrotor().to(device)
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

    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    flight = f"{flight_name}/{flight_yaw}/{flight_speed}/"
    data = load_flight(flight_name, flight_yaw, flight_speed)
    splitting_config = pickle.load(open(SPLITTING_CONFIG_PATH, "rb"))  # train_flights, validation_flights, test_flights
    assert flight in splitting_config["train_flights"]
    print("Testing on training")

    # return {"t": data[0:1], "state": data[1:13], "motor": data[13:17], "imu": data[17:23], "next_t": data[23:24],
    #         "dstate": data[24:36], "next_state": data[36:]}
    state = torch.from_numpy(data[:, 1:13])
    motor = torch.from_numpy(data[:, 13:17])
    time = torch.from_numpy(data[:, 23])
    dt = torch.from_numpy(data[:, 23]) - torch.from_numpy(data[:, 0])
    next_state = torch.from_numpy(data[:, 36:])
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


if __name__ == "__main__":
    # si_blackbird_quadrotor()
    test_model_on_training()
