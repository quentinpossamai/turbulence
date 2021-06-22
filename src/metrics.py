import torch
import os
import pickle
from tqdm import tqdm

from src.data.blackbird.loading import DATASET_PATH


def load_flights(mode):
    # assert mode in ["validation", "test"]

    ## Dataset location
    datadir = "/storage/Datasets/BlackBird/"  # Dataset location on cluster
    if not os.path.isdir(datadir):
        datadir = "/Volumes/Samsung_T5/BlackBird/"  # Dataset location (Steeven)
    if not os.path.isdir(datadir):
        datadir = DATASET_PATH  # Dataset location (Quentin)
    assert os.path.isdir(datadir)

    ## Load test/val flights
    with open(datadir + mode + ".txt", "r") as file:
        flight_list = file.read().split("\n")[:-1]
    flights = {}
    for flight in flight_list:
        flight_data = pickle.load(open(f"{datadir}intermediate/{flight}data.pkl", 'rb'))
        flights[flight] = flight_data
    return flights


def mse_error(f, horizon, mode="test"):
    """ Return the mean squared error on the selected split, controlled by the horizon
    args:
        - f(s_i, u_i) : the predictor (expect a Pytorch Module).
                        s_i is the state at time i (p, q, v, w) and u_i the RPM
                        Should return the future state : s_(i+1).
                            s_i.shape = (Batch, 13)
                            u_i.shape(Batch, 4)
        - H : The prediction horizon (1, 10, 100, 1s, 2s, ...)
        - mode : allows to switch between the test and the validation dataset"""

    # assert mode in ["validation", "test"]

    flights = load_flights(mode)

    # Compute Error
    error_per_flight = {flight_name: 0 for flight_name in flights.keys()}

    for k in tqdm(flights.keys()):  # Extract flight states and command
        states = torch.FloatTensor(flights[k]["state"])
        rpm = torch.FloatTensor(flights[k]["motor"])
        time = torch.FloatTensor(flights[k]['t'])
        error_per_flight[k] = _compute_error(states, rpm, time, horizon, f)

    # Sum of error for all flights
    total_error = torch.FloatTensor(list(error_per_flight.values())).mean()
    return total_error


def _compute_error(states, rpm, time, H, f):
    crop_idx = (time.shape[0] - 1) // (H + 1)  # remove last timestep of states to fit N*H points

    # Batchify the trajectories to speed up computation
    # First step is to crop the trajectory to an whole number of horizon
    # Next, tensor are reshaped in batch of h-length array
    ground_truth = states[:crop_idx * (H + 1)].view(-1, H + 1, 13)
    command = rpm[:crop_idx * (H + 1)].view(-1, H + 1, 4)
    dt = (time[1:] - time[:-1])[:crop_idx * (H + 1)].view(-1, H + 1)
    with torch.no_grad():
        ## Prediction rollout
        prediction = [ground_truth[:, 0], ]
        for i in range(1, H + 1):
            prediction.append(f(prediction[-1], command[:, i], dt[:, i].unsqueeze(-1)))

    prediction = torch.stack(prediction, dim=1)
    error = ((prediction - ground_truth) ** 2).mean()
    return error


def dummy_f(s_i, u_i, dt):
    return s_i.clone()


def visualise_metrics(f, H, mode="test", plotrange=None):
    """ Create visualization of the prediction made by the model
    args:
        - f(s_i, u_i) : the predictor (expect a Pytorch Module).
                        s_i is the state at time i (p, q, v, w) and u_i the RPM
                        Should return the future state : s_(i+1).
                            s_i.shape = (Batch, 13)
                            u_i.shape(Batch, 4)
        - H : The prediction horizon (1, 10, 100, 1s, 2s, ...)
        - mode : allows to switch between the test and the validation dataset
        - range : (min, max) crop the trajectories between these two bounds"""

    import matplotlib.pyplot as plt
    plt.style.use("seaborn")

    assert mode in ["validation", "test"]

    flights = load_flights(mode)

    # Compute Error

    for k in tqdm(flights.keys()):  # Extract flight states and command
        states = torch.FloatTensor(flights[k]["state"])
        rpm = torch.FloatTensor(flights[k]["motor"])
        crop_idx = states.shape[0] // (H + 1)  # remove last timestep of states to fit N*H points

        ground_truth = states[:crop_idx * (H + 1)].view(-1, H + 1, 13)  # Reformat with batch to go faster
        command = rpm[:crop_idx * (H + 1)].view(-1, H + 1, 4)

        ## Prediction rollout
        prediction = [ground_truth[:, 0], ]
        for i in range(1, H + 1):
            prediction.append(f(prediction[-1], command[:, i]))

        prediction = torch.stack(prediction, dim=1).view(-1, 13)
        if plotrange is not None:
            states = states[plotrange[0]:plotrange[1]]
            prediction = prediction[plotrange[0]:plotrange[1]]

        fig = plt.figure()
        ax_pos = [fig.add_subplot(4, 4, 1), fig.add_subplot(4, 4, 2), fig.add_subplot(4, 4, 3)]
        ax_quat = [fig.add_subplot(4, 4, 5), fig.add_subplot(4, 4, 6), fig.add_subplot(4, 4, 7),
                   fig.add_subplot(4, 4, 8)]
        ax_vel = [fig.add_subplot(4, 4, 9), fig.add_subplot(4, 4, 10), fig.add_subplot(4, 4, 11)]
        ax_rpy = [fig.add_subplot(4, 4, 13), fig.add_subplot(4, 4, 14), fig.add_subplot(4, 4, 15)]

        t = torch.arange(states.shape[0]) * 1 / 260
        ax_pos[0].set_title("X", pad=0, fontsize=8)
        ax_pos[1].set_title("Y", pad=0, fontsize=8)
        ax_pos[2].set_title("Z", pad=0, fontsize=8)

        ax_quat[0].set_title("Qx", pad=0, fontsize=8)
        ax_quat[1].set_title("Qy", pad=0, fontsize=8)
        ax_quat[2].set_title("Qz", pad=0, fontsize=8)
        ax_quat[3].set_title("Qw", pad=0, fontsize=8)

        ax_vel[0].set_title("Vx", pad=0, fontsize=8)
        ax_vel[1].set_title("Vy", pad=0, fontsize=8)
        ax_vel[2].set_title("Vz", pad=0, fontsize=8)

        ax_rpy[0].set_title("Wx", pad=0, fontsize=8)
        ax_rpy[1].set_title("Wy", pad=0, fontsize=8)
        ax_rpy[2].set_title("Wz", pad=0, fontsize=8)
        for i in range(3):
            ax_pos[i].plot(t, states[:, i], label="GT")
            ax_pos[i].plot(t, prediction[:, i], label="Prediction")
            ax_vel[i].plot(t, states[:, 7 + i])
            ax_vel[i].plot(t, prediction[:, 7 + i])
            ax_rpy[i].plot(t, states[:, 10 + i])
            ax_rpy[i].plot(t, prediction[:, 10 + i])
            ax_pos[i].tick_params(axis='both', labelsize=5)
            ax_vel[i].tick_params(axis='both', labelsize=5)
            ax_rpy[i].tick_params(axis='both', labelsize=5)

        for i in range(4):
            ax_quat[i].plot(t, states[:, 3 + i])
            ax_quat[i].plot(t, prediction[:, 3 + i])
            ax_quat[i].tick_params(axis='both', labelsize=5)

        handles, labels = ax_pos[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.8, 0.85), frameon=True)
        plt.tight_layout()

        plt.show()


if __name__ == '__main__':
    mse_error(dummy_f, 10, "test")
