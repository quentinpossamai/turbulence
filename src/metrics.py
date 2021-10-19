import torch
import os
import pickle
from tqdm import tqdm

import src.data.blackbird.loading as blackbird
import src.data.euroc_mav.loading as euroc


def load_blackbird_flights(mode):
    # assert mode in ["validation", "test"]

    # Dataset location
    datadir = "/storage/Datasets/BlackBird/"  # Dataset location on cluster
    if not os.path.isdir(datadir):
        datadir = "/Volumes/Samsung_T5/BlackBird/blackbird/"  # Dataset location (Steeven)
    if not os.path.isdir(datadir):
        datadir = blackbird.DATASET_PATH  # Dataset location (Quentin)
    assert os.path.isdir(datadir)

    # Load test/val flights
    with open(datadir + mode + ".txt", "r") as file:
        flight_list = file.read().split("\n")[:-1]
    flights = {}
    for flight in flight_list:
        flight_data = pickle.load(open(f"{datadir}intermediate/{flight}data.pkl", 'rb'))
        flights[flight] = flight_data
    return flights


def load_euroc_mav_flights(mode):
    assert mode in ["train", "validation", "test"]
    split = [e.split(",") for e in open(euroc.INTERMEDIATE_DATASET_PATH + f"split.txt", "r")]
    split = {e[0]: (int(e[1]), int(e[2]), int(e[3])) for e in split}
    flight_list = list(split.keys())

    flights = {flight: pickle.load(open(euroc.INTERMEDIATE_DATASET_PATH + flight + mode + ".pkl",
                                        "rb")) for flight in flight_list}
    return flights


def mse_error(model, horizon, past, dataset, split):
    """ Return the mean squared error on the selected split, controlled by the horizon
    args:
    :param model: f(s_i, u_i) : the predictor (expect a Pytorch Module).
                        s_i is the state at time i (p, q, v, w) and u_i the RPM
                        Should return the future state : s_(i+1).
                            s_i.shape = (Batch, 13)
                            u_i.shape(Batch, 4)
    :param horizon: The prediction horizon (1, 10, 100, 1s, 2s, ...)
    :param past: The number of past elements given as an input to the model.
    :param dataset: The name of the dataset to use. euroc_mav or blackbird
    :param split: Allows to switch between the train, validation, test set.
    """

    assert split in ["train", "validation", "test"]
    assert dataset in ["blackbird", "euroc_mav"]

    if dataset == "blackbird":
        flights = load_blackbird_flights(split)
    else:
        assert dataset == "euroc_mav"
        flights = load_euroc_mav_flights(split)

    # Compute Error
    error_per_flight = {flight_name: 0 for flight_name in flights.keys()}

    for k in tqdm(flights.keys()):  # Extract flight states and command
        state = torch.FloatTensor(flights[k]["state"])
        motor = torch.FloatTensor(flights[k]["motor"])
        time = torch.FloatTensor(flights[k]['t'])
        depth_emb = torch.FloatTensor(flights[k]['depth_emb'])
        norm_emb = torch.FloatTensor(flights[k]['norm_emb'])

        # Todo
        sh = list(cropped_states.shape)
        sh[-1] = 1
        dt = torch.FloatTensor([[1 / 200]]).repeat(sh)

        error_per_flight[k] = _compute_error(
            model=model,
            horizon=horizon,
            past=past,
            state=state,
            motor=motor,
            time=time,
            depth_emb=depth_emb,
            norm_emb=norm_emb,
        )

    # Sum of error for all flights
    total_error = torch.FloatTensor(list(error_per_flight.values())).mean()
    return total_error


def _compute_error(**kwargs):
    model = kwargs['model']
    horizon = kwargs['horizon']
    past = kwargs['past']
    state = kwargs['state']
    motor = kwargs['motor']
    time = kwargs['time']
    dt = kwargs['dt']
    depth_emb = kwargs['depth_emb']
    norm_emb = kwargs['norm_emb']

    crop_idx = (time.shape[0] - 1) // (past + horizon)  # remove last time step of states to fit N*(H+1) points

    # Batch-ify the trajectories to speed up computation
    # First step is to crop the trajectory to an whole number of horizon
    # Next, tensor are reshaped in batch of h-length array
    motor_size = motor.shape[1]
    cropped_states = state[:crop_idx * (past + horizon)].view(-1, past + horizon, 13)
    cropped_motor = motor[:crop_idx * (past + horizon)].view(-1, past + horizon, motor_size)
    cropped_depth_emb = depth_emb[:crop_idx * (past + horizon)].view(-1, past + horizon, 128)
    cropped_norm_emb = norm_emb[:crop_idx * (past + horizon)].view(-1, past + horizon, 128)
    cropped_dt = dt[:crop_idx * (past + horizon)].view(-1, past + horizon, 1)
    error, error_rollout, pred = model.rollout(
            motor=cropped_motor,
            state=cropped_states,
            dt=cropped_dt,
            horizon=horizon,
            depth_emb=cropped_depth_emb,
            norm_emb=cropped_norm_emb,
        )
    return error


def dummy_f(s_i, u_i, dt):
    return s_i.clone()


def visualise_metrics(f, h, mode="test", plot_range=None):
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

    flights = load_blackbird_flights(mode)

    # Compute Error

    for k in tqdm(flights.keys()):  # Extract flight states and command
        states = torch.FloatTensor(flights[k]["state"])
        rpm = torch.FloatTensor(flights[k]["motor"])
        time = torch.FloatTensor(flights[k]['t'])

        crop_idx = (time.shape[0] - 1) // (h + 1)  # remove last time step of states to fit N*H points

        ground_truth = states[:crop_idx * (h + 1)].view(-1, h + 1, 13)  # Reformat with batch to go faster
        command = rpm[:crop_idx * (h + 1)].view(-1, h + 1, 4)
        dt = (time[1:] - time[:-1])[:crop_idx * (h + 1)].view(-1, h + 1)

        with torch.no_grad():
            # Prediction rollout
            prediction = [ground_truth[:, 0], ]
            for i in range(1, h + 1):
                prediction.append(f(prediction[-1], command[:, i], dt[:, i].unsqueeze(-1)))

        prediction = torch.stack(prediction, dim=1).view(-1, 13)
        if plot_range is not None:
            states = states[plot_range[0]:plot_range[1]]
            prediction = prediction[plot_range[0]:plot_range[1]]

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
    visualise_metrics(dummy_f, 100, "test", plot_range=[15000, 15500])
