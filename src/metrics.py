import torch
import os
import pickle
from tqdm import tqdm

from src.data.blackbird.loading import DATASET_PATH


def load_flights(mode):
    assert mode in ["validation", "test"]

    ## Dataset location
    datadir = "/storage/Datasets/BlackBird/"  # Dataset location on cluster
    if not os.path.isdir(datadir):
        datadir = "/Volumes/Samsung_T5/BlackBird/"  # Dataset location (Steeven)
    if not os.path.isdir(datadir):
        datadir = DATASET_PATH + "intermediate/"  # TODO : mettre le path vers les données sur le Mac de Quentin
    assert os.path.isdir(datadir)

    ## Load test/val flights
    with open(datadir + mode + ".txt", "r") as file:
        flight_list = file.read().split("\n")[:-1]
    flights = {}
    for flight in flight_list:
        flight_data = pickle.load(open(f"{datadir}intermediate/{flight}data.pkl", 'rb'))
        flights[flight] = flight_data
    return flights


def MSE_error(f, H, mode="test"):
    """ Return the mean squared error on the selected split, controlled by the horizon
    args:
        - f(s_i, u_i) : the predictor (expect a Pytorch Module).
                        s_i is the state at time i (p, q, v, w) and u_i the RPM
                        Should return the future state : s_(i+1).
                            s_i.shape = (Batch, 13)
                            u_i.shape(Batch, 4)
        - H : The prediction horizon (1, 10, 100, 1s, 2s, ...)
        - mode : allows to switch between the test and the validation dataset"""

    assert mode in ["validation", "test"]

    flights = load_flights(mode)

    ## Compute Error
    error_per_flight = {flight_name: 0 for flight_name in flights.keys()}

    for k in flights.keys():  # Extract flight states and command
        states = torch.FloatTensor(flights[k]["state"])
        rpm = torch.FloatTensor(flights[k]["motor"])
        error_per_flight[k] = _compute_error(states, rpm, H, f)

    ## Sum of error for all flights
    total_error = torch.FloatTensor(list(error_per_flight.values())).mean()
    return total_error


def _compute_error(states, rpm, H, f):
    crop_idx = states.shape[0] // (H + 1)  # remove last timestep of states to fit N*H points

    ground_truth = states[:crop_idx * (H + 1)].view(-1, H + 1, 13)  # Reformat with batch to go faster
    command = rpm[:crop_idx * (H + 1)].view(-1, H + 1, 4)

    ## Prediction rollout
    prediction = [ground_truth[:, 0], ]
    for i in range(1, H + 1):
        prediction.append(f(prediction[-1], command[:, i]))

    prediction = torch.stack(prediction, dim=1)
    error = ((prediction - ground_truth) ** 2).mean()
    return error


def dummy_f(s_i, u_i):
    return s_i.clone()


if __name__ == '__main__':
    print(MSE_error(dummy_f, 1))
