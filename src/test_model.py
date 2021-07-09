import numpy as np
import torch

# Data
import src.data.blackbird.loading as blackbird
import src.data.euroc_mav.loading as euroc

# Models
from src.drone_models.analytical_models import quadrotor_4parms, hexarotor_4params
from src.drone_models.mlp import iii_mlp_past
from src.drone_models.mlp.ii_mlp import MLP
from src.drone_models.mlp.i_old_mlp_quadrotor import OldMLPQuadrotor
import src.utils.trajectory
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist
from src import metrics

BLACKBIRD_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
EUROC_MAV_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_learned_hexarotor_training/"
mkdir_if_not_exist([BLACKBIRD_ANALYSIS_PATH])


def test_blackbird_mlp(horizon, train_validation_test, model_name):
    assert train_validation_test in ["train", "validation", "test"]

    weights = torch.load(BLACKBIRD_ANALYSIS_PATH + f"training/{model_name}")
    model = MLP(motor_size=4)
    model.load_state_dict(weights)
    model.eval()

    flight = list(e[:-2] for e in open(blackbird.DATASET_PATH + f"{train_validation_test}.txt", "r"))[0]
    print(f"Testing on {train_validation_test}")
    flight_name, flight_yaw, flight_speed = flight.split("/")
    data = blackbird.load_flight(flight_name, flight_yaw, flight_speed)

    state = torch.from_numpy(data["state"][:-1]).float()
    motor = torch.from_numpy(data["motor"][:-1]).float()
    time = torch.from_numpy(data["t"][:-1]).float()
    next_time = torch.from_numpy(data["t"][1:]).float()
    next_state = torch.from_numpy(data["state"][1:]).float()
    dt = (next_time - time).float()

    next_state_hat = src.utils.trajectory.gen_traj(model, state, motor, dt, horizon=horizon)

    start = 0
    if horizon == 1:
        length = 100
    else:
        length = horizon
    src.utils.trajectory.compare_traj(next_state, next_state_hat, next_time, start, start + length,
                                      train_validation_test,
                                      EUROC_MAV_ANALYSIS_PATH + f"training/{model_name.split('/')[0]}/"
                                                                f"traj_{train_validation_test}_horizon{horizon}.svg")
    # visualise_metrics(model, 500, "test", plot_range=[15000, 15500])


def test_euroc_mav_mlp(horizon, train_validation_test, model_name):
    assert train_validation_test in ["train", "validation", "test"]
    assert horizon < 3156, "horizon too big, maximum possible horizon is 3156"
    train_flight = euroc.EurocMAVDatasetHorizon("train", traj_len=3155)[0]
    validation_flight = euroc.EurocMAVDatasetHorizon("validation", traj_len=3155)[0]
    test_flight = euroc.EurocMAVDatasetHorizon("test", traj_len=3155)[0]

    weights = torch.load(EUROC_MAV_ANALYSIS_PATH + f"training/{model_name}")
    model = MLP(motor_size=6)
    model.load_state_dict(weights)
    model.eval()

    if train_validation_test == "train":
        data = train_flight
    elif train_validation_test == "validation":
        data = validation_flight
    else:
        data = test_flight
    state = data["state"][:-1]
    motor = data["motor"][:-1]
    time = data["t"][:-1]
    next_state = data["state"][1:]
    next_time = data["t"][1:]
    dt = next_time - time

    next_state_hat = src.utils.trajectory.gen_traj(model, state, motor, dt, horizon=horizon)

    start = 0
    if horizon == 1:
        length = 100
    else:
        length = horizon
    src.utils.trajectory.compare_traj(next_state, next_state_hat, next_time, start, start + length,
                                      train_validation_test,
                                      EUROC_MAV_ANALYSIS_PATH + f"training/{model_name.split('/')[0]}/"
                                                                f"traj_{train_validation_test}_horizon{horizon}.svg")


def test_blackbird_mlp_metrics(model_name):
    weights = torch.load(BLACKBIRD_ANALYSIS_PATH + f"training/{model_name}")
    model = MLP(motor_size=4)
    model.load_state_dict(weights)
    model.eval()
    test_model(model, "blackbird", "test", "normal")


def test_euroc_mav_mlp_metrics(model_name):
    weights = torch.load(EUROC_MAV_ANALYSIS_PATH + f"training/{model_name}")
    model = MLP(motor_size=6)
    model.load_state_dict(weights)
    model.eval()
    test_model(model, "euroc_mav", "test", "normal")


def test_model(model, dataset, train_test_validation, mode):
    """
    Print the error on 0.025s, 0.5s and 1s of one model on one dataset split.
    :param model: The model to evaluate.
    :param dataset: The dataset to use for evaluation. "blackbird" or "euroc_mav".
    :param train_test_validation: The split to evaluate on. "train", "validation" or "test".
    :param mode: The split to evaluate on. "normal", "residual" or "past".
    """
    assert train_test_validation in ["train", "validation", "test"]
    assert dataset in ["blackbird", "euroc_mav"]
    assert mode in ["normal", "residual", "past"]
    # Test metric
    res = np.zeros(3)
    if mode == "normal":
        metric_func = metrics.mse_error
    elif mode == "residual":
        metric_func = metrics.mse_error_residual
    else:  # mode == "past"
        metric_func = metrics.mse_error_past

    if dataset == "euroc_mav":
        for i, e_i in enumerate([5, 100, 200]):
            res[i] = (metric_func(model, e_i, 'euroc_mav', train_test_validation).item())
    else:  # dataset == "blackbird"
        for i, e_i in enumerate([9, 180, 360]):
            res[i] = (metric_func(model, e_i, 'blackbird', train_test_validation).item())
    print(f"Model {model.__class__.__name__} on {train_test_validation} "
          f"E(0.025s): {res[0]:0.4e}, E(0.5s): {res[1]:0.4e}, E(1s): {res[2]:0.4e}")


def main():
    # BLACKBIRD
    # phi_paper_param = quadrotor_4parms.DynamicalModel(paper=True)
    # test_model(phi_paper_param, "blackbird", "test", "normal")

    # phi_mse_param = quadrotor_4parms.DynamicalModel(paper=False)
    # test_model(phi_mse_param, "blackbird", "test", "normal")

    # mlp_int = "20210623_113455_OldMLPQuadrotor_bs_64000/8004_weights.pt"
    # test_blackbird_mlp_metrics(mlp_int)

    # mlp_symplectic = "20210621_145132_MLPQuadrotor_bs_64000/11136_weights.pt"
    # test_blackbird_mlp_metrics(mlp_symplectic)

    # EUROC MAV
    # phi_paper_param = hexarotor_4params.DynamicalModel(paper=True)
    # test_model(phi_paper_param, "euroc_mav", "test", "normal")

    # phi_mse_param = hexarotor_4params.DynamicalModel(paper=False)
    # test_model(phi_mse_param, "euroc_mav", "test", "normal")

    # mlp_symplectic = "20210630_144438_MLP_bs_32/9416_weights.pt"
    # test_euroc_mav_mlp_metrics(mlp_symplectic)

    # mlp_symplectic_batch_norm = "20210708_153028_MLP_bs_32/103576_weights.pt"
    # weights = torch.load(EUROC_MAV_ANALYSIS_PATH + f"training/{mlp_symplectic_batch_norm}")
    # n_past = 1
    # model = iii_mlp_past.MLP(6, n_past)
    # model.load_state_dict(weights)
    # model.eval()
    # test_model(model, "euroc_mav", "test", "past")

    mlp_symplectic_past = "20210709_093134_MLP_bs_32/82320_weights.pt"
    weights = torch.load(EUROC_MAV_ANALYSIS_PATH + f"training/{mlp_symplectic_past}")
    n_past = 10
    model = iii_mlp_past.MLP(6, n_past)
    model.load_state_dict(weights)
    model.eval()
    test_model(model, "euroc_mav", "test", "past")

    pass


if __name__ == '__main__':
    main()
