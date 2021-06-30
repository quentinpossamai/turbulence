import torch

import src.data.blackbird.loading as blackbird
import src.data.euroc_mav.loading as euroc
from src.drone_models.mlp.mlp_quadrotor import MLPQuadrotor
from src.drone_models.mlp.old_mlp_quadrotor import OldMLPQuadrotor
import src.utils.trajectory
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist
from src.metrics import mse_error

BLACKBIRD_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
EUROC_MAV_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_learned_quadrotor_training/"
mkdir_if_not_exist([BLACKBIRD_ANALYSIS_PATH])


def test_blackbird_mlp(horizon, train_validation_test, model_name):
    assert train_validation_test in ["train", "validation", "test"]

    weights = torch.load(BLACKBIRD_ANALYSIS_PATH + f"training/{model_name}")
    model = MLPQuadrotor(motor_size=4)
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
    train_flight = euroc.EurocMAVDatasetHorizon("train", horizon=3155)[0]
    validation_flight = euroc.EurocMAVDatasetHorizon("validation", horizon=3155)[0]
    test_flight = euroc.EurocMAVDatasetHorizon("test", horizon=3155)[0]

    weights = torch.load(EUROC_MAV_ANALYSIS_PATH + f"training/{model_name}")
    model = MLPQuadrotor(motor_size=6)
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
    model = MLPQuadrotor(motor_size=4)
    model.load_state_dict(weights)
    model.eval()
    for i in [9, 180, 360]:
        print(f"Metric {i} on test: {mse_error(model, i, 'blackbird', 'test'):.4E}")


def test_euroc_mav_mlp_metrics(model_name):
    weights = torch.load(EUROC_MAV_ANALYSIS_PATH + f"training/{model_name}")
    model = MLPQuadrotor(motor_size=6)
    model.load_state_dict(weights)
    model.eval()

    for i in [5, 100, 200]:
        print(f"Metric {i} on test: {mse_error(model, i, 'euroc_mav', 'test'):.4E}")
    # visualise_metrics(f, 1000, mode="test", plot_range=[15000, 16000])


def main():
    # BLACKBIRD
    # mlp_symplectic_blackbird = "20210621_145132_MLPQuadrotor_bs_64000/11136_weights.pt"
    old_mlp_blackbird = "20210623_113455_OldMLPQuadrotor_bs_64000/8004_weights.pt"
    test_blackbird_mlp_metrics(old_mlp_blackbird)

    # EUROC MAV
    # mlp_symplectic_euroc_mav = "20210630_144438_MLPQuadrotor_bs_32/9416_weights.pt"
    # for horizon in [1, 100]:
    #     for dataset in ["train", "validation", "test"]:
    #         test_euroc_mav_mlp(horizon=horizon, train_validation_test=dataset,
    #                            model_name=model_name)
    # test_euroc_mav_mlp_metrics(model_name=mlp_symplectic_euroc_mav)


if __name__ == '__main__':
    main()
