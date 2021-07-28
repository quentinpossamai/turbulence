import matplotlib.pyplot as plt
import numpy as np
import torch

# Data
import src.data.blackbird.loading as blackbird
import src.data.euroc_mav.loading as euroc

# Models
import src.drone_models as drone_models
import src.utils.trajectory
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist
from src import metrics
from src.learning import residual_estimation

BLACKBIRD_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
EUROC_MAV_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_learned_hexarotor_training/"
EUROC_MAV_RESIDUAL = residual_estimation.ANALYSIS_PATH
mkdir_if_not_exist([BLACKBIRD_ANALYSIS_PATH])


def test_model(model, residual_model, past, dataset, train_test_validation):
    """
    Print the error on 0.025s, 0.5s and 1s of one model on one dataset split.
    :param model: The model to evaluate.
    :param residual_model: Model to compute residual of the first one.
    :param past: Number of samples to give to the model
    :param dataset: The dataset to use for evaluation. "blackbird" or "euroc_mav".
    :param train_test_validation: The split to evaluate on. "train", "validation" or "test".
    """
    assert train_test_validation in ["train", "validation", "test"]
    assert dataset in ["blackbird", "euroc_mav"]

    res = np.zeros(3)
    if dataset == "euroc_mav":
        for i, e_i in enumerate([5, 100, 200]):
            res[i] = (metrics.mse_error(model=model, residual_model=residual_model, horizon=e_i, past=past,
                                        dataset="euroc_mav", split=train_test_validation).item())
    else:  # dataset == "blackbird"
        for i, e_i in enumerate([9, 180, 360]):
            res[i] = (metrics.mse_error(model=model, residual_model=residual_model, horizon=e_i, past=past,
                                        dataset="blackbird", split=train_test_validation).item())
    print(f"Model {model.__class__.__name__} on {train_test_validation} "
          f"E(0.025s): {res[0]:0.4e}, E(0.5s): {res[1]:0.4e}, E(1s): {res[2]:0.4e}")


def plot_residual(model, residual_model, past, dataset, train_test_validation):
    residual_model.eval()
    if dataset == "euroc_mav":
        traj_len = past + 1
        data = euroc.EurocMAVDatasetHorizon(train_validation_test=train_test_validation,
                                            traj_len=traj_len, dtype=torch.float32)
        time = data[0]["t"]
        state = data[0]["state"].unsqueeze(0)
        dt = torch.FloatTensor([[1 / 200]]).unsqueeze(0)
        motor = data[0]["motor"].unsqueeze(0)
        residual_gt = data[0]["residual"][past:]

        with torch.no_grad():
            residual_hat = []
            for i in range(past, traj_len):
                residual_hat.append(residual_model(motor=motor[:, i-past:i], state=state[:, i-past:i], dt=dt))
            residual_hat = torch.stack(residual_hat, dim=0)
        ((residual_hat - residual_gt) ** 2).mean()

        i = 3
        name = ["Fa x", "Fa y", "Fa z", "Ta x", "Ta y", "Ta z"]
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        ax.plot(time[past:], residual_gt[:, i], label=f"{name[i]} GT")
        ax.plot(time[past:], residual_hat[:, i], label=f"{name[i]} hat")
        ax.set_ylabel("Fa [m/s2]" if i // 3 == 0 else "Ta [rad/s2]")
        ax.set_xlabel("Time [s]")
        ax.grid()
        ax.legend()
    else:
        raise NotImplementedError


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

    # mlp_symplectic_past = "20210709_093134_MLP_bs_32/82320_weights.pt"
    # weights = torch.load(EUROC_MAV_ANALYSIS_PATH + f"training/{mlp_symplectic_past}")
    # past = 10
    # model = drone_models.mlp.iii_mlp_past.MLP(motor_size=6, n_past=past)
    # model.load_state_dict(weights)
    # model.eval()
    # test_model(model=model, residual_model=None, past=past, dataset="euroc_mav", train_test_validation="test",
    #            mode="normal")

    cnn_residual = "20210723_180758_CNNResidual_bs256_epch10000_lr0.001_h1_p20_us50/636216_weights.pt"
    past = 20
    weights = torch.load(EUROC_MAV_RESIDUAL + f"training/{cnn_residual}")
    residual_model = drone_models.residual.cnn.CNNResidual(motor_size=6, past=past, unit_size=50)
    residual_model.load_state_dict(weights)
    residual_model.eval()
    model = drone_models.analytical_models.hexarotor_4params.DynamicalModel(paper=False, dtype=torch.float32)
    # test_model(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
    #            train_test_validation="test")
    plot_residual(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
                  train_test_validation="train")
    pass


if __name__ == '__main__':
    main()
