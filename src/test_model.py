import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as f
# Data
import src.data.blackbird.loading as blackbird
import src.data.euroc_mav.loading as euroc

# Models
import src.drone_models as drone_models
import src.utils.trajectory
from src.utils.paths import GENERAL_ANALYSIS_PATH, mkdir_if_not_exist
from src import metrics
from src.learning import residual_estimation
from src.log import Logger

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


def plot_residual(model, residual_model, horizon, past, dataset, train_test_validation):
    residual_model.eval()
    if dataset == "euroc_mav":
        traj_len = 3000
        data = euroc.EurocMAVDatasetHorizon(train_validation_test=train_test_validation,
                                            traj_len=traj_len, dtype=torch.float32, for_loader=False)
        time = data[0]["t"].squeeze()
        state = data[0]["state"]
        dt = torch.FloatTensor([[1 / 200]]).unsqueeze(0)
        motor = data[0]["motor"]
        residual_gt = data[0]["residual"][past:].squeeze()

        # Horizon 1
        # with torch.no_grad():
        #     residual_hat = []
        #     for i in range(past, traj_len):
        #         residual_hat.append(residual_model(motor=motor[:, i-past:i], state=state[:, i-past:i], dt=dt))
        #     residual_hat = torch.stack(residual_hat, dim=0)
        # print(f"{((residual_hat - residual_gt) ** 2).mean()=}")
        with torch.no_grad():
            prediction = [state[:, past-1]]
            for i in range(traj_len - past):
                past_states = torch.cat([state[:, i:past - 1],
                                         torch.stack(prediction, dim=1)[:, -past:]],
                                        dim=1)
                past_motor = motor[:, i:past + i]
                # past_dt = cropped_dt[:, i:past + i]

                if residual_model is not None:
                    residual = residual_model(state=past_states, motor=past_motor, dt=dt).view(1, 6)
                    next_state = model(motor=past_motor[:, -1], state=past_states[:, -1], dt=dt.view(1, 1),
                                       residual=residual)
                else:
                    next_state = model(motor=past_motor, state=past_states, dt=dt)
                prediction.append(next_state)

            prediction = torch.stack(prediction, dim=1)
            # error = ((prediction - ground_truth) ** 2).mean()

        i = 0
        name = ["Fa x", "Fa y", "Fa z", "Ta x", "Ta y", "Ta z"]
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        ax.plot(time[past:], residual_gt[:, i], label=f"{name[i]} GT")
        ax.plot(time[past:], prediction[:, i], label=f"{name[i]} hat")  # TODO Check if prediction should be plotted
        ax.set_ylabel("Fa [m/s2]" if i // 3 == 0 else "Ta [rad/s2]")
        ax.set_xlabel("Time [s]")
        ax.grid()
        ax.legend()
        plt.show()

        i = 0
        name = ["Fa x", "Fa y", "Fa z", "Ta x", "Ta y", "Ta z"]
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        ax.plot(time[past:], residual_gt[:, i], label=f"{name[i]} GT")
        ax.plot(time[past:], prediction[:, i], label=f"{name[i]} hat")
        ax.set_ylabel("Fa [m/s2]" if i // 3 == 0 else "Ta [rad/s2]")
        ax.set_xlabel("Time [s]")
        ax.grid()
        ax.legend()
        plt.show()
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

    # cnn_residual = "20210803_115051_CNNResidual_bs32_epch10000_lr0.001_h1_p20_us50/567525_weights.pt"
    # model = drone_models.analytical_models.hexarotor_4params.DynamicalModel(paper=False, dtype=torch.float32)
    # past = 20
    # weights = torch.load(EUROC_MAV_RESIDUAL + f"training/{cnn_residual}", map_location=torch.device('cpu'))
    # residual_model = drone_models.residual.cnn.CNNResidual(model=model, past=past, unit_size=50, activation=f.relu,
    #                                                        device="cpu")
    # residual_model.load_state_dict(weights)
    # residual_model.eval()
    # test_model(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
    #            train_test_validation="test")
    # plot_residual(model=model, residual_model=residual_model, horizon=1, past=past, dataset="euroc_mav",
    #               train_test_validation="validation")

    # cnn_residual = "20210831_120906_CNNResidual_bs128_epch10000_lr0.0001_h1_p20_us50_relu/1241_weights.pt"
    # # cnn_residual = "20210831_120905_CNNResidual_bs128_epch10000_lr0.0001_h1_p10_us50_relu/1787_weights.pt"
    # model = drone_models.analytical_models.hexarotor_4params.DynamicalModel(paper=False, dtype=torch.float32)
    # past = 20
    # weights = torch.load(EUROC_MAV_RESIDUAL + f"training/{cnn_residual}", map_location=torch.device('cpu'))
    # residual_model = drone_models.residual.cnn.CNNResidual(model=model, past=past, unit_size=50, activation=f.relu,
    #                                                        device="cpu")
    # residual_model.load_state_dict(weights)
    # residual_model.eval()
    # test_model(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
    #            train_test_validation="test")
    # plot_residual(model=model, residual_model=residual_model, horizon=1, past=past, dataset="euroc_mav",
    #               train_test_validation="validation")

    cnn_residual = "l3/"
    # cnn_residual = "20210902_120510_CNNResidual_bs128_epch10000_lr0.0001_h1_p60_us50_tanh/9982_weights.pt"
    model = drone_models.analytical_models.hexarotor_4params.DynamicalModel(paper=False, dtype=torch.float32)
    past = 60
    layer = 3
    weights = torch.load(EUROC_MAV_RESIDUAL + f"training/{cnn_residual}", map_location=torch.device('cpu'))
    for old_key, new_key in [["conv2d1", "conv2d_end"]]:
        weights[f"{new_key}.weight"] = weights.pop(f"{old_key}.weight")
        weights[f"{new_key}.bias"] = weights.pop(f"{old_key}.bias")
    residual_model = drone_models.residual.cnn.CNNResidual(model=model, past=past, unit_size=50, activation=torch.tanh,
                                                           layer=layer, embedding="None", device="cpu")
    residual_model.load_state_dict(weights)
    residual_model.eval()
    test_model(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
               train_test_validation="test")


    # # Testing embeddings
    # cnn_residual = "20210902_120510_CNNResidual_bs128_epch10000_lr0.0001_h1_p60_us50_tanh/9982_weights.pt"
    # model = drone_models.analytical_models.hexarotor_4params.DynamicalModel(paper=False, dtype=torch.float32)
    # past = 60
    # layer = 2
    # weights = torch.load(EUROC_MAV_RESIDUAL + f"training/{cnn_residual}", map_location=torch.device('cpu'))
    # for old_key, new_key in [["conv2d1", "conv2d_end"]]:
    #     weights[f"{new_key}.weight"] = weights.pop(f"{old_key}.weight")
    #     weights[f"{new_key}.bias"] = weights.pop(f"{old_key}.bias")
    # residual_model = drone_models.residual.cnn.CNNResidual(model=model, past=past, unit_size=50, activation=torch.tanh,
    #                                                        layer=layer, embedding=None, device="cpu")
    # residual_model.load_state_dict(weights)
    # residual_model.eval()
    # test_model(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
    #            train_test_validation="test")


if __name__ == '__main__':
    main()
