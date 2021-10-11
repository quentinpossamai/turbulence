import matplotlib.pyplot as plt
import numpy as np
import torch

# Data
# import src.data.blackbird.loading as blackbird
import src.data.euroc_mav.loading as euroc

# Models
from src import metrics
from src.utils.paths import ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING


# BLACKBIRD_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "blackbird_learned_quadrotor_training/"
# EUROC_MAV_ANALYSIS_PATH = GENERAL_ANALYSIS_PATH + "euroc_mav_learned_hexarotor_training/"
# mkdir_if_not_exist([BLACKBIRD_ANALYSIS_PATH])


def test_model(model, past, dataset, train_validation_test):
    """
    Print the error on 0.025s, 0.5s and 1s of one model on one dataset split.
    :param model: The model to evaluate.
    :param past: Number of samples to give to the model
    :param dataset: The dataset to use for evaluation. "blackbird" or "euroc_mav".
    :param train_validation_test: The split to evaluate on. "train", "validation" or "test".
    """
    assert train_validation_test in ["train", "validation", "test"]
    assert dataset in ["blackbird", "euroc_mav"]

    res = np.zeros(3)
    if dataset == "euroc_mav":
        for i, e_i in enumerate([5, 100, 200]):
            res[i] = (metrics.mse_error(model=model, horizon=e_i, past=past,
                                        dataset="euroc_mav", split=train_validation_test).item())
    else:  # dataset == "blackbird"
        for i, e_i in enumerate([9, 180, 360]):
            res[i] = (metrics.mse_error(model=model, horizon=e_i, past=past,
                                        dataset="blackbird", split=train_validation_test).item())
    print(f"Model {model.__class__.__name__} on {train_validation_test} "
          f"E(0.025s): {res[0]:0.4e}, E(0.5s): {res[1]:0.4e}, E(1s): {res[2]:0.4e}")


def plot_rollout(model, horizon, past, dataset, train_validation_test, plot_range=None):
    if dataset == "euroc_mav":
        traj_len = past + horizon
        data = euroc.EurocMAVDatasetHorizon(train_validation_test=train_validation_test,
                                            traj_len=traj_len, dtype=torch.float32, batch_dim=False)
        trajectory = data[0]
        t = trajectory["t"]
        motor = trajectory["motor"]
        state = trajectory["state"]
        depth_emb = trajectory["depth_emb"]
        norm_emb = trajectory["norm_emb"]

        dt_shape = list(state.shape)
        dt_shape[-1] = 1
        dt = torch.FloatTensor([[1 / 200]]).repeat(dt_shape)

        with torch.no_grad():
            loss, rollout_loss, prediction = model.rollout(motor=motor, state=state, dt=dt, depth_emb=depth_emb,
                                                           norm_emb=norm_emb, horizon=horizon)
        state = state[past:]
        t = t[past:]
        if plot_range is not None:
            state = state[plot_range[0]:plot_range[1]]
            prediction = prediction[plot_range[0]:plot_range[1]]
            t = t[plot_range[0]:plot_range[1]]

        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        ax_pos = [fig.add_subplot(4, 4, 1), fig.add_subplot(4, 4, 2), fig.add_subplot(4, 4, 3)]
        ax_quat = [fig.add_subplot(4, 4, 5), fig.add_subplot(4, 4, 6), fig.add_subplot(4, 4, 7),
                   fig.add_subplot(4, 4, 8)]
        ax_vel = [fig.add_subplot(4, 4, 9), fig.add_subplot(4, 4, 10), fig.add_subplot(4, 4, 11)]
        ax_rpy = [fig.add_subplot(4, 4, 13), fig.add_subplot(4, 4, 14), fig.add_subplot(4, 4, 15)]

        ax_pos[0].set_title("X")
        ax_pos[1].set_title("Y")
        ax_pos[2].set_title("Z")

        ax_quat[0].set_title("Qx")
        ax_quat[1].set_title("Qy")
        ax_quat[2].set_title("Qz")
        ax_quat[3].set_title("Qw")

        ax_vel[0].set_title("Vx")
        ax_vel[1].set_title("Vy")
        ax_vel[2].set_title("Vz")

        ax_rpy[0].set_title("Wx")
        ax_rpy[1].set_title("Wy")
        ax_rpy[2].set_title("Wz")
        for i in range(3):
            ax_pos[i].plot(t, state[:, i], label="GT")
            ax_pos[i].plot(t, prediction[:, i], label="Prediction")
            ax_vel[i].plot(t, state[:, 7 + i])
            ax_vel[i].plot(t, prediction[:, 7 + i])
            ax_rpy[i].plot(t, state[:, 10 + i])
            ax_rpy[i].plot(t, prediction[:, 10 + i])
            # ax_pos[i].tick_params(axis='both', labelsize=5)
            # ax_vel[i].tick_params(axis='both', labelsize=5)
            # ax_rpy[i].tick_params(axis='both', labelsize=5)
            ax_pos[i].grid()
            ax_vel[i].grid()
            ax_rpy[i].grid()

        for i in range(4):
            ax_quat[i].plot(t, state[:, 3 + i])
            ax_quat[i].plot(t, prediction[:, 3 + i])
            # ax_quat[i].tick_params(axis='both', labelsize=5)
            ax_quat[i].grid()

        handles, labels = ax_pos[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc=(0.8, 0.85), frameon=True)
        plt.tight_layout()

        plt.show()

        # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        # ax.plot(time[past:], state[past:, :3], label=f"GT")
        # ax.plot(time[past:], prediction[:, :3], label=f"hat")
        # ax.set_ylabel("Position [m]")
        # ax.set_xlabel("Time [s]")
        # ax.grid()
        # ax.legend()
        # plt.show()
        #
        # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        # ax.plot(time[past:], state[past:, 3:7], label=f"GT")
        # ax.plot(time[past:], prediction[:, 3:7], label=f"hat")
        # ax.set_ylabel("Quaternion")
        # ax.set_xlabel("Time [s]")
        # ax.grid()
        # ax.legend()
        # plt.show()
        #
        # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        # ax.plot(time[past:], state[past:, 7:10], label=f"GT")
        # ax.plot(time[past:], prediction[:, 7:10], label=f"hat")
        # ax.set_ylabel("Linear speed [m]")
        # ax.set_xlabel("Time [s]")
        # ax.grid()
        # ax.legend()
        # plt.show()
        #
        # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
        # ax.plot(time[past:], state[past:, 10:13], label=f"GT")
        # ax.plot(time[past:], prediction[:, 10:13], label=f"hat")
        # ax.set_ylabel("Angular velocity [m]")
        # ax.set_xlabel("Time [s]")
        # ax.grid()
        # ax.legend()
        # plt.show()
    else:
        raise NotImplementedError


def plot_residual(model, traj_len, dataset, train_validation_test):
    if dataset == "euroc_mav":
        data = euroc.EurocMAVDatasetHorizon(train_validation_test=train_validation_test,
                                            traj_len=traj_len, dtype=torch.float32, batch_dim=False)
        time = []
        residual_hat = []
        residual_gt = []
        with torch.no_grad():
            for i in range(2000):
                trajectory = data[i]
                motor = trajectory["motor"][1:]
                state = trajectory["state"][:-1]
                depth_emb = trajectory["depth_emb"][:-1]
                norm_emb = trajectory["norm_emb"][:-1]

                dt_shape = list(state.shape)
                dt_shape[-1] = 1
                dt = torch.FloatTensor([[1 / 200]]).repeat(dt_shape)

                time.append(trajectory["t"][-1])
                residual_gt.append(trajectory["residual"][-1])
                residual_hat.append(model(motor=motor, state=state, dt=dt, depth_emb=depth_emb, norm_emb=norm_emb))

        time = torch.stack(time, dim=0)
        residual_gt = torch.stack(residual_gt, dim=0)
        residual_hat = torch.stack(residual_hat, dim=0)
        loss = ((residual_gt - residual_hat) ** 2).mean().item()
        print(f"loss={loss}")

        for i in range(6):
            name = ["Fa x [m/s2]", "Fa y [m/s2]", "Fa z [m/s2]", "Ta x [rad/s2]", "Ta y [rad/s2]", "Ta z [rad/s2]"]
            fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
            ax.plot(time, residual_gt[:, i], label=f"GT")
            ax.plot(time, residual_hat[:, i], label=f"hat")
            ax.set_ylabel(name[i])
            ax.set_xlabel("Time [s]")
            ax.grid()
            ax.legend()
            plt.show()

    else:
        raise NotImplementedError


def presence_classification_test(classifier, dataset, train_validation_test):
    # Computing model prediction
    possible_dataset = ['euroc_mav', 'blackbird']
    assert dataset in possible_dataset, f'Possible dataset are {possible_dataset}'
    if dataset == 'euroc_mav':
        data = euroc.EurocMAVDatasetTurbulence(train_validation_test, dtype=torch.float32, batch_dim=False)
        prediction = []
        truth = []
        with torch.no_grad():
            for batch in data:
                prediction.append(classifier(depth_emb=batch['depth_emb'], norm_emb=batch['depth_emb']))
                truth.append(batch['fa_label'])
        prediction = torch.cat(prediction, dim=0)
        prediction = torch.round(prediction)
        truth = torch.cat(truth, dim=0)

        # Analysis
        total = len(truth)
        truth_is_0 = torch.where(truth == 0)[0]
        truth_is_1 = torch.where(truth == 1)[0]
        tp = (prediction[truth_is_1] == 1).float().sum()  # true positive, prediction is positive and it is true
        fp = (prediction[truth_is_0] == 1).float().sum()  # false positive, prediction is positive but it is false
        tn = (prediction[truth_is_0] == 0).float().sum()  # true negative, prediction is negative and it is true
        fn = (prediction[truth_is_1] == 0).float().sum()  # false negative, prediction is negative but it is false
        assert tp + fp + tn + fn == total
        print(f'{train_validation_test}: True positive rate :{tp.item() / total * 100:0.2f}%')
        print(f'{train_validation_test}: False positive rate: {fp.item() / total * 100:0.2f}%')
        print(f'{train_validation_test}: True negative rate: {tn.item() / total * 100:0.2f}%')
        print(f'{train_validation_test}: False negative rate: {fn.item() / total * 100:0.2f}%')

        # accuracy = (1 - abs(prediction - label)).mean()
        tpr = tp / (tp + fn)  # sensitivity, recall, hit rate, or true positive rate (TPR)
        tnr = tn / (tn + fp)  # specificity, selectivity or true negative rate (TNR)
        ppv = tp / (tp + fp)  # precision or positive predictive value (PPV)
        npv = tn / (tn + fn)  # negative predictive value (NPV)
        fnr = fn / (fn + tp)  # miss rate or false negative rate (FNR)
        fpr = fp / (fp + tn)  # fall-out or false positive rate (FPR)
        fdr = fp / (fp + tp)  # false discovery rate (FDR)
        for_ = fn / (fn + tn)  # false omission rate (FOR)
        # pt = (torch.sqrt(tpr * (-tnr + 1)) + (tnr - 1)) / (tpr + tnr - 1)  # prevalence threshold (PT)
        # ts = tp / (tp + fn + fp)  # threat score (TS) or critical success index (CSI)
        # accuracy = (tp + tn) / (tp + fp + tn + fn)
        # fp1_score = 2 * (ppv * tpr) / (ppv + tpr)
        assert torch.allclose(tpr, 1 - fnr)
        assert torch.allclose(tnr, 1 - fpr)
        assert torch.allclose(ppv, 1 - fdr)
        assert torch.allclose(npv, 1 - for_)
    else:
        raise NotImplementedError


def main():
    # # EUROC MAV
    # from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
    # print('Dynamical model paper')
    # model = DynamicalModel(param="paper", dtype=torch.float32)
    # test_model(model=model, past=1, dataset="euroc_mav", train_test_validation="test")
    # print('\nDynamical model identification')
    # model = DynamicalModel(param="identification", dtype=torch.float32)
    # test_model(model=model, past=1, dataset="euroc_mav", train_test_validation="test")

    # --- TODO update
    # - BLACKBIRD
    # phi_paper_param = quadrotor_4parms.DynamicalModel(paper=True)
    # test_model(phi_paper_param, "blackbird", "test", "normal")

    # phi_mse_param = quadrotor_4parms.DynamicalModel(paper=False)
    # test_model(phi_mse_param, "blackbird", "test", "normal")

    # mlp_int = "20210623_113455_OldMLPQuadrotor_bs_64000/8004_weights.pt"
    # test_blackbird_mlp_metrics(mlp_int)
    ##
    # EUROC MAV
    # mlp_symplectic = "20210621_145132_MLPQuadrotor_bs_64000/11136_weights.pt"
    # test_blackbird_mlp_metrics(mlp_symplectic)
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
    # weights = torch.load(ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING +
    # f"training/{cnn_residual}", map_location=torch.device('cpu'))
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
    # weights = torch.load(ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING +
    # f"training/{cnn_residual}", map_location=torch.device('cpu'))
    # residual_model = drone_models.residual.cnn.CNNResidual(model=model, past=past, unit_size=50, activation=f.relu,
    #                                                        device="cpu")
    # residual_model.load_state_dict(weights)
    # residual_model.eval()
    # test_model(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
    #            train_test_validation="test")
    # plot_residual(model=model, residual_model=residual_model, horizon=1, past=past, dataset="euroc_mav",
    #               train_test_validation="validation")

    # cnn_residual = "l3/"
    # # cnn_residual = "20210902_120510_CNNResidual_bs128_epch10000_lr0.0001_h1_p60_us50_tanh/9982_weights.pt"
    # model = drone_models.analytical_models.hexarotor_4params.DynamicalModel(paper=False, dtype=torch.float32)
    # past = 60
    # layer = 3
    # weights = torch.load(ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING +
    # f"training/{cnn_residual}", map_location=torch.device('cpu'))
    # for old_key, new_key in [["conv2d1", "conv2d_end"]]:
    #     weights[f"{new_key}.weight"] = weights.pop(f"{old_key}.weight")
    #     weights[f"{new_key}.bias"] = weights.pop(f"{old_key}.bias")
    # residual_model = drone_models.residual.cnn.CNNResidual(model=model, past=past, unit_size=50,
    #                                                        activation=torch.tanh,
    #                                                        layer=layer, embedding="None", device="cpu")
    # residual_model.load_state_dict(weights)
    # residual_model.eval()
    # test_model(model=model, residual_model=residual_model, past=past, dataset="euroc_mav",
    #            train_test_validation="test")
    # --- TODO update

    # # Testing embeddings
    # print('\nResidual model GRU CNN no embeddings')
    # gru_cnn = ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING + "GRUCNNResidual_None/default/"
    # gru_cnn += "20210928_090235_GRUCNNResidual_bs64_epch1000_lr1e-06_h1_p60_us50_tanh_l3_None/checkpoints/"
    # gru_cnn += "epoch=999-step=584999.ckpt"
    # from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
    # from src.drone_models.residual.gru_cnn import GRUCNNResidual
    # model = DynamicalModel(param='identification', dtype=torch.float32)
    # past = 60
    # residual_model = GRUCNNResidual.load_from_checkpoint(
    #         gru_cnn,
    #         activation='tanh',
    #         embedding=None,
    #         layer=3,
    #         # learning_rate=1e-6,
    #         model=model,
    #         motor_size=6,
    #         past=past,
    #         # unit_size=50,
    #     )
    #
    # residual_model.eval()
    # # test_model(model=residual_model, past=past, dataset="euroc_mav", train_test_validation="test")
    # plot_rollout(residual_model, horizon=300, past=past, dataset='euroc_mav', train_test_validation='test',
    #              plot_range=[0, 300])
    # # plot_residual(residual_model, traj_len=past + 1, dataset='euroc_mav', train_test_validation="validation")

    # Testing embeddings
    print('\nResidual model Presence Classifier')
    mlp = ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING + "PresenceClassifier_norm/default/"
    mlp += '20211010_155504_PresenceClassifier_tanh_bs256_norml3_lr1e-05_epch10000_p1_us50_/checkpoints/'
    mlp += "epoch=2286-step=331614.ckpt"
    from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
    from src.drone_models.residual.presence_classifier import PresenceClassifier
    model = DynamicalModel(param='identification', dtype=torch.float32)
    past = 1
    residual_model = PresenceClassifier.load_from_checkpoint(
        mlp,
        activation='tanh',
        embedding='norm',
        layer=3,
        # learning_rate=1e-6,
        model=model,
        motor_size=6,
        past=past,
        # unit_size=50,  # not
    )

    residual_model.eval()
    presence_classification_test(classifier=residual_model, dataset='euroc_mav', train_validation_test='train')
    presence_classification_test(classifier=residual_model, dataset='euroc_mav', train_validation_test='validation')
    presence_classification_test(classifier=residual_model, dataset='euroc_mav', train_validation_test='test')
    # test_model(model=residual_model, past=past, dataset="euroc_mav", train_test_validation="test")
    # plot_rollout(residual_model, horizon=300, past=past, dataset='euroc_mav', train_test_validation='test',
    #              plot_range=[0, 300])
    # plot_residual(residual_model, traj_len=past + 1, dataset='euroc_mav', train_test_validation="validation")


if __name__ == '__main__':
    main()
