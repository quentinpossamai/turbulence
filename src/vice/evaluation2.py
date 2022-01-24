import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import pickle

import src.data.euroc_mav.loading as euroc
import src.vice.euroc_vice as euroc_vice

import src.data.locobot.loading as locobot
import src.vice.locobot_vice as locobot_vice

import src.data.blackbird.loading as blackbird
import src.vice.blackbird_vice as blackbird_vice

from src.utils.transform import Transform as Tf, Rotation


def angle(position3d_label: np.ndarray, position3d_pred: np.ndarray) -> float:
    """
    Compute the angle hat{ABC} of the triangle defined with:
    A = position3d_label
    B = the camera position (0, 0, 0)
    C = position3d_pred
    param position3d_label: (3,) 3d point. 2D label projected in 3D.
    param position3d_pred: (3,) 3d point. 2D prediction projected in 3D.
    return: The angle in degrees
    """
    assert np.linalg.norm(position3d_label) != 0
    assert np.linalg.norm(position3d_pred) != 0
    v1 = position3d_label / np.linalg.norm(position3d_label)
    v2 = position3d_pred / np.linalg.norm(position3d_pred)
    return np.arccos(np.clip(v1 @ v2, -1, 1)) / np.pi * 180


def roe(rot_gt: np.ndarray, rot_pred: np.ndarray, k: int) -> np.ndarray:
    """
    Compute the relative orientation error defined in:

    Zhang, Z. and Scaramuzza, D. (2018) ‘A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial)
    Odometry’, in 2018 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2018 IEEE/RSJ
    International Conference on Intelligent Robots and Systems (IROS), pp. 7244–7251. doi:10.1109/IROS.2018.8593941.

    param rot_gt: (n, 3, 3).  Trajectory ground truth as rotation matrices.

    param rot_pred: (n, 3, 3). Trajectory predicted as rotation matrices.

    param k: Number of to sample to take uniformly in the trajectories.

    return: (k,). k relative orientation errors.
    """
    n = len(rot_gt)
    assert rot_gt.shape == (n, 3, 3) and rot_pred.shape == (n, 3, 3)

    res = np.zeros(k)
    for k_ind, i in enumerate(np.round(np.linspace(0, n - 2, k)).astype(int)):
        dr = rot_gt[i] @ rot_gt[i + 1].T
        dr_hat = rot_pred[i] @ rot_pred[i + 1].T
        prod = dr @ dr_hat
        a = 1 / 2 * (prod - prod.T)
        norm_a = np.sqrt(-1 / 2 * np.trace(a @ a))
        log_r = (np.arcsin(norm_a) / norm_a).reshape((-1, 1, 1)) * a
        res[k_ind] = np.linalg.norm(log_r)
    return res


def aoe(rot_gt: np.ndarray, rot_pred: np.ndarray, no_mean: bool) -> float:
    """
    Compute the relative orientation error defined in:

    Brossard, M., Bonnabel, S. and Barrau, A. (2020) ‘Denoising IMU Gyroscopes with Deep Learning for Open-Loop Attitude
    Estimation’, IEEE Robotics and Automation Letters, PP, pp. 1–1. doi:10.1109/LRA.2020.3003256.

    param rot_gt: (n, 3, 3).  Trajectory ground truth as rotation matrices.

    param rot_pred: (n, 3, 3). Trajectory predicted as rotation matrices.

    param no_mean: If True, return the entire tab, else return the rooted mean.

    return: The averaged absolute orientation error over the trajectory.
    """
    n = len(rot_gt)
    assert rot_gt.shape == (n, 3, 3) and rot_pred.shape == (n, 3, 3)

    dr = rot_gt @ rot_pred.transpose((0, 2, 1))
    a = 1 / 2 * (dr - dr.transpose((0, 2, 1)))
    norm_a = np.sqrt(-1 / 2 * np.trace(a @ a, axis1=1, axis2=2))
    log_r = (np.arcsin(norm_a) / norm_a).reshape((-1, 1, 1)) * a
    if no_mean:
        return np.sqrt(np.linalg.norm(log_r.reshape(-1, 9), axis=1) ** 2)
    else:
        # first element is initialised with gt
        return np.sqrt((np.linalg.norm(log_r.reshape(-1, 9), axis=1) ** 2)[1:].mean())


def vice_error(label, prediction):
    squared_2d_error = []
    index_2d = []
    for i in range(len(prediction)):
        try:
            point, is_keyframe = label[i]
            not_label = is_keyframe
        except KeyError:
            not_label = False
            # continue
        p3, not_prediction = prediction[i]
        if not not_prediction:
            assert not not_label
            squared_2d_error.append(np.sum((point - p3.astype(float)) ** 2))
            index_2d.append(i)

    squared_2d_error = np.array(squared_2d_error)
    index_2d = np.array(index_2d)
    print(f'Error 2D length: {len(squared_2d_error)}')
    rmse_2d = np.sqrt(np.mean(squared_2d_error))
    print(f'(p1) RMSE 2D (px): {rmse_2d:.3f}')
    return rmse_2d, squared_2d_error, index_2d


def specific_euroc_mav_evaluation(flight: str, ground: bool, imu: bool, label_number: int):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    labels = pickle.load(open(f'{vice_path}labels.pkl', 'rb'))
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name
    name = f'{name}{label_number}'

    prediction = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))
    # estimator = euroc_vice.build_estimator(flight, ground, imu, label_number=)

    rmse_2d, squared_2d_error, index_2d = vice_error(labels[label_number], prediction)
    return rmse_2d, squared_2d_error, index_2d


def euroc_mav_evaluation():
    for flight in euroc.FLIGHT_LIST:
        for imu in [False, True]:
            for ground in [True, False]:
                print(f'Flight: {flight} | ground: {ground} | imu: {imu}')

                key1 = f'imu' if imu else f'mocap'
                key2 = f'_ground' if ground else f'_depth'
                key3 = f'_{flight[:-1]}'
                plot_angle.euroc_vice_error[f'{key1}{key2}{key3}'] = {}
                for lab_num in range(4):
                    rmse_2d, squared_2d_error, index_2d = specific_euroc_mav_evaluation(flight,
                                                                                        ground=ground,
                                                                                        imu=imu,
                                                                                        label_number=lab_num)
                    plot_angle.euroc_vice_error[f'{key1}{key2}{key3}'][lab_num] = (rmse_2d, squared_2d_error, index_2d)
                mean0 = np.mean(plot_angle.euroc_vice_error[f'{key1}{key2}{key3}'][0][1])
                mean1 = np.mean(plot_angle.euroc_vice_error[f'{key1}{key2}{key3}'][1][1])
                mean2 = np.mean(plot_angle.euroc_vice_error[f'{key1}{key2}{key3}'][2][1])
                mean3 = np.mean(plot_angle.euroc_vice_error[f'{key1}{key2}{key3}'][3][1])

                p0 = np.sqrt(mean0)
                p1 = np.sqrt(mean1)
                p2 = np.sqrt(mean2)
                p3 = np.sqrt(mean3)

                p01 = np.sqrt(np.mean([mean0, mean1]))
                p12 = np.sqrt(np.mean([mean1, mean2]))
                p23 = np.sqrt(np.mean([mean2, mean3]))
                p30 = np.sqrt(np.mean([mean3, mean0]))
                p02 = np.sqrt(np.mean([mean0, mean2]))
                p13 = np.sqrt(np.mean([mean1, mean3]))

                p012 = np.sqrt(np.mean([mean0, mean1, mean2]))
                p013 = np.sqrt(np.mean([mean0, mean1, mean3]))
                p023 = np.sqrt(np.mean([mean0, mean2, mean3]))
                p123 = np.sqrt(np.mean([mean1, mean2, mean3]))

                print(f'Mean p1: {np.mean([p0, p1, p2, p3]):.1f}$\\pm${np.std([p0, p1, p2, p3]):.1f}')
                print(f'Mean p2: {np.mean([p01, p12, p23, p30, p02, p13]):.1f}$\\pm$'
                      f'{np.std([p01, p12, p23, p30, p02, p13]):.1f}')
                print(f'Mean p3: {np.mean([p012, p013, p023, p123]):.1f}$\\pm$'
                      f'{np.std([p012, p013, p023, p123]):.1f}\n\n')

    # MSE IMU MOCAP
    for i in range(3):
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{euroc.FLIGHT_LIST[i]}vice/'
        input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
        plot_angle.euroc_data[euroc.FLIGHT_LIST[i][:-1]] = input_data

        trans_mc = Tf.from_matrix(input_data['pose']).as_trans()
        trans_imu = Tf.from_matrix(input_data['imu']).as_trans()
        rot_mc = Tf.from_matrix(input_data['pose']).as_rot_mat()
        rot_imu = Tf.from_matrix(input_data['imu']).as_rot_mat()

        rmse_error = np.sqrt(np.mean(np.sum((trans_mc - trans_imu) ** 2, axis=1), axis=0))

        aoe_error = aoe(rot_mc, rot_imu, no_mean=False)

        roe_error = roe(rot_mc, rot_imu, 100)
        roe_error = roe_error.mean()

        print(f'flight: {euroc.FLIGHT_LIST[i]} | mse error: {rmse_error:.3f} | aoe error: {aoe_error:.2f} | '
              f'roe error: {roe_error:.3f} \n\n')


def specific_locobot_evaluation(flight: str, ground: bool, pose: str, label_number: int):
    assert pose in ['odom', 'slam']

    # 1-INITIALISATION
    vice_path = f'{locobot.LOCOBOT_DATA_PATH}{flight}'
    print("VICE for ", flight)

    labels = pickle.load(open(f"{vice_path}labels.pkl", 'rb'))

    name = f'ground_{pose}' if ground else f'depth_{pose}'
    prediction = pickle.load(open(f'{vice_path}{name}{label_number}.pkl', 'rb'))

    print(f'Flight: {flight} | ground: {ground} | pose: {pose}')

    rmse_2d, squared_2d_error, index_2d = vice_error(labels[label_number], prediction)
    return rmse_2d, squared_2d_error, index_2d


def locobot_evaluation():
    for pose in ['odom', 'slam']:
        for ground in [True, False]:
            print(f'Locobot | ground: {ground} | pose: {pose}')
            result = {}
            for lab_num in range(4):
                rmse_2d, squared_2d_error, index_2d = specific_locobot_evaluation(locobot.TRAJ_LIST[1], ground=ground,
                                                                                  pose=pose, label_number=lab_num)
                result[lab_num] = (rmse_2d, squared_2d_error, index_2d)
            mean0 = np.mean(result[0][1])
            mean1 = np.mean(result[1][1])
            mean2 = np.mean(result[2][1])
            mean3 = np.mean(result[3][1])

            p0 = np.sqrt(mean0)
            p1 = np.sqrt(mean1)
            p2 = np.sqrt(mean2)
            p3 = np.sqrt(mean3)

            p01 = np.sqrt(np.mean([mean0, mean1]))
            p12 = np.sqrt(np.mean([mean1, mean2]))
            p23 = np.sqrt(np.mean([mean2, mean3]))
            p30 = np.sqrt(np.mean([mean3, mean0]))
            p02 = np.sqrt(np.mean([mean0, mean2]))
            p13 = np.sqrt(np.mean([mean1, mean3]))

            p012 = np.sqrt(np.mean([mean0, mean1, mean2]))
            p013 = np.sqrt(np.mean([mean0, mean1, mean3]))
            p023 = np.sqrt(np.mean([mean0, mean2, mean3]))
            p123 = np.sqrt(np.mean([mean1, mean2, mean3]))

            print(f'Mean p1: {np.mean([p0, p1, p2, p3]):.1f}$\\pm${np.std([p0, p1, p2, p3]):.1f}')
            print(f'Mean p2: {np.mean([p01, p12, p23, p30, p02, p13]):.1f}$\\pm$'
                  f'{np.std([p01, p12, p23, p30, p02, p13]):.1f}')
            print(f'Mean p3: {np.mean([p012, p013, p023, p123]):.1f}$\\pm$'
                  f'{np.std([p012, p013, p023, p123]):.1f}\n\n')

    # vice_path = f'{locobot.LOCOBOT_DATA_PATH}{locobot.TRAJ_LIST[1]}'
    # input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    # plot_angle.locobot_data['data'] = input_data


def specific_blackbird_evaluation(ground: bool, imu: bool, label_number: int):
    vice_path = f'{blackbird.VICE_PATH}'
    labels = pickle.load(open(f'{vice_path}labels.pkl', 'rb'))
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name

    prediction = pickle.load(open(f'{vice_path}{name}{label_number}.pkl', 'rb'))

    print(f'Flight: {vice_path} | ground: {ground} | imu: {imu}')
    rmse_2d, squared_2d_error, index_2d = vice_error(labels[label_number], prediction)
    return rmse_2d, squared_2d_error, index_2d,


def blackbird_evaluation():
    for imu in [False, True]:  # True not implemented yet
        for ground in [True, False]:
            print(f'Blackbird | ground: {ground} | imu: {imu}')
            result = {}
            for lab_num in range(4):
                rmse_2d, squared_2d_error, index_2d = specific_blackbird_evaluation(ground=ground, imu=imu,
                                                                                    label_number=lab_num)
                result[lab_num] = rmse_2d, squared_2d_error, index_2d
            mean0 = np.mean(result[0][1])
            mean1 = np.mean(result[1][1])
            mean2 = np.mean(result[2][1])
            mean3 = np.mean(result[3][1])

            p0 = np.sqrt(mean0)
            p1 = np.sqrt(mean1)
            p2 = np.sqrt(mean2)
            p3 = np.sqrt(mean3)

            p01 = np.sqrt(np.mean([mean0, mean1]))
            p12 = np.sqrt(np.mean([mean1, mean2]))
            p23 = np.sqrt(np.mean([mean2, mean3]))
            p30 = np.sqrt(np.mean([mean3, mean0]))
            p02 = np.sqrt(np.mean([mean0, mean2]))
            p13 = np.sqrt(np.mean([mean1, mean3]))

            p012 = np.sqrt(np.mean([mean0, mean1, mean2]))
            p013 = np.sqrt(np.mean([mean0, mean1, mean3]))
            p023 = np.sqrt(np.mean([mean0, mean2, mean3]))
            p123 = np.sqrt(np.mean([mean1, mean2, mean3]))

            print(f'Mean p1: {np.mean([p0, p1, p2, p3]):.1f}$\\pm${np.std([p0, p1, p2, p3]):.1f}')
            print(f'Mean p2: {np.mean([p01, p12, p23, p30, p02, p13]):.1f}$\\pm$'
                  f'{np.std([p01, p12, p23, p30, p02, p13]):.1f}')
            print(f'Mean p3: {np.mean([p012, p013, p023, p123]):.1f}$\\pm$'
                  f'{np.std([p012, p013, p023, p123]):.1f}\n\n')

    # MSE IMU MOCAP
    vice_path = blackbird.VICE_PATH
    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    # plot_angle.blackbird_data['data'] = input_data

    trans_mc = Tf.from_matrix(input_data['pose']).as_trans()
    trans_imu = Tf.from_matrix(input_data['imu']).as_trans()
    rot_mc = Tf.from_matrix(input_data['pose']).as_rot_mat()
    rot_imu = Tf.from_matrix(input_data['imu']).as_rot_mat()

    rmse = np.sqrt(np.mean(np.sum((trans_mc - trans_imu) ** 2, axis=1), axis=0))

    aoe_error = aoe(rot_mc, rot_imu, no_mean=False)

    roe_error = roe(rot_mc, rot_imu, 100)
    roe_error = roe_error.mean()

    print(f'flight: {vice_path} | mse error: {rmse:.3f} | aoe error: {aoe_error:.2f} | '
          f'roe error: {roe_error:.3f} \n\n')


class PlotAngle:
    def __init__(self):
        self.euroc_data = {}
        self.euroc_vice_error = {}
        self.locobot_data = {}
        self.locobot_vice_error = {}
        self.blackbird_data = {}
        self.blackbird_vice_error = {}

    def plot(self):
        plt.style.use('seaborn')

        # plt.rcParams.update({"text.usetex": False,
        #                      "font.family": "sans-serif",
        #                      "font.sans-serif": ["Computer Modern Roman"],
        #                      'size':})
        import matplotlib
        matplotlib.rc('xtick', labelsize=12)
        matplotlib.rc('ytick', labelsize=12)
        plt.rcParams.update({'font.size': 12})

        # Euroc MAV
        i, ground = 0, True
        flight = euroc.FLIGHT_LIST[i]
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'

        em_data = self.euroc_data[euroc.FLIGHT_LIST[i][:-1]]
        mc_trans = Tf.from_matrix(em_data['pose']).as_trans()
        imu_trans = Tf.from_matrix(em_data['imu']).as_trans()
        mc_rot = Tf.from_matrix(em_data['pose']).as_rot_mat()
        imu_rot = Tf.from_matrix(em_data['imu']).as_rot_mat()
        em_ao = aoe(mc_rot, imu_rot, no_mean=True)
        squared_3d_error = np.sum((mc_trans - imu_trans) ** 2, axis=1)

        em_xx = np.linspace(0, len(em_data['pose']) / 10, len(em_data['pose']))  # Converting in sec
        em_start, em_stop = int(len(em_xx) * 0), int(len(em_xx) * 0.8)

        key3 = f'_{flight[:-1]}'
        key2 = f'_ground' if ground else f'_depth'

        imu = False
        key1 = f'imu' if imu else f'mocap'
        name = 'ground' if ground else 'depth'
        name = f'{name}imu' if imu else name
        tmp = self.euroc_vice_error[f'{key1}{key2}{key3}']  # euroc_mav mocap angle error
        mc_reset_ind = {}
        mc_squarred_error = {}
        mc_ind = {}
        mc_2d_check = {}
        for lab_num in range(3):
            mc_squarred_error[lab_num] = tmp[lab_num][1]
            mc_ind[lab_num] = tmp[lab_num][2]
            mc_p3 = pickle.load(open(f'{vice_path}{name}{lab_num}.pkl', 'rb'))
            mc_reset_ind[lab_num] = np.array(
                [i for i, p3 in enumerate(mc_p3) if isinstance(p3[0], np.ndarray) and p3[1]])
            mc_2d_check[lab_num] = [i for i, e in enumerate(mc_ind[lab_num]) if em_start < e < em_stop]

        imu = True
        key1 = f'imu' if imu else f'mocap'
        name = 'ground' if ground else 'depth'
        name = f'{name}imu' if imu else name
        tmp = self.euroc_vice_error[f'{key1}{key2}{key3}']  # euroc_mav imu angle error
        imu_reset_ind = {}
        imu_squarred_error = {}
        imu_ind = {}
        imu_2d_check = {}
        for lab_num in range(3):
            imu_squarred_error[lab_num] = tmp[lab_num][1]
            imu_ind[lab_num] = tmp[lab_num][2]
            imu_p3 = pickle.load(open(f'{vice_path}{name}{lab_num}.pkl', 'rb'))
            imu_reset_ind[lab_num] = np.array(
                [i for i, p3 in enumerate(imu_p3) if isinstance(p3[0], np.ndarray) and p3[1]])
            imu_2d_check[lab_num] = [i for i, e in enumerate(imu_ind[lab_num]) if em_start < e < em_stop]

        # - Plot
        # style_label = 'default'
        fontsize = 12
        fig: plt.Figure
        fig, axs = plt.subplots(2, 1, figsize=(2 * 10, 1 * 5))
        axs_twin = []
        legend0 = ([], [])

        # FIRST AXE ----------------------------
        ax: plt.Axes = axs[0]
        line = ax.plot(em_xx[em_start:em_stop], squared_3d_error[em_start:em_stop] ** 0.5, color='tab:blue')
        legend0[0].append(line[0])
        legend0[1].append('Absolute error')
        ax.set_ylabel(f'Absolute error', fontsize=fontsize, color='tab:blue')

        ax_twin = ax.twinx()
        axs_twin.append(ax_twin)
        line = ax_twin.plot(em_xx[em_start:em_stop], em_ao[em_start:em_stop] ** 0.5, color='tab:red')
        legend0[0].append(line[0])
        legend0[1].append('AOE')
        ax_twin.set_ylabel(f'AOE', fontsize=fontsize, color='tab:red')

        ax.set_xlabel(f'Time [s]', fontsize=fontsize)

        # SECOND AXE ----------------------------
        ax: plt.Axes = axs[1]
        legend1 = ([], [])
        mc_mean = []
        mc_std = []
        imu_mean = []
        imu_std = []
        for i in range(len(em_data['pose'])):
            if not em_start <= i < em_stop:
                continue
            mc_val = []
            imu_val = []
            for lab_num in range(3):
                if i in mc_2d_check[lab_num]:
                    mc_val.append(mc_squarred_error[lab_num][i] ** 0.5)
                if i in imu_2d_check[lab_num]:
                    imu_val.append(imu_squarred_error[lab_num][i] ** 0.5)
            if len(mc_val) >= 2:
                mc_mean.append([i, np.mean(mc_val)])
                mc_std.append([i, np.std(mc_val)])
                imu_mean.append([i, np.mean(imu_val)])
                imu_std.append([i, np.std(imu_val)])
        mc_mean = np.array(mc_mean)
        mc_std = np.array(mc_std)
        imu_mean = np.array(imu_mean)
        imu_std = np.array(imu_std)

        line = ax.plot(em_xx[mc_mean[:, 0].astype(int)], mc_mean[:, 1], color='tab:blue', zorder=1)
        ax.fill_between(em_xx[mc_mean[:, 0].astype(int)],
                        mc_mean[:, 1] - mc_std[:, 1],
                        mc_mean[:, 1] + mc_std[:, 1], color='tab:blue', alpha=0.3)
        legend1[0].append(line[0])
        legend1[1].append(f'MoCap')
        line = ax.plot(em_xx[imu_mean[:, 0].astype(int)], imu_mean[:, 1], color='tab:red', zorder=1)
        ax.fill_between(em_xx[imu_mean[:, 0].astype(int)],
                        imu_mean[:, 1] - imu_std[:, 1],
                        imu_mean[:, 1] + imu_std[:, 1], color='tab:red', alpha=0.3)
        legend1[0].append(line[0])
        legend1[1].append(f'IMU')
        ax.set_ylabel(f'Pixel Absolute error', fontsize=fontsize)
        for lab_num in range(3):
            for reset_ind in mc_reset_ind[lab_num]:
                if em_start <= reset_ind < em_stop:
                    ax.scatter(em_xx[reset_ind], mc_mean[:, 1][reset_ind], s=10, color='k', zorder=2)
            for reset_ind in imu_reset_ind[lab_num]:
                if em_start <= reset_ind < em_stop:
                    ax.scatter(em_xx[reset_ind], imu_mean[:, 1][reset_ind], s=10, color='k', zorder=2)

        # for lab_num in range(3):
        #     line = ax.plot(em_xx[np.array(mc_ind[lab_num])[mc_2d_check[lab_num]]],
        #                    np.array(mc_squarred_error[lab_num])[mc_2d_check[lab_num]] ** 0.5, color='b')
        #     legend1[0].append(line[0])
        #     legend1[1].append(f'MoCap {lab_num}')
        #     line = ax.plot(em_xx[np.array(imu_ind[lab_num])[imu_2d_check[lab_num]]],
        #                    np.array(imu_squarred_error[lab_num])[imu_2d_check[lab_num]] ** 0.5, color='g')
        #     legend1[0].append(line[0])
        #     legend1[1].append(f'IMU {lab_num}')
        #     ax.set_ylabel(f'Pixel Absolute error', fontsize=fontsize, color='b')

        # mc_reset_check = [i for i, e in enumerate(mc_reset_ind) if em_start < e < em_stop]
        # [ax.axvline(em_xx[mc_reset_ind[i]], alpha=0.3, linestyle='--', color='b') for i in mc_reset_check]

        # imu_reset_check = [i for i, e in enumerate(imu_reset_ind) if em_start < e < em_stop]
        # [ax.axvline(em_xx[imu_reset_ind[i]], alpha=0.5, linestyle='--', color='g') for i in imu_reset_check]
        ax.set_xlabel(f'Time [s]', fontsize=fontsize)

        # legend = []
        for i in range(2):
            ax: plt.Axes = axs[i]
            legend = legend0 if i == 0 else legend1
            ax.legend(legend[0], legend[1], loc=2, bbox_to_anchor=(1.03, 1., 0, 0), frameon=True, fontsize=fontsize)

        ax_twin: plt.Axes = axs_twin[0]
        ax_twin.grid()
        plt.subplots_adjust(left=0.03, bottom=0.1, right=0.87, top=0.98, wspace=0, hspace=0.3)
        plt.show()
        plt.savefig(f'{vice_path}fig_curve.pdf')


plot_angle = PlotAngle()


def main():
    euroc_mav_evaluation()
    locobot_evaluation()
    blackbird_evaluation()
    # plot_angle.plot()


if __name__ == '__main__':
    main()
