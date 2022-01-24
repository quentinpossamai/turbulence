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


def vice_error(estimator, label, prediction):
    squared_3d_error = []
    angle_error = []
    index_3d = []

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

            label_3d = estimator.image23d(i, point)
            prediction_3d = estimator.image23d(i, p3.astype(float))

            if isinstance(prediction_3d, np.ndarray) and isinstance(label_3d, np.ndarray):
                index_3d.append(i)
                # mse_error
                squared_3d_error.append(np.sum((label_3d - prediction_3d) ** 2))

                # angle error
                error2 = angle(label_3d, prediction_3d)
                angle_error.append(error2)

    squared_2d_error = np.array(squared_2d_error)
    index_2d = np.array(index_2d)
    print(f'Error 2D length: {len(squared_2d_error)}')
    rmse_2d = np.sqrt(np.mean(squared_2d_error))
    print(f'RMSE 2D (px): {rmse_2d:.3f}')

    index_3d = np.array(index_3d)
    squared_3d_error = np.array(squared_3d_error)
    angle_error = np.array(angle_error)
    print(f'Error 3D length: {len(squared_3d_error)}')
    rmse_3d, mean_angle_error = np.mean(squared_3d_error), np.mean(angle_error)
    print(f'RMSE 3D (m): {rmse_3d:.3f} | mean angle_error (ª): {mean_angle_error:.2f}\n\n')
    return (rmse_2d, squared_2d_error, index_2d,
            rmse_3d, squared_3d_error, mean_angle_error, angle_error, index_3d)


def specific_euroc_mav_evaluation(flight: str, ground: bool, imu: bool):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name

    prediction = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))
    estimator = euroc_vice.build_estimator(flight, ground, imu)

    print(f'Flight: {flight} | ground: {ground} | imu: {imu}')
    (rmse_2d, squared_2d_error, index_2d,
     rmse_3d, squared_3d_error, mean_angle_error, angle_error, index_3d) = vice_error(estimator, label, prediction)
    return (rmse_2d, squared_2d_error, index_2d,
            rmse_3d, squared_3d_error, mean_angle_error, angle_error, index_3d)


def euroc_mav_evaluation():
    for imu in [False, True]:
        for ground in [True, False]:
            for i in range(3):
                (rmse_2d, squared_2d_error, index_2d,
                 rmse_3d, squared_3d_error, mean_angle_error, angle_error,
                 index_3d) = specific_euroc_mav_evaluation(euroc.FLIGHT_LIST[i], ground=ground, imu=imu)
                key1 = f'imu' if imu else f'mocap'
                key2 = f'_ground' if ground else f'_depth'
                key3 = f'_{euroc.FLIGHT_LIST[i][:-1]}'
                plot_angle.euroc_vice_error[f'{key1}{key2}{key3}'] = (rmse_2d, squared_2d_error, index_2d,
                                                                      rmse_3d, squared_3d_error, mean_angle_error,
                                                                      angle_error, index_3d)

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
              f'roe error: {roe_error:.3f} ')


def specific_locobot_evaluation(flight: str, ground: bool, pose: str):
    assert pose in ['odom', 'slam']

    # 1-INITIALISATION
    vice_path = f'{locobot.LOCOBOT_DATA_PATH}{flight}'
    print("VICE for ", flight)

    label = pickle.load(open(f"{vice_path}label.pkl", 'rb'))
    estimator = locobot_vice.build_estimator(flight, ground, pose)

    name = f'ground_{pose}' if ground else f'depth_{pose}'
    prediction = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))

    print(f'Flight: {flight} | ground: {ground} | pose: {pose}')

    (rmse_2d, squared_2d_error, index_2d,
     rmse_3d, squared_3d_error, mean_angle_error, angle_error, index_3d) = vice_error(estimator, label, prediction)
    return (rmse_2d, squared_2d_error, index_2d,
            rmse_3d, squared_3d_error, mean_angle_error, angle_error, index_3d)


def locobot_evaluation():
    for pose in ['odom', 'slam']:
        for ground in [True, False]:
            (rmse_2d, squared_2d_error, index_2d,
             rmse_3d, squared_3d_error, mean_angle_error, angle_error,
             index_3d) = specific_locobot_evaluation(locobot.TRAJ_LIST[1], ground=ground, pose=pose)
            key1 = pose
            key2 = f'_ground' if ground else f'_depth'
            plot_angle.locobot_vice_error[f'{key1}{key2}'] = (rmse_2d, squared_2d_error, index_2d,
                                                              rmse_3d, squared_3d_error, mean_angle_error,
                                                              angle_error, index_3d)
    vice_path = f'{locobot.LOCOBOT_DATA_PATH}{locobot.TRAJ_LIST[1]}'
    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    plot_angle.locobot_data['data'] = input_data


def specific_blackbird_evaluation(ground: bool, imu: bool):
    vice_path = f'{blackbird.VICE_PATH}'
    label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name

    prediction = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))
    estimator = blackbird_vice.build_estimator(ground, imu)

    print(f'Flight: {vice_path} | ground: {ground} | imu: {imu}')
    (rmse_2d, squared_2d_error, index_2d,
     rmse_3d, squared_3d_error, mean_angle_error, angle_error, index_3d) = vice_error(estimator, label, prediction)
    return (rmse_2d, squared_2d_error, index_2d,
            rmse_3d, squared_3d_error, mean_angle_error, angle_error, index_3d)


def blackbird_evaluation():
    for imu in [False, True]:  # True not implemented yet
        for ground in [True, False]:
            (rmse_2d, squared_2d_error, index_2d,
             rmse_3d, squared_3d_error, mean_angle_error, angle_error,
             index_3d) = specific_blackbird_evaluation(ground=ground, imu=imu)
            key1 = f'imu' if imu else f'mocap'
            key2 = f'_ground' if ground else f'_depth'
            plot_angle.blackbird_vice_error[f'{key1}{key2}'] = (rmse_2d, squared_2d_error, index_2d,
                                                                rmse_3d, squared_3d_error, mean_angle_error,
                                                                angle_error, index_3d)

    # MSE IMU MOCAP
    vice_path = blackbird.VICE_PATH
    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    plot_angle.blackbird_data['data'] = input_data

    trans_mc = Tf.from_matrix(input_data['pose']).as_trans()
    trans_imu = Tf.from_matrix(input_data['imu']).as_trans()
    rot_mc = Tf.from_matrix(input_data['pose']).as_rot_mat()
    rot_imu = Tf.from_matrix(input_data['imu']).as_rot_mat()

    rmse = np.sqrt(np.mean(np.sum((trans_mc - trans_imu) ** 2, axis=1), axis=0))

    aoe_error = aoe(rot_mc, rot_imu, no_mean=False)

    roe_error = roe(rot_mc, rot_imu, 100)
    roe_error = roe_error.mean()

    print(f'flight: {vice_path} | mse error: {rmse:.3f} | aoe error: {aoe_error:.2f} | '
          f'roe error: {roe_error:.3f} ')


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

        plt.rcParams.update({"text.usetex": False,
                             "font.family": "serif",
                             "font.sans-serif": ["Computer Modern Roman"]})

        # Euroc MAV
        i, ground = 0, True
        flight = euroc.FLIGHT_LIST[i]
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
        key3 = f'_{flight[:-1]}'
        key2 = f'_ground' if ground else f'_depth'

        imu = False
        key1 = f'imu' if imu else f'mocap'
        name = 'ground' if ground else 'depth'
        name = f'{name}imu' if imu else name
        (_,  # rmse_2d
         mc_squared_2d_error,
         mc_index_2d,
         _,  # rmse_3d
         mc_squared_3d_error,
         _,  # mean_angle_error
         mc_angle_error,
         mc_index_3d) = self.euroc_vice_error[f'{key1}{key2}{key3}']  # euroc_mav mocap angle error
        mc_p3 = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))
        mc_reset_ind = np.array([i for i, p3 in enumerate(mc_p3) if isinstance(p3[0], np.ndarray) and p3[1]])

        imu = True
        key1 = f'imu' if imu else f'mocap'
        name = 'ground' if ground else 'depth'
        name = f'{name}imu' if imu else name
        (_,  # rmse_2d
         imu_squared_2d_error,
         imu_index_2d,
         _,  # rmse_3d
         imu_squared_3d_error,
         _,  # mean_angle_error
         imu_angle_error,
         imu_index_3d) = self.euroc_vice_error[f'{key1}{key2}{key3}']  # euroc_mav imu angle error
        imu_p3 = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))
        imu_reset_ind = np.array([i for i, p3 in enumerate(imu_p3) if isinstance(p3[0], np.ndarray) and p3[1]])

        em_data = self.euroc_data[euroc.FLIGHT_LIST[i][:-1]]
        mc_trans = Tf.from_matrix(em_data['pose']).as_trans()
        imu_trans = Tf.from_matrix(em_data['imu']).as_trans()
        mc_rot = Tf.from_matrix(em_data['pose']).as_rot_mat()
        imu_rot = Tf.from_matrix(em_data['imu']).as_rot_mat()
        em_ao = aoe(mc_rot, imu_rot, no_mean=True)
        squared_3d_error = np.sum((mc_trans - imu_trans) ** 2, axis=1)

        em_xx = np.linspace(0, len(em_data['pose']) / 20, len(em_data['pose']))  # Converting in sec
        em_start, em_stop = int(len(em_xx) * 0), int(len(em_xx) * 0.2)








        # - Plot
        # style_label = 'default'
        fontsize = 9
        fig: plt.Figure
        fig, axs = plt.subplots(2, 1, figsize=(2 * 5, 1 * 5))
        axs_twin = []
        legend0 = ([], [])


        # FIRST AXE ----------------------------
        ax: plt.Axes = axs[0]
        line = ax.plot(em_xx[em_start:em_stop], squared_3d_error[em_start:em_stop] ** 0.5, color='b')
        legend0[0].append(line[0])
        legend0[1].append('Absolute error')
        ax.set_ylabel(f'Absolute error', fontsize=fontsize, color='b')

        ax_twin = ax.twinx()
        axs_twin.append(ax_twin)
        line = ax_twin.plot(em_xx[em_start:em_stop], em_ao[em_start:em_stop] ** 0.5, color='g')
        legend0[0].append(line[0])
        legend0[1].append('AOE')
        ax_twin.set_ylabel(f'AOE', fontsize=fontsize, color='g')

        ax.set_xlabel(f'Time [s]', fontsize=fontsize)

        # SECOND AXE ----------------------------
        ax: plt.Axes = axs[1]
        legend1 = ([], [])
        mc_2d_check = [i for i, e in enumerate(mc_index_2d) if em_start < e < em_stop]
        mc_3d_check = [i for i, e in enumerate(mc_index_3d) if em_start < e < em_stop]
        imu_2d_check = [i for i, e in enumerate(imu_index_2d) if em_start < e < em_stop]
        imu_3d_check = [i for i, e in enumerate(imu_index_3d) if em_start < e < em_stop]
        line = ax.plot(em_xx[np.array(mc_index_2d)[mc_2d_check]], np.array(mc_squared_2d_error)[mc_2d_check] ** 0.5,
                       color='b')
        legend1[0].append(line[0])
        legend1[1].append('MoCap Pixel error')
        line = ax.plot(em_xx[np.array(imu_index_2d)[imu_2d_check]], np.array(imu_squared_2d_error)[imu_2d_check] ** 0.5,
                       color='g')
        legend1[0].append(line[0])
        legend1[1].append('IMU Pixel error')
        ax.set_ylabel(f'Pixel ABS error', fontsize=fontsize, color='b')

        ax_twin = ax.twinx()
        axs_twin.append(ax_twin)
        line = ax_twin.plot(em_xx[np.array(mc_index_3d)[mc_3d_check]], np.array(mc_squared_3d_error)[mc_3d_check] ** 0.5,
                            color='c')
        legend1[0].append(line[0])
        legend1[1].append('MoCap 3D error')
        line = ax_twin.plot(em_xx[np.array(imu_index_3d)[imu_3d_check]], np.array(imu_squared_3d_error)[imu_3d_check] ** 0.5,
                            color='lime')
        legend1[0].append(line[0])
        legend1[1].append('IMU 3D error')
        ax_twin.set_ylabel(f'3D ABS error', fontsize=fontsize, color='g')

        mc_reset_check = [i for i, e in enumerate(mc_reset_ind) if em_start < e < em_stop]
        [ax.axvline(em_xx[mc_reset_ind[i]], alpha=0.3, linestyle='--', color='b') for i in mc_reset_check]
        # imu_reset_check = [i for i, e in enumerate(imu_reset_ind) if em_start < e < em_stop]
        # [ax.axvline(em_xx[imu_reset_ind[i]], alpha=0.5, linestyle='--', color='g') for i in imu_reset_check]

        # ax.plot(em_xx[np.array(mc_index_3d)[mc_3d_check]], np.array(mc_angle_error)[mc_3d_check],
        #         label=r'MoCap angle error')
        # ax.plot(em_xx[np.array(imu_index_3d)[imu_3d_check]], np.array(imu_angle_error)[imu_3d_check],
        #         label=r'IMU angle error')
        ax.set_xlabel(f'Time [s]', fontsize=fontsize)

        # legend = []
        for i in range(2):
            ax: plt.Axes = axs[i]
            legend = legend0 if i == 0 else legend1
            ax.legend(legend[0], legend[1], loc=2, bbox_to_anchor=(1.08, 1., 0, 0), frameon=True, fontsize='x-small')

            ax_twin: plt.Axes = axs_twin[i]
            ax_twin.grid()
        plt.subplots_adjust(left=0.07, bottom=0.1, right=0.8, top=0.98, wspace=0, hspace=0.3)
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
