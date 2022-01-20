import numpy as np
import pickle

import src.data.euroc_mav.loading as euroc
import src.vice.euroc_vice as euroc_vice

import src.data.locobot.loading as locobot
import src.vice.locobot_vice as locobot_vice

from src.vice.vice import ErrorEstimation
from src.utils.transform import Transform


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


def aoe(rot_gt: np.ndarray, rot_pred: np.ndarray) -> float:
    """
    Compute the relative orientation error defined in:

    Brossard, M., Bonnabel, S. and Barrau, A. (2020) ‘Denoising IMU Gyroscopes with Deep Learning for Open-Loop Attitude
    Estimation’, IEEE Robotics and Automation Letters, PP, pp. 1–1. doi:10.1109/LRA.2020.3003256.

    param rot_gt: (n, 3, 3).  Trajectory ground truth as rotation matrices.

    param rot_pred: (n, 3, 3). Trajectory predicted as rotation matrices.

    return: The averaged absolute orientation error over the trajectory.
    """
    n = len(rot_gt)
    assert rot_gt.shape == (n, 3, 3) and rot_pred.shape == (n, 3, 3)

    dr = rot_gt @ rot_pred.transpose((0, 2, 1))
    a = 1 / 2 * (dr - dr.transpose((0, 2, 1)))
    norm_a = np.sqrt(-1 / 2 * np.trace(a @ a, axis1=1, axis2=2))
    log_r = (np.arcsin(norm_a) / norm_a).reshape((-1, 1, 1)) * a
    return np.sqrt(np.linalg.norm(log_r.reshape(-1, 9), axis=0)[1:].mean())  # first element is initialised with gt


def vice_error(estimator, label, prediction):
    mrse_error = []
    angle_error = []
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
            label_3d = estimator.image23d(i, point)
            prediction_3d = estimator.image23d(i, p3.astype(float))

            if isinstance(prediction_3d, np.ndarray) and isinstance(label_3d, np.ndarray):
                # mse_error
                error1 = np.linalg.norm(label_3d - prediction_3d)
                mrse_error.append(error1)

                # angle error
                error2 = angle(label_3d, prediction_3d)
                angle_error.append(error2)

    print(f'Error length: {len(mrse_error)}')
    mrse_error, angle_error = np.mean(mrse_error), np.mean(angle_error)
    print(f'mean root squared error (m): {mrse_error:.3f} | mean angle_error (ª): {angle_error:.2f}\n\n')
    return mrse_error, angle_error


def specific_euroc_mav_evaluation(flight: str, ground: bool, imu: bool):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name

    prediction = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))
    estimator = euroc_vice.build_estimator(flight, ground, imu)

    print(f'Flight: {flight} | ground: {ground} | imu: {imu}')
    mrse_error, angle_error = vice_error(estimator, label, prediction)
    return mrse_error, angle_error


def euroc_mav_evaluation():
    # for imu in [False, True]:
    #     for ground in [True, False]:
    #         for i in range(3):
    #             specific_euroc_mav_evaluation(euroc.FLIGHT_LIST[i], ground=ground, imu=imu)

    # MSE IMU MOCAP
    for i in range(3):
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{euroc.FLIGHT_LIST[i]}vice/'
        input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))

        trans_mc = Transform.from_matrix(input_data['pose']).as_trans()
        trans_imu = Transform.from_matrix(input_data['imu']).as_trans()
        rot_mc = Transform.from_matrix(input_data['pose']).as_rot_mat()
        rot_imu = Transform.from_matrix(input_data['imu']).as_rot_mat()

        mrse_error = np.linalg.norm(trans_mc - trans_imu, axis=1).mean()

        aoe_error = aoe(rot_mc, rot_imu)

        roe_error = roe(rot_mc, rot_imu, 100)
        roe_error = roe_error.mean()

        print(f'flight: {euroc.FLIGHT_LIST[i]} | mse error: {mrse_error:.3f} | aoe error: {aoe_error:.2f} | '
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

    mrse_error, angle_error = vice_error(estimator, label, prediction)
    return mrse_error, angle_error


def locobot_evaluation():
    for pose in ['odom', 'slam']:
        for ground in [True, False]:
            specific_locobot_evaluation(locobot.TRAJ_LIST[1], ground=ground, pose=pose)


def main():
    locobot_evaluation()


if __name__ == '__main__':
    main()
