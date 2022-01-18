import json
import numpy as np
import os.path
import pickle
import yaml

import src.data.euroc_mav.loading as euroc
from src.vice.vice import ErrorEstimation
from src.utils.transform import Transform, Rotation
from src.utils.quaternion import qmul_np
from src.utils.paths import mkdir_if_not_exist


def angle(position3d_label: np.ndarray, position3d_pred: np.ndarray) -> float:
    v1 = position3d_label / np.linalg.norm(position3d_label)
    v2 = position3d_pred / np.linalg.norm(position3d_pred)

    return np.arccos(np.clip(v1 @ v2, -1, 1))


def roe(tf_gt: np.ndarray, tf_pred: np.ndarray, k: int) -> np.ndarray:
    n = len(tf_gt)
    assert tf_gt.shape == (n, 4, 4) and tf_pred.shape == (n, 4, 4)
    r_gt = Transform.from_matrix(tf_gt).as_rot_mat()
    r_pred = Transform.from_matrix(tf_pred).as_rot_mat()

    res = np.zeros(k)
    for k_ind, i in enumerate(np.round(np.linspace(0, n - 2, k)).astype(int)):
        dr = r_gt[i] @ r_gt[i + 1].T
        dr_hat = r_pred[i] @ r_pred[i + 1].T
        prod = dr @ dr_hat
        a = 1 / 2 * (prod - prod.T)
        norm_a = np.sqrt(-1 / 2 * np.trace(a @ a))
        log_r = (np.arcsin(norm_a) / norm_a).reshape((-1, 1, 1)) * a
        res[k_ind] = np.linalg.norm(log_r)
    return res


def aoe(tf_gt: np.ndarray, tf_pred: np.ndarray) -> float:
    n = len(tf_gt)
    assert tf_gt.shape == (n, 4, 4) and tf_pred.shape == (n, 4, 4)
    r_gt = Transform.from_matrix(tf_gt).as_rot_mat()
    r_pred = Transform.from_matrix(tf_pred).as_rot_mat()

    dr = r_gt @ r_pred.transpose((0, 2, 1))
    a = 1 / 2 * (dr - dr.transpose((0, 2, 1)))
    norm_a = np.sqrt(-1 / 2 * np.trace(a @ a, axis1=1, axis2=2))
    log_r = (np.arcsin(norm_a) / norm_a).reshape((-1, 1, 1)) * a
    return np.sqrt(np.linalg.norm(log_r.reshape(-1, 9), axis=0)[1:].mean())  # first element is initialised with gt


def vice_evaluation(flight: str, ground: bool, imu: bool):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    cam_prms = pickle.load(open(f"{vice_path}camera_parameters.pkl", 'rb'))
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name
    if imu:
        input_data["pose"] = input_data["imu"]
        del input_data["imu"]
    else:
        del input_data["imu"]
    prediction = pickle.load(open(f'{vice_path}{name}.pkl', 'rb'))
    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                distortion_coefficient=cam_prms['D'],
                                width=752,
                                height=480,
                                use_depth=not ground,
                                label=label,
                                max_distance=1000.,
                                depth_coef=7.48)
    mse_error = []
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

            # mse_error
            if isinstance(prediction_3d, np.ndarray) and isinstance(label_3d, np.ndarray):
                error1 = ((label_3d - prediction_3d) ** 2).mean()
                mse_error.append(error1)

                # angle error
                error2 = angle(label_3d, prediction_3d)
                angle_error.append(error2)
                # print(f'i: {i} | mse_error: {error1} | angle error: {error2}')

    err1, err2 = np.mean(mse_error), np.mean(angle_error)
    print(f'Flight: {flight} | ground: {ground} | imu: {imu}')
    print(f'Error length: {len(mse_error)}')
    print(f'mse_error: {err1:1.4e} | angle_error: {err2:1.4e}\n\n')
    return err1, err2


def evaluation():
    for imu in [True]:
        for ground in [True, False]:
            sum_mse = 0
            sum_ang = 0
            for i in range(3):
                mse, ang = vice_evaluation(euroc.FLIGHT_LIST[i], ground=ground, imu=imu)
                sum_mse += mse
                sum_ang += ang
            print(f'Total mse ground: {sum_mse:1.4e} | Total ang ground: {sum_ang:1.4e}\n')

    # MSE IMU MOCAP
    for i in range(3):
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{euroc.FLIGHT_LIST[i]}vice/'
        input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))

        trans_gt = Transform.from_matrix(input_data['pose']).as_trans()
        trans_pred = Transform.from_matrix(input_data['imu']).as_trans()
        mse_error = ((trans_gt - trans_pred) ** 2).mean()

        aoe_error = aoe(input_data['pose'], input_data['imu'])

        roe_error = roe(input_data['pose'], input_data['imu'], 100)
        roe_error = roe_error.mean()

        print(f'flight: {euroc.FLIGHT_LIST[i]} | mse error: {mse_error:1.4e} | aoe error: {aoe_error:1.4e} | '
              f'roe error: {roe_error:1.4e} ')


def main():
    evaluation()


if __name__ == '__main__':
    main()
