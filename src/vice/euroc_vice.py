import json
import numpy as np
import os.path
import pickle
import yaml

import src.data.euroc_mav.loading as euroc
from src.vice.posture_error_estimation import ErrorEstimation
from src.utils.transform import Transform, Rotation
from src.utils.quaternion import qmul_np
from src.utils.paths import mkdir_if_not_exist


def euroc_mav_pre_processing(flight: str):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    print("Processing: ", flight)
    # 1-LOADING
    # 1.1-Loading camera parameters
    with open(f"{euroc.DATASET_PATH}raw/{flight}cam0/sensor.yaml", 'r') as stream:
        # /Users/quentin/phd/turbulence/data/euroc_mav/raw/V1_01_easy/cam0
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # K camera matrix : intrinsic parameters
    k = np.eye(3)
    k[0, 0] = yaml_file['intrinsics'][0]
    k[1, 1] = yaml_file['intrinsics'][1]
    k[0, 2] = yaml_file['intrinsics'][2]
    k[1, 2] = yaml_file['intrinsics'][3]
    tf_body_cam = np.array(yaml_file['T_BS']['data']).reshape((4, 4))
    cam_prms = {'K': k,
                'D': np.array([e for e in yaml_file['distortion_coefficients']]),
                'fps': yaml_file['rate_hz'],
                'tf_body_cam': tf_body_cam}  # wrt. the body-frame

    # 1.2-Loading motion capture parameters
    with open(f"{euroc.DATASET_PATH}raw/{flight}vicon0/sensor.yaml", 'r') as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    tf_vicon0_body = Transform.from_matrix(np.array(yaml_file['T_BS']['data']).reshape((4, 4))).inv().as_matrix()
    cam_prms['tf_vicon0_cam'] = Transform.from_matrix(tf_vicon0_body @ cam_prms['tf_body_cam']).as_matrix()
    del cam_prms['tf_body_cam']

    # 1.3-Loading camera trajectory
    data = euroc.load_flight(flight)  # dict_keys(['t', 'imu_ang_acc', 'motor', 'img', 'state', 'residual'])
    trans = data["state"][:, :3] + euroc.TRANS_MEAN[flight].reshape((1, 3))
    tf_origin_vicon0 = Transform.from_quat_trans(quat=data["state"][:, 3:7], trans=trans).as_matrix()
    tf_camera_origin = Transform.from_matrix(tf_origin_vicon0 @ cam_prms["tf_vicon0_cam"]).inv().as_matrix()

    # 1.5-Cutting at the image frequency
    cut_img = np.unique(data["cam0"])
    tfpose_camera_origin = np.zeros((len(cut_img), 4, 4))
    cut_t = np.zeros((len(cut_img), 1))
    cut_imu = np.zeros((len(cut_img), 6))

    img_path = []
    depth_path = []
    for i, img_timestamp in enumerate(cut_img):
        ind = np.abs(img_timestamp - data["timestamp"]).argmin()
        tfpose_camera_origin[i] = tf_camera_origin[ind]
        cut_t[i] = data["t"][ind]
        cut_imu[i] = data['imu_ang_acc'][ind]
        img_path.append(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{img_timestamp}.png")
        depth_path.append(f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}'
                          f'depth_estimation/{str(data["t"][ind]).replace(".", "_")}.png')

        assert os.path.isfile(depth_path[i])

    # - Computing pose from IMU
    dt = cut_t[1:] - cut_t[:-1]
    ang_vel = cut_imu[:, :3]
    acc = cut_imu[:, 3:]
    rot_flu_imu = np.round(Rotation.from_euler('zyx', angles=np.array([180, 90, 0]), degrees=True).as_matrix().T)
    # from IMU reference frame to FLU reference frame
    ang_vel = (rot_flu_imu @ ang_vel.T).T
    acc = (rot_flu_imu @ acc.T).T

    # Integration
    quat, trans = Transform.from_matrix(tfpose_camera_origin).as_quat_trans()
    imu_quat = [quat[0]]
    for i in range(1, len(quat)):
        dquat_dt = 1 / 2 * qmul_np(imu_quat[-1], np.concatenate([ang_vel[i], np.zeros(1)]))
        next_quat = dquat_dt * dt[i - 1] + imu_quat[-1]
        next_quat = next_quat / np.linalg.norm(next_quat)
        imu_quat.append(next_quat)
    imu_quat = np.stack(imu_quat)
    imu_lin_vel = np.concatenate([np.zeros((1, 3)), np.cumsum(acc[:-1] * dt, axis=0)], axis=0)
    imu_trans = np.concatenate([trans[0].reshape((1, 3)), np.cumsum(imu_lin_vel[:-1] * dt, axis=0)], axis=0)
    tf2_origin_cam = Transform.from_quat_trans(imu_quat, imu_trans).as_matrix() @ tf_body_cam
    tfimu_cam_origin = Transform.from_matrix(tf2_origin_cam).inv().as_matrix()
    # tf_vicon0_body

    # - Saving preprocessing
    mkdir_if_not_exist([vice_path])
    input_data = {"t": cut_t,
                  "pose": tfpose_camera_origin,
                  "imu": tfimu_cam_origin,
                  "img": img_path,
                  "depth": depth_path}
    pickle.dump(input_data, open(f"{vice_path}vice_data.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{vice_path}camera_parameters.pkl", 'wb'))


def error_estimation(flight: str, ground: bool, imu: bool):
    # 1-INITIALISATION
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    print("VICE for ", flight)

    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    cam_prms = pickle.load(open(f"{vice_path}camera_parameters.pkl", 'rb'))
    label = pickle.load(open(f"{vice_path}label.pkl", 'rb'))
    if imu:
        input_data["pose"] = input_data["imu"]
        del input_data["imu"]
    else:
        del input_data["imu"]
    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                distortion_coefficient=cam_prms['D'],
                                width=752,
                                height=480,
                                use_depth=not ground,
                                label=label,
                                max_distance=10.,
                                depth_coef=7.48)
    # 2-GENERATE P3 POINTS
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name
    estimator.p3_generator(saving_path=f'{vice_path}{name}.pkl')

    # 3-GENERATING THE VIDEO
    estimator.build_video(fps=20, saving_path=f'{vice_path}{name}.mkv')


def position_2d_error(flight):
    # 1. Loading VICE result
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    vice_point = pickle.load(open(f'{vice_path}p3_points.pkl', 'rb'))
    vice_point = np.stack(vice_point, axis=0).astype(int)

    # 2. Loading annotations
    with open(f'{vice_path}task_v1_01_easy_backup_2021_10_25_14_25_23/annotations.json') as json_file:
        data = json.load(json_file)
    key_points = np.zeros((len(data[0]['tracks'][0]['shapes']), 3))  # frame_id, point_x, point_y
    for i, e in enumerate(data[0]['tracks'][0]['shapes']):
        key_points[i] = np.array([e['frame']] + e['points'])
    point_x = np.round(np.interp(range(len(vice_point)), key_points[:, 0], key_points[:, 1])).astype(int)
    point_y = np.round(np.interp(range(len(vice_point)), key_points[:, 0], key_points[:, 2])).astype(int)
    annotation_point = np.stack([point_x, point_y], axis=1)

    # 3. Computing error
    return np.linalg.norm(vice_point - annotation_point)


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
                                max_distance=10.,
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
            if isinstance(prediction_3d, np.ndarray):
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
    # for imu in [False, True]:
    #     for ground in [True, False]:
    #         sum_mse = 0
    #         sum_ang = 0
    #         for i in range(3):
    #             mse, ang = vice_evaluation(euroc.FLIGHT_LIST[i], ground=ground, imu=imu)
    #             sum_mse += mse
    #             sum_ang += ang
    #         print(f'Total mse ground: {sum_mse:1.4e} | Total ang ground: {sum_ang:1.4e}\n')

    mse_sum = 0
    aoe_sum = 0
    _sum = 0
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
    # flight = euroc.FLIGHT_LIST[0]
    # case = {0: (True, False),
    #         1: (False, False),
    #         2: (True, True),
    #         3: (False, True)}
    # ground, imu = case[3]
    # euroc_mav_pre_processing(flight)
    # error_estimation(flight, ground=ground, imu=imu)

    evaluation()


if __name__ == '__main__':
    main()
