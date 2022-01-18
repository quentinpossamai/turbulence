import csv
import imageio
import numpy as np
import os.path
import pickle
from scipy.signal import correlate, correlation_lags
from tqdm import tqdm
from typing import Union
import yaml

import src.data.euroc_mav.loading as euroc
from src.vice.vice import ErrorEstimation
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
    cut_camera_origin = np.zeros((len(cut_img), 4, 4))
    cut_t = np.zeros((len(cut_img), 1))
    cut_imu = np.zeros((len(cut_img), 6))
    cut_openvins = np.zeros((len(cut_img), 4, 4))

    # timestamp, x, y, z, qw, qx, qy, qz
    imu_open_vins = np.loadtxt(f'{euroc.DATASET_PATH}OpenVINS/{flight[:-1]}.csv', delimiter=',', skiprows=1)
    time, trans = imu_open_vins[:, 0], imu_open_vins[:, 1:4]
    quat = np.stack([imu_open_vins[:, 5], imu_open_vins[:, 6], imu_open_vins[:, 7], imu_open_vins[:, 4]], axis=1)
    open_vins_dict = {'t': time,
                      'pose': np.concatenate([trans, quat], axis=1),
                      'timestamp': (time * 1e9).astype(int)}

    img_path = []
    depth_path = []
    shape = (480, 752)
    img_array = np.zeros(len(cut_img), shape[0], shape[1], 3)
    depth_array = np.zeros(len(cut_img), shape[0], shape[1])
    for i, img_timestamp in enumerate(cut_img):
        ind = np.abs(img_timestamp - data["timestamp"]).argmin()
        cut_camera_origin[i] = tf_camera_origin[ind]
        cut_t[i] = data["t"][ind]
        # cut_imu[i] = data['imu_ang_acc'][ind]
        img_path.append(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{img_timestamp}.png")
        img = np.array(imageio.imread(img_path[i]))
        color_frame = np.uint8(np.zeros((shape[0], shape[1], 3)))
        for j in range(3):
            color_frame[:, :, j] = img
        img_array[i] = color_frame

        depth_path.append(f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}'
                          f'depth_estimation/{str(data["t"][ind]).replace(".", "_")}.png')
        depth_array[i] = np.array(imageio.imread(depth_path[i]))
        assert os.path.isfile(depth_path[i])
        # imu synchronisation
        open_vins_ind = np.abs(img_timestamp - open_vins_dict["timestamp"]).argmin()
        trans_quat = open_vins_dict['pose'][open_vins_ind]
        tmp = Transform.from_quat_trans(trans_quat[3:], trans_quat[:3]).as_matrix() @ tf_body_cam
        cut_openvins[i] = Transform.from_matrix(tmp).inv().as_matrix()

    quat, trans = Transform.from_matrix(cut_openvins).as_quat_trans()
    cut_openvins = Transform.from_quat_trans(-quat, trans).as_matrix()
    # quat_gt, trans_gt = Transform.from_matrix(cut_camera_origin).as_quat_trans()

    # # - Computing pose from IMU
    # dt = cut_t[1:] - cut_t[:-1]
    # ang_vel = cut_imu[:, :3]
    # acc = cut_imu[:, 3:]
    # rot_flu_imu = np.round(Rotation.from_euler('zyx', angles=np.array([180, 90, 0]), degrees=True).as_matrix().T)
    # # from IMU reference frame to FLU reference frame
    # ang_vel = (rot_flu_imu @ ang_vel.T).T
    # acc = (rot_flu_imu @ acc.T).T

    # # Integration
    # quat, trans = Transform.from_matrix(tfpose_camera_origin).as_quat_trans()
    # imu_quat = [quat[0]]
    # for i in range(1, len(quat)):
    #     dquat_dt = 1 / 2 * qmul_np(imu_quat[-1], np.concatenate([ang_vel[i], np.zeros(1)]))
    #     next_quat = dquat_dt * dt[i - 1] + imu_quat[-1]
    #     next_quat = next_quat / np.linalg.norm(next_quat)
    #     imu_quat.append(next_quat)
    # imu_quat = np.stack(imu_quat)
    # imu_lin_vel = np.concatenate([np.zeros((1, 3)), np.cumsum(acc[:-1] * dt, axis=0)], axis=0)
    # imu_trans = np.concatenate([trans[0].reshape((1, 3)), np.cumsum(imu_lin_vel[:-1] * dt, axis=0)], axis=0)
    # tf2_origin_cam = Transform.from_quat_trans(imu_quat, imu_trans).as_matrix() @ tf_body_cam
    # tfimu_cam_origin = Transform.from_matrix(tf2_origin_cam).inv().as_matrix()
    # # tf_vicon0_body

    # - Saving preprocessing
    mkdir_if_not_exist([vice_path])
    input_data = {"t": cut_t,
                  "pose": cut_camera_origin,
                  "imu": cut_openvins,
                  "img": img_array,
                  "depth": depth_array}
    name = f'vice_data'
    pickle.dump(input_data, open(f"{vice_path}{name}.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{vice_path}camera_parameters.pkl", 'wb'))
    return input_data, cam_prms


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


def synchronisation_sensibility(flight: str, ground: bool, imu: bool):
    delay = 40
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    input_data, cam_prms = euroc_mav_pre_processing(flight, delay=delay)
    print("VICE for ", flight)

    label = pickle.load(open(f"{vice_path}label.pkl", 'rb'))
    label_delayed = {}
    for key, val in label.items():
        label_delayed[key - delay] = val

    if imu:
        input_data["pose"] = input_data["imu"]
        del input_data["imu"]
    else:
        del input_data["imu"]

    n = 100
    error = np.zeros(n)
    for i in tqdm(range(n)):
        input_data0 = dict(input_data)
        input_data0['pose'] = input_data0['pose'][i:]
        estimator = ErrorEstimation(input_data=input_data0,
                                    calibration_matrix=cam_prms['K'],
                                    distortion_coefficient=cam_prms['D'],
                                    width=752,
                                    height=480,
                                    use_depth=not ground,
                                    label=label,
                                    max_distance=10.,
                                    depth_coef=7.48)
        # 2-GENERATE P3 POINTS
        mocap_vice = estimator.p3_generator(saving_path=None)

        point_gt = []
        point_mocap = []
        for j in range(len(mocap_vice)):
            broken = mocap_vice[j][1]
            if not broken:
                point_gt.append(label[j][0])
                point_mocap.append(mocap_vice[j][0])
        point_gt = np.stack(point_gt)
        point_mocap = np.stack(point_mocap)

        error[i] = np.mean((point_gt - point_mocap) ** 2)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.plot(error, label=f'pixel mse')
    ax.set_ylabel(f'Error')
    ax.set_xlabel(f'Indices lags')
    plt.grid()
    plt.legend()
    plt.show()

    # lags = correlation_lags(len(point_gt), len(point_mocap))
    #
    # corr_x = correlate(point_gt[:, 0], point_mocap[:, 0])
    # corr_x = corr_x / np.max(corr_x)
    # corr_y = correlate(point_gt[:, 1], point_mocap[:, 1])
    # corr_y = corr_y / np.max(corr_y)
    #
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(lags, corr_x, label=f'corr x')
    # ax.plot(lags, corr_y, label=f'corr y')
    # delay_x = lags[np.argmax(corr_x)]
    # ax.axvline(delay_x, label=f'delay found on x: {delay_x}', color='g')
    # delay_y = lags[np.argmax(corr_y)]
    # ax.axvline(delay_y, label=f'delay found on y: {delay_y}', color='r', linestyle='-.')
    # ax.set_ylabel(f'Cross-correlation')
    # ax.set_xlabel(f'Indices lags')
    # plt.grid()
    # plt.legend()
    # plt.show()


def main():
    flight = euroc.FLIGHT_LIST[1]
    euroc_mav_pre_processing(flight)
    case = {0: (True, False),
            1: (False, False),
            2: (True, True),
            3: (False, True)}
    ground, imu = case[3]
    error_estimation(flight, ground=ground, imu=imu)
    # flight = euroc.FLIGHT_LIST[0]
    # synchronisation_sensibility(flight, ground=True, imu=False)


if __name__ == '__main__':
    main()
