import numpy as np
import os.path
import pickle
from scipy.signal import correlate
from typing import Union
import yaml

import src.data.locobot.loading as locobot
from src.vice.posture_error_estimation import ErrorEstimation
from src.utils.transform import Transform, Rotation
from src.utils.quaternion import qmul_np
from src.utils.paths import mkdir_if_not_exist


def locobot_pre_processing(traj: str, delay: float):
    vice_path = f'{locobot.LOCOBOT_DATA_PATH}{traj}/'
    print("Processing: ", traj)
    # 1-LOADING
    # 1.1-Loading camera parameters

    rgb_cam = np.array([[607.0703735351562, 0.0, 332.4421691894531],
                        [0.0, 607.4769897460938, 238.6558837890625],
                        [0.0, 0.0, 1.0]])

    depth_cam = np.array([[382.2403564453125, 0.0, 320.8781433105469],
                          [0.0, 382.2403564453125, 237.9492950439453],
                          [0.0, 0.0, 1.0]])
    # K camera matrix : intrinsic parameters
    k = rgb_cam
    cam_prms = {'K': k,
                'D': [0, 0, 0, 0, 0, 0, 0, 0],
                'fps': 30}  # wrt. the body-frame

    # 1.3-Loading camera trajectory
    data = locobot.loading_traj(traj)
    tf_camera_origin = Transform.from_matrix().inv().as_matrix()

    # 1.5-Cutting at the image frequency
    cut_img = np.unique(data["cam0"])
    tfpose_camera_origin = np.zeros((len(cut_img), 4, 4))
    cut_t = np.zeros((len(cut_img), 1))
    cut_imu = np.zeros((len(cut_img), 6))

    img_path = []
    depth_path = []
    for i, img_timestamp in enumerate(cut_img):
        img_timestamp = img_timestamp + delay
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
    name = f'vice_data' if delay == 0. else f'vice_data_delay'
    pickle.dump(input_data, open(f"{vice_path}{name}.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{vice_path}camera_parameters.pkl", 'wb'))


def error_estimation(flight: str, ground: bool, imu: bool, sensibility: Union[str, None]):
    assert sensibility in [None, 'camera_matrix', 'distortion', 'tf_camera_body', 'pose_noise']
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
    delay_gt = 0
    euroc_mav_pre_processing(flight, delay_gt)
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    print("VICE for ", flight)

    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    time = input_data['t']
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
    out_gt = estimator.p3_generator(None)
    point_gt = np.stack([e[0] for e in out_gt if not e[1]], axis=0)

    delay = 4
    euroc_mav_pre_processing(flight, delay)
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    print("VICE for ", flight)

    input_data = pickle.load(open(f"{vice_path}vice_data_delay.pkl", 'rb'))
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
    out_delay = estimator.p3_generator(saving_path=None)
    point_delay = np.stack([e[0] for e in out_delay if not e[1]], axis=0)
    corr = correlate(point_gt, point_delay)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1)
    ax.plot(time, corr, label=f't={delay}')
    plt.grid()
    plt.show()
    print()
    # correlate(, )


def main():
    traj = locobot.TRAJ_LIST[0]
    case = {0: (True, False),
            1: (False, False),
            2: (True, True),
            3: (False, True)}
    ground, imu = case[0]
    locobot_pre_processing(traj, 0)
    error_estimation(traj, ground=ground, imu=imu, sensibility=None)


if __name__ == '__main__':
    main()
