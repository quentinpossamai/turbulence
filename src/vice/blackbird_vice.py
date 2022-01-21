import cv2
import imageio
import numpy as np
import pickle
from PIL import Image
from scipy import interpolate
from tqdm import tqdm
from typing import Union

import src.data.blackbird.loading as blackbird
from src.vice.vice import ErrorEstimation
from src.utils.transform import Transform as Tf

coef_multiplicator_depth = 50

START, STOP = 15556 // (4 * 60 + 19) * (2 * 60 + 40), 15556 // (4 * 60 + 19) * (3 * 60)
SHAPE = [768, 1024]
FPS = 30


def video_to_png():
    vice_path = blackbird.VICE_PATH
    video = cv2.VideoCapture(f'{vice_path}rgb/compressed.m4v')
    rgb_time = np.loadtxt(f'{vice_path}rgb/nSecTimestamps.txt', dtype=int)

    progress = tqdm(range(len(rgb_time)), desc=f'Converting video to image')
    frame_number = 0
    is_reading, frame = video.read()  # read video is in bgr.
    Image.fromarray(frame[:, :, ::-1]).save(f'{vice_path}rgb/images/{rgb_time[frame_number]}.png')
    while is_reading:
        progress.update()
        frame_number += 1
        is_reading, frame = video.read()
        Image.fromarray(frame[:, :, ::-1]).save(f'{vice_path}rgb/images/{rgb_time[frame_number]}.png')


def blackbird_pre_processing():
    # - Loading data
    print(f'Loading {blackbird.VICE_PATH}')
    vice_path = blackbird.VICE_PATH
    rgb_time = np.loadtxt(f'{vice_path}rgb/nSecTimestamps.txt', dtype=int)
    depth_time = np.loadtxt(f'{vice_path}depth/nSecTimestamps.txt', dtype=int)
    mc_time = np.loadtxt(f'{vice_path}blackbird_slash_state.csv', dtype=int, delimiter=',', skiprows=1,
                         usecols=[0])
    mc_state = np.loadtxt(f'{vice_path}blackbird_slash_state.csv', dtype=float, delimiter=',', skiprows=1,
                          usecols=[9, 10, 11, 13, 14, 15, 16])
    # imu_time = np.loadtxt(f'{vice_path}blackbird_slash_imu.csv', dtype=int, delimiter=',', skiprows=1,
    #                       usecols=[0])
    # imu_ang_acc = np.loadtxt(f'{vice_path}blackbird_slash_imu.csv', dtype=float, delimiter=',', skiprows=1,
    #                  usecols=[22, 23, 24, 35, 36, 37])
    imu_dict = pickle.load(open(f'{vice_path}imu.pkl', 'rb'))
    imu_time = imu_dict['time']
    imu_state = np.concatenate([imu_dict['integrated_pos'], imu_dict['integrated_quat']], axis=1)

    assert (depth_time == rgb_time).sum() == len(rgb_time)  # No changes on depth time

    # - Cut trajectory
    rgb_time = rgb_time[START:STOP]
    assert (imu_time == rgb_time).sum() == len(rgb_time)  # No changes on depth time

    # Synchronization
    mc_tf_mocap_body = np.zeros((len(rgb_time), 4, 4))
    imu_tf_mocap_body = np.zeros((len(rgb_time), 4, 4))
    img_path = []
    depth_path = []
    for i, img_timestamp in enumerate(rgb_time):
        img_path.append(f'{blackbird.VICE_PATH}rgb/images/{img_timestamp}.png')

        depth_timestamp = depth_time[np.abs(img_timestamp - depth_time).argmin()]
        depth_path.append(f'{blackbird.VICE_PATH}depth/lossless/{depth_timestamp}.png')

        state_ind = np.abs(img_timestamp - mc_time).argmin()
        mc_tf_mocap_body[i] = Tf.from_quat_trans(mc_state[state_ind][3:7], mc_state[state_ind][:3]).as_matrix()

        imu_ind = np.abs(img_timestamp - imu_time).argmin()
        imu_tf_mocap_body[i] = Tf.from_quat_trans(imu_state[imu_ind][3:7], imu_state[imu_ind][:3]).as_matrix()

    tf_180 = Tf.from_euler_trans(seq='xyz', angles=np.array([180, 0, 0]), degrees=True, trans=np.zeros(3)).as_matrix()
    tf_fgenv_norm = Tf.from_quat_trans(np.eye(4)[0], np.array([-0.04, -0.09, -0.5])).as_matrix()  # for patrick
    tf_norm_mocap = Tf.from_quat_trans(np.eye(4)[0], np.array([0.0, -0.5, -0.883718])).as_matrix()
    tf_body_camera = Tf.from_euler_trans(seq='xyz', angles=np.array([90, 0, 90]), degrees=True,
                                         trans=np.array([0, -0.05, 0])).as_matrix()

    mc_tf_fgenv_camera = tf_180 @ tf_fgenv_norm @ tf_norm_mocap @ mc_tf_mocap_body @ tf_body_camera
    mc_tf_camera_fgenv = Tf.from_matrix(mc_tf_fgenv_camera).inv().as_matrix()

    imu_tf_fgenv_camera = tf_180 @ tf_fgenv_norm @ tf_norm_mocap @ imu_tf_mocap_body @ tf_body_camera
    imu_tf_camera_fgenv = Tf.from_matrix(imu_tf_fgenv_camera).inv().as_matrix()

    # quat_mc, trans_mc = Tf.from_matrix(mc_tf_camera_fgenv).as_quat_trans()
    # quat_imu, trans_imu = Tf.from_matrix(imu_tf_camera_fgenv).as_quat_trans()
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(3, 2)
    # for i in range(3):
    #     ax[i][0].plot(rgb_time, trans_mc[:, i], label="GT")
    #     ax[i][1].plot(rgb_time, quat_mc[:, i], label="GT")
    #
    #     ax[i][0].plot(rgb_time, trans_imu[:, i], label="IMU")
    #     ax[i][1].plot(rgb_time, quat_imu[:, i], label="IMU")
    #
    #     ax[i][0].legend()
    #     ax[i][1].legend()
    # plt.show()

    cam_prms = {'K': np.array([[665.107510106, 0, 511.5],
                               [0, 665.107510106, 383.5],
                               [0, 0, 1]]),
                'D': [0, 0, 0, 0, 0, 0],
                'fps': FPS}

    # Cutting at the image frequency
    input_data = {"pose": mc_tf_camera_fgenv,
                  "imu": imu_tf_camera_fgenv,
                  "img_path": img_path,
                  "depth_path": depth_path}

    pickle.dump(input_data, open(f"{blackbird.VICE_PATH}vice_data.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{blackbird.VICE_PATH}camera_parameters.pkl", 'wb'))


def build_estimator(ground: bool, imu: bool):
    vice_path = f'{blackbird.VICE_PATH}'
    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    cam_prms = pickle.load(open(f"{vice_path}camera_parameters.pkl", 'rb'))
    label = pickle.load(open(f"{vice_path}label.pkl", 'rb'))
    # label_point = np.array([np.array(SHAPE) // 2] * len(input_data['pose'])).astype(float)
    # label = {i: (label_point[i], False) for i in range(len(input_data['pose']))}
    # label[0] = (label_point[0], True)
    if imu:
        input_data["pose"] = input_data["imu"]
        del input_data["imu"]
    else:
        del input_data["imu"]

    def load_img(idx):
        return np.array(imageio.imread(input_data['img_path'][idx]))

    def depth_func(x_prime, y_prime, idx):
        depth_path = input_data['depth_path'][idx]
        depth_map = np.array(imageio.imread(depth_path)).astype(float) * 50 / 2 ** 16.  # from mm to m

        u_depth = x_prime * cam_prms['K'][0, 0] + cam_prms['K'][0, 2]
        v_depth = y_prime * cam_prms['K'][1, 1] + cam_prms['K'][1, 2]
        u_depth = np.round(u_depth).astype(int)
        v_depth = np.round(v_depth).astype(int)
        return depth_map[v_depth, u_depth]

    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                depth_func=depth_func,
                                load_img=load_img,
                                distortion_coefficient=cam_prms['D'],
                                width=SHAPE[1],
                                height=SHAPE[0],
                                use_depth=not ground,
                                label=label,
                                max_distance=10.)
    return estimator


def error_estimation(ground: bool, imu: bool):
    # 1-INITIALISATION
    vice_path = f'{blackbird.VICE_PATH}'
    print("VICE for ", vice_path)
    estimator = build_estimator(ground, imu)

    # 2-GENERATE P3 POINTS
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name
    estimator.p3_generator(saving_path=f'{vice_path}{name}.pkl')

    # 3-GENERATING THE VIDEO
    estimator.build_video(fps=FPS, saving_path=f'{vice_path}{name}.mkv')


def main():
    # video_to_png()
    # blackbird_pre_processing()
    case = {
            0: (True, False),
            1: (False, False),
            2: (True, True),
            3: (False, True),
            }
    for ground, imu in case.values():
        error_estimation(ground=ground, imu=imu)


if __name__ == '__main__':
    main()
