import imageio
import numpy as np
import pickle
from scipy import interpolate
from tqdm import tqdm
from typing import Union

import src.data.blackbird.loading as blackbird
from src.vice.vice import ErrorEstimation
from src.utils.transform import Transform as Tf

coef_multiplicator_depth = 50


def blackbird_pre_processing():
    rgb_time = np.loadtxt(f'{blackbird.VICE_PATH}rgb/nSecTimestamps.txt', dtype=int)
    depth_time = np.loadtxt(f'{blackbird.VICE_PATH}depth/nSecTimestamps.txt', dtype=int)
    state_time = np.loadtxt(f'{blackbird.VICE_PATH}blackbird_slash_state.csv', dtype=int, delimiter=',', skiprows=1,
                            usecols=[0])
    state = np.loadtxt(f'{blackbird.VICE_PATH}blackbird_slash_state.csv', dtype=float, delimiter=',', skiprows=1,
                       usecols=[9, 10, 11, 13, 14, 15, 16])

    # Synchronization
    sync_depth_time = np.zeros_like(rgb_time)
    sync_state_time = np.zeros_like(rgb_time)
    sync_state = np.zeros((len(rgb_time), 7))
    depth = np.zeros(len(rgb_time), )
    # for i, time in enumerate(tqdm(rgb_time, desc=f'Synchronization')):
    for i, time in enumerate(rgb_time):
        sync_depth_time[i] = depth_time[np.abs(time - depth_time).argmin()]
        depth[i] = imageio.imread(f'{blackbird.VICE_PATH}depth/{sync_depth_time[i]}.png')

        state_ind = np.abs(time - state_time).argmin()
        sync_state_time[i] = state_time[state_ind]
        sync_state[i] = state[state_ind]
    assert (sync_depth_time == rgb_time).sum() == len(rgb_time)  # No changes on depth time

    tf_180 = Tf.from_euler_trans(seq='xyz', angles=np.array([180, 0, 0]), degrees=True, trans=np.zeros(3)).as_matrix()

    tf_mocap_body = Tf.from_quat_trans(sync_state[:, 3:7], sync_state[:, :3]).as_matrix()
    tf_fgenv_norm = Tf.from_quat_trans(np.eye(4)[0], np.array([-0.04, -0.09, -0.5])).as_matrix()  # for patrick
    tf_norm_mocap = Tf.from_quat_trans(np.eye(4)[0], np.array([0.0, -0.5, -0.883718])).as_matrix()
    tf_body_camera = Tf.from_euler_trans(seq='xyz', angles=np.array([90, 0, 90]), degrees=True,
                                         trans=np.array([0, -0.05, 0])).as_matrix()

    tf_fgenv_camera = tf_180 @ tf_fgenv_norm @ tf_norm_mocap @ tf_mocap_body @ tf_body_camera
    tf_camera_fgenv = Tf.from_matrix(tf_fgenv_camera).inv().as_matrix()
    print()

    cam_prms = {'K': np.array([[665.107510106, 0, 511.5],
                               [0, 665.107510106, 383.5],
                               [0, 0, 1]]),
                'D': [0, 0, 0, 0, 0, 0],
                'fps': 30}

    # Cut trajectory, subsample ?
    start, stop = 4000, 5000

    # Cutting at the image frequency
    input_data = {"pose": tf_camera_fgenv,
                  "img": ,
                  "depth": }

    # Subsampling at 4 speed
    less_input_data = {}
    for key in input_data:
        less_input_data[key] = np.stack([e for i, e in enumerate(input_data[key]) if i % 4 == 0])

    pickle.dump(less_input_data, open(f"{blackbird.VICE_PATH}vice_data.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{blackbird.VICE_PATH}camera_parameters.pkl", 'wb'))


def error_estimation(flight: str, ground: bool, pose: str):
    assert pose in ['odom', 'slam']
    # 1-INITIALISATION
    vice_path = f'{locobot.LOCOBOT_DATA_PATH}{flight}'
    print("VICE for ", flight)

    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    cam_prms = pickle.load(open(f"{vice_path}camera_parameters.pkl", 'rb'))
    label = pickle.load(open(f"{vice_path}label.pkl", 'rb'))
    if pose == 'odom':
        input_data["pose"] = input_data["odom"]
    else:
        assert pose == 'slam'
        input_data["pose"] = input_data["slam"]
    del input_data["odom"]
    del input_data["slam"]

    def depth_func(x_prime, y_prime, idx):
        u_depth = x_prime * cam_prms['K_depth'][0, 0] + cam_prms['K_depth'][0, 2]
        v_depth = y_prime * cam_prms['K_depth'][1, 1] + cam_prms['K_depth'][1, 2]
        u_depth = np.round(u_depth).astype(int)
        v_depth = np.round(v_depth).astype(int)
        return input_data['depth'][idx, v_depth, u_depth]

    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                depth_func=depth_func,
                                distortion_coefficient=cam_prms['D'],
                                width=640,
                                height=480,
                                use_depth=not ground,
                                label=label,
                                max_distance=10.)
    # 2-GENERATE P3 POINTS
    name = f'ground_{pose}' if ground else f'depth_{pose}'
    estimator.p3_generator(saving_path=f'{vice_path}{name}.pkl')

    # 3-GENERATING THE VIDEO
    estimator.build_video(fps=30, saving_path=f'{vice_path}{name}.mkv')


def main():
    case = {0: True, 1: False}
    blackbird_pre_processing()
    # error_estimation(ground=case[0])


if __name__ == '__main__':
    main()
