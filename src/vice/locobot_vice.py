import numpy as np
import pickle
from typing import Union

import src.data.locobot.loading as locobot
from src.vice.vice import ErrorEstimation
from src.utils.transform import Transform


def locobot_pre_processing(traj: str):
    vice_path = f'{locobot.LOCOBOT_DATA_PATH}{traj}/'
    print("Processing: ", traj)
    # 1-LOADING
    # 1.1-Loading camera parameters

    rgb_prms = np.array([[607.0703735351562, 0.0, 332.4421691894531],
                        [0.0, 607.4769897460938, 238.6558837890625],
                        [0.0, 0.0, 1.0]])

    depth_prms = np.array([[382.2403564453125, 0.0, 320.8781433105469],
                          [0.0, 382.2403564453125, 237.9492950439453],
                          [0.0, 0.0, 1.0]])
    # K camera matrix : intrinsic parameters
    cam_prms = {'K': rgb_prms,
                'K_depth': depth_prms,
                'D': [0, 0, 0, 0, 0, 0, 0, 0],
                'fps': 30}  # wrt. the body-frame

    # 1.3-Loading camera trajectory
    data = locobot.loading_traj(traj)
    odom_tf_camera_origin = Transform.from_matrix(np.stack([e['odom'] for e in data])).inv().as_matrix()
    slam_tf_camera_origin = Transform.from_matrix(np.stack([e['slam'] for e in data])).inv().as_matrix()
    img = np.stack([e['rgb'] for e in data])
    depth = np.stack([e['depth'] for e in data])

    # 1.5-Cutting at the image frequency
    input_data = {"odom": odom_tf_camera_origin,
                  "slam": slam_tf_camera_origin,
                  "img": img,
                  "depth": depth}
    name = f'vice_data'
    pickle.dump(input_data, open(f"{vice_path}{name}.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{vice_path}camera_parameters.pkl", 'wb'))


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
    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                calibration_matrix_depth=cam_prms['K_depth'],
                                distortion_coefficient=cam_prms['D'],
                                width=752,
                                height=480,
                                use_depth=not ground,
                                label=label,
                                max_distance=10.,
                                depth_coef=7.48)
    # 2-GENERATE P3 POINTS
    name = 'ground' if ground else 'depth'
    estimator.p3_generator(saving_path=f'{vice_path}{name}.pkl')

    # 3-GENERATING THE VIDEO
    estimator.build_video(fps=30, saving_path=f'{vice_path}{name}.mkv')


def main():
    traj = locobot.TRAJ_LIST[0]
    case = {0: True,
            1: False}
    ground, imu = case[0]
    locobot_pre_processing(traj)
    error_estimation(traj, ground=ground, pose='odom')


if __name__ == '__main__':
    main()
