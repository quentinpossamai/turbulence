import numpy as np
import pickle
from scipy import interpolate
from tqdm import tqdm
from typing import Union

import src.data.locobot.loading as locobot
from src.vice.vice import ErrorEstimation
from src.utils.transform import Transform


def locobot_pre_processing(traj: str):
    vice_path = f'{locobot.LOCOBOT_DATA_PATH}{traj}/'
    print("Processing: ", traj)
    # Loading camera parameters

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

    # Loading camera trajectory
    data = locobot.loading_traj(traj)
    odom_tf_camera_origin = Transform.from_matrix(np.stack([e['odom'] for e in data])).inv().as_matrix()
    slam_tf_camera_origin = Transform.from_matrix(np.stack([e['slam'] for e in data])).inv().as_matrix()
    img = np.stack([e['rgb'] for e in data])
    depth = np.stack([e['depth'] for e in data]) / 1000  # from mm to m

    # Cutting at the image frequency
    input_data = {"odom": odom_tf_camera_origin,
                  "slam": slam_tf_camera_origin,
                  "img": img,
                  "depth": depth}

    # Subsampling at 4 speed
    less_input_data = {}
    for key in input_data:
        less_input_data[key] = np.stack([e for i, e in enumerate(input_data[key]) if i % 4 == 0])

    # Complete holes in depth_map
    filled_depth = np.zeros_like(less_input_data['depth'])
    for i, z_image in enumerate(tqdm(less_input_data['depth'], desc=f'Filling depth')):
        # Fill hole in images with linear interp
        z_image = np.where(z_image == 0, np.nan, z_image)
        valid_mask = (1 - np.isnan(z_image)).astype(bool)
        coord = np.array(np.nonzero(valid_mask)).T

        values_z = z_image[valid_mask]
        it = interpolate.LinearNDInterpolator(coord, values_z, fill_value=np.mean(values_z))
        filled_z = it(list(np.ndindex(z_image.shape))).reshape(z_image.shape)
        filled_depth[i] = filled_z
    less_input_data['depth'] = filled_depth

    pickle.dump(less_input_data, open(f"{vice_path}vice_data.pkl", 'wb'))
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
    traj = locobot.TRAJ_LIST[1]
    locobot_pre_processing(traj)

    case = {
        # 0: True,
        1: False
    }
    for pose in ['odom', 'slam']:
        for ground in case.values():
            error_estimation(traj, ground=ground, pose=pose)


if __name__ == '__main__':
    main()
