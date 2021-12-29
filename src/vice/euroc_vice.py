import json
import numpy as np
import os.path
import pickle
import yaml

import src.data.euroc_mav.loading as euroc
from src.vice.posture_error_estimation import ErrorEstimation
from src.utils.transform import Transform
from src.utils.paths import mkdir_if_not_exist


def euroc_mav_pre_processing(flight):
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

    cam_prms = {'K': k,
                'D': np.array([e for e in yaml_file['distortion_coefficients']]),
                'fps': yaml_file['rate_hz'],
                'tf_body_cam': np.array(yaml_file['T_BS']['data']).reshape((4, 4))}  # wrt. the body-frame

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
    cut_tf_camera_origin = np.zeros((len(cut_img), 4, 4))
    cut_t = np.zeros((len(cut_img), 1))

    img_path = []
    depth_path = []
    for i, img_timestamp in enumerate(cut_img):
        ind = np.abs(img_timestamp - data["timestamp"]).argmin()
        cut_tf_camera_origin[i] = tf_camera_origin[ind]
        cut_t[i] = data["t"][ind]
        img_path.append(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{img_timestamp}.png")
        depth_path.append(f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}'
                          f'depth_estimation/{str(data["t"][ind]).replace(".", "_")}.png')

        assert os.path.isfile(depth_path[i])

    # 1.6 Saving preprocessing
    mkdir_if_not_exist([vice_path])
    input_data = {"t": cut_t, "pose": cut_tf_camera_origin, "img": img_path, "depth": depth_path}
    pickle.dump(input_data, open(f"{vice_path}vice_data.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{vice_path}camera_parameters.pkl", 'wb'))


def error_estimation(flight):
    # 1-INITIALISATION
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    print("VICE for ", flight)

    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    cam_prms = pickle.load(open(f"{vice_path}camera_parameters.pkl", 'rb'))
    # vice_point = np.stack(pickle.load(open(f'{vice_path}p3_points_backup.pkl', 'rb')), axis=0)
    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                distortion_coefficient=cam_prms['D'],
                                width=752,
                                height=480,
                                use_depth=False,
                                initial_point=None,
                                max_distance=10.,
                                depth_coef=7.48)
    # 2-GENERATE P3 POINTS
    estimator.p3_generator(saving_path=f'{vice_path}p3_points.pkl')

    # 3-GENERATING THE VIDEO
    estimator.build_video(fps=20, saving_path=f'{vice_path}ground.mkv')


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


def main():
    flight = euroc.FLIGHT_LIST[2]
    euroc_mav_pre_processing(flight=flight)
    error_estimation(flight)
    # position_2d_error(flight)

    # for flight in euroc.FLIGHT_LIST:
    #     euroc_mav_pre_processing(flight=flight)
    #     error_estimation(flight, need_csv=False)


if __name__ == '__main__':
    main()
