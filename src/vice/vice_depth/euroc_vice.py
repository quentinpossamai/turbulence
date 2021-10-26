import json
import numpy as np
import os.path
import pickle
import yaml

import src.data.euroc_mav.loading as euroc
from src.vice.vice_depth.posture_error_estimation import ErrorEstimation
from src.utils.transform import Transform


def euroc_mav_pre_processing(flight):
    flight_path = euroc.INTERMEDIATE_DATASET_PATH + flight
    print("Processing: ", flight_path)
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

    # 1.5-Cutting a the image frequency
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
    input_data = {"t": cut_t, "pose": cut_tf_camera_origin, "img": img_path, "depth": depth_path}
    pickle.dump(input_data, open(f"{flight_path}vice_data.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{flight_path}camera_parameters.pkl", 'wb'))


def error_estimation(flight):
    # 1-INITIALISATION
    flight_path = euroc.INTERMEDIATE_DATASET_PATH + flight
    print("VICE for ", flight_path)

    input_data = pickle.load(open(f"{flight_path}vice_data.pkl", 'rb'))
    cam_prms = pickle.load(open(f"{flight_path}camera_parameters.pkl", 'rb'))
    fps = 20
    width = 752
    height = 480

    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                distortion_coefficient=cam_prms['D'],
                                width=width,
                                height=height,
                                depth_coef=7.48)
    # 2-GENERATE P3 POINTS
    estimator.p3_generator(saving_path=f'{flight_path}p3_points.pkl')

    # 3-GENERATING THE VIDEO
    estimator.build_video(fps=fps, saving_path=flight_path + 'pose_camera_correspondence_video.mkv')


def pixel_error(flight):
    flight_path = euroc.INTERMEDIATE_DATASET_PATH + flight
    # p3 = pickle.load(open(f'{flight_path}p3_points.pkl', 'rb'))
    p3_len = 2719  # = len(p3)

    with open(f'{flight_path}task_v1_01_easy_backup_2021_10_25_14_25_23/annotations.json') as json_file:
        data = json.load(json_file)

    key_points = np.zeros((len(data[0]['tracks'][0]['shapes']), 3))  # frame_id, pixel_x, pixel_y
    for i, e in enumerate(data[0]['tracks'][0]['shapes']):
        key_points[i] = np.array([e['frame']] + e['points'])

    pixel_x = np.round(np.interp(range(p3_len), key_points[:, 0], key_points[:, 1])).astype(int)
    pixel_y = np.round(np.interp(range(p3_len), key_points[:, 0], key_points[:, 2])).astype(int)

    # Todo compute error


def main():
    flight = euroc.FLIGHT_LIST[0]
    euroc_mav_pre_processing(flight=flight)
    error_estimation(flight)
    # pixel_error(flight)

    # for flight in euroc.FLIGHT_LIST:
    #     euroc_mav_pre_processing(flight=flight)
    #     error_estimation(flight, need_csv=False)


if __name__ == '__main__':
    main()
