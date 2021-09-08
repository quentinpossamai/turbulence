import csv
import imageio
import numpy as np
import pandas as pd
import pickle
import yaml
import matplotlib.pyplot as plt

import src.data.euroc_mav.loading as euroc
from src.vice.posture_error_estimation import ErrorEstimation
from src.utils.transform import Transform, Rotation


def euroc_mav_pre_processing(flight):
    flight_path = euroc.INTERMEDIATE_DATASET_PATH + flight
    print(flight_path)
    # 1-LOADING
    # 1.1-Loading camera parameters
    with open(f"{euroc.DATASET_PATH}raw/{flight}cam0/sensor.yaml", 'r') as stream:
        # /Users/quentin/phd/turbulence/data/euroc_mav/raw/V1_01_easy/cam0
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    cam_prms = {'K': np.eye(3),
                # D distortion model
                'D': np.array([e for e in yaml_file['distortion_coefficients']]),
                'fps': yaml_file['rate_hz']}

    # K camera matrix : intrinsic parameters
    cam_prms['K'][0, 0] = yaml_file['intrinsics'][0]
    cam_prms['K'][1, 1] = yaml_file['intrinsics'][1]
    cam_prms['K'][0, 2] = yaml_file['intrinsics'][2]
    cam_prms['K'][1, 2] = yaml_file['intrinsics'][3]

    # Camera body transformation
    cam_prms['tf_body_cam'] = np.array(yaml_file['T_BS']['data']).reshape((4, 4))  # wrt. the body-frame

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

    # 1.4-Loading images
    input_data = {"t": data["t"], "pose": tf_camera_origin, "img": data["img"]}

    pickle.dump(input_data, open(f"{flight_path}vice_data.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{flight_path}camera_parameters.pkl", 'wb'))


def error_estimation(flight, need_csv):
    flight_path = euroc.INTERMEDIATE_DATASET_PATH + flight
    print(flight_path)

    input_data = pickle.load(open(f"{flight_path}vice_data.pkl", 'rb'))
    camera_parameters = pickle.load(open(f"{flight_path}camera_parameters.pkl", 'rb'))

    estimator = posture_error_estimation.ErrorEstimation(input_data=input_data,
                                                         calibration_matrix=camera_parameters['K'],
                                                         distortion_coefficient=camera_parameters['D'],
                                                         max_distance=5)
    estimator.p3_generator()
    # saving_path = f.folders['results'][flight_number] + 'raw_video.mkv'
    saving_path = f.folders['results'][flight_number] + 'pose_camera_correspondence_video.mkv'
    # estimator.p32video(fps=camera_parameters['fps'], saving_path=saving_path)

    if need_csv:
        # p3 saved as csv
        point_position_in_image = estimator.p3
        with open(f.folders['results'][flight_number] + "points.csv", 'w', newline="") as csv_file:
            wr = csv.writer(csv_file, dialect="excel", quoting=csv.QUOTE_NONE)
            wr.writerow(["p3 [px] (red dot position in image)",
                         "x [m] (position of the camera relative to global reference frame)",
                         "y [m] (position of the camera relative to global reference frame)",
                         "z [m] (position of the camera relative to global reference frame)",
                         "a [quaternion] (orientation of the camera relative to global reference frame)",
                         "b [quaternion] (orientation of the camera relative to global reference frame)",
                         "c [quaternion] (orientation of the camera relative to global reference frame)",
                         "w [quaternion] (orientation of the camera relative to global reference frame)",
                         "phi [deg] (orientation of the camera relative to global reference frame)",
                         "theta [deg] (orientation of the camera relative to global reference frame)",
                         "psi [deg] (orientation of the camera relative to global reference frame)"])
            for i in range(len(estimator.df)):
                p3 = point_position_in_image[i]
                xyz = estimator.df["pose"][i].inv().as_trans()
                quat = estimator.df["pose"][i].inv().as_rot_quat()
                ang = estimator.df["pose"][i].inv().as_rot_euler(seq='xyz', degrees=True)
                if isinstance(p3, np.ndarray):
                    wr.writerow([p3, xyz[0], xyz[1], xyz[2], quat[0], quat[1], quat[2], quat[3],
                                 ang[0], ang[1], ang[2]])
                else:
                    if "\n" in p3:
                        p3 = p3.replace("\n", " ")
                    wr.writerow([p3, xyz[0], xyz[1], xyz[2], quat[0], quat[1], quat[2], quat[3],
                                 ang[0], ang[1], ang[2]])
        print()


def main():
    for flight in euroc.FLIGHT_LIST:
        # euroc_mav_pre_processing(flight=flight)
        error_estimation(flight, need_csv=False)


if __name__ == '__main__':
    main()
