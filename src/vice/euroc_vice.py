import numpy as np
import pickle
import yaml
# import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import imageio

import src.data.euroc_mav.loading as euroc
from src.vice.posture_error_estimation import ErrorEstimation
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

    for i, img_timestamp in enumerate(cut_img):
        ind = np.abs(img_timestamp - data["timestamp"]).argmin()
        cut_tf_camera_origin[i] = tf_camera_origin[ind]
        cut_t[i] = data["t"][ind]

    input_data = {"t": cut_t, "pose": cut_tf_camera_origin, "cam0": cut_img}

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
                                max_distance=5,
                                width=width,
                                height=height)
    # 2-GENERATE P3 POINTS
    estimator.p3_generator()

    # 3-GENERATING THE VIDEO
    # saving_path = f.folders['results'][flight_number] + 'raw_video.mkv'
    saving_path = flight_path + 'pose_camera_correspondence_video.mkv'
    codec = VideoWriter_fourcc(*'H264')
    # name = ABSOLUTE_PATH + 'error_estimation_videos/' + part1 + '_' + part2 + '_error_estimation.mkv'
    video = VideoWriter(saving_path, codec, float(fps), (width, height), True)
    print(f'Video will be saved at : {saving_path}')
    shape = (height, width, 3)
    for i, image_time in enumerate(tqdm(input_data['cam0'], desc="Creating a video")):
        image_name = str(image_time)
        img = np.array(imageio.imread(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{image_name}.png"))
        color_frame = np.uint8(np.zeros((shape[0], shape[1], 3)))
        for j in range(3):
            color_frame[:, :, j] = img
        img = color_frame
        p3_img = estimator.p32video_get_frame(i=i, image=img)
        video.write(p3_img[:, :, ::-1])  # Because VideoWriter.write take BGR images
    video.release()
    cv2.destroyAllWindows()


def main():
    flight = euroc.FLIGHT_LIST[0]
    euroc_mav_pre_processing(flight=flight)
    error_estimation(flight)

    # for flight in euroc.FLIGHT_LIST:
    #     euroc_mav_pre_processing(flight=flight)
    #     error_estimation(flight, need_csv=False)


if __name__ == '__main__':
    main()
