import imageio
import numpy as np
import os.path
import pickle
import yaml

import src.data.euroc_mav.loading as euroc
from src.vice.vice import ErrorEstimation
from src.utils.transform import Transform as Tf
import src.utils.quaternion as quaternion
from src.utils.path import mkdir


def euroc_mav_pre_processing(flight: str):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    print("Processing: ", flight)
    # - Loading camera parameters
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
    tf_body_cam0 = np.array(yaml_file['T_BS']['data']).reshape((4, 4))
    cam_prms = {'K': k,
                'D': np.array([e for e in yaml_file['distortion_coefficients']]),
                'fps': 10}  # 20 = yaml_file['rate_hz']

    # - Loading motion capture transform to body
    with open(f"{euroc.DATASET_PATH}raw/{flight}vicon0/sensor.yaml", 'r') as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    tf_vicon0_body = Tf.from_matrix(np.array(yaml_file['T_BS']['data']).reshape((4, 4))).inv().as_matrix()
    tf_vicon0_cam0 = tf_vicon0_body @ tf_body_cam0

    # - Loading camera trajectory measured by mc (MoCap)
    mc_data = euroc.load_flight(flight)  # dict_keys(['t', 'imu_ang_acc', 'motor', 'img', 'state', 'residual'])
    trans = mc_data["state"][:, :3] + euroc.TRANS_MEAN[flight].reshape((1, 3))
    mc_tf_global_vicon0 = Tf.from_quat_trans(quat=mc_data["state"][:, 3:7], trans=trans).as_matrix()
    mc_tf_cam0_global = Tf.from_matrix(mc_tf_global_vicon0 @ tf_vicon0_cam0).inv().as_matrix()
    mc_tf_global_body = mc_tf_global_vicon0 @ tf_vicon0_body

    # - Loading camera trajectory measured by ov (OpenVINS)
    # timestamp, x, y, z, qw, qx, qy, qz
    # Loaded data is ov_tf_global_body
    ov_data = np.loadtxt(f'{euroc.DATASET_PATH}OpenVINS/{flight[:-1]}_state_estimate.txt', delimiter=',', skiprows=1,
                         usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    ov_timestamp, trans = (ov_data[:, 0] * 1e9).astype(int), ov_data[:, 5:8]
    trans = trans + Tf.from_matrix(mc_tf_global_body).as_trans()[0]  # Setting OpenVINS initial point
    quat = ov_data[:, 1:5]
    quat = quaternion.qfix(quat)
    ov_tf_global_body = Tf.from_quat_trans(quat, trans).as_matrix()
    ov_tf_cam0_global = Tf.from_matrix(ov_tf_global_body @ tf_body_cam0).inv().as_matrix()

    # - Synchronising and subsampling at the image frequency
    cut_img_timestamp = np.unique(mc_data["cam0"])
    cut_mc = np.zeros((len(cut_img_timestamp), 4, 4))
    cut_ov = np.zeros((len(cut_img_timestamp), 4, 4))
    img_path = []
    depth_path = []

    for i, img_timestamp in enumerate(cut_img_timestamp):
        # MoCap sync
        mc_ind = np.abs(img_timestamp - mc_data["timestamp"]).argmin()
        cut_mc[i] = mc_tf_cam0_global[mc_ind]

        # img load
        img_path.append(f"{euroc.DATASET_PATH}raw/{flight}cam0/data/{img_timestamp}.png")

        # depth load
        depth_path.append(f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}'
                          f'depth_estimation/{str(mc_data["t"][mc_ind]).replace(".", "_")}.png')

        # OpenVINS sync
        cut_ov[i] = ov_tf_cam0_global[np.abs(img_timestamp - ov_timestamp).argmin()]

    # - Saving preprocessing
    mkdir([vice_path])

    input_data = {"pose": cut_mc,
                  "imu": cut_ov,
                  "img_path": img_path,
                  "depth_path": depth_path}

    # Subsampling at half speed and cutting 30s
    less_input_data = {}
    for key in input_data:
        less_input_data[key] = np.stack([e for i, e in enumerate(input_data[key]) if i % 2 == 0])
        less_input_data[key] = less_input_data[key][:300]

    name = f'vice_data'
    pickle.dump(less_input_data, open(f"{vice_path}{name}.pkl", 'wb'))
    pickle.dump(cam_prms, open(f"{vice_path}camera_parameters.pkl", 'wb'))
    return input_data, cam_prms


def build_estimator(flight: str, ground: bool, imu: bool, label_number: int):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    input_data = pickle.load(open(f"{vice_path}vice_data.pkl", 'rb'))
    cam_prms = pickle.load(open(f"{vice_path}camera_parameters.pkl", 'rb'))
    labels = pickle.load(open(f"{vice_path}labels.pkl", 'rb'))

    if imu:
        input_data["pose"] = input_data["imu"]
        del input_data["imu"]
    else:
        del input_data["imu"]

    def load_img(idx):
        shape = (480, 752)
        img = np.array(imageio.imread(input_data['img_path'][idx]))
        color_frame = np.uint8(np.zeros((shape[0], shape[1], 3)))
        for j in range(3):
            color_frame[:, :, j] = img
        return color_frame

    def depth_func(x_prime, y_prime, idx):
        depth_path = input_data['depth_path'][idx]
        assert os.path.isfile(depth_path)
        depth = np.array(imageio.imread(depth_path))
        assert np.all(depth[..., 0] == depth[..., 1]) and np.all(depth[..., 0] == depth[..., 2]) \
               and np.all(depth[..., 3] == 255)
        depth_map = np.array(imageio.imread(depth_path))[:, :, 0] / 255 * 7.48

        rr = x_prime ** 2 + y_prime ** 2
        k1, k2 = cam_prms['D'][0], cam_prms['D'][1]  # Extracting distortion coefficients to find depth
        distortion = (1 + k1 * rr + k2 * rr * rr)
        u_depth = x_prime * distortion * cam_prms['K'][0, 0] + cam_prms['K'][0, 2]
        v_depth = y_prime * distortion * cam_prms['K'][1, 1] + cam_prms['K'][1, 2]
        u_depth = np.floor(0.5 * u_depth).astype(int)  # reduce value by 0.5 because in euroc mav depth are two
        # time smaller than rgb in height and width.
        v_depth = np.floor(0.5 * v_depth).astype(int)

        return depth_map[v_depth, u_depth]

    estimator = ErrorEstimation(input_data=input_data,
                                calibration_matrix=cam_prms['K'],
                                depth_func=depth_func,
                                load_img=load_img,
                                distortion_coefficient=cam_prms['D'],
                                width=752,
                                height=480,
                                use_depth=not ground,
                                label=labels[label_number],
                                max_distance=10.)
    return estimator


def error_estimation(flight: str, ground: bool, imu: bool, label_number: int):
    estimator = build_estimator(flight, ground, imu, label_number)
    # 1-INITIALISATION
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    print("VICE for ", flight)

    # 2-GENERATE P3 POINTS
    name = 'ground' if ground else 'depth'
    name = f'{name}imu' if imu else name
    name = f'{name}{label_number}'
    estimator.p3_generator(saving_path=f'{vice_path}{name}.pkl')

    # 3-GENERATING THE VIDEO
    # estimator.build_video(fps=10, saving_path=f'{vice_path}{name}.mkv')


def main():
    for flight in euroc.FLIGHT_LIST:
        euroc_mav_pre_processing(flight)

    # case = {0: (True, False),
    #         1: (False, False),
    #         2: (True, True),
    #         3: (False, True)}
    # for flight in euroc.FLIGHT_LIST:
    #     for label_number in range(4):
    #         for ground, imu in case.values():
    #             error_estimation(flight, ground=ground, imu=imu, label_number=label_number)


if __name__ == '__main__':
    main()
