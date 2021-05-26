import csv
import imageio
import numpy as np
import pandas as pd
import pathlib
import pickle
import os
import yaml

from src import utils
from src import posture_error_estimation

DATASET_PATH = os.getcwd() + "/data/blackbird/"


def data_processing():
    """
    Process data from the blackbird dataset.
    1. Load data.
    2. Synchronized them.
    3.
    Downloaded from: http://blackbird-dataset.mit.edu/BlackbirdDatasetData/.
    Github: https://github.com/mit-fast/Blackbird-Dataset
    Paper: https://arxiv.org/abs/1810.01987
    """
    # Trajectories name classification
    flight_name = "ampersand"
    flight_yaw = "yawForward"
    flight_speed = "maxSpeed1p0"
    environment = "Small_Apartment"  # Small_Apartment, Large_Apartment_Night_Near_Couches
    # raw folder is used to store raw data
    raw_flight_path = DATASET_PATH + f"raw/{flight_name}/{flight_yaw}/{flight_speed}/"

    # Load tf_norm_mocap
    with open(raw_flight_path + "flightNormalizationOffset.csv", newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        trans = np.array(next(reader))
    tf_norm_mocap = utils.Transformation().from_pose(trans, np.array([0, 0, 0, 1]))

    # Load tf_fgenv_norm for each environment
    with open(DATASET_PATH + "raw/trajectoryOffsets.yaml", 'r') as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # Finding the correct environment
    offset = None
    for e in yaml_file["unitySettings"][flight_name]:
        if e["name"] == environment:
            offset = e["offset"]
    if offset is None:
        print(f"Environment not found: {environment}")
        raise ValueError(environment)
    trans, angles = np.array(offset[:3]), np.array([offset[3], 0, 0])
    tf_fgenv_norm = utils.Transformation().from_euler(trans, "zyx", angles, degrees=True)

    # Intermediate folder is used to store processed data
    intermediate_flight_path = DATASET_PATH + f"intermediate/{flight_name}/{flight_yaw}/{flight_speed}/"
    data_path = intermediate_flight_path + "rosbag.pkl"

    # Load sensors data
    # Extracting /Users/quentin/phd/turbulence/data/blackbird/raw/ampersand/yawForward/maxSpeed1p0/rosbag.bag
    # List of topic to be extracted :
    # - /blackbird/pwm | Type : blackbird/MotorPWM | Message count : 38893
    # - /blackbird/rotor_rpm | Type : blackbird/MotorRPM | Message count : 38897
    # - /blackbird/imu | Type : sensor_msgs/Imu | Message count : 21257
    # - /camera_d/camera_info | Type : sensor_msgs/CameraInfo | Message count : 25717
    # - /camera_l/camera_info | Type : sensor_msgs/CameraInfo | Message count : 25717
    # - /camera_r/camera_info | Type : sensor_msgs/CameraInfo | Message count : 25717
    # - /tf | Type : tf/tfMessage | Message count : 77148
    # - /blackbird/state | Type : geometry_msgs/PoseStamped | Message count : 77148
    # - /blackbird/pose_ref | Type : geometry_msgs/PoseStamped | Message count : 35090
    # Encoding is latin1 because pickle with python 2.7 was used to make the .pkl
    data_raw = pickle.load(open(data_path, 'rb'), encoding='latin1')
    renaming_dict = {'/blackbird/pwm__blackbird/MotorPWM': "motor_pwm",
                     '/blackbird/rotor_rpm__blackbird/MotorRPM': "motor_rpm",
                     '/camera_d/camera_info__sensor_msgs/CameraInfo': "camera_info_d",
                     '/camera_l/camera_info__sensor_msgs/CameraInfo': "camera_info_l",
                     '/camera_r/camera_info__sensor_msgs/CameraInfo': "camera_info_r",
                     '/blackbird/imu__sensor_msgs/Imu': "imu",
                     '/blackbird/pose_ref__geometry_msgs/PoseStamped': "pose_ref",
                     '/tf__tf/tfMessage': "tf",
                     '/blackbird/state__geometry_msgs/PoseStamped': "state"}

    # Renaming data keys for easier use
    for old_key, new_key in renaming_dict.items():
        data_raw[new_key] = data_raw.pop(old_key)

    csv_filenames = ["blackbird_slash_imu.csv",
                     "blackbird_slash_pose_ref.csv",
                     "blackbird_slash_pwm.csv",
                     "blackbird_slash_rotor_rpm.csv",
                     "blackbird_slash_state.csv",
                     "camera_d_slash_camera_info.csv",
                     "camera_l_slash_camera_info.csv",
                     "camera_r_slash_camera_info.csv",
                     "tf.csv"]

    ground_truth = {}
    with open(raw_flight_path + "groundTruthPoses.csv", newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(reader):
            ground_truth[i] = [float(e) for e in row]

    tf_imu_body

    # drone_tf_camera
    ground_truth = np.array(yaml_file['T_BS']['data']).reshape((4, 4))  # Sensor extrinsic wrt. the body-frame

    # The extrinsic parameter of the
    camera_parameters['camera_tf_drone'] = utils.Transformation().from_matrix(ground_truth).inv()

    # Vicon0 parameters
    # f.get_files_paths('.yaml', f.folders['raw'][vol_number], 'vicon0')
    with open(f.get_unique_file_path('.yaml', f.folders['raw'][flight_number], 'vicon0'), 'r') as stream:
        try:
            yaml_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # drone_tf_vicon0 : Sensor extrinsic wrt. the body-frame
    drone_tf_vicon0 = utils.Transformation().from_matrix(np.array(yaml_file['T_BS']['data']).reshape((4, 4)))
    camera_parameters['camera_tf_vicon0'] = camera_parameters['camera_tf_drone'] @ drone_tf_vicon0
    del camera_parameters['camera_tf_drone']

    # mc default columns : '#timestamp [ns]', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []',
    #                                         'q_RS_x []', 'q_RS_y []', 'q_RS_z []'
    mc = pd.read_csv(f.get_unique_file_path('.csv', f.folders['raw'][flight_number], 'vicon0'), sep=',', header=0,
                     names=['pose_time', 'x', 'y', 'z', 'wn', 'a', 'b', 'c'])  # Renaming column for simplification
    p = utils.Progress(len(mc), "Preparing mc data")
    ground_truth = {}
    for index, row in mc.iterrows():
        # vicon0_tf_origin
        vicon0_tf_origin = utils.Transformation().from_pose(trans=row[['x', 'y', 'z']].to_numpy(),
                                                            quat=row[['a', 'b', 'c', 'wn']].to_numpy()).inv()
        camera_tf_origin = camera_parameters['camera_tf_vicon0'] @ vicon0_tf_origin
        ground_truth[index] = camera_tf_origin
        p.update()
    mc['pose'] = pd.Series(ground_truth)  # camera_tf_origin

    # image default columns : '#timestamp [ns]', 'filename'
    file_path = f.get_unique_file_path('.csv', f.folders['raw'][flight_number], 'cam0')
    image = pd.read_csv(file_path, sep=',', header=0, names=['image_time', 'filename'])  # Renaming column for
    # simplification
    p = utils.Progress(len(image), "Preparing camera data")
    image_path = utils.get_folder_path(file_path) + 'data/'
    ground_truth = {}
    shape = np.array(imageio.imread(image_path + image['filename'][0])).shape
    for index, filename in image['filename'].items():
        measure = np.array(imageio.imread(image_path + filename))
        color_frame = np.uint8(np.zeros((shape[0], shape[1], 3)))
        for j in range(3):
            color_frame[:, :, j] = measure
        ground_truth[index] = color_frame
        p.update()
    image['image'] = pd.Series(ground_truth)

    # Synchro
    time_min = min(mc['pose_time'][0], image['image_time'][0])
    mc['pose_time'] = (mc['pose_time'] - time_min) * 1e-9
    image['image_time'] = (image['image_time'] - time_min) * 1e-9
    mc_ids, image_ids = utils.merge_two_arrays(mc['pose_time'], image['image_time'])
    input_data = pd.concat([mc.loc[mc_ids, ['pose_time', 'pose']].reset_index(drop=True),
                            image.loc[image_ids, ['image_time', 'image']].reset_index(drop=True)],
                           axis=1)
    pickle.dump(input_data, open(f.folders['intermediate'][flight_number] + 'posture_error_input_data.pkl', 'wb'))
    pickle.dump(camera_parameters, open(f.folders['intermediate'][flight_number] + 'camera_parameters.pkl', 'wb'))


def error_estimation():
    f = utils.DataFolder('euroc_mav')
    for flight_number in [0, 1, 2, 3, 4, 5]:
        print(f"Flight : {flight_number}")

        input_data = pickle.load(open(f.folders['intermediate'][flight_number] + 'posture_error_input_data.pkl', 'rb'))
        camera_parameters = pickle.load(open(f.folders['intermediate'][flight_number] + 'camera_parameters.pkl', 'rb'))

        estimator = posture_error_estimation.ErrorEstimation(input_data=input_data,
                                                             calibration_matrix=camera_parameters['K'],
                                                             distortion_coefficient=camera_parameters['D'],
                                                             max_distance=5)
        estimator.p3_generator()
        # saving_path = f.folders['results'][flight_number] + 'raw_video.mkv'
        saving_path = f.folders['results'][flight_number] + 'pose_camera_correspondence_video.mkv'

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
                xyz = estimator.df["pose"][i].inv().get_trans()
                quat = estimator.df["pose"][i].inv().get_quat()
                ang = estimator.df["pose"][i].inv().get_rot_euler(seq='xyz', degrees=True)
                if isinstance(p3, np.ndarray):
                    wr.writerow([p3, xyz[0], xyz[1], xyz[2], quat[0], quat[1], quat[2], quat[3],
                                 ang[0], ang[1], ang[2]])
                else:
                    if "\n" in p3:
                        p3 = p3.replace("\n", " ")
                    wr.writerow([p3, xyz[0], xyz[1], xyz[2], quat[0], quat[1], quat[2], quat[3],
                                 ang[0], ang[1], ang[2]])

        # estimator.p32video(fps=camera_parameters['fps'], saving_path=saving_path)
        print()


def dataset_creation(flight_number: int):
    f = utils.DataFolder('euroc_mav')
    flight_number = flight_number

    # Motor speed
    data = f.pickle_load_file('.pkl', f.folders['raw_python'][flight_number], None, True)

    renaming = {'/vicon/firefly_sbx/firefly_sbx__geometry_msgs/TransformStamped': 'vicon_pose',
                '/fcu/motor_speed__asctec_hl_comm/MotorSpeed': 'motor_speed',
                '/cam0/image_raw__sensor_msgs/Image': 'cam0_img',
                '/cam1/image_raw__sensor_msgs/Image': 'cam1_img',
                '/imu0__sensor_msgs/Imu': 'imu0',
                '/fcu/imu__sensor_msgs/Imu': 'fcu_imu'}

    correspondence = {}
    for key in data:
        _, msg_type = key.split('__')
        correspondence[key] = [renaming[key], renaming[key] + '_time', msg_type]

    to_synchro = []
    for data_column, (name, _, msg_type) in correspondence.items():
        if msg_type == 'sensor_msgs/Image':
            tmp = {'time': [], name: []}
            for measure in data[data_column].values():
                tmp['time'].append(measure['t'])
                tmp[name].append(measure['image'])
        elif msg_type == 'geometry_msgs/TransformStamped':
            tmp = {'time': [], name: []}
            for measure in data[data_column].values():
                tmp['time'].append(measure['t'])
                trans, quat = measure['translation'], measure['rotation']
                tmp[name].append(utils.Transformation().from_pose(trans, quat).inv())
        elif msg_type == 'asctec_hl_comm/MotorSpeed':
            tmp = {'time': [], name: []}
            for measure in data[data_column].values():
                tmp['time'].append(measure['t'])
                tmp[name].append(measure['motor_speed'])
        elif msg_type == 'sensor_msgs/Imu':
            tmp = {'time': [],
                   name + '_angular_velocity': [],
                   name + '_angular_velocity_covariance': [],
                   name + '_linear_acceleration': [],
                   name + '_linear_acceleration_covariance': [],
                   name + '_orientation': [],
                   name + '_orientation_covariance': []}
            for measure in data[data_column].values():
                tmp['time'].append(measure['t'])
                tmp[name + '_angular_velocity'].append(measure['angular_velocity'])
                tmp[name + '_angular_velocity_covariance'].append(measure['angular_velocity_covariance'])
                tmp[name + '_linear_acceleration'].append(measure['linear_acceleration'])
                tmp[name + '_linear_acceleration_covariance'].append(measure['linear_acceleration_covariance'])
                tmp[name + '_orientation'].append(measure['orientation'])
                tmp[name + '_orientation_covariance'].append(measure['orientation_covariance'])
        else:
            raise TypeError(f'ROS type not implemented : {msg_type}')
        to_synchro.append(pd.DataFrame(tmp))

    freq = []
    for i in range(len(to_synchro)):
        freq.append(len(to_synchro[i]) / (to_synchro[i]['time'].iloc[-1] - to_synchro[i]['time'].iloc[0]))
    ind_freq = sorted(range(len(freq)), key=lambda k: freq[k])

    ids_0 = list(range(len(to_synchro[ind_freq[0]])))
    for i in ind_freq[1:]:
        ids_0, ids_i = utils.merge_two_arrays(to_synchro[ind_freq[0]]['time'], to_synchro[i]['time'])
        to_synchro[ind_freq[0]] = to_synchro[ind_freq[0]].iloc[ids_0].reset_index(drop=True)
        to_synchro[i] = to_synchro[i].iloc[ids_i].reset_index(drop=True)
        to_synchro[i]['time'] = to_synchro[ind_freq[0]]['time']

    for i in range(len(to_synchro)):
        to_synchro[i] = to_synchro[i].iloc[ids_0].reset_index(drop=True)
        if 0 != i:
            to_synchro[i] = to_synchro[i].drop(['time'], axis=1)

    data_clean = pd.concat(to_synchro, axis=1)
    np.set_printoptions(threshold=np.inf)
    # data_clean.to_csv(f.folders["intermediate"][flight_number] + "sensors_synchronised.csv", sep=";")
    data_clean.to_pickle(f.folders["intermediate"][flight_number] + "sensors_synchronised.pkl")


def _test():
    for i, e in enumerate(ground_truth.values()):
        gt_trans = np.array([e[1], e[2], e[3]])
        gt_quat = np.array([e[5], e[6], e[7], e[4]])
        gt_time_like_gt = e[0]

        tf_secs = data["tf"][i]["secs"]
        tf_nsecs = data["tf"][i]["nsecs"]
        tf_trans = data["tf"][i]["translation"]
        tf_quat = data["tf"][i]["rotation"]

        state_secs = data["state"][i]["secs"]
        state_nsecs = data["state"][i]["nsecs"]
        state_trans = data["state"][i]["translation"]
        state_quat = data["state"][i]["rotation"]

        assert tf_secs == state_secs
        assert tf_nsecs == state_nsecs
        assert all(tf_trans == state_trans)
        assert all(tf_quat == state_quat)

        tf_time_like_gt = tf_secs * 1e6 + np.around(tf_nsecs * 1e-3, 0)
        tf_trans = np.around(data["tf"][i]["translation"], 6)
        tf_quat = np.around(data["tf"][i]["rotation"], 6)

        assert tf_time_like_gt == gt_time_like_gt
        assert all(tf_trans == gt_trans)
        assert all(tf_quat == gt_quat)

    for camera_name in ["camera_info_d", "camera_info_l", "camera_info_r"]:
        D = data[camera_name][0]["D"]
        K = data[camera_name][0]["K"]
        P = data[camera_name][0]["P"]
        R = data[camera_name][0]["R"]
        for e in data[camera_name].values():
            np.testing.assert_equal(D, e["D"])
            np.testing.assert_equal(K, e["K"])
            np.testing.assert_equal(P, e["P"])
            np.testing.assert_equal(R, e["R"])
    print("blackbird.py is being tested.")


if __name__ == "__main__":
    data_processing()
    # _test()
