import pickle

import posture_error_estimation
import util
import pandas as pd
import imageio
import numpy as np
import yaml


def data_processing():
    f = util.DataFolder('euroc_mav')
    for flight_number in [0, 1, 2, 3, 4, 5]:

        # Camera parameters
        f.get_files_paths('.yaml', f.folders['raw'][flight_number], 'cam0')
        with open(f.get_unique_file_path('.yaml', f.folders['raw'][flight_number], 'cam0'), 'r') as stream:
            try:
                yaml_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        camera_parameters = {'K': np.eye(3),
                             # D distortion model
                             'D': np.array([e for e in yaml_file['distortion_coefficients']]),
                             'fps': yaml_file['rate_hz']}

        # K camera matrix : intrinsic parameters
        camera_parameters['K'][0, 0] = yaml_file['intrinsics'][0]
        camera_parameters['K'][1, 1] = yaml_file['intrinsics'][1]
        camera_parameters['K'][0, 2] = yaml_file['intrinsics'][2]
        camera_parameters['K'][1, 2] = yaml_file['intrinsics'][3]

        # drone_tf_camera
        tmp = np.array(yaml_file['T_BS']['data']).reshape((4, 4))  # Sensor extrinsic wrt. the body-frame

        # The extrinsic parameter of the
        camera_parameters['camera_tf_drone'] = util.Transformation().from_matrix(tmp).inv()

        # Vicon0 parameters
        # f.get_files_paths('.yaml', f.folders['raw'][vol_number], 'vicon0')
        with open(f.get_unique_file_path('.yaml', f.folders['raw'][flight_number], 'vicon0'), 'r') as stream:
            try:
                yaml_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # drone_tf_vicon0 : Sensor extrinsic wrt. the body-frame
        drone_tf_vicon0 = util.Transformation().from_matrix(np.array(yaml_file['T_BS']['data']).reshape((4, 4)))
        camera_parameters['camera_tf_vicon0'] = camera_parameters['camera_tf_drone'] @ drone_tf_vicon0
        del camera_parameters['camera_tf_drone']

        # mc default columns : '#timestamp [ns]', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []',
        #                                         'q_RS_x []', 'q_RS_y []', 'q_RS_z []'
        mc = pd.read_csv(f.get_unique_file_path('.csv', f.folders['raw'][flight_number], 'vicon0'), sep=',', header=0,
                         names=['pose_time', 'x', 'y', 'z', 'wn', 'a', 'b', 'c'])  # Renaming column for simplification
        p = util.Progress(len(mc), "Preparing mc data")
        tmp = {}
        for index, row in mc.iterrows():
            # vicon0_tf_origin
            vicon0_tf_origin = util.Transformation().from_pose(trans=row[['x', 'y', 'z']].to_numpy(),
                                                               quat=row[['a', 'b', 'c', 'wn']].to_numpy()).inv()
            camera_tf_origin = camera_parameters['camera_tf_vicon0'] @ vicon0_tf_origin
            tmp[index] = camera_tf_origin
            p.update()
        mc['pose'] = pd.Series(tmp)  # camera_tf_origin

        # image default columns : '#timestamp [ns]', 'filename'
        file_path = f.get_unique_file_path('.csv', f.folders['raw'][flight_number], 'cam0')
        image = pd.read_csv(file_path, sep=',', header=0, names=['image_time', 'filename'])  # Renaming column for
        # simplification
        p = util.Progress(len(image), "Preparing camera data")
        image_path = util.get_folder_path(file_path) + 'data/'
        tmp = {}
        shape = np.array(imageio.imread(image_path + image['filename'][0])).shape
        for index, filename in image['filename'].items():
            measure = np.array(imageio.imread(image_path + filename))
            color_frame = np.uint8(np.zeros((shape[0], shape[1], 3)))
            for j in range(3):
                color_frame[:, :, j] = measure
            tmp[index] = color_frame
            p.update()
        image['image'] = pd.Series(tmp)

        # Synchro
        time_min = min(mc['pose_time'][0], image['image_time'][0])
        mc['pose_time'] = (mc['pose_time'] - time_min) * 1e-9
        image['image_time'] = (image['image_time'] - time_min) * 1e-9
        mc_ids, image_ids = util.merge_two_arrays(mc['pose_time'], image['image_time'])
        input_data = pd.concat([mc.loc[mc_ids, ['pose_time', 'pose']].reset_index(drop=True),
                                image.loc[image_ids, ['image_time', 'image']].reset_index(drop=True)],
                               axis=1)
        pickle.dump(input_data, open(f.folders['intermediate'][flight_number] + 'posture_error_input_data.pkl', 'wb'))
        pickle.dump(camera_parameters, open(f.folders['intermediate'][flight_number] + 'camera_parameters.pkl', 'wb'))


def error_estimation():
    f = util.DataFolder('euroc_mav')
    for flight_number in [0, 1, 2, 3, 4, 5]:
        print(f"Flight : {flight_number}")

        input_data = pickle.load(open(f.folders['intermediate'][flight_number] + 'posture_error_input_data.pkl', 'rb'))
        camera_parameters = pickle.load(open(f.folders['intermediate'][flight_number] + 'camera_parameters.pkl', 'rb'))

        estimator = posture_error_estimation.ErrorEstimation(input_data=input_data,
                                                             calibration_matrix=camera_parameters['K'],
                                                             distortion_coefficient=camera_parameters['D'],
                                                             max_distance=5)
        estimator.p3_generator()
        saving_path = f.folders['results'][flight_number] + 'raw_video.mkv'
        estimator.p32video(fps=camera_parameters['fps'], saving_path=saving_path)
        print()


def excel_creation():
    f = util.DataFolder('euroc_mav')
    flight_number = 2

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
                tmp[name].append(util.Transformation().from_pose(trans, quat).inv())
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
        ids_0, ids_i = util.merge_two_arrays(to_synchro[ind_freq[0]]['time'], to_synchro[i]['time'])
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


if __name__ == '__main__':
    # data_processing()
    error_estimation()
    # excel_creation()
