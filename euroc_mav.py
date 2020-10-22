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
        camera_parameters['camera_tf_drone'] = util.Transform().from_matrix(tmp).inv()  # The extrinsic parameter of the

        # Vicon0 parameters
        # f.get_files_paths('.yaml', f.folders['raw'][vol_number], 'vicon0')
        with open(f.get_unique_file_path('.yaml', f.folders['raw'][flight_number], 'vicon0'), 'r') as stream:
            try:
                yaml_file = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        # drone_tf_vicon0 : Sensor extrinsic wrt. the body-frame
        drone_tf_vicon0 = util.Transform().from_matrix(np.array(yaml_file['T_BS']['data']).reshape((4, 4)))
        camera_parameters['camera_tf_vicon0'] = camera_parameters['camera_tf_drone'] @ drone_tf_vicon0
        del camera_parameters['camera_tf_drone']

        # mc default columns : '#timestamp [ns]', 'p_RS_R_x [m]', 'p_RS_R_y [m]', 'p_RS_R_z [m]', 'q_RS_w []',
        #                                         'q_RS_x []', 'q_RS_y []', 'q_RS_z []'
        mc = pd.read_csv(f.get_unique_file_path('.csv', f.folders['raw'][flight_number], 'vicon0'), sep=',', header=0,
                         names=['pose_time', 'x', 'y', 'z', 'w', 'a', 'b', 'c'])  # Renaming column for simplification
        p = util.Progress(len(mc), "Preparing mc data")
        tmp = {}
        for index, row in mc.iterrows():
            # vicon0_tf_origin
            vicon0_tf_origin = util.Transform().from_pose(trans=row[['x', 'y', 'z']].to_numpy(),
                                                          quat=row[['a', 'b', 'c', 'w']].to_numpy()).inv()
            camera_tf_origin = camera_parameters['camera_tf_vicon0'] @ vicon0_tf_origin
            tmp[index] = camera_tf_origin
            p.update_pgr()
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
            p.update_pgr()
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
        saving_path = f.folders['results'][flight_number] + 'pose_camera_correspondence_video.mkv'
        estimator.p32video(fps=camera_parameters['fps'], saving_path=saving_path)
        print()


def excel_creation():
    f = util.DataFolder('euroc_mav')
    flight_number = 0

    # Motor speed
    data = f.pickle_load_file('.pkl', f.folders['raw_python'][flight_number], None, True)
    tmp = {'motor_speed': [], 'motor_speed_time': []}
    for idx, measure in data['/fcu/motor_speed'].items():
        tmp['motor_speed'].append(measure['motor_speed'])
        tmp['motor_speed_time'].append(measure['t'])
    motor_speed = pd.DataFrame(tmp)
    motor_speed['motor_speed_time'] = motor_speed['motor_speed_time'] - motor_speed['motor_speed_time'][0]

    img_pose_data = pickle.load(open(f.folders['intermediate'][flight_number] + 'posture_error_input_data.pkl', 'rb'))

    ids_pose, ids_ms = util.merge_two_arrays(img_pose_data['pose_time'], motor_speed['motor_speed_time'])



if __name__ == '__main__':
    # data_processing()
    # error_estimation()
    excel_creation()
