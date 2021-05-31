import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
from tqdm import tqdm
from typing import Dict, Any

from src import utils

DATASET_PATH = os.getcwd() + "/data/blackbird/"


def raw_importing_cleaning(raw_flight_path, intermediate_flight_path):
    """
    Process data from the blackbird dataset.
    1. Load data.
    2. Format data.

    Downloaded from: http://blackbird-dataset.mit.edu/BlackbirdDatasetData/.
    Github: https://github.com/mit-fast/Blackbird-Dataset
    Paper: https://arxiv.org/abs/1810.01987

    Will save in intermediate/[flight_directory]/rosbag.pkl
    """
    print("Loading data")
    # Load tf_norm_mocap
    with open(raw_flight_path + "flightNormalizationOffset.csv", newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        trans = np.array(next(reader))
    tf_norm_mocap = utils.Transform().from_pose(trans, np.array([0, 0, 0, 1]))

    # # We do not use images so this transform is useless
    # # Load tf_fgenv_norm for each environment
    # with open(DATASET_PATH + "raw/trajectoryOffsets.yaml", 'r') as stream:
    #     try:
    #         yaml_file = yaml.safe_load(stream)
    #     except yaml.YAMLError as exc:
    #         print(exc)
    # # Finding the correct environment
    # offset = None
    # for e in yaml_file["unitySettings"][flight_name]:
    #     if e["name"] == environment:
    #         offset = e["offset"]
    # if offset is None:
    #     print(f"Environment not found: {environment}")
    #     raise ValueError(environment)
    # trans, angles = np.array(offset[:3]), np.array([offset[3], 0, 0])
    # tf_fgenv_norm = utils.Transform().from_euler(trans, "zyx", angles, degrees=True)

    # We will not use this one
    # tf_fgenvflu_fgenv = utils.Transform().from_euler(trans=np.zeros(3), seq="zyx",
    #                                                  angles=np.array([0, 0, 180]), degrees=True)
    # We will use this one instead
    # tf_normflu_norm = utils.Transform().from_euler(trans=np.zeros(3), seq="zyx",
    #                                                angles=np.array([0, 0, 180]), degrees=True)
    # tf_drone_droneflu = utils.Transform().from_euler(trans=np.zeros(3), seq="zyx",
    #                                                  angles=np.array([0, 0, 180]), degrees=True)

    # Intermediate folder is used to store processed data
    data_path = intermediate_flight_path + "0_rosbag.pkl"

    # Load sensors data
    raw_data = pickle.load(open(data_path, 'rb'), encoding='latin1')
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
    renaming_dict = {'/blackbird/pwm__blackbird/MotorPWM': "motor_pwm",
                     '/blackbird/rotor_rpm__blackbird/MotorRPM': "motor_rpm",
                     '/camera_d/camera_info__sensor_msgs/CameraInfo': "camera_info_d",
                     '/camera_l/camera_info__sensor_msgs/CameraInfo': "camera_info_l",
                     '/camera_r/camera_info__sensor_msgs/CameraInfo': "camera_info_r",
                     '/blackbird/imu__sensor_msgs/Imu': "imu",
                     '/blackbird/pose_ref__geometry_msgs/PoseStamped': "pose_ref",
                     '/tf__tf/tfMessage': "mc_pose",
                     '/blackbird/state__geometry_msgs/PoseStamped': "state"}
    # Renaming data keys for easier use
    for old_key, new_key in renaming_dict.items():
        raw_data[new_key] = raw_data.pop(old_key)

    # Assuming data timeseries keys are sorted by time and index is start at 0 and is incrementing by 1
    tmp = {}
    for name, timeserie in raw_data.items():
        tmp[name] = {k: v for k, v in zip(range(len(timeserie)),
                                          sorted(timeserie.values(),
                                                 key=lambda ts_msg: ts_msg["secs"] + 1e-9 * ts_msg["nsecs"]))}
    raw_data = tmp

    # # Let's search IMU camera body Transforms
    # calibration_data = pickle.load(open(DATASET_PATH +
    #                                     "intermediate/cameraCalibration/imuCameraCalibration.pkl", 'rb'),
    #                                encoding='latin1')
    # # dict_keys(['/camera_d/camera_info__sensor_msgs/CameraInfo',
    # # '/camera_l/camera_info__sensor_msgs/CameraInfo'
    # # '/camera_r/camera_info__sensor_msgs/CameraInfo',
    # # '/blackbird/imu__sensor_msgs/Imu',
    # # '/blackbird/state__geometry_msgs/PoseStamped',  mocap_NED
    # # '/tf__tf2_msgs/TFMessage',
    # # ])

    # All data have been loaded, formatting begins
    data = {"motor_speed_0": {},
            "motor_speed_1": {},
            "motor_speed_2": {},
            "motor_speed_3": {},
            "imu": {},
            "mc_pose": {},
            "state": {}}
    data: Dict[str][Dict[int][Dict[str][Any]]]
    # data structure
    # data = {
    #   "motor_speed_0": {idx(int): {"t": float, "angular_velocity": float}},
    #   "motor_speed_1": {idx(int): {"t": float, "angular_velocity": float}},
    #   "motor_speed_2": {idx(int): {"t": float, "angular_velocity": float}},
    #   "motor_speed_3": {idx(int): {"t": float, "angular_velocity": float}},
    #   "imu": {idx(int): {"t": float, "linear_acceleration": ndarray, "angular_velocity": ndarray}},
    #   "mc_pose": {idx(int): {"t": float, "translation": ndarray, "rotation": ndarray}},
    #   "state":  {idx(int): {"t": float, "dt": float, "state": ndarray}}

    # Motor speed
    for idx, msg in tqdm(raw_data["motor_rpm"].items(), desc="Computing motor speed"):
        data["motor_speed_0"][idx] = {"t": msg[0][0].secs + 1e-9 * msg[0][0].nsecs,
                                      "angular_velocity": np.pi / 30 * msg[0][1]}
        data["motor_speed_1"][idx] = {"t": msg[1][0].secs + 1e-9 * msg[1][0].nsecs,
                                      "angular_velocity": np.pi / 30 * msg[1][1]}
        data["motor_speed_2"][idx] = {"t": msg[2][0].secs + 1e-9 * msg[2][0].nsecs,
                                      "angular_velocity": np.pi / 30 * msg[2][1]}
        data["motor_speed_3"][idx] = {"t": msg[3][0].secs + 1e-9 * msg[3][0].nsecs,
                                      "angular_velocity": np.pi / 30 * msg[3][1]}
    # Assuming data timeseries keys are sorted by time and index is start at 0 and is incrementing by 1
    for name in ["motor_speed_0", "motor_speed_1", "motor_speed_2", "motor_speed_3"]:
        data[name] = {k: v for k, v in zip(range(len(data[name])),
                                           sorted(data[name].values(), key=lambda ts_msg: ts_msg["t"]))}

    # Compute IMU data in the correct reference frame
    for idx, msg in tqdm(raw_data["imu"].items(), desc="Computing imu"):
        # This transform is not confirmed, it is an HYPOTHESIS
        tf_imu_drone = utils.Transform().from_euler(trans=np.array([0, 0, 0]), seq="zyx",
                                                    angles=np.array([-90, 0, 0]), degrees=True)
        # rot_drone_imu = (tf_imu_drone @ tf_drone_droneflu).inv().get_rot_matrix()
        rot_drone_imu = tf_imu_drone.inv().get_rot_matrix()
        data["imu"][idx] = {"t": msg["secs"] + 1e-9 * msg["nsecs"],
                            "linear_acceleration": rot_drone_imu @ msg["linear_acceleration"],
                            "angular_velocity": rot_drone_imu @ msg["angular_velocity"]}

    # Compute pose data in the correct reference frame
    for idx, msg in tqdm(raw_data["mc_pose"].items(), desc="Computing pose"):
        # mc_pose is tf_mocap_drone
        tf_mocap_drone = utils.Transform().from_pose(trans=msg["translation"], quat=msg["rotation"])
        # tf_norm_drone = tf_normflu_norm @ tf_norm_mocap @ tf_mocap_drone @ tf_drone_droneflu
        tf_norm_drone = tf_norm_mocap @ tf_mocap_drone
        # mc_pose is tf_norm_drone
        data["mc_pose"][idx] = {"t": msg["secs"] + 1e-9 * msg["nsecs"], "transform": tf_norm_drone}

    # Resetting drone position
    tf_dronestart_norm = utils.Transform().from_pose(trans=data["mc_pose"][0]["transform"].get_trans(),
                                                     quat=np.array([0, 0, 0, 1])).inv()
    for idx in tqdm(data["mc_pose"], desc="Resetting first position"):
        tf_norm_drone = data["mc_pose"][idx]["transform"]
        tf_dronestart_drone = tf_dronestart_norm @ tf_norm_drone
        data["mc_pose"][idx]["transform"] = tf_dronestart_drone  # mc_pose is tf_dronestartFRD_droneFRD !

    # Compute the 12 dimension state data of the drone
    for idx in tqdm(range(1, len(data["mc_pose"])), desc="Computing state"):
        dt = data["mc_pose"][idx]["t"] - data["mc_pose"][idx - 1]["t"]
        # Compute euler angles in global frame from quaternion in global frame
        euler_orientation = data["mc_pose"][idx]["transform"].get_rot_euler(seq="zyx", degrees=False)[::-1]

        # Compute angular velocity in drone's frame from quaternion in global frame
        qn1 = data["mc_pose"][idx]["transform"].get_quat()
        qn = data["mc_pose"][idx - 1]["transform"].get_quat()
        # Graf, Basile. ‘Quaternions and Dynamics’. ArXiv:0811.2889 [Math-Ph], 18 November 2008.
        # https://arxiv.org/abs/0811.2889. Use a different convention than scipy.spatial.transform.Rotation.from_quat¶
        dq_dt = (qn1 - qn) / dt
        dq_dt = np.array([dq_dt[3], dq_dt[0], dq_dt[1], dq_dt[2]])
        q0, q1, q2, q3 = qn1[3], qn1[0], qn1[1], qn1[2]
        g_matrix = np.array([[-q1, q0, q3, -q2],
                             [-q2, -q3, q0, q1],
                             [-q3, q2, -q1, q0]])
        angular_velocity = (2 * g_matrix @ dq_dt.reshape((4, 1))).reshape((3,))

        # Compute linear speed in drone's frame from drone's position in global frame
        rot_drone_dronestart = data["mc_pose"][idx]["transform"].inv().get_rot_matrix()
        linear_speed = rot_drone_dronestart @ (data["mc_pose"][idx]["transform"].get_trans() -
                                               data["mc_pose"][idx - 1]["transform"].get_trans()) / dt

        position = data["mc_pose"][idx]["transform"].get_trans()
        data["state"][idx - 1] = {"t": data["mc_pose"][idx]["t"],
                                  "dt": dt,
                                  "state": np.hstack((euler_orientation, angular_velocity, linear_speed, position))}

    pickle.dump(data, open(intermediate_flight_path + "1_before_synchro.pkl", "wb"))


def synchronization(flight_name, flight_yaw, flight_speed, intermediate_flight_path, fig_path):
    print("Loading data")
    data = pickle.load(open(intermediate_flight_path + "1_before_synchro.pkl", "rb"))

    # data = {"motor_speed_0": {0: {"t": 1}, 1: {"t": 2}, 2: {"t": 3}, 3: {"t": 4}, 4: {"t": 5}},
    #         "motor_speed_1": {0: {"t": 1}, 1: {"t": 1}, 2: {"t": 3}, 3: {"t": 3}},
    #         "motor_speed_2": {0: {"t": 1}, 1: {"t": 2}, 2: {"t": 3}, 3: {"t": 4}, 4: {"t": 5}},
    #         "motor_speed_3": {0: {"t": 1}, 1: {"t": 2}, 2: {"t": 3}, 3: {"t": 4}, 4: {"t": 5}},
    #         "imu": {0: {"t": 1}, 1: {"t": 2}, 2: {"t": 3}, 3: {"t": 4}, 4: {"t": 5}},
    #         "mc_pose": {0: {"t": 1}, 1: {"t": 2}, 2: {"t": 3}, 3: {"t": 4}, 4: {"t": 5}},
    #         "state": {0: {"t": 1}, 1: {"t": 2}, 2: {"t": 3}, 3: {"t": 4}, 4: {"t": 5}}}

    # Synchronisation
    print("Synchronisation")
    # Assuming data timeseries keys are sorted by time and index is start at 0 and is incrementing by 1
    for ts_name, timeserie in data.items():
        times = np.array([timeserie[i]["t"] for i in range(len(timeserie))])
        sorted_times = np.sort(times)
        assert all(sorted_times == times), f"Timeseries: {ts_name} is not sorted by time"
    assert "t" not in data, "'t' key will be used for synchronisation"

    synchronisation_master = {}
    for ts_name in list(data.keys()):
        synchronisation_master[ts_name] = {"max": len(data[ts_name]), "idx": 0, "buffer": None}

    def is_one_timeserie_done():
        for max_idx_buffer in synchronisation_master.values():
            if max_idx_buffer["max"] == max_idx_buffer["idx"]:
                return True
        return False

    data_synced = []
    while not is_one_timeserie_done():
        ts_name, ts_stat = min(synchronisation_master.items(), key=lambda elem: data[elem[0]][elem[1]["idx"]]["t"])

        # for name in data.keys():
        #     print(data[name][synchronisation_master[name]["idx"]])
        synchronisation_master[ts_name]["buffer"] = data[ts_name][ts_stat["idx"]]

        # When buffer is full store sample with the most recent timestamp
        if all(val["buffer"] is not None for val in synchronisation_master.values()):
            # Create a sample from the buffer
            data_sample = {k: v["buffer"] for k, v in synchronisation_master.items()}

            # Timestamp kept for the sample will be the last addition to the complete buffer
            data_sample["t"] = data[ts_name][ts_stat["idx"]]["t"]
            data_synced.append(data_sample)

            # Resetting buffer
            for ts_name2 in synchronisation_master:
                synchronisation_master[ts_name2]["buffer"] = None

        synchronisation_master[ts_name]["idx"] = synchronisation_master[ts_name]["idx"] + 1

    # Check synchronization results
    data_keys = ['motor_speed_0', 'motor_speed_1', 'motor_speed_2', 'motor_speed_3', 'imu', 'mc_pose', 'state']
    var_time = []
    for sample in data_synced:
        var_time.append(np.var([sample[e]["t"] for e in data_keys]))
        # for data_type, element in sample.items():
        #     timestamp -
    var_time = np.array(var_time)
    tol = 1e-4
    bad_synchros = list(np.where(var_time > tol))[0]

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.plot([e for i, e in enumerate(var_time) if i not in bad_synchros])
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    ax.set_ylabel("Variance of timestamp considered as synced [s-2]")
    ax.set_xlabel("Sample number")
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "blackbird_synchro_one_flight.png")
    plt.show()

    print(f"Bad synchronisation happened at indices {bad_synchros} with a tolerance of {tol} of "
          f"the variance of the timestamps.")
    assert bad_synchros == [0], "Not only the first point is badly synchronised."
    kept_data = [e for i, e in enumerate(data_synced) if i not in bad_synchros]
    pickle.dump(kept_data, open(intermediate_flight_path + "2_data_synced.pkl", "wb"))


def plot_data(flight_name, flight_yaw, flight_speed, intermediate_flight_path, fig_path):
    print("Loading data")
    data = pickle.load(open(intermediate_flight_path + "2_data_synced.pkl", "rb"))

    # data_names = ['motor_speed_0', 'motor_speed_1', 'motor_speed_2', 'motor_speed_3', 'imu', 'mc_pose', 'state']
    time = np.array([e["t"] for e in data]) - data[0]["t"]

    # Motor speeds
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i in range(4):
        ax.plot(time, [e[f"motor_speed_{i}"]["angular_velocity"] for e in data], label=f"Motor {i}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Motor speed [rad/s]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "motor_speed.png")
    plt.show()

    # IMU angular velocity
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("x", "y", "z")):
        ax.plot(time, [e[f"imu"]["angular_velocity"][i] for e in data], label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("IMU angular velocity [rad/s]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "imu_angular_velocity.png")
    plt.show()

    # IMU linear acceleration
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("x", "y", "z")):
        ax.plot(time, [e[f"imu"]["linear_acceleration"][i] for e in data], label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("IMU linear acceleration [m/s2]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "imu_linear_acceleration.png")
    plt.show()

    # State orientation
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("phi", "theta", "psi")):
        ax.plot(time, [e[f"state"]["state"][i] for e in data], label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State orientation [rad]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # State angular velocity
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("p", "q", "r")):
        ax.plot(time, [e[f"state"]["state"][i + 3] for e in data], label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State angular velocity [rad/s]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_angular_velocity.png")
    plt.show()

    # State linear speed
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("u", "v", "w")):
        ax.plot(time, [e[f"state"]["state"][i + 6] for e in data], label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State linear speed [m/s]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_linear_speed.png")
    plt.show()

    # State position
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("x", "y", "z")):
        ax.plot(time, [e[f"state"]["state"][i + 9] for e in data], label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State position[m]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_position.png")
    plt.show()


# def _test():
#     # csv_filenames = ["blackbird_slash_imu.csv",
#     #                  "blackbird_slash_pose_ref.csv",
#     #                  "blackbird_slash_pwm.csv",
#     #                  "blackbird_slash_rotor_rpm.csv",
#     #                  "blackbird_slash_state.csv",
#     #                  "camera_d_slash_camera_info.csv",
#     #                  "camera_l_slash_camera_info.csv",
#     #                  "camera_r_slash_camera_info.csv",
#     #                  "tf.csv"]
#
#     # Check tf contains always the same transform
#     transform_list = []
#     for e in raw_data["mc_pose"].values():
#         frame1 = e["child_frame_id"]
#         frame2 = e["frame_id"]
#         transform = (frame1, frame2)
#         if transform not in transform_list:
#             transform_list.append(transform)
#
#     ground_truth_data = {}
#     with open(raw_flight_path + "groundTruthPoses.csv", newline='') as csv_file:
#         reader = csv.reader(csv_file, delimiter=',')
#         for i, row in enumerate(reader):
#             ground_truth_data[i] = [float(e) for e in row]
#     for i, e in enumerate(ground_truth.values()):
#         gt_trans = np.array([e[1], e[2], e[3]])
#         gt_quat = np.array([e[5], e[6], e[7], e[4]])
#         gt_time_like_gt = e[0]
#
#         tf_secs = data["tf"][i]["secs"]
#         tf_nsecs = data["tf"][i]["nsecs"]
#         tf_trans = data["tf"][i]["translation"]
#         tf_quat = data["tf"][i]["rotation"]
#
#         state_secs = data["state"][i]["secs"]
#         state_nsecs = data["state"][i]["nsecs"]
#         state_trans = data["state"][i]["translation"]
#         state_quat = data["state"][i]["rotation"]
#
#         assert tf_secs == state_secs
#         assert tf_nsecs == state_nsecs
#         assert all(tf_trans == state_trans)
#         assert all(tf_quat == state_quat)
#
#         tf_time_like_gt = tf_secs * 1e6 + np.around(tf_nsecs * 1e-3, 0)
#         tf_trans = np.around(data["tf"][i]["translation"], 6)
#         tf_quat = np.around(data["tf"][i]["rotation"], 6)
#
#         assert tf_time_like_gt == gt_time_like_gt
#         assert all(tf_trans == gt_trans)
#         assert all(tf_quat == gt_quat)
#
#     for camera_name in ["camera_info_d", "camera_info_l", "camera_info_r"]:
#         D = data[camera_name][0]["D"]
#         K = data[camera_name][0]["K"]
#         P = data[camera_name][0]["P"]
#         R = data[camera_name][0]["R"]
#         for e in data[camera_name].values():
#             np.testing.assert_equal(D, e["D"])
#             np.testing.assert_equal(K, e["K"])
#             np.testing.assert_equal(P, e["P"])
#             np.testing.assert_equal(R, e["R"])
#     print("blackbird.py is being tested.")
#
#     xyz = np.stack([e["transform"].get_trans() for e in data["mc_pose"].values()])
#     for i in range(3):
#         fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
#         ax.plot(xyz[:, i])
#     plt.show()
#
#     motor_speed = np.stack([e["angular_velocity"] for e in data["motor_speed_0"].values()])
#     fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
#     ax.plot(motor_speed)
#     plt.show()


def main():
    # Trajectories name classification

    # flight_names = ['ampersand','bentDice','cameraCalibration','clover','dice','egg','halfMoon','mouse','oval',
    # 'patrick','picasso','sid','sphinx','star','thrice','tiltedThrice','winter']

    # flight_yaws = ['yawForward', 'yawConstant']

    # flight_speed = ['maxSpeed0p5', 'maxSpeed1p0', 'maxSpeed2p0', 'maxSpeed3p0', 'maxSpeed4p0', 'maxSpeed5p0',
    # 'maxSpeed6p0', 'maxSpeed7p0']

    # environments = ['Small_Apartment', 'Large_Apartment_Night_Near_Couches']+

    flight_name = "ampersand"
    flight_yaw = "yawForward"
    flight_speed = "maxSpeed1p0"
    # environment = "Small_Apartment"

    # Paths
    raw_flight_path = DATASET_PATH + f"raw/{flight_name}/{flight_yaw}/{flight_speed}/"
    intermediate_flight_path = DATASET_PATH + f"intermediate/{flight_name}/{flight_yaw}/{flight_speed}/"
    fig_path = os.getcwd() + "/reports/blackbird_processed_data/"

    raw_importing_cleaning(raw_flight_path, intermediate_flight_path)
    synchronization(flight_name, flight_yaw, flight_speed, intermediate_flight_path, fig_path)
    plot_data(flight_name, flight_yaw, flight_speed, intermediate_flight_path, fig_path)

    # _test()


if __name__ == "__main__":
    main()
