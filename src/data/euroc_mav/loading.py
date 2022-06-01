from collections import OrderedDict
import numpy as np
import pathlib
import pickle
import os
from scipy import interpolate
from scipy import signal
# from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from typing import Union

# import src.data.euroc_mav.data_vis as data_vis
from src.drone_model.analytical_model.hexarotor_4params import DynamicalModel
from src.utils.path import WD_PATH, mkdir
from src.utils import quaternion

# Path
DATASET_PATH = WD_PATH + "data/euroc_mav/"
RAW_PYTHON_DATASET_PATH = DATASET_PATH + f"raw_python/"
RAW_DATASET_PATH = DATASET_PATH + f"raw/"
INTERMEDIATE_DATASET_PATH = DATASET_PATH + f"intermediate/"

mkdir([DATASET_PATH, RAW_PYTHON_DATASET_PATH, INTERMEDIATE_DATASET_PATH, RAW_DATASET_PATH])

FLIGHT_LIST = ["V1_01_easy/", "V1_02_medium/", "V1_03_difficult/"]
# FLIGHT_LIST = ["V2_01_easy/", "V2_02_medium/", "V2_03_difficult/"]

TRANS_MEAN = {"V1_01_easy/": np.array([0.36256679, 0.37336331, 1.52168557]),
              "V1_02_medium/": np.array([0.00806092, 1.18968812, 1.60374988]),
              "V1_03_difficult/": np.array([-0.0128916, 1.04127094, 1.75094365])}


def load_clean_data(flight):
    raw_python_flight_path = RAW_PYTHON_DATASET_PATH + flight
    intermediate_flight_path = INTERMEDIATE_DATASET_PATH + flight
    print("Loading", flight)
    # 1-LOADING DATA
    # dict_keys(['/cam1/image_raw__sensor_msgs/Image', '/vicon/firefly_sbx/firefly_sbx__geometry_msgs/TransformStamped',
    # '/fcu/motor_speed__asctec_hl_comm/MotorSpeed', '/cam0/image_raw__sensor_msgs/Image', '/fcu/imu__sensor_msgs/Imu',
    # '/imu0__sensor_msgs/Imu'])
    raw = pickle.load(open(raw_python_flight_path + "data.pkl", "rb"), encoding="latin1")

    # 1.1-IMU Loading
    # dict_keys(['orientation_covariance', 'orientation', 'angular_velocity_covariance', 'nsecs',
    # 'linear_acceleration_covariance', 'frame_id', 'secs', 'linear_acceleration', 'angular_velocity'])
    imu_key = "/fcu/imu__sensor_msgs/Imu"  # '/imu0__sensor_msgs/Imu' also exist
    # orientation = np.array([raw[imu_key][i]["orientation"] for i in range(len(raw[imu_key]))])
    orientation_covariance = np.array([raw[imu_key][i]["orientation_covariance"] for i in range(len(raw[imu_key]))])
    assert np.linalg.norm(orientation_covariance) == 0  # not usable
    ang_vel = np.array([raw[imu_key][i]["angular_velocity"] for i in range(len(raw[imu_key]))])
    ang_vel_cov = np.array([raw[imu_key][i]["angular_velocity_covariance"] for i in range(len(raw[imu_key]))])
    assert np.linalg.norm(ang_vel_cov) != 0  # usable
    lin_acc = np.array([raw[imu_key][i]["linear_acceleration"] for i in range(len(raw[imu_key]))])
    lin_acc_cov = np.array([raw[imu_key][i]["linear_acceleration_covariance"] for i in range(len(raw[imu_key]))])
    assert np.linalg.norm(lin_acc_cov) != 0  # usable
    frame_id = np.array([raw[imu_key][i]["frame_id"] for i in range(len(raw[imu_key]))])
    assert np.sum("fcu" == frame_id) == len(frame_id)  # All the time "fcu"
    time = np.array([raw[imu_key][i]["secs"] + raw[imu_key][i]["nsecs"] * 1e-9 for i in range(len(raw[imu_key]))])
    timestamp = np.array([raw[imu_key][i]["secs"] * int(1e9) +
                          raw[imu_key][i]["nsecs"] for i in range(len(raw[imu_key]))])
    assert (timestamp == sorted(timestamp)).all()
    imu_dict = {"t": time, "ang_acc": np.concatenate([ang_vel, lin_acc], axis=1), "timestamp": timestamp}

    # # timestamp, x, y, z, qw, qx, qy, qz
    # imu_open_vins = np.loadtxt(f'{DATASET_PATH}OpenVINS/{flight[:-1]}.csv', delimiter=',', skiprows=1)
    # time, trans = imu_open_vins[:, 0], imu_open_vins[:, 1:4]
    # quat = np.stack([imu_open_vins[:, 5], imu_open_vins[:, 6], imu_open_vins[:, 7], imu_open_vins[:, 4]], axis=1)
    # imu_dict = {'t': time,
    #             'pose': np.concatenate([trans, quat], axis=1),
    #             'timestamp': (time * 1e9).astype(int)}

    # 1.2-MOTOR SPEED Loading
    # dict_keys(['frame_id', 'secs', 'motor_speed', 'nsecs'])
    ms_key = '/fcu/motor_speed__asctec_hl_comm/MotorSpeed'
    frame_id = np.array([raw[ms_key][i]["frame_id"] for i in range(len(raw[ms_key]))])
    assert np.sum("fcu" == frame_id) == len(frame_id)  # All the time "fcu"
    time = np.array([raw[ms_key][i]["secs"] + raw[ms_key][i]["nsecs"] * 1e-9 for i in range(len(raw[ms_key]))])
    timestamp = np.array([raw[ms_key][i]["secs"] * int(1e9) + raw[ms_key][i]["nsecs"] for i in range(len(raw[ms_key]))])
    assert (timestamp == sorted(timestamp)).all()
    ms = np.array([raw[ms_key][i]["motor_speed"] for i in range(len(raw[ms_key]))], dtype=float)  # Assumed in rad/s
    motor_dict = {"t": time, "motor": ms, "timestamp": timestamp}

    # 1.3-STATE Loading
    # dict_keys(['child_frame_id', 'nsecs', 'frame_id', 'secs', 'translation', 'rotation'])
    state_key = "/vicon/firefly_sbx/firefly_sbx__geometry_msgs/TransformStamped"
    child_frame_id = np.array([raw[state_key][i]["child_frame_id"] for i in range(len(raw[state_key]))])
    # Check if it is all the time 'vicon/firefly_sbx/firefly_sbx'
    assert np.sum("vicon/firefly_sbx/firefly_sbx" == child_frame_id) == len(child_frame_id)
    time = np.array([raw[state_key][i]["secs"] + raw[state_key][i]["nsecs"] * 1e-9 for i in range(len(raw[state_key]))])
    timestamp = np.array([raw[state_key][i]["secs"] * int(1e9) +
                          raw[state_key][i]["nsecs"] for i in range(len(raw[state_key]))])
    assert (timestamp == sorted(timestamp)).all()
    frame_id = np.array([raw[state_key][i]["frame_id"] for i in range(len(raw[state_key]))])
    assert np.sum("/world" == frame_id) == len(frame_id)  # All the time "/world"
    trans = np.array([raw[state_key][i]["translation"] for i in range(len(raw[state_key]))])
    quat = np.array([raw[state_key][i]["rotation"] for i in range(len(raw[state_key]))])  # x, y, z, w
    pose_dict = {"t": time, "pose": np.concatenate([trans, quat], axis=1), "timestamp": timestamp}

    # 1.4-CAM0 image name loading
    # dict_keys(['nsecs', 'image', 'step', 'height', 'width', 'secs', 'is_bigendian'])
    cam0_key = '/cam0/image_raw__sensor_msgs/Image'
    timestamp = np.array(sorted([int(e[:-4])
                                 for e in os.listdir(f"{RAW_DATASET_PATH}{flight}/cam0/data/")]))
    # time = np.array([raw[cam0_key][i]["secs"] + raw[cam0_key][i]["nsecs"] * 1e-9 for i in range(len(raw[cam0_key]))])
    # image = np.array([raw[cam0_key][i]["image"] for i in range(len(raw[cam0_key]))])
    step = np.array([raw[cam0_key][i]["step"] for i in range(len(raw[cam0_key]))])
    assert (step == 752).all()  # step not use
    height = np.array([raw[cam0_key][i]["height"] for i in range(len(raw[cam0_key]))])
    assert (height == 480).all()  # height not use
    width = np.array([raw[cam0_key][i]["width"] for i in range(len(raw[cam0_key]))])
    assert (width == 752).all()  # width not use
    is_bigendian = np.array([raw[cam0_key][i]["is_bigendian"] for i in range(len(raw[cam0_key]))])
    assert is_bigendian.sum() == 0  # is_bigendian not use
    cam0_dict = {"timestamp": timestamp}

    # 2-DENOISING
    # 2.1-Denoising motor speed's timestamp and motor speed
    den_motor_dict = {}
    for key, val in motor_dict.items():
        den_motor_dict[key] = val.copy()
    # Denoising motor timestamp
    period = 1 / 200
    den_motor_dict["t"] = (period * np.array(range(len(motor_dict["t"]))) + motor_dict["t"][0])

    # Butterworth on motor speed
    sos = signal.butter(N=2, Wn=2.8, btype='lowpass', fs=1 / period, output='sos')
    den_motor_dict["motor"] = signal.sosfiltfilt(sos, motor_dict["motor"], axis=0)

    # 2.2-Denoising IMU timestamp
    den_imu_dict = {}
    period = np.mean(imu_dict["t"][1:] - imu_dict["t"][:-1])
    den_imu_dict["t"] = (period * np.array(range(len(imu_dict["t"]))) + imu_dict["t"][0])  # Denoising IMU timestamp
    den_imu_dict["timestamp"] = imu_dict["timestamp"]
    den_imu_dict["ang_acc"] = imu_dict["ang_acc"]  # No denoising for IMU data

    # den_imu_dict["ang_acc"] = imu_dict["ang_acc"]  # No denoising for IMU data
    # # Butterworth on imu angular_velocity
    # sos = signal.butter(N=2, Wn=4.04, btype='lowpass', fs=1/period, output='sos')
    # den_imu_dict["ang_acc"][:, :3] = signal.sosfiltfilt(sos,  imu_dict["ang_acc"][:, :3], axis=0)
    # # Butterworth on imu acceleration
    # sos = signal.butter(N=2, Wn=2.8, btype='lowpass', fs=1/period, output='sos')
    # den_imu_dict["ang_acc"][:, 3:5] = signal.sosfiltfilt(sos,  imu_dict["ang_acc"][:, 3:5], axis=0)
    # sos = signal.butter(N=2, Wn=5.14, btype='lowpass', fs=1/period, output='sos')
    # den_imu_dict["ang_acc"][:, 5] = signal.sosfiltfilt(sos,  imu_dict["ang_acc"][:, 5])

    # 2.3-Computing a denoised motion capture's translation and quaternion.
    # 2.3.1-Initialization, normalization and centering
    # Body and World reference frame are in FLU! Nice!
    den_pose_dict = {"timestamp": pose_dict["timestamp"]}
    quat = pose_dict["pose"][:, 3:7]
    trans = pose_dict["pose"][:, :3]
    time = pose_dict["t"]
    n = len(time)

    quat = quat / np.linalg.norm(quat, axis=1).reshape((-1, 1))  # Normalize quaternion
    quat = quaternion.qfix(quat)  # Fix quaternion swap in continuity

    # Centering position
    trans = trans - np.mean(trans, axis=0)

    # 2.3.2 Denoising timestamp
    period = np.mean(time[1:] - time[:-1])
    den_pose_dict["t"] = (period * np.array(range(n)) + time[0])

    # 2.3.3-Denoising quaternion
    # dt = (den_pose_dict["t"][1:] - den_pose_dict["t"][:-1]).reshape((-1, 1))
    sos = signal.butter(N=2, Wn=2.92, btype='lowpass', fs=1 / period, analog=False, output='sos')
    den_quat = signal.sosfiltfilt(sos, quat, axis=0)
    den_quat = den_quat / np.linalg.norm(den_quat, axis=1).reshape((-1, 1))  # Normalize quaternions

    # 2.3.4-Denoising translation
    # Ideal denoising
    # data_vis.plot_butterworth_comparison(trans[:, 3].reshape((-1, 1)), ideal_f=5, period=period, do_norm=False, dt=dt)
    sos = signal.butter(N=2, Wn=2.92, btype='lowpass', fs=1 / period, analog=False, output='sos')
    den_trans = signal.sosfiltfilt(sos, trans, axis=0)

    den_pose_dict["pose"] = np.concatenate([den_trans, den_quat], axis=1)

    # 3-CUT: Cut data and keep only when the drone is flying
    # data_vis.found_fa(flight, den_motor_dict, den_pose_dict, cam0_dict)
    cut_motor_idx = {FLIGHT_LIST[0]: [1113, 28294],
                     FLIGHT_LIST[1]: [954, 16503],
                     FLIGHT_LIST[2]: [1510, 20855]}

    start, stop = cut_motor_idx[flight]
    for key in motor_dict:
        den_motor_dict[key] = den_motor_dict[key][start:stop]
    print(f'Cut len: {stop - start}')

    # 4-SYNCHRONISATION    
    # Highest frequency comes from the fcu (flight control unit). Motor timestamp will be taken for synchronisation
    # 4.1.1-Spline interpolation method
    # Imu
    imu_sync = np.zeros((len(den_motor_dict["t"]), 6))
    for i in range(6):
        rep = interpolate.splrep(den_imu_dict["t"], den_imu_dict["ang_acc"][:, i], k=5)
        imu_sync[:, i] = interpolate.splev(den_motor_dict["t"], rep)
    # Motion capture
    pose_sync = np.zeros((len(den_motor_dict["t"]), 7))
    for i in range(7):
        rep = interpolate.splrep(den_pose_dict["t"], den_pose_dict["pose"][:, i], k=5)
        pose_sync[:, i] = interpolate.splev(den_motor_dict["t"], rep)
    # Cam0
    cam0_sync = np.zeros(len(den_motor_dict["timestamp"]), dtype=int)  # The timestamp of the image
    for i, timestamp in enumerate(den_motor_dict["timestamp"]):
        cam0_sync[i] = cam0_dict["timestamp"][np.abs(timestamp - cam0_dict["timestamp"]).argmin()]

    # 4.1.2-Linear interpolation method
    # imu_sync = np.stack([interpolate.interp1d(den_imu_dict["t"], den_imu_dict["ang_acc"][:, i], bounds_error=False,
    #                                           fill_value=(den_imu_dict["ang_acc"][0, i],
    #                                                       den_imu_dict["ang_acc"][-1, i]))(
    #     den_motor_dict["t"]) for i in range(6)], axis=1)
    # pose_sync = np.stack([interpolate.interp1d(den_pose_dict["t"], den_pose_dict["pose"][:, i], bounds_error=False,
    #                                            fill_value=(den_pose_dict["pose"][0, i],
    #                                                        den_pose_dict["pose"][-1, i]))(
    #     den_motor_dict["t"]) for i in range(7)], axis=1)
    syn = {"t": den_motor_dict["t"], "timestamp": den_motor_dict["timestamp"], "imu_ang_acc": imu_sync,
           "motor": den_motor_dict["motor"], "pose": pose_sync, "cam0": cam0_sync}

    # 4.2-Setting initial parameters and normalizing quaternion
    syn["t"] = syn["t"] - syn["t"][0]
    syn["pose"][:, 3:7] = syn["pose"][:, 3:7] / np.linalg.norm(syn["pose"][:, 3:7], axis=1).reshape((-1, 1))

    # 5-COMPUTE DERIVATIVE
    dt = 1 / 200
    quat = syn["pose"][:, 3:7]
    trans = syn["pose"][:, :3]

    # 5.1-Compute angular velocity
    # 5.1.1-Derivative of the quaternion
    dquat = (quat[1:] - quat[:-1]) / dt  # Euler derivative
    dquat = np.concatenate([np.zeros((1, 4)), dquat], axis=0)

    # 5.1.2-Quaternion to angular velocity
    # 5.1.2.1-Quaternion multiplication method
    ang_vel = 2. * quaternion.qmul_np(quat * np.array([[-1., -1., -1., 1.]]), dquat)[:, :3]

    # 5.1.2.2-Matrix method
    # dq_dt_convention = dquat[:, [3, 0, 1, 2]]
    # q0, q1, q2, q3 = (quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2])
    # g_matrix = np.array([[-q1, q0, q3, -q2],
    #                      [-q2, -q3, q0, q1],
    #                      [-q3, q2, -q1, q0]]).transpose((2, 0, 1))
    # ang_vel2 = (2 * g_matrix @ dq_dt_convention.reshape((-1, 4, 1))).squeeze()
    #
    # print("mean: ", abs(ang_vel2 - ang_vel).mean())  # < 1e-17 with qmul product method from ref
    # print("std: ", abs(ang_vel2 - ang_vel).std())

    # 5.1.2.3-With Lie exponential
    # https://math.stackexchange.com/questions/2282938/
    # converting-from-quaternion-to-angular-velocity-then-back-to-quaternion
    # ang_vel3 = 2. * (quaternion.qmul_np(quat[:-1] * np.array([[-1, -1, -1, 1]]), quat[1:]))[:, :3] / dt
    # ang_vel3 = np.concatenate([np.zeros((1, 3)), ang_vel3], axis=0)
    # def exp_so3(w):
    #     norm = np.linalg.norm(w, axis=1).reshape((-1, 1))
    #     return np.concatenate([np.cos(1/2 * norm), w / norm * np.sin(1/2*norm)], axis=1)
    # next_quat = quaternion.qmul_np(quat[:-1], exp_so3(ang_vel * dt))
    #
    # print("mean: ", abs(ang_vel3 - ang_vel).mean())  # < 3e-15 with qmul product method from ref
    # print("std: ", abs(ang_vel3 - ang_vel).std())

    # 5.2 Compute linear speed
    dtrans = (trans[1:] - trans[:-1]) / dt  # Euler derivative
    dtrans = np.concatenate([np.zeros((1, 3)), dtrans], axis=0)

    # 5.3 Merge and normalize
    syn["state"] = np.concatenate([trans, quat, dtrans, ang_vel], axis=1)
    del syn["pose"]
    syn["state"][:, 3:7] = syn["state"][:, 3:7] / np.linalg.norm(syn["state"][:, 3:7], axis=1).reshape((-1, 1))
    syn['dt'] = np.zeros_like(syn['t']) + 1 / 200

    for e in syn:
        syn[e] = syn[e][1:]  # Reject first speed

    # 6. Save dict as with pickle
    pathlib.Path(intermediate_flight_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(syn, open(intermediate_flight_path + "data.pkl", "wb"))


def euroc_mav_residual():
    dtype = torch.float64
    model = DynamicalModel(param="identification", dtype=dtype)
    for flight in FLIGHT_LIST:
        # Importing data
        data = load_flight(flight)
        # time = torch.from_numpy(data["t"][:-1])
        motor = torch.from_numpy(data["motor"][1:])
        dt = torch.from_numpy(data["dt"][1:]).reshape((-1, 1))
        state = torch.from_numpy(data["state"][:-1])
        next_state = torch.from_numpy(data["state"][1:])

        # Test-1 Quaternion norm is 1 and standard deviation is 1e-18
        # state[:, 3:7] = state[:, 3:7] / torch.norm(state[:, 3:7], dim=1).view(-1, 1)
        # next_state[:, 3:7] = next_state[:, 3:7] / torch.norm(next_state[:, 3:7], dim=1).view(-1, 1)
        norm = torch.norm(state[:, 3:7], dim=1)
        print("Quaternion's norm mean: ", torch.mean(norm))
        print("Quaternion's norm std: ", torch.std(norm))

        # Test-2 integrating perfect derivative lead to perfect pose
        assert abs((state[1:, :3] - state[:-1, :3]) / dt[1:] - state[1:, 7:10]).sum() == 0

        for key in data:
            data[key] = data[key][1:]

        # Computing residual
        residual_torsor = model.residual_dynamic(motor=motor, state=state, dt=dt, next_state=next_state)
        data["residual"] = residual_torsor.numpy()

        # plot_residual_torsor(time, data["residual"], flight, xlim=(0, time[-1]))

        # Saving
        intermediate_flight_path = INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
        pickle.dump(data, open(intermediate_flight_path, "wb"))


def merge_embeddings():
    for flight in FLIGHT_LIST:
        # Importing data
        data = load_flight(flight)

        # list[dict_keys(['timestamp', 'embedding', 'MSE', 'GradMSE'])]
        depth_emb = pickle.load(open(f"{INTERMEDIATE_DATASET_PATH}{flight}depth_embedding.pkl", "rb"))
        depth_emb = sorted(depth_emb, key=lambda x: x["timestamp"])
        depth_t = np.array([e["timestamp"] for e in depth_emb])
        assert (data["t"] == depth_t).all
        data["depth_emb"] = np.array([e["embedding"].flatten() for e in depth_emb]).astype(float)

        # list[dict_keys(['timestamp', 'embedding', 'MSE', 'CrossProduct'])]
        norm_emb = pickle.load(open(f"{INTERMEDIATE_DATASET_PATH}{flight}normal_embedding.pkl", "rb"))
        norm_emb = sorted(norm_emb, key=lambda x: x["timestamp"])
        assert (data["t"] == np.array([e["timestamp"] for e in norm_emb])).all
        data["norm_emb"] = np.array([e["embedding"].flatten() for e in norm_emb]).astype(float)

        # Saving
        intermediate_flight_path = INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
        pickle.dump(data, open(intermediate_flight_path, "wb"))


def label_surface_influence():
    for flight in FLIGHT_LIST:
        data = load_flight(flight)
        last_idx = len(data['t'])
        fa_presence = {FLIGHT_LIST[0]: [range(25943, 26213), range(26513, last_idx)],
                       FLIGHT_LIST[1]: [range(0, 250), range(15240, last_idx)],
                       FLIGHT_LIST[2]: [range(18626, last_idx)]}
        fa_presence_array = np.zeros(len(data['state']))
        for presence_range in fa_presence[flight]:
            fa_presence_array[presence_range] = 1
        data['fa_label'] = fa_presence_array
        intermediate_flight_path = INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
        pickle.dump(data, open(intermediate_flight_path, "wb"))
        print(f'''flight={flight} | Portion of 1: {data['fa_label'].mean()}''')


def splitting():
    with open(INTERMEDIATE_DATASET_PATH + "split.txt", "w") as file:
        for flight in FLIGHT_LIST:
            print("Splitting: ", flight)
            data = load_flight(flight)  # dict_keys(['t', 'imu_ang_acc', 'motor', 'state', 'residual'])

            assert all(np.array([len(data[key]) for key in data]) == len(data["t"])), f"All Data key must have same" \
                                                                                      f" length"
            flight_len = len(data["t"])
            print(flight, flight_len)

            train_val_idx = round(flight_len * 0.6)
            val_test_idx = round(flight_len * 0.8)
            file.write(f"{flight},{train_val_idx},{val_test_idx},{flight_len}\n")

            train_data = {key: data[key][:train_val_idx] for key in data}
            pickle.dump(train_data, open(INTERMEDIATE_DATASET_PATH + flight + "/train.pkl", "wb"))
            validation_data = {key: data[key][train_val_idx:val_test_idx] for key in data}
            pickle.dump(validation_data, open(INTERMEDIATE_DATASET_PATH + flight + "/validation.pkl", "wb"))
            test_data = {key: data[key][val_test_idx:flight_len] for key in data}
            pickle.dump(test_data, open(INTERMEDIATE_DATASET_PATH + flight + "/test.pkl", "wb"))


DATASET_RANGE = OrderedDict({  # [stat, stop , size]
    "t": [0, 1, 1],
    "state": [1, 14, 13],
    "motor": [14, 20, 6],
    "residual": [20, 26, 6],
    "imu_ang_acc": [26, 32, 6],
    "depth_emb": [32, 160, 128],
    "norm_emb": [160, 288, 128],
    "fa_label": [288, 289, 1],
    "dt": [289, 290, 1],
})


class EurocMAVDatasetHorizon(Dataset):
    def __init__(self, train_validation_test: str, traj_len: int, dtype, batch_dim: bool):
        """
        Create a torch Dataset to be used in training with Dataloader.
        :param train_validation_test: train, validation or test to load the appropriate dataset
        :param traj_len: length of the trajectory to give each index
        :param dtype: Datatype of the model.
        :param batch_dim: If the Dataset will be used with a Dataloader or not.

        One sample is time:1, state:13, motor:6, residual:6, imu_ang_acc:6, depth_emb:128, norm_emb:128, fa_label:1,
        dt: 1
        """
        print(f'loading.py EurocMAVDatasetHorizon __init__ start train_validation_test={train_validation_test}'
              f'traj_len={traj_len} dtype={dtype} batch_dim={batch_dim}')
        assert dtype in [torch.float32, torch.float64]
        self.batch_dim = batch_dim
        self.dtype = dtype
        if train_validation_test not in ["train", "validation", "test"]:
            raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
        self.train_validation_test = train_validation_test
        split = [e.split(",") for e in open(INTERMEDIATE_DATASET_PATH + f"split.txt", "r")]
        split = {e[0]: (int(e[1]), int(e[2]), int(e[3])) for e in split}
        self.flight_list = list(split.keys())

        loaded_data = {flight: pickle.load(open(INTERMEDIATE_DATASET_PATH + flight + train_validation_test + ".pkl",
                                                "rb")) for flight in self.flight_list}
        self.flight_length = {flight: len(loaded_data[flight]["t"]) for flight in self.flight_list}
        minimum_length = np.min([[split[flight][0],
                                  split[flight][1] - split[flight][0],
                                  split[flight][2] - split[flight][1]] for flight in self.flight_list])

        assert isinstance(traj_len, int)
        assert traj_len < minimum_length, f"horizon too big, maximum possible horizon is {minimum_length - 1}"
        self.traj_len = traj_len

        self.cum_flight_idx = np.cumsum([self.flight_length[name] for name in self.flight_list])
        self.cum_flight_idx = np.concatenate([np.array([0]), self.cum_flight_idx])
        self.cum_dataset_idx = np.cumsum([self.flight_length[name] - self.traj_len for name in self.flight_list])
        self.cum_dataset_idx = np.concatenate([np.array([0]), self.cum_dataset_idx])

        # Loading data
        for flight, val in loaded_data.items():
            concatenation = []
            for data_key, dr in DATASET_RANGE.items():
                concatenation.append(val[data_key].reshape((-1, 1)) if dr[2] == 1 else val[data_key])
            loaded_data[flight] = np.concatenate(concatenation, axis=1)

        self.data = np.concatenate([loaded_data[flight] for flight in self.flight_list], axis=0)
        self.data = torch.from_numpy(self.data).type(self.dtype)

    def __len__(self):
        return self.cum_dataset_idx[-1]

    def __getitem__(self, idx):
        flight_idx = np.where(idx <= self.cum_dataset_idx[1:])[0][0]
        local_idx = idx - self.cum_dataset_idx[flight_idx] + self.cum_flight_idx[flight_idx]
        start = local_idx
        stop = local_idx + self.traj_len
        data = self.data[start:stop]
        assert len(data) == self.traj_len
        if self.batch_dim:
            return {key: data[:, dr[0]:dr[1]].view((1, -1, dr[2])) for key, dr in DATASET_RANGE.items()}
        else:
            return {key: data[:, dr[0]:dr[1]] for key, dr in DATASET_RANGE.items()}


class EurocMAVDatasetTurbulence(Dataset):
    def __init__(self, train_validation_test: str, dtype: Union[type, torch.dtype], batch_dim: bool):
        """
        Create a torch Dataset to be used in training with Dataloader.

        :param train_validation_test: train, validation or test to load the appropriate dataset

        :param dtype: Datatype of the model.

        :param batch_dim: If the Dataset will be used with a Dataloader or not.

        One sample is time:1, state:13, motor:6, residual:6, imu_ang_acc:6, depth_emb:128, norm_emb:128, fa_label:1,
        dt: 1
        """
        assert dtype in [torch.float32, torch.float64]
        self.batch_dim = batch_dim
        self.dtype = dtype
        if train_validation_test not in ["train", "validation", "test"]:
            raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
        self.train_validation_test = train_validation_test
        if self.train_validation_test in ['train']:
            loaded_data = {flight: load_flight(flight) for flight in [FLIGHT_LIST[0], FLIGHT_LIST[2]]}
        else:
            loaded_data = {flight: load_flight(flight) for flight in [FLIGHT_LIST[1]]}
        # Loading data
        for flight, val in loaded_data.items():
            concatenation = []
            for data_key, dr in DATASET_RANGE.items():
                concatenation.append(val[data_key].reshape((-1, 1)) if dr[2] == 1 else val[data_key])
            loaded_data[flight] = np.concatenate(concatenation, axis=1)

        self.data = np.concatenate([loaded_data[flight] for flight in loaded_data.keys()], axis=0)
        self.data = torch.from_numpy(self.data).type(self.dtype)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        if self.batch_dim:
            return {key: data[dr[0]:dr[1]].view((1, -1, dr[2])) for key, dr in DATASET_RANGE.items()}
        else:
            return {key: data[dr[0]:dr[1]] for key, dr in DATASET_RANGE.items()}


def load_flight(flight):
    """
    :param flight: The flight to from FLIGHT_LIST
    :return: dict_keys(['t', 'imu_ang_acc', 'motor', 'img', 'state', 'residual', 'depth_emb', 'norm_emb', 'fa_label',
                        'dt'])
    """
    #     data = {"t", "imu_ang_acc", "motor", "state"}
    intermediate_flight_path = INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
    data = pickle.load(open(intermediate_flight_path, "rb"))
    return data


def main():
    load_clean_data(FLIGHT_LIST[0])
    load_clean_data(FLIGHT_LIST[1])
    load_clean_data(FLIGHT_LIST[2])
    euroc_mav_residual()
    merge_embeddings()
    label_surface_influence()
    splitting()

    # from torch.utils.data import DataLoader
    # dataset = EurocMAVDatasetHorizon(train_validation_test="train", traj_len=201, dtype=torch.float32, batch_dim=False)
    # batch = next(iter(DataLoader(dataset, batch_size=64)))
    # print(len(batch))

    # from torch.utils.data import DataLoader
    #
    # all_data = {}
    # for key in ['train', 'validation', 'test']:
    #     dataset = EurocMAVDatasetTurbulence(train_validation_test=key, dtype=torch.float32, batch_dim=False)
    #     all_data[key] = next(iter(DataLoader(dataset, batch_size=len(dataset))))
    # for emb in ['depth_emb', 'norm_emb']:
    #     for data_to_test in ['validation', 'test']:
    #         for i, sample in enumerate(all_data['train'][emb]):
    #             assert len(torch.where((sample == all_data[data_to_test][emb]).all(dim=1))[0]) == 0


if __name__ == '__main__':
    main()
