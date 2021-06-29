import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import random
from scipy.fft import fft, fftfreq, ifft
from scipy import interpolate
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src import trajectory
from src.utils.transform import Transform
import src.utils.paths
from src.utils import quaternion

# Path
ANALYSIS_SECTION_PATH = src.utils.paths.GENERAL_ANALYSIS_PATH + "euroc_mav_processing_data/"
DATASET_PATH = src.utils.paths.WD_PATH + "data/euroc_mav/"
RAW_DATASET_PATH = DATASET_PATH + f"raw_python/"
INTERMEDIATE_DATASET_PATH = DATASET_PATH + f"intermediate/"

src.utils.paths.mkdir_if_not_exist([DATASET_PATH, ANALYSIS_SECTION_PATH, RAW_DATASET_PATH, INTERMEDIATE_DATASET_PATH])

FLIGHT_LIST = ["V1_01_easy/V1_01_easy.pkl", "V1_02_medium/V1_02_medium.pkl", "V1_03_difficult/V1_03_difficult.pkl"]


def _load_clean_data(raw_flight_path, intermediate_flight_path):
    # dict_keys(['/cam1/image_raw__sensor_msgs/Image', '/vicon/firefly_sbx/firefly_sbx__geometry_msgs/TransformStamped',
    # '/fcu/motor_speed__asctec_hl_comm/MotorSpeed', '/cam0/image_raw__sensor_msgs/Image', '/fcu/imu__sensor_msgs/Imu',
    # '/imu0__sensor_msgs/Imu'])
    raw = pickle.load(open(raw_flight_path, "rb"), encoding="latin1")

    # IMU Loading
    # dict_keys(['orientation_covariance', 'orientation', 'angular_velocity_covariance', 'nsecs',
    # 'linear_acceleration_covariance', 'frame_id', 'secs', 'linear_acceleration', 'angular_velocity'])
    imu_key = "/fcu/imu__sensor_msgs/Imu"  # '/imu0__sensor_msgs/Imu' also exist
    orientation = np.array([raw[imu_key][i]["orientation"] for i in range(len(raw[imu_key]))])
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
    # plt.plot(time - time[0])
    imu_dict = {"t": time, "ang_acc": np.concatenate([ang_vel, lin_acc], axis=1)}

    # MOTOR SPEED Loading
    # dict_keys(['frame_id', 'secs', 'motor_speed', 'nsecs'])
    ms_key = '/fcu/motor_speed__asctec_hl_comm/MotorSpeed'
    frame_id = np.array([raw[ms_key][i]["frame_id"] for i in range(len(raw[ms_key]))])
    assert np.sum("fcu" == frame_id) == len(frame_id)  # All the time "fcu"
    time = np.array([raw[ms_key][i]["secs"] + raw[ms_key][i]["nsecs"] * 1e-9 for i in range(len(raw[ms_key]))])
    ms = np.array([raw[ms_key][i]["motor_speed"] for i in range(len(raw[ms_key]))], dtype=float)  # Assumed in rad/s
    motor_dict = {"t": time, "ms": ms}

    # STATE Loading
    # dict_keys(['child_frame_id', 'nsecs', 'frame_id', 'secs', 'translation', 'rotation'])
    state_key = "/vicon/firefly_sbx/firefly_sbx__geometry_msgs/TransformStamped"
    child_frame_id = np.array([raw[state_key][i]["child_frame_id"] for i in range(len(raw[state_key]))])
    # Check if it is all the time 'vicon/firefly_sbx/firefly_sbx'
    assert np.sum("vicon/firefly_sbx/firefly_sbx" == child_frame_id) == len(child_frame_id)
    time = np.array([raw[state_key][i]["secs"] + raw[state_key][i]["nsecs"] * 1e-9 for i in range(len(raw[state_key]))])
    frame_id = np.array([raw[state_key][i]["frame_id"] for i in range(len(raw[state_key]))])
    assert np.sum("/world" == frame_id) == len(frame_id)  # All the time "/world"
    trans = np.array([raw[state_key][i]["translation"] for i in range(len(raw[state_key]))])
    quat = np.array([raw[state_key][i]["rotation"] for i in range(len(raw[state_key]))])  # x, y, z, w
    pose_dict = {"t": time}

    # Body and World reference frame are in FLU! Nice!
    # Fix quaternion discontinuity
    quat = quaternion.qfix(quat)

    # Centering position
    trans = trans - np.mean(trans, axis=0)

    # trajectory.trace_trajectory3d(trans, quat)  # Check data by tracing drone trajectory

    # FOURIER noise reduction
    def low_pass_filter(fft_coef, fourier_freq, freq_cut):
        output = fft_coef.copy()
        output[np.abs(fourier_freq) > freq_cut] = 0
        return output

    # trajectory.timestamp_analysis(pose_dict["t"])  # Check timestamps for each timeseries
    # trajectory.timestamp_analysis(imu_dict["t"])
    # trajectory.timestamp_analysis(motor_dict["t"])
    # Highest frequency comes from the fcu (flight control unit). Motor timestamp will be taken for synchronisation

    # Computing a denoised state.
    period = np.mean(pose_dict["t"][1:] - pose_dict["t"][:-1])
    n = len(pose_dict["t"])
    frequencies = fftfreq(n, d=period)

    # DENOISING with Fourier
    fc_rot = 5  # Hz
    fourier_quat = fft(quat, axis=0)
    fourier_dquat = 2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)) * fourier_quat
    # fourier_d2quat = 2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)) * fourier_dquat
    den_quat = np.real(ifft(low_pass_filter(fourier_quat, frequencies, fc_rot), axis=0))
    den_dquat = np.real(ifft(low_pass_filter(fourier_dquat, frequencies, fc_rot), axis=0))
    # den_d2quat = np.real(ifft(low_pass_filter(fourier_d2quat, frequencies, fc_rot), axis=0))

    # Compute angular velocity
    dq_dt_convention = den_dquat[:, [3, 0, 1, 2]]
    q0, q1, q2, q3 = den_quat[:, 3], den_quat[:, 0], den_quat[:, 1], den_quat[:, 2]
    g_matrix = np.array([[-q1, q0, q3, -q2],
                         [-q2, -q3, q0, q1],
                         [-q3, q2, -q1, q0]]).transpose((2, 0, 1))
    den_ang_vel = (2 * g_matrix @ dq_dt_convention.reshape((-1, 4, 1))).squeeze()

    # Compute linear speed
    fc_trans = 5  # Hz
    fourier_trans = fft(trans, axis=0)
    fourier_dtrans = 2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)) * fourier_trans
    # fourier_d2trans = 2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)) * fourier_dtrans
    den_trans = np.real(ifft(low_pass_filter(fourier_trans, frequencies, fc_trans), axis=0))
    den_dtrans = np.real(ifft(low_pass_filter(fourier_dtrans, frequencies, fc_trans), axis=0))
    # den_d2trans = np.real(ifft(low_pass_filter(fourier_d2trans, frequencies, fc_trans), axis=0))

    # Denoising motor speed
    period = np.mean(motor_dict["t"][1:] - motor_dict["t"][:-1])
    n = len(motor_dict["t"])
    frequencies = fftfreq(n, d=period)
    den_motor_dict = {}
    for key, val in motor_dict.items():
        den_motor_dict[key] = val.copy()
    for i in range(6):
        fc_motor = 5  # Hz
        fourier_motor = fft(motor_dict["ms"][:, i])
        # den_motor.append(np.real(ifft(low_pass_filter(fourier_motor, frequencies, fc_motor))))
        den_motor_dict["ms"][:, i] = np.real(ifft(low_pass_filter(fourier_motor, frequencies, fc_motor)))
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
    # time = motor_dict["t"] - motor_dict["t"][0]
    # # [ax.plot(time, motor_dict["ms"][:, i], label="raw") for i in range(6)]
    # # [ax.plot(time, den_motor_dict["ms"][:, i], label="denoised") for i in range(6)]
    # ax.plot(time, motor_dict["ms"][:, 0], label="raw")
    # ax.plot(time, den_motor_dict["ms"][:, 0], label="denoised")
    # ax.grid()
    # ax.set_title(raw_flight_path.split("/")[-1])
    # ax.set_ylabel("[rad/s]")
    # ax.set_xlabel("Time [s]")
    # ax.legend()
    state = np.concatenate([den_trans, den_quat, den_dtrans, den_ang_vel], axis=1)
    pose_dict["state"] = state

    # Synchronisation
    imu_sync = np.stack([interpolate.interp1d(imu_dict["t"], imu_dict["ang_acc"][:, i], bounds_error=False,
                                              fill_value=(imu_dict["ang_acc"][0, i], imu_dict["ang_acc"][-1, i]))(
        den_motor_dict["t"]) for i in range(6)], axis=1)
    state_sync = np.stack([interpolate.interp1d(pose_dict["t"], pose_dict["state"][:, i], bounds_error=False,
                                                fill_value=(pose_dict["state"][0, i], pose_dict["state"][-1, i]))(
        den_motor_dict["t"]) for i in range(13)], axis=1)

    synchronisation = {"t": den_motor_dict["t"], "imu_ang_acc": imu_sync, "motor": den_motor_dict["ms"],
                       "state": state_sync}

    # Take data only when drone is flying
    start, stop = 0, 0
    has_started = False
    for i, ms in enumerate(synchronisation["motor"]):
        if (ms > 120).any() and not has_started:
            start = i
            has_started = True
        if has_started and (ms < 80).any():
            stop = i
            break
    assert start != stop

    # return start, stop
    cut_data = {}
    for key in synchronisation:
        cut_data[key] = synchronisation[key][start:stop]

    # Setting initial parameters
    cut_data["t"] = cut_data["t"] - cut_data["t"][0]

    save_path = intermediate_flight_path + "data.pkl"
    pathlib.Path(intermediate_flight_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(cut_data, open(save_path, "wb"))


def _splitting_txt():
    idx = list(range(len(FLIGHT_LIST)))
    random.shuffle(idx)
    idx = np.array(idx)
    train_val_idx = round(len(idx) * 0.6)
    val_test_idx = round(len(idx) * 0.8)
    shuffled_list = np.asarray(FLIGHT_LIST)[idx]

    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    special_flight = f"{flight_name}/{flight_yaw}/{flight_speed}/"

    while special_flight not in shuffled_list[:train_val_idx]:
        random.shuffle(idx)
        shuffled_list = np.asarray(FLIGHT_LIST)[idx]

    for name, flight_list in zip(["train.txt", "validation.txt", "test.txt"],
                                 [sorted(shuffled_list[:train_val_idx]),
                                  sorted(shuffled_list[train_val_idx:val_test_idx]),
                                  sorted(shuffled_list[val_test_idx:])]):
        with open(INTERMEDIATE_DATASET_PATH + name, "w") as file:
            for flight in flight_list:
                file.write(flight + "\n")


def _torch_splitting():
    train_list = list(e[:-2] for e in open(INTERMEDIATE_DATASET_PATH + "train.txt", "r"))
    validation_list = list(e[:-2] for e in open(INTERMEDIATE_DATASET_PATH + "validation.txt", "r"))
    test_list = list(e[:-2] for e in open(INTERMEDIATE_DATASET_PATH + "test.txt", "r"))

    def concatenate(data_pkl):
        t = data_pkl["t"][:-1].reshape((-1, 1))
        state = data_pkl["state"][:-1]
        motor = data_pkl["motor"][:-1]
        imu = data_pkl["imu_ang_acc"][:-1]
        next_t = data_pkl["t"][1:].reshape((-1, 1))
        next_state = data_pkl["state"][1:]
        return np.concatenate([t, state, motor, imu, next_t, next_state], axis=1)

    train = []
    for flight in train_list:
        intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}/"
        train.append(concatenate(pickle.load(open(intermediate_flight_path + "data.pkl", "rb"))))
    train = torch.from_numpy(np.concatenate(train, axis=0))
    torch.save(train, INTERMEDIATE_DATASET_PATH + "train.pt")

    validation = []
    for flight in validation_list:
        intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}/"
        validation.append(concatenate(pickle.load(open(intermediate_flight_path + "data.pkl", "rb"))))
    validation = torch.from_numpy(np.concatenate(validation, axis=0))
    torch.save(validation, INTERMEDIATE_DATASET_PATH + "validation.pt")

    test = []
    for flight in test_list:
        intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}/"
        test.append(concatenate(pickle.load(open(intermediate_flight_path + "data.pkl", "rb"))))
    test = torch.from_numpy(np.concatenate(test, axis=0))
    torch.save(test, INTERMEDIATE_DATASET_PATH + "test.pt")


class BlackbirdDatasetHorizon(Dataset):
    def __init__(self, train_validation_test: str, horizon: int):
        if train_validation_test not in ["train", "validation", "test"]:
            raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
        self.flight_length = pickle.load(open(INTERMEDIATE_DATASET_PATH + "flight_length.pkl", "rb"))
        self.flight_list = list(e[:-2] for e in open(DATASET_PATH + f"{train_validation_test}.txt", "r"))

        assert isinstance(horizon, int) and (horizon + 1) < min(self.flight_length.values())
        self.h = horizon
        self.cum_flight_idx = np.cumsum([self.flight_length[name] for name in self.flight_list]) - 1 - self.h
        self.cum_dataset_idx = np.cumsum([self.flight_length[name] - self.h for name in self.flight_list]) - 1

        # Loading data
        loading = {flight: pickle.load(open(DATASET_PATH + f"intermediate/{flight}/data.pkl", "rb"))
                   for flight in self.flight_list}
        for key, val in loading.items():
            loading[key] = np.concatenate([val["t"].reshape((-1, 1)), val["state"], val["motor"]], axis=1)
            # t = self.data1[0:1], state = self.data1[1:14], motor = self.data1[14:18]
        self.data = np.concatenate([loading[flight] for flight in self.flight_list], axis=0)

    def __len__(self):
        return self.cum_dataset_idx[-1] + 1

    def __getitem__(self, idx):
        """
        One sample is time:1, state:13, motor:4, imu_ang_acc:6, time_next_state:1, next_state:13 = 38

        """
        flight_idx = np.where(idx <= self.cum_dataset_idx)[0][0]
        data = self.data[self.cum_flight_idx[flight_idx]:self.cum_flight_idx[flight_idx] + self.h + 1, :]

        return {"t": torch.from_numpy(data[:, :1]).float(),
                "state": torch.from_numpy(data[:, 1:14]).float(),
                "motor": torch.from_numpy(data[:, 14:18]).float()}


# class BlackbirdDataset(Dataset):
#     def __init__(self, train_validation_test: str):
#         if train_validation_test not in ["train", "validation", "test"]:
#             raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
#         self.data = torch.load(DATASET_PATH + f"intermediate_torch_dataset/{train_validation_test}.pt")
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         """
#         One sample is time:1, state:13, motor:4, imu_ang_acc:6, time_next_state:1, next_state:13 = 38
#
#         """
#         data = self.data[idx]
#         return {"t": data[0:1], "state": data[1:14], "motor": data[14:18], "imu": data[18:24], "next_t": data[24:25],
#                 "next_state": data[25:]}


def load_flight(flight_name, flight_yaw, flight_speed):
    flight = f"/{flight_name}/{flight_yaw}/{flight_speed}/"

    intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
    data = pickle.load(open(intermediate_flight_path + "data.pkl", "rb"))
    return data


def main():
    for flight in FLIGHT_LIST:
        raw_flight_path = RAW_DATASET_PATH + flight
        intermediate_flight_path = INTERMEDIATE_DATASET_PATH + flight
        print(flight)
        _load_clean_data(raw_flight_path, intermediate_flight_path)

    # print(f"ANALYSIS_SECTION_PATH: ", ANALYSIS_SECTION_PATH)
    # for i in tqdm(range(len(FLIGHT_LIST))):
    #     flight = FLIGHT_LIST[i]
    #     raw_flight_path = DATASET_PATH + f"raw/{flight}"
    #     intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
    #     _load_clean_data(raw_flight_path, intermediate_flight_path)

    # _torch_splitting()


def _madiha_laurent_csv():
    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    flight = f"/{flight_name}/{flight_yaw}/{flight_speed}/"
    # return {"t": data[0:1], "state": data[1:13], "motor": data[13:17], "imu": data[17:23], "next_t": data[23:24],
    #         "dstate": data[24:36], "next_state": data[36:]}

    data = None  # Must load from raw not intermediate because of denoising.
    t = data[:, 0]
    state = data[:, 1:13]
    motor = data[:, 13:17]
    imu_ang_acc = data[:, 17:23]
    header = ["t[s]",
              "phi[rad]", "theta[rad]", "psi[rad]",
              "x[m]", "y[m]", "z[m]",
              "motor0[rad/s]", "motor1[rad/s]", "motor2[rad/s]", "motor3[rad/s]",
              "imu_ang_vel_x[rad/s]", "imu_ang_vel_y[rad/s]", "imu_ang_vel_z[rad/s]",
              "imu_acc_x[m/s2]", "imu_acc_y[m/s2]", "imu_acc_z[m/s2]"]
    ml_data = np.concatenate([t.reshape((-1, 1)),
                              state[:, 0:3],
                              state[:, 9:12],
                              motor,
                              imu_ang_acc], axis=1)
    assert len(header) == ml_data.shape[1]
    with open(DATASET_PATH + "intermediate/madiha_laurent.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in ml_data:
            writer.writerow(row)


def _test_plot_final_data():
    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    flight = f"/{flight_name}/{flight_yaw}/{flight_speed}/"

    intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
    data = pickle.load(open(intermediate_flight_path + "data.pkl", "rb"))
    # return {"t": data[0:1], "state": data[1:13], "motor": data[13:17], "imu": data[17:23], "next_t": data[23:24],
    #         "dstate": data[24:36], "next_state": data[36:]}

    # State orientation
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    dt = 1 / 360
    dphi = (data[1:, 0] - data[:-1, 0]) / dt
    dtheta = (data[1:, 1] - data[:-1, 1]) / dt
    dpsi = (data[1:, 2] - data[:-1, 2]) / dt

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("phi", "theta", "psi")):
        ax.plot(range(len(data)), data[:, 0 + i], label=name)
    ax.set_xlabel("Sample number")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # Autre derivee
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for array, name in zip([dphi, dtheta, dpsi], ("phi", "theta", "psi")):
        ax.plot(list(range(len(data)))[1:], array, label=name)
    ax.set_xlabel("Sample number")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # State orientation
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("phi", "theta", "psi")):
        ax.plot(range(len(data)), data[:, 0 + i], label=name)
    ax.set_xlabel("Sample number")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # State angular velocity
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("phi", "theta", "psi")):
        ax.plot(range(len(data)), data[:, 3 + i], label=name)
    ax.set_xlabel("Sample number")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    # _madiha_laurent_csv()
    # _test_plot_final_data()
