import numpy as np
import pathlib
import pickle
from scipy import interpolate
from scipy.fft import fft, fftfreq, ifft
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset

from src.utils.transform import Transform
import src.utils.paths
from src.utils import quaternion

# Path
ANALYSIS_SECTION_PATH = src.utils.paths.GENERAL_ANALYSIS_PATH + "euroc_mav_processing_data/"
DATASET_PATH = src.utils.paths.WD_PATH + "data/euroc_mav/"
RAW_DATASET_PATH = DATASET_PATH + f"raw_python/"
INTERMEDIATE_DATASET_PATH = DATASET_PATH + f"intermediate/"

src.utils.paths.mkdir_if_not_exist([DATASET_PATH, ANALYSIS_SECTION_PATH, RAW_DATASET_PATH, INTERMEDIATE_DATASET_PATH])

FLIGHT_LIST = ["V1_01_easy/", "V1_02_medium/", "V1_03_difficult/"]


# FLIGHT_LIST = ["V2_01_easy/", "V2_02_medium/", "V2_03_difficult/"]


def _load_clean_data(raw_flight_path, intermediate_flight_path):
    # dict_keys(['/cam1/image_raw__sensor_msgs/Image', '/vicon/firefly_sbx/firefly_sbx__geometry_msgs/TransformStamped',
    # '/fcu/motor_speed__asctec_hl_comm/MotorSpeed', '/cam0/image_raw__sensor_msgs/Image', '/fcu/imu__sensor_msgs/Imu',
    # '/imu0__sensor_msgs/Imu'])
    raw = pickle.load(open(raw_flight_path + "data.pkl", "rb"), encoding="latin1")

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
    quat = Rotation.from_quat(quat).as_quat()
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

    print(f"start: {start} | stop: {stop} | total: {len(synchronisation['t'])}")

    # return start, stop
    cut_data = {}
    for key in synchronisation:
        cut_data[key] = synchronisation[key][start:stop]

    # Setting initial parameters
    cut_data["t"] = cut_data["t"] - cut_data["t"][0]

    save_path = intermediate_flight_path + "data.pkl"
    pathlib.Path(intermediate_flight_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(cut_data, open(save_path, "wb"))


def splitting():
    with open(INTERMEDIATE_DATASET_PATH + "split.txt", "w") as file:
        for flight in FLIGHT_LIST:
            print("Splitting: ", flight)
            data = load_flight(flight)  # dict_keys(['t', 'imu_ang_acc', 'motor', 'state', 'residual'])
            assert all(np.array([len(data[key]) for key in data]) == len(data["t"]))  # All keys have the same length
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


class EurocMAVDatasetHorizon(Dataset):
    def __init__(self, train_validation_test: str, traj_len: int):
        """
        Create a torch Dataset to be used in training with Dataloader.
        :param train_validation_test: train, validation or test to load the appropriate dataset
        :param traj_len: number of next state to load
        """
        if train_validation_test not in ["train", "validation", "test"]:
            raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
        split = [e.split(",") for e in open(INTERMEDIATE_DATASET_PATH + f"split.txt", "r")]
        split = {e[0]: (int(e[1]), int(e[2]), int(e[3])) for e in split}
        self.flight_list = list(split.keys())

        loading = {flight: pickle.load(open(INTERMEDIATE_DATASET_PATH + flight + train_validation_test + ".pkl",
                                            "rb")) for flight in self.flight_list}
        self.flight_length = {flight: len(loading[flight]["t"]) for flight in self.flight_list}
        minimum_length = np.min([[split[flight][0], split[flight][1] - split[flight][0],
                                  split[flight][2] - split[flight][1]] for flight in self.flight_list])

        assert isinstance(traj_len, int)
        assert (traj_len + 1) < minimum_length, f"horizon too big, maximum possible horizon is {minimum_length - 2}"
        self.h = traj_len

        self.cum_flight_idx = np.cumsum([self.flight_length[name] for name in self.flight_list]) - self.h - 1
        self.cum_dataset_idx = np.cumsum([self.flight_length[name] - self.h for name in self.flight_list]) - 1

        # Loading data
        for key, val in loading.items():
            loading[key] = np.concatenate([val["t"].reshape((-1, 1)), val["state"], val["motor"], val["residual"]],
                                          axis=1)

        self.data = np.concatenate([loading[flight] for flight in self.flight_list], axis=0)

        # print(np.min(self.data[:, 20:], axis=0))
        # print(np.max(self.data[:, 20:], axis=0))

        # self.min_res = torch.tensor([-2.56919551, -1.6980865, -2.45562077,
        #                              -14.32858562, -11.89167309, -4.79519939]).view(1, 6)
        # self.max_res = torch.tensor([4.07108498, 2.69354105, 2.62223434,
        #                              13.02334404, 15.27588463, 6.88414383]).view(1, 6)

    def __len__(self):
        return self.cum_dataset_idx[-1] + 1

    def __getitem__(self, idx):
        """
        One sample is time:1, state:13, motor:6, residual:6, not implemented (imu_ang_acc:6)

        """
        flight_idx = np.where(idx <= self.cum_dataset_idx)[0][0]
        data = self.data[self.cum_flight_idx[flight_idx]:self.cum_flight_idx[flight_idx] + self.h + 1, :]
        # "residual": (torch.from_numpy(data[:, 20:]).float() - self.min_res) / (self.max_res - self.min_res)}
        res = {"t": torch.from_numpy(data[:, :1]).float(),
               "state": torch.from_numpy(data[:, 1:14]).float(),
               "motor": torch.from_numpy(data[:, 14:20]).float(),
               # "residual": (torch.from_numpy(data[:, 20:]).float() - self.min_res) / (self.max_res - self.min_res)}
               "residual": torch.from_numpy(data[:, 20:]).float()}

        return res


def load_flight(flight):
    #     data = {"t", "imu_ang_acc", "motor", "state"}
    intermediate_flight_path = INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
    data = pickle.load(open(intermediate_flight_path, "rb"))
    return data


def main():
    # for flight in FLIGHT_LIST:
    #     raw_flight_path = RAW_DATASET_PATH + flight
    #     intermediate_flight_path = INTERMEDIATE_DATASET_PATH + flight
    #     print("Loading", flight)
    #     _load_clean_data(raw_flight_path, intermediate_flight_path)
    # splitting()
    data = EurocMAVDatasetHorizon("train", traj_len=1)


if __name__ == '__main__':
    main()
    # _madiha_laurent_csv()
    # _test_plot_final_data()
