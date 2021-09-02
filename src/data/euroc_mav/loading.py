import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
from scipy import interpolate, signal
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
    # 1-LOADING DATA
    # dict_keys(['/cam1/image_raw__sensor_msgs/Image', '/vicon/firefly_sbx/firefly_sbx__geometry_msgs/TransformStamped',
    # '/fcu/motor_speed__asctec_hl_comm/MotorSpeed', '/cam0/image_raw__sensor_msgs/Image', '/fcu/imu__sensor_msgs/Imu',
    # '/imu0__sensor_msgs/Imu'])
    raw = pickle.load(open(raw_flight_path + "data.pkl", "rb"), encoding="latin1")

    # 1.1-IMU Loading
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
    imu_dict = {"t": time, "ang_acc": np.concatenate([ang_vel, lin_acc], axis=1)}

    # 1.2-MOTOR SPEED Loading
    # dict_keys(['frame_id', 'secs', 'motor_speed', 'nsecs'])
    ms_key = '/fcu/motor_speed__asctec_hl_comm/MotorSpeed'
    frame_id = np.array([raw[ms_key][i]["frame_id"] for i in range(len(raw[ms_key]))])
    assert np.sum("fcu" == frame_id) == len(frame_id)  # All the time "fcu"
    time = np.array([raw[ms_key][i]["secs"] + raw[ms_key][i]["nsecs"] * 1e-9 for i in range(len(raw[ms_key]))])
    ms = np.array([raw[ms_key][i]["motor_speed"] for i in range(len(raw[ms_key]))], dtype=float)  # Assumed in rad/s
    motor_dict = {"t": time, "motor": ms}

    # 1.3-STATE Loading
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
    pose_dict = {"t": time, "pose": np.concatenate([trans, quat], axis=1)}

    # 2-DENOISING
    def low_pass_filter(fft_coef, fourier_freq, freq_cut):  # Fourier low pass filter
        output = fft_coef.copy()
        output[np.abs(fourier_freq) > freq_cut] = 0
        return output

    # 2.1-Denoising motor speed's timestamp and motor speed
    den_motor_dict = {}
    for key, val in motor_dict.items():
        den_motor_dict[key] = val.copy()
    # Denoising motor speed with low pass filter on Fourier's coefficients
    # period = np.mean(motor_dict["t"][1:] - motor_dict["t"][:-1])
    period = 1 / 200
    n = len(motor_dict["t"])
    frequencies = fftfreq(n, d=period)
    den_motor_dict["t"] = (period * np.array(range(len(motor_dict["t"]))) + motor_dict["t"][0])

    for i in range(6):
        # den_motor_dict["motor"][:, i] = motor_dict["motor"][:, i]  # no denoising
        fc_motor = 5  # Hz
        # Ideal denoising
        fourier_motor = fft(motor_dict["motor"][:, i])
        den_motor_dict["motor"][:, i] = np.real(ifft(low_pass_filter(fourier_motor, frequencies, fc_motor)))
        # Butterworth
        # sos = signal.butter(1, fc_motor, 'lowpass', fs=200, output='sos')
        # den_motor_dict["motor"][:, i] = signal.sosfilt(sos, motor_dict["motor"][:, i])

    # 2.2-Denoising IMU timestamp
    den_imu_dict = {}
    period = np.mean(imu_dict["t"][1:] - imu_dict["t"][:-1])
    den_imu_dict["t"] = (period * np.array(range(len(imu_dict["t"]))) + imu_dict["t"][0])  # Denoising IMU timestamp
    den_imu_dict["ang_acc"] = imu_dict["ang_acc"]  # No denoising for IMU data

    # 2.3-Computing a denoised translation and quaternion.
    # Body and World reference frame are in FLU! Nice!
    den_pose_dict = {}
    quat = pose_dict["pose"][:, 3:7]
    trans = pose_dict["pose"][:, :3]
    time = pose_dict["t"]
    n = len(time)

    quat = quat / np.linalg.norm(quat, axis=1).reshape((-1, 1))  # Normalize quaternion
    quat = quaternion.qfix(quat)  # Fix quaternion swap in continuity

    # Centering position  # TODO
    # trans = trans - np.mean(trans, axis=0)

    # Denoising timestamp
    period = np.mean(time[1:] - time[:-1])
    den_pose_dict["t"] = (period * np.array(range(n)) + time[0])

    # Denoising quaternion
    # den_quat = quat  # no denoising
    # Ideal denoising
    frequencies = fftfreq(n, d=period)
    fc_quat = 5  # Hz
    fourier_quat = fft(quat, axis=0)
    den_quat = np.real(ifft(low_pass_filter(fourier_quat, frequencies, fc_quat), axis=0))
    # fourier_dquat = 2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)) * fourier_quat
    # Butterworth
    # den_quat = np.zeros_like(quat)
    # sos = signal.butter(1, fc_quat, 'lowpass', fs=1 / period, output='sos')
    # for i in range(4):
    #     den_quat[:, i] = signal.sosfilt(sos, quat[:, i])
    den_quat = den_quat / np.linalg.norm(den_quat, axis=1).reshape((-1, 1))  # Normalize quaternions

    # Denoising translation
    # den_trans = trans  # no denoising
    # Ideal denoising
    fc_trans = 5  # Hz
    fourier_trans = fft(trans, axis=0)
    den_trans = np.real(ifft(low_pass_filter(fourier_trans, frequencies, fc_trans), axis=0))
    # fourier_dtrans = 2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)) * fourier_trans
    # fourier_d2trans = 2 * np.pi * complex(0, 1) * frequencies.reshape((-1, 1)) * fourier_dtrans
    # Butterworth
    # den_trans = np.zeros_like(trans)
    # sos = signal.butter(1, fc_trans, 'lowpass', fs=1 / period, output='sos')
    # for i in range(3):
    #     den_trans[:, i] = signal.sosfilt(sos, trans[:, i])
    den_pose_dict["pose"] = np.concatenate([den_trans, den_quat], axis=1)

    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
    # ax.plot(motor_dict["t"] - motor_dict["t"][0], motor_dict["motor"][:, i], label="Raw")
    # ax.plot(den_motor_dict["t"] - den_motor_dict["t"][0], np.real(ifft(low_pass_filter(fourier_motor, frequencies, fc_motor))), label="Ideal low pass")
    # ax.plot(den_motor_dict["t"] - den_motor_dict["t"][0], den_motor_dict["motor"][:, i], label="Butterworth")
    # ax.legend()
    # ax.grid()
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Motor speed")

    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
    # i = 0
    # ax.plot(pose_dict["t"] - pose_dict["t"][0], quat, label="Raw")
    # ax.plot(den_pose_dict["t"] - den_pose_dict["t"][0], np.real(ifft(low_pass_filter(fourier_quat, frequencies, fc_quat), axis=0)), label="Ideal low pass")
    # ax.plot(den_pose_dict["t"] - den_pose_dict["t"][0], den_quat, label="Butterworth")
    # ax.legend()
    # ax.grid()
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Quaternion")
    #
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8))
    # ax.plot(pose_dict["t"] - pose_dict["t"][0], trans, label="Raw")
    # ax.plot(den_pose_dict["t"] - den_pose_dict["t"][0], np.real(ifft(low_pass_filter(fourier_trans, frequencies, fc_trans), axis=0)), label="Ideal low pass")
    # ax.plot(den_pose_dict["t"] - den_pose_dict["t"][0], den_trans, label="Butterworth")
    # ax.legend()
    # ax.grid()
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Translations [m]")

    # 3-CUT: Cut data and keep only when the drone is flying
    start, stop = 0, 0
    has_started = False
    for i, ms in enumerate(den_motor_dict["motor"]):
        if (ms > 120).any() and not has_started:
            start = i
            has_started = True
        if has_started and (ms < 80).any():
            stop = i
            break
    assert start != stop
    print(f"""start: {start} | stop: {stop} | total: {len(den_motor_dict["t"])}""")
    for key in motor_dict:
        den_motor_dict[key] = den_motor_dict[key][start:stop]

    # 4-SYNCHRONISATION    
    # Highest frequency comes from the fcu (flight control unit). Motor timestamp will be taken for synchronisation
    # 4.1.1-Spline interpolation
    imu_sync = np.zeros((len(den_motor_dict["t"]), 6))
    for i in range(6):
        rep = interpolate.splrep(den_imu_dict["t"], den_imu_dict["ang_acc"][:, i], k=5)
        imu_sync[:, i] = interpolate.splev(den_motor_dict["t"], rep)
    pose_sync = np.zeros((len(den_motor_dict["t"]), 7))
    for i in range(7):
        rep = interpolate.splrep(den_pose_dict["t"], den_pose_dict["pose"][:, i], k=5)
        pose_sync[:, i] = interpolate.splev(den_motor_dict["t"], rep)

    # 4.1.2-Linear interpolation
    # imu_sync = np.stack([interpolate.interp1d(den_imu_dict["t"], den_imu_dict["ang_acc"][:, i], bounds_error=False,
    #                                           fill_value=(den_imu_dict["ang_acc"][0, i],
    #                                                       den_imu_dict["ang_acc"][-1, i]))(
    #     den_motor_dict["t"]) for i in range(6)], axis=1)
    # pose_sync = np.stack([interpolate.interp1d(den_pose_dict["t"], den_pose_dict["pose"][:, i], bounds_error=False,
    #                                            fill_value=(den_pose_dict["pose"][0, i],
    #                                                        den_pose_dict["pose"][-1, i]))(
    #     den_motor_dict["t"]) for i in range(7)], axis=1)
    syn = {"t": den_motor_dict["t"], "imu_ang_acc": imu_sync, "motor": den_motor_dict["motor"],
           "pose": pose_sync}

    # 4.2-Setting initial parameters and normalizing quaternion
    syn["t"] = syn["t"] - syn["t"][0]
    syn["pose"][:, 3:7] = syn["pose"][:, 3:7] / np.linalg.norm(syn["pose"][:, 3:7], axis=1).reshape((-1, 1))

    # 5-COMPUTE DERIVATIVE
    dt = 1 / 200
    quat = syn["pose"][:, 3:7]
    trans = syn["pose"][:, :3]

    # 5.1 Compute angular velocity
    # Derivative of the quaternion
    dquat = (quat[1:] - quat[:-1]) / dt  # Euler derivative
    dquat = np.concatenate([np.zeros((1, 4)), dquat], axis=0)

    # 5.1.1 Quaternion multiplication method
    ang_vel = 2. * quaternion.qmul_np(quat * np.array([[-1., -1., -1., 1.]]), dquat)[:, :3]

    # 5.1.2 Matrix method
    # dq_dt_convention = dquat[:, [3, 0, 1, 2]]
    # q0, q1, q2, q3 = (quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2])
    # g_matrix = np.array([[-q1, q0, q3, -q2],
    #                      [-q2, -q3, q0, q1],
    #                      [-q3, q2, -q1, q0]]).transpose((2, 0, 1))
    # ang_vel2 = (2 * g_matrix @ dq_dt_convention.reshape((-1, 4, 1))).squeeze()
    #
    # print("mean: ", abs(ang_vel2 - ang_vel).mean())  # < 1e-17 with qmul product method from ref
    # print("std: ", abs(ang_vel2 - ang_vel).std())

    # 5.1.3 With Lie exponential
    # https://math.stackexchange.com/questions/2282938/converting-from-quaternion-to-angular-velocity-then-back-to-quaternion
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

    syn["state"] = np.concatenate([trans, quat, dtrans, ang_vel], axis=1)
    del syn["pose"]

    syn["state"][:, 3:7] = syn["state"][:, 3:7] / np.linalg.norm(syn["state"][:, 3:7], axis=1).reshape((-1, 1))

    save_path = intermediate_flight_path + "depth_data.pkl"
    pathlib.Path(intermediate_flight_path).mkdir(parents=True, exist_ok=True)
    pickle.dump(syn, open(save_path, "wb"))


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
    def __init__(self, train_validation_test: str, traj_len: int, dtype, for_loader: bool):
        """
        Create a torch Dataset to be used in training with Dataloader.
        :param train_validation_test: train, validation or test to load the appropriate dataset
        :param traj_len: length of the trajectory to give each index
        :param dtype: Datatype of the model.
        :param for_loader: If the Dataset will be used with a Dataloader or not.
        """
        assert dtype in [torch.float32, torch.float64]
        self.for_loader = for_loader
        self.dtype = dtype
        if train_validation_test not in ["train", "validation", "test"]:
            raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
        split = [e.split(",") for e in open(INTERMEDIATE_DATASET_PATH + f"split.txt", "r")]
        split = {e[0]: (int(e[1]), int(e[2]), int(e[3])) for e in split}
        self.flight_list = list(split.keys())

        loaded_data = {flight: pickle.load(open(INTERMEDIATE_DATASET_PATH + flight + train_validation_test + ".pkl",
                                                "rb")) for flight in self.flight_list}
        self.flight_length = {flight: len(loaded_data[flight]["t"]) for flight in self.flight_list}
        minimum_length = np.min([[split[flight][0], split[flight][1] - split[flight][0],
                                  split[flight][2] - split[flight][1]] for flight in self.flight_list])

        assert isinstance(traj_len, int)
        assert traj_len < minimum_length, f"horizon too big, maximum possible horizon is {minimum_length - 1}"
        self.traj_len = traj_len

        self.cum_flight_idx = np.cumsum([self.flight_length[name] for name in self.flight_list])
        self.cum_flight_idx = np.concatenate([np.array([0]), self.cum_flight_idx])
        self.cum_dataset_idx = np.cumsum([self.flight_length[name] - self.traj_len for name in self.flight_list])
        self.cum_dataset_idx = np.concatenate([np.array([0]), self.cum_dataset_idx])

        # Loading data
        for key, val in loaded_data.items():
            loaded_data[key] = np.concatenate([val["t"].reshape((-1, 1)), val["state"], val["motor"], val["residual"]],
                                              axis=1)

        self.data = np.concatenate([loaded_data[flight] for flight in self.flight_list], axis=0)
        self.data = torch.from_numpy(self.data).type(self.dtype)

        # print(np.min(self.data[:, 20:], axis=0))
        # print(np.max(self.data[:, 20:], axis=0))

        # self.min_res = torch.tensor([-2.56919551, -1.6980865, -2.45562077,
        #                              -14.32858562, -11.89167309, -4.79519939]).view(1, 6)
        # self.max_res = torch.tensor([4.07108498, 2.69354105, 2.62223434,
        #                              13.02334404, 15.27588463, 6.88414383]).view(1, 6)

        if self.for_loader:
            self.get_func = lambda x: {"t": x[:, :1],
                                       "state": x[:, 1:14],
                                       "motor": x[:, 14:20],
                                       "residual": x[:, 20:]}
        else:
            self.get_func = lambda x: {"t": x[:, :1].view((1, -1, 1)),
                                       "state": x[:, 1:14].view((1, -1, 13)),
                                       "motor": x[:, 14:20].view((1, -1, 6)),
                                       "residual": x[:, 20:].view((1, -1, 6))}

    def __len__(self):
        return self.cum_dataset_idx[-1]

    def __getitem__(self, idx):
        """
        One sample is time:1, state:13, motor:6, residual:6, not implemented (imu_ang_acc:6)

        """
        flight_idx = np.where(idx <= self.cum_dataset_idx[1:])[0][0]
        local_idx = idx - self.cum_dataset_idx[flight_idx] + self.cum_flight_idx[flight_idx]
        start = local_idx
        stop = local_idx + self.traj_len
        data = self.data[start:stop]
        assert len(data) == self.traj_len
        return self.get_func(data)


def load_flight(flight):
    #     data = {"t", "imu_ang_acc", "motor", "state"}
    intermediate_flight_path = INTERMEDIATE_DATASET_PATH + f"{flight}/data.pkl"
    data = pickle.load(open(intermediate_flight_path, "rb"))
    return data


def main():
    for flight in FLIGHT_LIST:
        # flight = FLIGHT_LIST[2]
        raw_flight_path = RAW_DATASET_PATH + flight
        intermediate_flight_path = INTERMEDIATE_DATASET_PATH + flight
        print("Loading", flight)
        _load_clean_data(raw_flight_path, intermediate_flight_path)
    # splitting()

    # data = EurocMAVDatasetHorizon(train_validation_test="test", traj_len=10, dtype=torch.float32, for_loader=False)
    # # dict_keys(['t', 'state', 'motor', 'residual'])
    # test = data[0]
    #
    # data = EurocMAVDatasetHorizon(train_validation_test="test", traj_len=10, dtype=torch.float32, for_loader=True)
    # dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)
    # batch = next(iter(dataloader))


if __name__ == '__main__':
    main()
