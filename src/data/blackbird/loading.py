import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.signal as signal
from src.utils.transform import Transform
from scipy.fft import fft, fftfreq, ifft
import csv
from src.utils import quaternion
from scipy.spatial.transform import Rotation
from scipy import interpolate
import pickle

from src.utils.paths import WD_PATH
# plt.style.use('seaborn')

DATASET_PATH = "/Volumes/Samsung_T5/BlackBird/blackbird/raw/"
VICE_PATH = f'{WD_PATH}data/blackbird/raw/patrick/yawConstant/maxSpeed3p0/csv/'


def get_flight_list():
    flight_list = []
    for trajectory in os.listdir(DATASET_PATH):
        for orientation in os.listdir(DATASET_PATH + trajectory):
            for speed in os.listdir(DATASET_PATH + trajectory + "/" + orientation):
                flight_list.append(f"{DATASET_PATH}{trajectory}/{orientation}/{speed}/")
    return flight_list


def load_from_csv(flight):
    name = "blackbird_slash_imu"
    raw = {'imu': {}, "motor": {}, 'state': {}}
    with open(flight + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)
        data = [e for e in csv_reader]
        time = np.array([float(e[0]) * 1e-9 for e in data])
        imu_angular_velocity = np.array([np.array([float(e[14]), float(e[15]), float(e[16])]) for e in data])
        imu_linear_acceleration = np.array([np.array([float(e[19]), float(e[20]), float(e[21])]) for e in data])
        raw["imu"] = {"t": time, "angular_velocity": imu_angular_velocity,
                      "linear_acceleration": imu_linear_acceleration}

    name = "blackbird_slash_rotor_rpm"
    with open(flight + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)
        data = [e for e in csv_reader]
        time = np.array([float(e[0]) * 1e-9 for e in data])
        time0 = np.array([float(e[8]) + float(e[9]) * 1e-9 for e in data])
        time1 = np.array([float(e[10]) + float(e[11]) * 1e-9 for e in data])
        time2 = np.array([float(e[12]) + float(e[13]) * 1e-9 for e in data])
        time3 = np.array([float(e[14]) + float(e[15]) * 1e-9 for e in data])
        motors = np.array([np.array(e[16].strip("[]").split(","), dtype=float) for e in data]) * np.pi / 30
        raw["motor"] = {"m0": motors[:, 0], 'time0': time0,
                        "m1": motors[:, 1], 'time1': time1,
                        "m2": motors[:, 2], 'time2': time2,
                        "m3": motors[:, 3], 'time3': time3,
                        't': time}

    name = "blackbird_slash_state"
    with open(flight + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)
        data = [e for e in csv_reader]
        time = np.array([float(e[0]) * 1e-9 for e in data])
        trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
        quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
        quat = Rotation.from_quat(quat).as_quat()
        raw["state"] = {"time": time, "translation": trans, "rotation": quat}
    return raw


def flatten_timestamp(time):
    """Calcule le dt moyen, puis génère une array uniforme avec ce dt"""
    dt = np.mean(time[1:] - time[:-1])
    return dt * np.arange(0, time.shape[0]) + time[0]


def process_raw_data(imu_dict, pose_dict, motors_dict, step=18, order=2, wn=5, mode="butter"):
    def low_pass_filter(x, dt):  # Fourier low pass filter
        frequencies = fftfreq(len(time), d=float(dt))
        fc_quat = wn  # Hz
        fourier_quat = fft(x, axis=0)
        output = fourier_quat.copy()
        output[np.abs(frequencies) > fc_quat] = 0
        x = np.real(ifft(output, axis=0))
        return x

    def low_pass_butter(x, dt):
        sos = signal.butter(N=order, Wn=wn, btype='lowpass', fs=1 / dt, analog=False, output='sos')
        return signal.sosfiltfilt(sos, x, axis=0)

    # FLATTENING TIME
    imu_dict['t'] = flatten_timestamp(imu_dict['t'])
    pose_dict['t'] = flatten_timestamp(pose_dict['t'])
    motors_dict['t'] = flatten_timestamp(motors_dict['t'])
    for i in range(4):
        motors_dict['time' + str(i)] = flatten_timestamp(motors_dict['time' + str(i)])

    # STATE PROCESSING
    quat = pose_dict["pose"][::step, 3:7]
    trans = pose_dict["pose"][::step, :3]
    time = pose_dict["t"][::step]
    dt = np.mean(time[1:] - time[:-1])

    quat = quat / np.linalg.norm(quat, axis=1).reshape((-1, 1))  # Normalize quaternion
    quat = quaternion.qfix(quat)  # Fix quaternion swap in continuity
    quat = low_pass_butter(quat, dt) if mode == "butter" else low_pass_filter(quat, dt)
    quat = quat / np.linalg.norm(quat, axis=1).reshape((-1, 1))  # Normalize quaternions
    trans = low_pass_butter(trans, dt) if mode == "butter" else low_pass_filter(trans, dt)

    # 5-COMPUTE DERIVATIVE
    dquat = (quat[1:] - quat[:-1]) / dt  # Euler derivative
    dquat = np.concatenate([np.zeros((1, 4)), dquat], axis=0)
    ang_vel = 2. * quaternion.qmul_np(quat * np.array([[-1., -1., -1., 1.]]), dquat)[:, :3]

    dtrans = (trans[1:] - trans[:-1]) / dt  # Euler derivative
    dtrans = np.concatenate([np.zeros((1, 3)), dtrans], axis=0)

    state = np.concatenate([trans, quat, dtrans, ang_vel], axis=1)

    # 4-SYNCHRONISATION
    imu_sync = np.stack([interpolate.interp1d(imu_dict["t"], imu_dict["ang_acc"][:, i], bounds_error=False,
                                              fill_value=(imu_dict["ang_acc"][0, i], imu_dict["ang_acc"][-1, i]))(time)
                         for i in range(6)], axis=1)
    motor_sync = np.stack([interpolate.interp1d(motors_dict['t'], motors_dict[f"m{i}"],
                                                bounds_error=False,
                                                fill_value=(motors_dict[f"m{i}"][0], motors_dict[f"m{i}"][-1]))(time)
                           for i in range(4)], axis=1)

    motor_sync = low_pass_butter(motor_sync, dt)
    imu_sync = low_pass_butter(imu_sync, dt)

    synchronisation = {"t": time, "imu_ang_acc": imu_sync, "motor": motor_sync, "state": state}

    # Crop de 5 secondes
    # fig = plt.figure()
    #
    # axs = [fig.add_subplot(2, 2, i) for i in range(1, 5)]
    # for k in range(4):
    #     axs[k].plot(synchronisation['t'], synchronisation["motor"][:, k])
    # plt.tight_layout()
    # plt.show()

    # CUT
    motors = synchronisation["motor"]
    start = max([np.nonzero(motors[:, i] > 1000)[0][0] for i in range(4)])
    stop = min([np.nonzero(motors[:, i] > 1000)[0][-1] for i in range(4)])

    cut_data = {}
    for key in synchronisation:
        cut_data[key] = synchronisation[key][start:stop]

    # Setting initial parameters
    cut_data["t"] = cut_data["t"] - cut_data["t"][0]

    return cut_data


def load_clean_data(flight):
    print("Loading", flight)
    # 1-LOADING DATA
    raw = load_from_csv(flight)

    # 1.1-IMU Loading
    ang_vel = np.array(raw["imu"]["angular_velocity"])
    lin_acc = np.array(raw["imu"]["linear_acceleration"])
    time = np.array(raw["imu"]["t"])
    imu_dict = {"t": time, "ang_acc": np.concatenate([ang_vel, lin_acc], axis=1)}

    # 1.2-MOTOR Loading
    motor_dict = raw['motor']

    # 1.3-STATE Loading
    time = np.array(raw["state"]["time"])
    trans = np.array(raw["state"]["translation"])
    quat = np.array(raw["state"]["rotation"])
    pose = Transform.from_quat_trans(quat, trans).as_matrix()
    pi_rotation = Transform.from_euler_trans("zyx", np.array([0, 0, np.pi]),
                                             degrees=False, trans=np.array([0, 0, 0])).as_matrix()
    pose = pi_rotation @ pose @ pi_rotation
    quat = Transform.from_matrix(pose).as_rot_quat()
    trans = Transform.from_matrix(pose).as_trans()
    pose_dict = {"t": time, "pose": np.concatenate([trans, quat], axis=1)}

    cleaned_data_butter = process_raw_data(imu_dict, pose_dict, motor_dict, mode="butter", wn=5, order=2,
                                           step=18)
    dir = flight.replace("raw", "intermediate_V2")
    os.makedirs(dir, exist_ok=True)
    pickle.dump(cleaned_data_butter, open(dir + "/data.pkl", 'wb'))


def main():
    flight_list = get_flight_list()
    for flight in flight_list:
        load_clean_data(flight)


if __name__ == '__main__':
    main()
