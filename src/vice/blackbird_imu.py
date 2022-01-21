import csv
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import scipy.signal as signal
from tqdm import tqdm

import src.data.blackbird.loading as blackbird
from src.utils.quaternion import qfix, qmul_np, qrot_np

plt.style.use("seaborn")
PATH = blackbird.VICE_PATH


def get_states():
    def numpyfy(d):
        for k in d:
            d[k] = np.array(d[k])
        return d

    state = {"time": [], "position": [], "quaternion": []}
    with open(PATH + "blackbird_slash_state.csv", 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(tqdm(reader, desc="State")):
            if i > 0:
                assert row[0] == row[4] + row[5].rjust(9, '0')  # Make sure secs+nsec = rosbag timestamp
                position = np.array([float(x) for x in row[9:12]])
                quaternion = np.array([float(x) for x in row[13:]])
                quaternion = quaternion / np.sqrt(np.sum(quaternion ** 2))

                state["time"].append(float(row[0]))
                state["position"].append(position)
                state["quaternion"].append(quaternion)
    state = numpyfy(state)
    dt = np.mean(state['time'][1:] - state['time'][:-1]) * 1e-9

    sos = signal.butter(N=2, Wn=1, btype='lowpass', fs=1 / dt, analog=False, output='sos')
    state["quaternion"] = signal.sosfiltfilt(sos, qfix(state["quaternion"]), axis=0)
    state["position"] = signal.sosfiltfilt(sos, state["position"], axis=0)

    print(f"State : {state['time'].shape[0]} timestep, dt = {dt} ")

    return state


def get_imu():
    def numpyfy(d):
        for k in d:
            d[k] = np.array(d[k])
        return d

    imu = {"time": [], "acceleration": [], "rpy": []}

    rotation = Rotation.from_euler("zyx", [-np.pi / 2, 0, 0])
    with open(PATH + "blackbird_slash_imu.csv", 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(tqdm(reader, desc="State")):
            if i > 0:
                assert row[0] == row[4] + row[5].rjust(9, '0')  # Make sure secs+nsec = rosbag timestamp
                acceleration = np.array([float(x) for x in row[19:22]])
                rpy = np.array([float(x) for x in row[14:17]])

                acceleration = rotation.apply(acceleration) * np.array([-1, -1, 1])
                rpy = rotation.apply(rpy) * np.array([-1, -1, 1])

                imu["time"].append(int(row[0]))
                imu["acceleration"].append(acceleration)
                imu["rpy"].append(rpy)
    imu = numpyfy(imu)
    dt = np.mean(imu['time'][1:] - imu['time'][:-1]) * 1e-9
    sos = signal.butter(N=2, Wn=2.92, btype='lowpass', fs=1 / dt, analog=False, output='sos')
    imu["acceleration"] = signal.sosfiltfilt(sos, imu["acceleration"], axis=0)
    imu["rpy"] = signal.sosfiltfilt(sos, imu["rpy"], axis=0)

    print(f"IMU : {imu['time'].shape[0]} timestep, dt = {dt} ")

    return imu


def compute_derivatives(state):
    time, position, quaternion = state["time"], state["position"], state["quaternion"]
    dt = np.mean(time[1:] - time[:-1]) * 1e-9

    speed = np.concatenate([np.zeros((1, 3)), (position[1:] - position[:-1]) / dt], axis=0)
    acceleration = np.concatenate([np.zeros((1, 3)), (speed[1:] - speed[:-1]) / dt], axis=0)

    state["speed"] = speed
    state["acceleration"] = acceleration

    dquat = (quaternion[1:] - quaternion[:-1]) / dt  # Euler derivative
    dquat = np.concatenate([np.zeros((1, 4)), dquat], axis=0)

    # 5.1.2-Quaternion to angular velocity
    # 5.1.2.1-Quaternion multiplication method
    angular_velocity = 2. * qmul_np(quaternion * np.array([[-1., -1., -1., 1.]]), dquat)[:, :3]

    state['rpy'] = angular_velocity


def synchronize(state, imu):
    # Synchronization
    vice_path = blackbird.VICE_PATH
    rgb_time = np.loadtxt(f'{vice_path}rgb/nSecTimestamps.txt', dtype=int)
    start, stop = 15556 // (4 * 60 + 19) * (2 * 60 + 40), 15556 // (4 * 60 + 19) * (3 * 60)
    rgb_time = rgb_time[start:stop]

    # imu_ind = np.zeros(len(rgb_time))
    imu_acc = np.zeros((len(rgb_time), 3))
    imu_rpy = np.zeros((len(rgb_time), 3))
    for k, img_timestamp in enumerate(rgb_time):
        imu_ind = np.abs(img_timestamp - imu['time']).argmin()
        imu_acc[k] = imu['acceleration'][imu_ind]
        imu_rpy[k] = imu['rpy'][imu_ind]
    imu["time"] = rgb_time
    imu['acceleration'] = imu_acc
    imu['rpy'] = imu_rpy

    time_state = state["time"]
    time_imu = imu["time"]
    for item in ["position", "quaternion", "rpy", "speed", "acceleration"]:
        data = []
        for i in range(state[item].shape[-1]):
            data.append(interp1d(time_state, state[item][:, i], bounds_error=False)(time_imu))
        state[item] = np.stack(data, axis=-1)
    state["time"] = time_imu


def integrate(p, q, v, imu_acc, imu_rpy, dt):
    w_quat = np.array([*imu_rpy, 0])
    qdot = 0.5 * qmul_np(q, w_quat)
    next_q = q + dt * qdot
    next_w = imu_rpy

    acc = qrot_np(q.copy(), imu_acc.copy()) + np.array([0, 0, 9.81])

    next_v = v + dt * acc
    next_p = p + dt * v

    return next_p, next_q, next_v, next_w, acc


if __name__ == '__main__':
    state = get_states()
    imu = get_imu()
    compute_derivatives(state)
    synchronize(state, imu)

    dt = np.mean(imu["time"][1:] - imu["time"][:-1]) * 1e-9

    integrated_pos, integrated_quat, integrated_speed, integrated_rpy = [state["position"][0]], [
        state["quaternion"][0]], [state["speed"][0]], [state["rpy"][0]]
    integrated_acc = []

    for t in range(imu["time"].shape[0]):
        next_p, next_q, next_v, next_w, acc = integrate(integrated_pos[-1], integrated_quat[-1],
                                                        integrated_speed[-1],
                                                        imu["acceleration"][t], imu["rpy"][t], dt)

        integrated_pos.append(next_p)
        integrated_quat.append(next_q)
        integrated_speed.append(next_v)
        integrated_rpy.append(next_w)
        integrated_acc.append(acc)

    integrated_pos = np.stack(integrated_pos, axis=0)[1:]
    integrated_quat = np.stack(integrated_quat, axis=0)[1:]
    integrated_speed = np.stack(integrated_speed, axis=0)[1:]
    integrated_rpy = np.stack(integrated_rpy, axis=0)[1:]
    integrated_acc = np.stack(integrated_acc, axis=0)

    imu['integrated_pos'] = integrated_pos
    imu['integrated_quat'] = integrated_quat
    imu['integrated_speed'] = integrated_speed
    imu['integrated_rpy'] = integrated_rpy
    imu['integrated_acc'] = integrated_acc
    imu['corrected_speed'] = state["speed"]
    imu['corrected_rpy'] = state["rpy"]
    pickle.dump(imu, open(f'{blackbird.VICE_PATH}imu.pkl', 'wb'))

    fig, ax = plt.subplots(3, 2)
    for i in range(3):
        ax[i][0].plot(state["time"], integrated_pos[:, i], label="GT")
        ax[i][1].plot(state["time"], state["quaternion"][:, i], label="GT")

        ax[i][0].plot(imu["time"], state['position'][:, i], label="IMU")
        ax[i][1].plot(imu["time"], integrated_quat[:, i], label="IMU")

        ax[i][0].legend()
        ax[i][1].legend()
    plt.show()
