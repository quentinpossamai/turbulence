import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import src.data.euroc_mav.loading as euroc

plt.style.use("seaborn")

# parser = argparse.ArgumentParser()
# parser.add_argument('--path_dataset', default='/Volumes/Samsung_T5/BlackBird/blackbird/intermediate', type=str)
# args = parser.parse_args()
#
# PATH = args.path_dataset
# FLIGHT_LIST = []
# for flight in os.listdir(PATH):
#     for orientation in os.listdir(PATH + "/" + flight):
#         for speed in os.listdir(PATH + "/" + flight + "/" + orientation):
#             FLIGHT_LIST.append(flight + "/" + orientation + "/" + speed + "/data.pkl")


def load_flight(flight_name):
    data = pickle.load(open(f"{PATH}/{flight_name}", 'rb'))
    return data


def integrate(state, imu, dt):
    position = state[:3]
    quaternion = state[3:7]
    speed = state[7:10]
    rpy = state[10:]

    sensor_rpy = imu[:3]
    sensor_acc = imu[3:]

    qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    G = np.array([[qw, qz, -qy, -qx],
                  [-qz, qw, qx, -qy],
                  [qy, -qx, qw, -qz]]).transpose()
    dot_q = (0.5 * G @ sensor_rpy.reshape((3, 1))).reshape((4))

    next_quaternion = quaternion + dt * dot_q
    next_quaternion = next_quaternion / np.linalg.norm(next_quaternion)

    r = Rotation.from_quat(next_quaternion)
    sensor_acc = r.apply(sensor_acc) + np.array([0, 0, -9.81])

    next_speed = speed + dt * sensor_acc
    next_position = position + dt * next_speed

    next_state = np.concatenate([next_position, next_quaternion, next_speed, sensor_rpy, sensor_acc])
    return next_state


def orientation_check(quaternion_gt, pred):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    def get_body_frame_point(pos, quat):
        width = 1
        d1 = np.array([1, 0, 0])
        d2 = np.array([0, 1, 0])
        d3 = np.array([0, 0, 1])

        rot = Rotation.from_quat(quat).as_matrix()
        return pos + rot @ d1 * width, pos + rot @ d2 * width, pos + rot @ d3 * width

    for t in range(0, quaternion_gt.shape[0], 100):
        ax.clear()
        ax.set_title(str(t))
        r_gt = Rotation.from_quat(quaternion_gt[t])
        p1, p2, p3 = get_body_frame_point(np.zeros((3)), quaternion_gt[t])

        ax.plot([0, p1[0]], [0, p1[1]], [0, p1[2]], c='red')
        ax.plot([0, p2[0]], [0, p2[1]], [0, p2[2]], c='green')
        ax.plot([0, p3[0]], [0, p3[1]], [0, p3[2]], c='blue')

        p1, p2, p3 = get_body_frame_point(np.zeros((3)), pred[t])

        ax.plot([0, p1[0]], [0, p1[1]], [0, p1[2]], c='magenta')
        ax.plot([0, p2[0]], [0, p2[1]], [0, p2[2]], c='yellow')
        ax.plot([0, p3[0]], [0, p3[1]], [0, p3[2]], c='magenta')

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        plt.pause(0.0001)


def main():
    # data = load_flight(FLIGHT_LIST[k])
    data = euroc.load_flight(euroc.FLIGHT_LIST[0])

    START = 10_000
    LENGTH = 5_000
    time = data["t"][START:START + LENGTH]
    dt = (time[1:] - time[:-1]).mean()
    state = data["state"][START:START + LENGTH]
    imu = data["imu_ang_acc"][START:START + LENGTH]

    pi_rotation = Rotation.from_euler("zyx", np.array([np.pi / 2, 0, np.pi]), degrees=False)
    imu_rpy = pi_rotation.apply(imu[:, :3])
    # imu_rpy = imu_rpy[:, [1, 0, 2]] * np.array([1, -1, 1])
    imu_acc = pi_rotation.apply(imu[:, 3:])
    # imu_acc = imu_acc[:, [1, 0, 2]] * np.array([1, -1, 1])
    imu = np.concatenate([imu_rpy, imu_acc], axis=-1)

    acc = np.concatenate([np.zeros((1, 3)), (state[1:, 7:10] - state[:-1, 7:10]) / dt], axis=0)
    state = np.concatenate([state, acc], axis=-1)
    integration = [state[0]]
    for k in range(time.shape[0] - 1):
        integration.append(integrate(integration[-1], imu[k], dt))

    integration = np.stack(integration, axis=0)
    # orientation_check(state[:, 3:7], integration)

    fig, ax = plt.subplots(4, 4)
    label = ["X", "Y", "Z"]
    for k in range(3):
        ax[0][k].plot(time, state[:, k], label="gt")
        ax[0][k].plot(time, integration[:, k], label="pred")
        ax[0][k].set_title(label[k])
        ax[0][k].legend()

    label = ["Qx", "Qy", "Qz", "Qw"]
    for k in range(4):
        ax[1][k].plot(time, state[:, 3 + k], label="gt")
        ax[1][k].plot(time, integration[:, 3 + k], label="pred")
        ax[1][k].set_title(label[k])
        ax[1][k].legend()

    label = ["Vx", "Vy", "Vz"]
    for k in range(3):
        ax[2][k].plot(time, state[:, k + 7], label="gt")
        ax[2][k].plot(time, integration[:, k + 7], label="pred")
        ax[2][k].set_title(label[k])
        ax[2][k].legend()

    label = ["Wx", "Wy", "Wz"]
    for k in range(3):
        ax[3][k].plot(time, state[:, k + 10], label="gt")
        ax[3][k].plot(time, integration[:, k + 10], label="pred")
        ax[3][k].set_title(label[k])
        ax[3][k].legend()
    fig.tight_layout()
    plt.show()

    # plt.plot(integration)
    plt.show()


if __name__ == '__main__':
    main()
