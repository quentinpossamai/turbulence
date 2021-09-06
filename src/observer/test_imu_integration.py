import argparse
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform.rotation import Rotation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

plt.style.use("seaborn")

parser = argparse.ArgumentParser()
parser.add_argument('--path_dataset', default='/Volumes/Samsung_T5/BlackBird/blackbird/intermediate', type=str)
args = parser.parse_args()

PATH = args.path_dataset
FLIGHT_LIST = []
for flight in os.listdir(PATH):
    for orientation in os.listdir(PATH + "/" + flight):
        for speed in os.listdir(PATH + "/" + flight + "/" + orientation):
            FLIGHT_LIST.append(flight + "/" + orientation + "/" + speed + "/data.pkl")


def load_flight(flight_name):
    data = pickle.load(open(f"{PATH}/{flight_name}", 'rb'))
    return data


def integrate(position, quaternion, imu, dt):
    sensor_rpy = imu[:3]
    sensor_acc = imu[3:]

    qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    G = np.array([[qw, qz, -qy, -qx],
                  [-qz, qw, qx, -qy],
                  [qy, -qx, qw, -qz]]).transpose()
    dot_q = -(0.5 * G @ sensor_rpy.reshape((3, 1))).reshape((4))

    next_quaternion = quaternion + dt * dot_q
    next_quaternion = next_quaternion / np.linalg.norm(next_quaternion)
    r = Rotation.from_quat(next_quaternion).inv()
    # sensor_acc_frame = r.apply(sensor_acc)
    return next_quaternion


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


if __name__ == '__main__':
    data = load_flight(FLIGHT_LIST[2])

    time = data["t"]
    dt = (time[1:] - time[:-1]).mean()
    state = data["state"]
    imu = data["imu_ang_acc"]

    integration = [state[0, 3:7]]
    for k in range(time.shape[0]):
        integration.append(integrate(None, integration[-1], imu[k], dt))

    integration = np.array(integration)
    # orientation_check(state[:, 3:7], integration)
    fig, ax = plt.subplots(4, 1)
    for k in range(4):
        ax[k].plot(state[:, 3 + k], label='mocap')
        ax[k].plot(integration[:, k], label='IMU')
        ax[k].set_title("Q_" + str(k))
    fig.tight_layout()
    # plt.plot(integration)
    plt.show()
