"""Renders depth_estimation maps from EuRoC MAV point clouds.
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
Some of the rooms of the EuRoC MAV dataset have point clouds. This script
renders depth_estimation maps from the point clouds.

TO GET THE DEPTH MAP AND NORMAL MAP BACK :
- DEPTH MAP : Read with plt.imread(), then multiple by ZMAX
- NORMAL MAP : Read with plt.imread(), then n*2 -1"""

import collections
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
from scipy import interpolate
from tqdm import tqdm
import pickle
from src.data.euroc_mav.loading import load_flight, FLIGHT_LIST, INTERMEDIATE_DATASET_PATH, TRANS_MEAN
import src.utils.quaternion as quaternion
from src.utils.transform import Transform

# 4D Camera transformation from yaml
TF_BODY_CAM0 = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                         [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                         [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                         [0.0, 0.0, 0.0, 1.0]])

# Transformation BODY_VICON0 from yaml
TF_BODY_VICON0 = (np.array([[0.33638, -0.01749, 0.94156, 0.06901],
                            [-0.02078, -0.99972, -0.01114, -0.02781],
                            [0.94150, -0.01582, -0.33665, -0.12395],
                            [0.0, 0.0, 0.0, 1.0]]))

# Intrinsic parameters of camera
FX = 320
FY = 320
X0 = 640 // 2
Y0 = 480 // 2
K1 = 0
K2 = 0
H = 480
W = 640

# Number of parallel process
NUM_PROCESS = 1


class GroundTruthInterpolator(object):
    """Interpolates MAV position and orientation groundtruth to a timestamp."""

    def __init__(self):
        """Creates an instance.
        Args:
          filename: A string, filepath of the state_groundtruth_estimate0.csv file. Can also be the room number.
        """

        self.data = {key: pickle.load(open(INTERMEDIATE_DATASET_PATH + key + "/data.pkl", 'rb')) for key in
                     ["V1_01_easy/",
                      "V1_02_medium/",
                      "V1_03_difficult/"]}

        with open("depth_target.txt", 'r') as f:
            splits = f.readlines()
        self.splits = [(d.split(',')[0], int(d.split(',')[1])) for d in splits]

        data = []
        for (flight, t) in self.splits:
            idx = list(self.data[flight]["timestamp"]).index(t)
            time = self.data[flight]['t'][idx]
            data.append((flight, time, self.data[flight]["state"][idx]))

        self.data = data

    def get_transform(self, k):
        """Interpolates the MAV's transform matrix at a timestamp."""

        flight, timestamps, state = self.data[k]
        trans = state[:3] + TRANS_MEAN[flight].reshape((1, 3))
        tf_vicon0_origin = Transform.from_quat_trans(state[3:7],
                                                     trans).inv().as_matrix()
        tf_origin_body = Transform.from_matrix(TF_BODY_VICON0 @ tf_vicon0_origin).inv().as_matrix()
        quat, trans = Transform.from_matrix(tf_origin_body).as_quat_trans()
        quat = quaternion.qfix(quat.reshape((1, 4)))
        quat = quat / np.linalg.norm(quat, axis=1).reshape((-1, 1))
        quat = quat.reshape((4))
        x = trans[0]
        y = trans[1]
        z = trans[2]

        qw = quat[3]
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]

        # Creates a matrix
        transform = np.array([[
            1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw, x],
            [2 * qx * qy + 2 * qz * qw,
             1 - 2 * qx * qx - 2 * qz * qz,
             2 * qy * qz - 2 * qx * qw, y],
            [2 * qx * qz - 2 * qy * qw,
             2 * qy * qz + 2 * qx * qw,
             1 - 2 * qx * qx - 2 * qy * qy, z], [0.0, 0.0, 0.0, 1.0]])

        return timestamps, flight, transform


def gen_img(pipe, transform, xyz):
    # Move the point cloud to fit the camera position

    xyz = xyz[xyz[:, 2] < transform[2, -1]]
    dist = ((xyz[:, :2] - transform[:2, -1]) ** 2).sum(-1)
    closest = np.argmin(dist)
    closest_point = xyz[closest]

    z = transform[2, -1] - closest_point[-1]
    assert z > 0
    pipe.send(z)
    pipe.close()


def main():
    gti = GroundTruthInterpolator()
    xyz = np.load(INTERMEDIATE_DATASET_PATH + "depth_estimation/pointcloud.npy")

    # Create the output directories
    os.makedirs("/Users/steevenjanny/Desktop/depth/", exist_ok=True)

    data = {"V1_01_easy/": [], "V1_02_medium/": [], "V1_03_difficult/": []}
    # For each timestamp
    for k in tqdm(range(0, len(gti.splits), NUM_PROCESS)):
        timestamps, transforms, flights = [], [], []
        # Extract NUM_PROCESS position/orientation for computation
        for i in range(NUM_PROCESS):
            if k + i < len(gti.splits):
                time, flight, trans = gti.get_transform(k + i)
                timestamps.append(time)
                transforms.append(trans)
                flights.append(flight)

        childPipes, parentPipes = [], []
        processes = []
        heights = []

        for pr in range(NUM_PROCESS):  # Create pipes for multiprocessing
            parentPipe, childPipe = mp.Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(len(transforms)):  # Run the processes, ie compute normals and depths
            p = mp.Process(target=gen_img,
                           args=(childPipes[rank], transforms[rank], xyz))
            p.start()
            processes.append(p)

        for rank, p in enumerate(processes):  # Get the imgs from pipes
            z = parentPipes[rank].recv()
            heights.append(z)
            p.join()

        # Save images, use png for time/space saving
        for i in range(len(heights)):
            data[flights[i]].append([timestamps[i], heights[i]])
    for key in data.keys():
        data[key] = np.array(data[key])
    pickle.dump(data, open("/Volumes/Samsung_T5/EurocMav/depth_iros_test/height.pkl", 'wb'))


if __name__ == '__main__':
    main()
