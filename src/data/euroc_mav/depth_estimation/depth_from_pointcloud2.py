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
NUM_PROCESS = 16
INTERMEDIATE_DATASET_PATH = "/Volumes/Samsung_T5/EurocMAV/"


def get_camera_view_pointcloud(transform, xyz):
    overall_transform = np.linalg.inv(TF_BODY_CAM0).dot(np.linalg.inv(transform))
    transformed_xyz = xyz.dot(overall_transform.transpose())
    x, y, z, _ = _split(transformed_xyz)
    u, v = _project_and_distort(x, y, z)
    valid_mask = np.logical_and.reduce(
        (z > 0.0, u > -0.05 * W, u < W * 1.05, v > -0.05 * H, v < H * 1.05),
        axis=0)
    valid_points = valid_mask.nonzero()[0]
    return transformed_xyz[valid_points, :3]


def get_occluded_points(xyz, neighborhood_radius, z_threshold):
    def get_bin(xz, yz):
        return int(round(xz / neighborhood_radius)), int(round(yz / neighborhood_radius))

    xs, ys, zs = _split(xyz)
    xzs = xs / zs
    yzs = ys / zs
    grid = collections.defaultdict(lambda: np.inf)
    for ind in range(xyz.shape[0]):
        xbin, ybin = get_bin(xzs[ind], yzs[ind])
        for i in range(-1, 2):
            for j in range(-1, 2):
                grid[(xbin + i, ybin + j)] = min(grid[(xbin + i, ybin + j)], zs[ind])

    occluded_indices = []
    for ind in range(xyz.shape[0]):
        zmin = grid[get_bin(xzs[ind], yzs[ind])]
        if zmin < (1 - z_threshold) * zs[ind]:
            occluded_indices.append(ind)
    occluded_inds = set(occluded_indices)
    visible_indices = [
        i for i in range(xyz.shape[0]) if i not in occluded_inds
    ]
    return visible_indices


def render_z(xyz):
    x, y, z = _split(xyz)
    u, v = _project_and_distort(x, y, z)
    u = np.floor(u).astype(int)
    v = np.floor(v).astype(int)
    rendered_z = np.full((int(H), int(W)), -np.inf)
    rendered_z[v, u] = z
    rendered_z = np.where(rendered_z == -np.inf, np.nan, rendered_z)
    return rendered_z


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


def read_ply(filename):
    with open(filename) as f:
        lines = f.readlines()
    lines = lines[11:]
    xyz = []
    for line in lines:
        tokens = line.split(' ')
        xyz.append([float(t) for t in tokens[:3]])

    return np.array(xyz)


def filter_out_ot_frame_points(xyz):
    x, y, z = _split(xyz)
    u, v = _project_and_distort(x, y, z)
    u = np.floor(u).astype(int)
    v = np.floor(v).astype(int)
    valid_mask = np.logical_and.reduce((u >= 0, u < W, v >= 0, v < H), axis=0)
    valid_points = valid_mask.nonzero()[0]
    return valid_points


def gen_img(pipe, transform, xyz):
    # Move the point cloud to fit the camera position
    xyz_view = get_camera_view_pointcloud(transform, xyz)
    # Remove occluded points (point behind a wall)
    visible_indices = get_occluded_points(xyz_view, 0.02, 0.08)
    xyz_view = xyz_view[visible_indices, :]

    # Remove out of screen points
    visible_indices = filter_out_ot_frame_points(xyz_view)
    xyz_view = xyz_view[visible_indices, :]

    # Compute depth + fancy stuff on the normal
    z_image = render_z(xyz_view)

    # Fill hole in images with linear interp
    valid_mask = (1 - np.isnan(z_image)).astype(bool)
    coords = np.array(np.nonzero(valid_mask)).T

    values_z = z_image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values_z, fill_value=np.mean(values_z))
    filled_z = it(list(np.ndindex(z_image.shape))).reshape(z_image.shape)

    pipe.send((filled_z))
    pipe.close()


def main():
    gti = GroundTruthInterpolator()
    xyz = np.load(INTERMEDIATE_DATASET_PATH + "depth_estimation/pointcloud.npy")
    xyz_homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

    # Create the output directories
    os.makedirs("/Users/steevenjanny/Desktop/depth/", exist_ok=True)

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
        imgs_z = []

        for pr in range(NUM_PROCESS):  # Create pipes for multiprocessing
            parentPipe, childPipe = mp.Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(len(transforms)):  # Run the processes, ie compute normals and depths
            p = mp.Process(target=gen_img,
                           args=(childPipes[rank], transforms[rank], xyz_homogeneous))
            p.start()
            processes.append(p)

        for rank, p in enumerate(processes):  # Get the imgs from pipes
            z = parentPipes[rank].recv()
            imgs_z.append(z)
            p.join()

        # Save images, use png for time/space saving
        for i in range(len(imgs_z)):
            os.makedirs(f"/Users/steevenjanny/Desktop/depth/{flights[i]}", exist_ok=True)
            np.save(f"/Users/steevenjanny/Desktop/depth/{flights[i]}/{str(float(timestamps[i])).replace('.', '_')}.npy",
                    imgs_z[i])


def _split(matrix):
    return [
        np.squeeze(v, axis=1) for v in np.split(matrix, matrix.shape[1], axis=1)
    ]


def _project_and_distort(x, y, z):
    """Apply perspective projection and distortion on a point cloud.
    Args:
      x: A vector containing the x coordinates of the points.
      y: A vector containing the y coordinates of the points, same length as x.
      z: A vector containing the z coordinates of the points, same length as x.
    Returns:
      A tuple of two vectors of the same length as x, containing the image-plane
      coordinates (u, v) of the point cloud.
    """
    xz = (x / z)
    yz = (y / z)
    # 2. Apply radial camera distortion:
    rr = xz ** 2 + yz ** 2
    distortion = (1 + K1 * rr + K2 * rr * rr)
    xz *= distortion
    yz *= distortion
    # 3. Apply intrinsic matrix to get image coordinates:
    u = FX * xz + X0
    v = FY * yz + Y0
    return u, v


if __name__ == '__main__':
    main()
