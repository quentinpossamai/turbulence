# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Renders depth maps from EuRoC MAV point clouds.
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
Some of the rooms of the EuRoC MAV dataset have point clouds. This script
renders depth maps from the point clouds.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os

from absl import app
from absl import flags

from scipy import interpolate
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import open3d as o3d

flags.DEFINE_string('room_path', '/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy',
                    'Path to the EuRoC data for one of the rooms ')

flags.DEFINE_string('output_path', '/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy/',
                    'Path where to store the outputs.')

FLAGS = flags.FLAGS

# A 4D transform that connects the Cam0 to the body of the MAV. This is taken
# from the sensor.yaml file. To project the point cloud on Cam1, please replace
# with the respective extrinsic matrix. This is constant across all the rooms in
# the dataset.
CAM0_TO_BODY = np.array(
    [[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
     [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
     [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
     [0.0, 0.0, 0.0, 1.0]])

# Intrinsics of Cam0. This is taken from cam0/sensor.yaml and is the same for
# all rooms.
FX = 458.654
FY = 457.296
X0 = 367.215
Y0 = 248.375
K1 = -0.28340811
K2 = 0.07395907
H = 480
W = 752


def get_camera_view_pointcloud(transform, xyz):
    """Transform point cloud to camera view, prune points outside of the view.
    Args:
      transform: 4x4 transform matrix representing position and orientation of
        the body of the MAV.
      xyz: A 4xN matrix, point cloud in homogeneous coordinates. The k-th column
        is (x, y, z, 1), where x, y, z are the coordinates of the k-th point.
      greyscale_color: N-vector, vertex grayscale value. The k-th entry is the
        greyscale color of the k-th point.
    Returns:
      3xM (M < N) matrix representing the point cloud in the camera view.
      M vector, vertex grayscale value.
      Only points that fall within the camera viweing angle and are in front of
      the camera are kept.
    """

    overall_transform = np.linalg.inv(CAM0_TO_BODY).dot(np.linalg.inv(transform))
    transformed_xyz = xyz.dot(overall_transform.transpose())
    x, y, z, _ = _split(transformed_xyz)
    u, v = _project_and_distort(x, y, z)
    # Remove points that are out of frame. Keep some margin (1.05), to make sure
    # occlusions are addressed correctly at the edges of the field of view. For
    # example a point that is just slightly out of frame can occlude a neighboring
    # point inside the frame.
    valid_mask = np.logical_and.reduce(
        (z > 0.0, u > -0.05 * W, u < W * 1.05, v > -0.05 * H, v < H * 1.05),
        axis=0)
    valid_points = valid_mask.nonzero()[0]
    return transformed_xyz[valid_points, :3]


def get_occluded_points(xyz, neighborhood_radius, z_threshold):
    """Remove points that are occluded by others from a camera-view point cloud.
    Args:
      xyz: A 3xN matrix representing the point cloud in the camera view.
      neighborhood_radius: The radius around each point in which it occludes
        others.
      z_threshold: Minimum z distance betweem two points for them considered to
        be occluding each other. If two points are verty close in z, they likely
        belong to the same surface and thus do not occlude each other.
    Returns:
      A list of indices in xyz corresponding to points that are occluded.
    """

    def get_bin(xz, yz):
        xbin = int(round(xz / neighborhood_radius))
        ybin = int(round(yz / neighborhood_radius))
        return xbin, ybin

    xs, ys, zs = _split(xyz)
    xzs = xs / zs
    yzs = ys / zs
    grid = collections.defaultdict(lambda: np.inf)
    for ind in range(xyz.shape[0]):
        # Place each point in the bin where it belongs, and in the neighboring bins.
        # Keep only the closest point to the camera in each bin.
        xbin, ybin = get_bin(xzs[ind], yzs[ind])
        for i in range(-1, 2):
            for j in range(-1, 2):
                grid[(xbin + i, ybin + j)] = min(grid[(xbin + i, ybin + j)], zs[ind])

    occluded_indices = []
    for ind in range(xyz.shape[0]):
        # Loop over all points and see if they are occluded, by finding the closest
        # point to the camera within the same bin and testing for the occlusion
        # condition. A point is occluded if there is another point in the same bin
        # that is far enough in z, so that it cannot belong to the same surface,
        zmin = grid[get_bin(xzs[ind], yzs[ind])]
        if zmin < (1 - z_threshold) * zs[ind]:
            occluded_indices.append(ind)
    occluded_inds = set(occluded_indices)
    visible_indices = [
        i for i in range(xyz.shape[0]) if i not in occluded_inds
    ]
    return visible_indices


def render_z_and_normal(xyz, normal):
    """Given a colored cloud in camera coordinates, render a depth map.
    This function is useful for visualization / debugging.
    Args:
      xyz: A 3xN matrix representing the point cloud in the camera view.
    Returns:
      A rendered depth map.
    """
    x, y, z = _split(xyz)
    u, v = _project_and_distort(x, y, z)
    u = np.floor(0.5 * u).astype(int)
    v = np.floor(0.5 * v).astype(int)
    rendered_z = np.full((int(H / 2), int(W / 2)), -np.inf)
    rendered_n = np.full((int(H / 2), int(W / 2), 3), -np.inf)
    rendered_z[v, u] = z
    rendered_n[v, u] = normal
    rendered_z = np.where(rendered_z == -np.inf, np.nan, rendered_z)
    rendered_n = np.where(rendered_n == -np.inf, np.nan, rendered_n)
    return rendered_z, rendered_n


class GroundTruthInterpolator(object):
    """Interpolates MAV position and orientation groundtruth to a timestamp."""

    def __init__(self, filename):
        """Creates an instance.
        Args:
          filename: A string, filepath of the state_groundtruth_estimate0.csv file.
        """
        with open(filename) as f:
            lines = f.readlines()
        lines = lines[1:]  # skip the first line

        gt = []
        for l in lines:
            tokens = l.split(',')
            gt.append([float(t) for t in tokens[:8]])
        self.gt = np.array(gt)

    def get_transform(self, k):
        """Interpolates the MAV's transform matrix at a timestamp."""

        # self._gt[:, 0], the 0th column, is the timestamp. Columns 1-3 are x, y, z,
        # and columns 4-7 are quaternion components describing the rotation.
        timestamps = self.gt[:, 0]
        x = self.gt[k, 1]
        y = self.gt[k, 2]
        z = self.gt[k, 3]

        qw = self.gt[k, 4]
        qx = self.gt[k, 5]
        qy = self.gt[k, 6]
        qz = self.gt[k, 7]

        # Creates a matrix
        transform = np.array([[
            1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qz * qw,
            2 * qx * qz + 2 * qy * qw, x
        ],  # pylint: disable=bad-continuation
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw, y
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy, z
            ], [0.0, 0.0, 0.0, 1.0]])

        return self.gt[k, 0], transform


def read_ply(filename):
    """Reads a PLY file representing EuRoc's point cloud."""
    with open(filename) as f:
        lines = f.readlines()
    lines = lines[11:]
    xyz = []
    for l in lines:
        tokens = l.split(' ')
        xyz.append([float(t) for t in tokens[:3]])

    return np.array(xyz)


def filter_out_ot_frame_points(xyz):
    """Remove all points in a camera-view pointcloud that are out of frame.
    Args:
      xyz: A 3xN matrix representing the point cloud in the camera view.
      c: A N-long vector containing (greyscale) colors of the points.
    Returns:
      A 3xM matrix and a M-long vector representing the filtered colored point
      cloud.
    """
    x, y, z = _split(xyz)
    u, v = _project_and_distort(x, y, z)
    u = np.floor(u).astype(int)
    v = np.floor(v).astype(int)
    valid_mask = np.logical_and.reduce((u >= 0, u < W, v >= 0, v < H), axis=0)
    valid_points = valid_mask.nonzero()[0]
    return valid_points


def gen_img(pipe, transform, xyz):
    xyz_view = get_camera_view_pointcloud(transform, xyz)

    visible_indices = get_occluded_points(xyz_view, 0.02, 0.08)

    xyz_view = xyz_view[visible_indices, :]

    visible_indices = filter_out_ot_frame_points(xyz_view)

    xyz_view = xyz_view[visible_indices, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_view)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=1000),
                         fast_normal_computation=True)
    pcd.orient_normals_towards_camera_location(camera_location=transform[:-1, -1])
    pcd.normalize_normals()
    # o3d.visualization.draw_geometries([pcd])

    visible_normal = np.asarray(pcd.normals)

    z_image, n_image = render_z_and_normal(xyz_view, visible_normal)

    valid_mask = (1 - np.isnan(z_image)).astype(bool)
    coords = np.array(np.nonzero(valid_mask)).T

    values_z = z_image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values_z, fill_value=np.mean(values_z))
    filled_z = it(list(np.ndindex(z_image.shape))).reshape(z_image.shape)

    values_n = n_image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values_n)
    filled_n = it(list(np.ndindex(z_image.shape))).reshape(n_image.shape)
    #
    # import matplotlib.pyplot as plt
    # plt.matshow((filled_n + 1) / 2)
    # plt.show()

    pipe.send((filled_z, filled_n))
    pipe.close()


def main(argv):
    del argv  # unused
    gti = GroundTruthInterpolator(
        os.path.join(FLAGS.room_path, 'state_groundtruth_estimate0/data.csv'))
    print('Groundtruth loaded.')
    xyz = read_ply(os.path.join(FLAGS.room_path, 'pointcloud0/data.ply'))
    # xyz = np.load("pointcloud.npy")
    print('PLY loaded.')
    xyz_homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

    for k in tqdm(range(0, gti.gt.shape[0], 8)):
        timestamps, transforms = [], []
        gen_img(0, gti.get_transform(k)[1], xyz_homogeneous)
        for i in range(8):
            if k + i < gti.gt.shape[0]:
                time, trans = gti.get_transform(k + i)
                timestamps.append(time)
                transforms.append(trans)
        childPipes, parentPipes = [], []
        processes = []
        imgs_z, imgs_n = [], []
        for pr in range(8):  # Create pipes to get the simulation
            parentPipe, childPipe = mp.Pipe()
            parentPipes.append(parentPipe)
            childPipes.append(childPipe)

        for rank in range(len(transforms)):  # Run the processes
            p = mp.Process(target=gen_img,
                           args=(childPipes[rank], transforms[rank], xyz_homogeneous, normal_homogeneous))
            p.start()
            processes.append(p)

        for rank, p in enumerate(processes):  # Get the simulation
            z, n = parentPipes[rank].recv()
            imgs_z.append(z)
            imgs_n.append(n)
            p.join()

        for i in range(len(imgs_z)):
            np.save(os.path.join(FLAGS.output_path, 'depth/%d.npy' % int(timestamps[i])), imgs_z[i])
            np.save(os.path.join(FLAGS.output_path, 'normal/%d.npy' % int(timestamps[i])), imgs_n[i])


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
    app.run(main)
