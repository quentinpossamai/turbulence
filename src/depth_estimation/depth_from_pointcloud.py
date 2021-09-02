"""Renders depth_estimation maps from EuRoC MAV point clouds.
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
Some of the rooms of the EuRoC MAV dataset have point clouds. This script
renders depth_estimation maps from the point clouds.

TO GET THE DEPTH MAP AND NORMAL MAP BACK :
- DEPTH MAP : Read with plt.imread(), then multiple by ZMAX
- NORMAL MAP : Read with plt.imread(), then n*2 -1"""

import argparse
import collections
import csv
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import open3d as o3d
import os
from scipy import interpolate
from tqdm import tqdm

from src.utils.paths import WD_PATH
from src.data.euroc_mav.loading import load_flight_for_depth, FLIGHT_LIST, INTERMEDIATE_DATASET_PATH, RAW_DATASET_PATH
from src.utils.trajectory import trace_trajectory3d
import src.utils.quaternion as quaternion
from src.utils.transform import Transform

parser = argparse.ArgumentParser()
parser.add_argument('--room_path', default='/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy', type=str)
parser.add_argument('--output_path', default='/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy/',
                    help="Output location, where the depth/normal will be registerd", type=str)
parser.add_argument('--integrated', default=False, type=bool)
args = parser.parse_args()

# 4D Camera transformation
TF_BODY_CAM0 = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                         [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                         [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                         [0.0, 0.0, 0.0, 1.0]])
TF_BODY_VICON0 = np.array([[0.33638, -0.01749, 0.94156, 0.06901],
                           [-0.02078, -0.99972, -0.01114, -0.02781],
                           [0.94150, -0.01582, -0.33665, -0.12395],
                           [0.0, 0.0, 0.0, 1.0]])

# Intrinsic parameters of camera
FX = 458.654
FY = 457.296
X0 = 367.215
Y0 = 248.375
K1 = -0.28340811
K2 = 0.07395907
H = 480
W = 752

# Normalization factor for Depth
ZMAX = 7.48

# Number of parallel process
NUM_PROCESS = 8


def get_camera_view_pointcloud(transform, xyz):
    """Transform point cloud to camera view, prune points outside of the view.
    Args:
      transform: 4x4 transform matrix representing position and orientation of
        the body of the MAV.
      xyz: A 4xN matrix, point cloud in homogeneous coordinates. The k-th column
        is (x, y, z, 1), where x, y, z are the coordinates of the k-th point.
    Returns:
      3xM (M < N) matrix representing the point cloud in the camera view.
      M vector, vertex grayscale value.
      Only points that fall within the camera viweing angle and are in front of
      the camera are kept.
    """

    overall_transform = np.linalg.inv(TF_BODY_CAM0).dot(np.linalg.inv(transform))
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
        return int(round(xz / neighborhood_radius)), int(round(yz / neighborhood_radius))

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
    """Given a colored cloud in camera coordinates, render a depth_estimation map.
    This function is useful for visualization / debugging.
    Args:
      xyz: A 3xN matrix representing the point cloud in the camera view.
      normal: normal of each point in xyz
    Returns:
      A rendered depth_estimation map.
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
          filename: A string, filepath of the state_groundtruth_estimate0.csv file. Can also be the room number.
        """

        if type(filename) == int:
            # mean = {"V1_01_easy/": np.array([0.36256679, 0.37336331, 1.52168557]),
            #         "V1_02_medium/": np.array([0.00806092, 1.18968812, 1.60374988]),
            #         "V1_03_difficult/": np.array([-0.0128916, 1.04127094, 1.75094365])}

            data = load_flight_for_depth(FLIGHT_LIST[filename])

            # data["state"][:, :3] = data["state"][:, :3] + mean[FLIGHT_LIST[filename]].reshape((1, 3))
            trans, quat = data["state"][:, :3], data["state"][:, 3:7]
            state_body = TF_BODY_VICON0 @ Transform.from_quat_trans(quat, trans).as_matrix()
            quat, trans = Transform.from_matrix(state_body).as_quat_trans()
            quat = quaternion.qfix(quat)
            quat = quat / np.linalg.norm(quat, axis=1).reshape((-1, 1))
            self.gt = np.concatenate([data["t"].reshape((-1, 1)), trans, quat[:, [3, 0, 1, 2]]], axis=1)

            # Check csv
            filename = (WD_PATH + "data/euroc_mav/raw/" + FLIGHT_LIST[int(args.room_path)] +
                        "state_groundtruth_estimate0/data.csv")
            with open(filename) as f:
                lines = f.readlines()
            lines = lines[1:]  # skip the first line

            gt2 = []
            for line in lines:
                tokens = line.split(',')
                gt2.append([float(t) for t in tokens[:8]])
            gt2 = np.array(gt2)
            gt2[:, 4:8] = quaternion.qfix(gt2[:, 4:8])
            gt2[:, 4:8] = gt2[:, 4:8] / np.linalg.norm(gt2[:, 4:8], axis=1).reshape((-1, 1))

            # for i in range(4, 8):
            #     plt.plot(self.gt[:, i], label=f"Quentin {i}")
            #     plt.plot(gt2[:, i], label=f"Steeven {i}")
            # plt.legend()
            # plt.show()

            # print()
            # trace_trajectory3d(self.gt[:, 1:4], self.gt[:, 4:8])

        else:
            with open(filename) as f:
                lines = f.readlines()
            lines = lines[1:]  # skip the first line

            gt = []
            for line in lines:
                tokens = line.split(',')
                gt.append([float(t) for t in tokens[:8]])

            self.gt = np.array(gt)  # Should be an array Tx7 : [x, y, z, qw, qx, qy, qz]

    def get_transform(self, k):
        """Interpolates the MAV's transform matrix at a timestamp."""

        # self._gt[:, 0], the 0th column, is the timestamp. Columns 1-3 are x, y, z,
        # and columns 4-7 are quaternion components describing the rotation.
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
            2 * qx * qz + 2 * qy * qw, x],
            [2 * qx * qy + 2 * qz * qw,
             1 - 2 * qx * qx - 2 * qz * qz,
             2 * qy * qz - 2 * qx * qw, y],
            [2 * qx * qz - 2 * qy * qw,
             2 * qy * qz + 2 * qx * qw,
             1 - 2 * qx * qx - 2 * qy * qy, z], [0.0, 0.0, 0.0, 1.0]])

        return self.gt[k, 0], transform


def read_ply(filename):
    """Reads a PLY file representing EuRoc's point cloud."""
    with open(filename) as f:
        lines = f.readlines()
    lines = lines[11:]
    xyz = []
    for line in lines:
        tokens = line.split(' ')
        xyz.append([float(t) for t in tokens[:3]])

    return np.array(xyz)


def filter_out_ot_frame_points(xyz):
    """Remove all points in a camera-view pointcloud that are out of frame.
    Args:
      xyz: A 3xN matrix representing the point cloud in the camera view.
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
    """Generate depth and normals from the position/orientation :
    input : - pipe : pipe object for multiprocessing
            - transform : 3x4 transformation matrix from frame to camera
            - xyz : pointcloud with 1 on the last line"""

    # Move the point cloud to fit the camera position
    xyz_view = get_camera_view_pointcloud(transform, xyz)
    # Remove occluded points (point behind a wall)
    visible_indices = get_occluded_points(xyz_view, 0.02, 0.08)
    xyz_view = xyz_view[visible_indices, :]

    # Remove out of screen points
    visible_indices = filter_out_ot_frame_points(xyz_view)
    xyz_view = xyz_view[visible_indices, :]

    # Estimate normals using pcd lib
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_view)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1, max_nn=1000),
                         fast_normal_computation=True)
    pcd.orient_normals_towards_camera_location(camera_location=transform[:-1, -1])
    pcd.normalize_normals()
    visible_normal = np.asarray(pcd.normals)

    # Compute depth + fancy stuff on the normal
    z_image, n_image = render_z_and_normal(xyz_view, visible_normal)

    # Fill hole in images with linear interp
    valid_mask = (1 - np.isnan(z_image)).astype(bool)
    coords = np.array(np.nonzero(valid_mask)).T

    values_z = z_image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values_z, fill_value=np.mean(values_z))
    filled_z = it(list(np.ndindex(z_image.shape))).reshape(z_image.shape)

    values_n = n_image[valid_mask]
    it = interpolate.LinearNDInterpolator(coords, values_n)
    filled_n = it(list(np.ndindex(z_image.shape))).reshape(n_image.shape)

    # Normalization
    filled_z = (filled_z / ZMAX)
    filled_n = ((filled_n + 1) / 2)

    # fig, ax = plt.subplots(2, 2)
    # ax[0][0].imshow(z_image)
    # ax[0][1].imshow(filled_z)
    # ax[1][0].imshow(n_image)
    # ax[1][1].imshow(filled_n)
    #
    # ax[0][0].axis("off")
    # ax[0][1].axis("off")
    # ax[1][0].axis("off")
    # ax[1][1].axis("off")
    # plt.show()

    # plt.imsave("depth_estimation.png", np.stack([filled_z, filled_z, filled_z], axis=-1))
    # p = plt.imread("depth_estimation.png").astype(np.float64)[:, :, 0]
    #
    # delta = ((filled_z - p) ** 2)

    # np.save("depth_estimation.npy", filled_z.astype(np.float32))
    # np.save("normal.npy", filled_n.astype(np.float32))

    # plt.matshow((filled_n + 1) / 2)
    # plt.show()
    pipe.send((filled_z, filled_n))
    pipe.close()


def main():
    if args.integrated:
        gti = GroundTruthInterpolator(int(args.room_path))
        xyz = np.load(WD_PATH + "src/depth_estimation/pointcloud.npy")
    else:
        # GT position/orientation interpolator
        gti = GroundTruthInterpolator(os.path.join(args.room_path, 'state_groundtruth_estimate0/data.csv'))
        # 3D point cloud (as npy, faster loading)
        xyz = np.load("pointcloud.npy")
    xyz_homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1))], axis=1)

    # Create the output directories
    os.makedirs(args.output_path + "depth_estimation", exist_ok=True)
    os.makedirs(args.output_path + "normal", exist_ok=True)

    # For each timestamp
    for k in tqdm(range(0, gti.gt.shape[0], NUM_PROCESS)):
        timestamps, transforms = [], []
        # Extract NUM_PROCESS position/orientation for computation
        for i in range(NUM_PROCESS):
            if k + i < gti.gt.shape[0]:
                time, trans = gti.get_transform(k + i)
                timestamps.append(time)
                transforms.append(trans)

        childPipes, parentPipes = [], []
        processes = []
        imgs_z, imgs_n = [], []

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
            z, n = parentPipes[rank].recv()
            imgs_z.append(z)
            imgs_n.append(n)
            p.join()

        # Save images, use png for time/space saving
        for i in range(len(imgs_z)):
            plt.imsave(args.output_path + f"depth_estimation/{str(float(timestamps[i])).replace('.', '_')}.png",
                       np.stack([imgs_z[i], imgs_z[i], imgs_z[i]], axis=-1))
            plt.imsave(args.output_path + f"normal/{str(float(timestamps[i])).replace('.', '_')}.png", imgs_n[i])


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
    if args.integrated:
        args.output_path = INTERMEDIATE_DATASET_PATH + FLIGHT_LIST[int(args.room_path)]
    main()
