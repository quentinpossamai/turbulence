import glob
import os
import pickle
import sys
import time
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from cv2 import VideoWriter, VideoWriter_fourcc
from scipy.spatial.transform import Rotation
import scipy.integrate as integrate

from util import Progress, Transform, angle_arccos

major, _, _, _, _ = sys.version_info
assert major == 3

frame_width = 640
frame_height = 480


def aff3d(xyz_array, quat_array, video_path):
    assert len(xyz_array) == len(quat_array)
    print(video_path)
    x_pos = np.array([1, 0, 0])
    y_pos = np.array([0, 1, 0])
    z_pos = np.array([0, 0, 1])
    # minx, miny, minz = np.min(xyz_array, axis=0) - 1
    # maxx, maxy, maxz = np.max(xyz_array, axis=0) + 1

    width = 640
    height = 480

    codec = VideoWriter_fourcc(*'H264')
    fps = 30
    video = VideoWriter(video_path, codec, float(fps), (width, height), True)
    prg = Progress(len(xyz_array))

    for xyz, quat in zip(xyz_array, quat_array):
        ref_tf_pos = Transform().from_pose(xyz, quat).get_inv()

        x_ref = ref_tf_pos.get_rot() @ x_pos
        y_ref = ref_tf_pos.get_rot() @ y_pos
        z_ref = ref_tf_pos.get_rot() @ z_pos

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.quiver(xyz[0], xyz[1], xyz[2], x_ref[0], x_ref[1], x_ref[2], length=1, normalize=False, color='r')
        ax.quiver(xyz[0], xyz[1], xyz[2], y_ref[0], y_ref[1], y_ref[2], length=1, normalize=False, color='g')
        ax.quiver(xyz[0], xyz[1], xyz[2], z_ref[0], z_ref[1], z_ref[2], length=1, normalize=False, color='b')

        ax.scatter(xs=xyz[0] + 1, ys=xyz[1] + 1, zs=xyz[2] + 1, alpha=0)
        ax.scatter(xs=xyz[0] - 1, ys=xyz[1] - 1, zs=xyz[2] - 1, alpha=0)

        # ax.scatter(xs=minx, ys=miny, zs=minz, alpha=0)
        # ax.scatter(xs=maxx, ys=maxy, zs=maxz, alpha=0)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((height, width, 3))

        plt.close('all')
        # plt.show()

        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
        prg.update_pgr()

    video.release()
    cv2.destroyAllWindows()
    print()


class ErrorEstimation(object):
    def __init__(self, flight_number):
        self.tic = time.time()
        self.flight_number = flight_number
        # Import data
        flight_files = []
        i = 0
        for day in sorted(next(os.walk(ABSOLUTE_PATH))[1]):
            temp_path = ABSOLUTE_PATH + day + '/'
            for flight_name in sorted(next(os.walk(temp_path))[1]):
                flight_path = temp_path + flight_name + '/clean_data/'
                if i == self.flight_number:
                    self.flight_name = day + '/' + flight_name + '/'
                    for filepath in sorted(glob.glob(flight_path + '*.pkl')):
                        flight_files.append(filepath)
                    flight_files = sorted(flight_files)
                i += 1

        self.data = pickle.load(open(flight_files[0], 'rb'))

        self.time = self.data['time']
        self.xyz = self.data['xyz']
        self.quaternions = self.data['quaternions']
        self.cov = self.data['cov']
        self.left_frames = self.data['left_frames']
        self.right_frames = self.data['right_frames']
        self.camera_info_left = self.data['camera_info_left']
        self.camera_info_right = self.data['camera_info_right']

        self.frames_height = self.left_frames.shape[1]
        self.frames_width = self.left_frames.shape[2]

        self.flu0_tf_enu = Transform().from_pose(self.xyz[0], self.quaternions[0])

        # Vincent input (0.105, 0, 0, -1.57, 0.0, -2.0943) (convention x, y, z, roll, pitch, yaw)
        quat = Rotation.from_euler('xyz', np.array([-120, 0, -90]), degrees=True).as_quat()
        self.camera_tf_drone = Transform().from_pose(np.array([0.105, 0, 0]), quat)

        enu_tf_flu0 = self.flu0_tf_enu.get_inv()
        for i, (pos, quat) in enumerate(zip(self.xyz, self.quaternions)):
            drone_tf_enu = Transform().from_pose(pos, quat)
            drone_tf_flu0 = drone_tf_enu @ enu_tf_flu0
            self.xyz[i], self.quaternions[i] = drone_tf_flu0.get_pose()

        self.tac = time.time()
        print(f'Primary import done | '
              f'Time since beginning : {self.tac - self.tic:.03f}s | '
              f'Time since last step : {self.tac - self.tic:.03f}s')
        self.toc = time.time()

    def get_camera_angle_to_horizontal_plane(self, iteration):
        z_camera_in_camera = np.array([0, 0, 1])  # Vector only rotation are important and NOT translations

        drone_tf_camera = self.camera_tf_drone.get_inv()
        drone_rot_camera = drone_tf_camera.get_rot()
        z_camera_in_drone = drone_rot_camera @ z_camera_in_camera

        drone_tf_flu0 = Transform().from_pose(self.xyz[iteration], self.quaternions[iteration])
        flu0_tf_drone = drone_tf_flu0.get_inv()
        flu0_rot_drone = flu0_tf_drone.get_rot()
        z_camera_in_flu0 = flu0_rot_drone @ z_camera_in_drone

        res = np.arcsin(np.abs(z_camera_in_flu0[2]) / np.linalg.norm(z_camera_in_flu0))
        return res

    def image23d(self, idx, image_point):
        # self.xyz and self.quaternions are from FLU0 to Drone

        p_left = self.camera_info_left['P']
        fx = p_left[0, 0]
        fy = p_left[1, 1]
        cx = p_left[0, 2]
        cy = p_left[1, 2]

        image_x = image_point[0]
        image_y = image_point[1]
        temp_z = 10
        x = (temp_z / fx) * (image_x - cx)
        y = (temp_z / fy) * (image_y - cy)
        vect1 = np.array([x, y, temp_z])

        middle_x = cx
        middle_y = cy
        x = (temp_z / fx) * (middle_x - cx)
        y = (temp_z / fy) * (middle_y - cy)
        vect2 = np.array([x, y, temp_z])
        pixel_angle = angle_arccos(vect1, vect2) * 180 / np.pi

        # pixel_angle = 8.34  # ° between optical center (cx, cy) and (cx, cy + cy / 2)

        camera_angle_to_horizontal_plane = self.get_camera_angle_to_horizontal_plane(idx) * 180 / np.pi
        theta = (camera_angle_to_horizontal_plane + pixel_angle) * np.pi / 180

        h = np.round(self.xyz[idx][2], 3)
        h = (h if h > 0 else 0) + 0.1  # Todo to complete

        true_z = h / np.cos(np.pi / 2 - theta)

        x = (true_z / fx) * (image_x - cx)
        y = (true_z / fy) * (image_y - cy)

        return np.array([x, y, true_z, 1]).reshape((4, 1))

    def _is_out_of_image(self, image_point):
        assert len(image_point) == 2
        a = image_point
        return not ((0 <= a[0] < self.frames_width) and (0 <= a[1] < self.frames_height))

    def _space2image(self, point3d):
        p_matrix = self.camera_info_left['P']
        u, v, w = np.dot(p_matrix, point3d).flatten()
        return u / w, v / w

    def p3_generator(self):
        r0 = {'xyz': self.xyz[0], 'quat': self.quaternions[0]}  # in Reference Frame flu0 to drone
        r1 = r0
        cx, cy = self.camera_info_left['P'][0, 2], self.camera_info_left['P'][1, 2]
        p = (cx, cy + cy / 2)
        p1 = self.image23d(0, p)
        p3_list = []
        for i in range(len(self.time)):
            ri = {'xyz': self.xyz[i], 'quat': self.quaternions[i]}
            r1cam_tf_flu0 = self.camera_tf_drone @ Transform().from_pose(r1['xyz'], r1['quat'])
            ricam_tf_flu0 = self.camera_tf_drone @ Transform().from_pose(ri['xyz'], ri['quat'])
            ricam_tf_r1cam = ricam_tf_flu0 @ r1cam_tf_flu0.get_inv()
            p2 = ricam_tf_r1cam @ p1
            p3 = self._space2image(p2)
            if self._is_out_of_image(p3):
                r1 = {'xyz': self.xyz[i], 'quat': self.quaternions[i]}
                p1 = self.image23d(i, p)  # RF camera
                p3 = (40, 40)
            p3_list.append(p3)
        self.data['p3'] = p3_list
        pickle.dump(self.data, open(ABSOLUTE_PATH + self.flight_name + 'clean_data/data.pkl', 'wb'))

    def p32video(self):
        fps = 10
        codec = VideoWriter_fourcc(*'H264')
        part1, part2, _ = self.flight_name.split('/')
        name = ABSOLUTE_PATH + 'error_estimation_videos/' + part1 + '_' + part2 + '_error_estimation.mkv'
        video = VideoWriter(name, codec, float(fps), (self.frames_width, self.frames_height), True)
        print(name)

        frames = self.left_frames
        prg = Progress(len(frames))
        point_list = self.data['p3']
        shape = (self.frames_height, self.frames_width, 3)
        for i, frame in enumerate(frames):
            p = np.int32(np.round(point_list[i]))
            color_frame = np.uint8(np.zeros((self.frames_height, self.frames_width, 3)))
            for j in range(3):
                color_frame[:, :, j] = frame
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            plt.imshow(color_frame)
            angles = Rotation.from_quat(self.quaternions[i]).as_euler('xyz', degrees=True)
            plt.scatter(p[0], p[1], color='r')
            legend = (f'Drone pose - FLU0\n'
                      f'      x : {self.xyz[i][0]:8.3f}m\n'
                      f'      y : {self.xyz[i][1]:8.3f}m\n'
                      f'      z : {self.xyz[i][2]:8.3f}m\n'
                      f'Rot x : {angles[0]:8.3f}°\n'
                      f'Rot y : {angles[1]:8.3f}°\n'
                      f'Rot z : {angles[2]:8.3f}°')
            plt.annotate(legend, (450, 335), xycoords='axes points', size=12, ha='right', va='top',
                         bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
            ax.axis('off')
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
            plt.close('all')

            video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
            prg.update_pgr()

        video.release()
        cv2.destroyAllWindows()
        # del video


if __name__ == '__main__':
    def main():
        # pose_source possibilities : ['tubex_estimator, 'board_imu', 'mavros_imu']
        e = DataPreparation(flight_number=1, pose_source='mavros_imu')
        e.fusion()
        f = ErrorEstimation(flight_number=1)
        f.p3_generator()
        f.p32video()

        # for flight_number in range(0, 12):
        #     print(flight_number)
        #     if flight_number == 1:
        #         continue
        #     e = DataPreparation(flight_number=flight_number, pose_source='mavros_imu')
        #     e.fusion()
        #     f = ErrorEstimation(flight_number=flight_number)
        #     f.p3_generator()
        #     f.p32video()


    main()
