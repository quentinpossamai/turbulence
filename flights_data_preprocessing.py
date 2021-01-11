import glob
import os
import pickle
import time
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

from util import Transformation, DataFolder


def main():
    f = DataFolder('data_drone2')
    file_path = f.get_unique_file_path('.pkl', f.folders['raw_python'][0], 'tara')

    data = pickle.load(open(file_path, 'rb'))
    e = DataPreparation()
    e.fusion()


class DataPreparation(object):
    def __init__(self, flight_number: int, pose_source: str):
        self.tic = time.time()

        self.flight_number = flight_number
        # Import data
        flight_files = []
        i = 0
        for day in sorted(next(os.walk(ABSOLUTE_PATH))[1]):
            temp_path = ABSOLUTE_PATH + day + '/'
            for flight_name in sorted(next(os.walk(temp_path))[1]):
                flight_path = temp_path + flight_name + '/'
                if i == self.flight_number:
                    self.flight_name = day + '/' + flight_name + '/'
                    for filepath in sorted(glob.glob(flight_path + '*.npy')):
                        flight_files.append(filepath)
                    for filepath in sorted(glob.glob(flight_path + '*.pkl')):
                        flight_files.append(filepath)
                    flight_files = sorted(flight_files)
                i += 1
                # print(day + '/' + flight_name + '/')

        if pose_source == 'tubex_estimator':
            name = [e for e in flight_files if re.search('.*tara_info_left.*.pkl', e) is not None][0]
            self.camera_info_left = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_info_right.*.pkl', e) is not None][0]
            self.camera_info_right = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_left.*.npy', e) is not None][0]
            self.left_frames = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_left_time.*.npy', e) is not None][0]
            self.left_frames_time = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_right.*.npy', e) is not None][0]
            self.right_frames = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_right_time.*.npy', e) is not None][0]
            self.right_frames_time = np.load(name)

            name = [e for e in flight_files if re.search('.*tubex_estimator.*.pkl', e) is not None][0]
            self.poses_data = pickle.load(open(name, 'rb'), encoding='latin1')

        elif pose_source == 'board_imu':
            name = [e for e in flight_files if re.search('.*board_imu.*.pkl', e) is not None][0]
            imu = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_info_left.*.pkl', e) is not None][0]
            self.camera_info_left = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_info_right.*.pkl', e) is not None][0]
            self.camera_info_right = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_left.*.npy', e) is not None][0]
            self.left_frames = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_left_time.*.npy', e) is not None][0]
            self.left_frames_time = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_right.*.npy', e) is not None][0]
            self.right_frames = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_right_time.*.npy', e) is not None][0]
            self.right_frames_time = np.load(name)

            # angular velocity integration to quaternions
            quat = np.zeros((len(imu['orientation']), 4))
            quat[0][3] = 1

            ts = imu['time'][1:] - imu['time'][:-1]
            for i, (wx, wy, wz) in enumerate(imu['angular_velocity'][1:]):
                mat = np.eye(4) + 1 / 2 * ts[i] * np.array([[0, -wx, -wy, -wz],
                                                            [wx, 0, wz, -wy],
                                                            [wy, -wz, 0, wx],
                                                            [wz, wy, -wx, 0]])
                quat[i + 1] = mat @ quat[i]
                quat[i + 1] /= np.linalg.norm(quat[i + 1])

            # linear acceleration correction
            # IMU orientated in drone = FLU = FLUi i in [0, T]
            mat_a = np.array([[0.9930045566279224, -0.0007377157002538479, -0.007219469698671251],
                              [-0.0007377157002538617, 0.9957519644937433, -0.0003710755696976209],
                              [-0.007219469698671307, -0.0003710755696976209, 0.9959784223847098]])
            drone_rot_imu = np.array([[0.9997849927500873, 0.011378000779108768, 0.014941846703305539],
                                      [-0.01137537518424766, 0.9998406525646836, 0.006196493976405852],
                                      [-0.014940204374346889, -0.00630934708287653, 0.9998219158560427]])
            b = np.array([0.14347160191766697, 0.006715615184851986, -0.014026001440967934])
            g = 9.81
            acc_g_in_global = np.array([0, 0, g])
            for i, acc in enumerate(imu['linear_acceleration']):
                acc = mat_a @ (imu['linear_acceleration'][i] - b)  # Remove noise
                flu0_rot_drone = Transformation().from_pose(np.zeros(3), quat[i]).inv().get_rot()
                imu['linear_acceleration'][i] = flu0_rot_drone @ drone_rot_imu @ acc
                imu['linear_acceleration'][i] = imu['linear_acceleration'][i] - acc_g_in_global

            # linear acceleration double integration to position
            velocity = np.zeros_like(imu['linear_acceleration'])
            xyz = np.zeros_like(imu['linear_acceleration'])
            for i in range(3):
                velocity[:, i] = integrate.cumtrapz(y=imu['linear_acceleration'][:, i], x=imu['time'], initial=0.)
                xyz[:, i] = integrate.cumtrapz(y=velocity[:, i], x=imu['time'], initial=0.)

            # Video creation
            # part1, part2, _ = self.flight_name.split('/')
            # name = ABSOLUTE_PATH + 'integration_videos/' + part1 + '_' + part2 + '_integration.mkv'
            # aff3d(xyz, quat, video_path=name)
            # states.exit(0)

            self.poses_data = {'time': imu['time'],
                               'pose': {'xyz': xyz,
                                        'quaternions': quat,
                                        'cov': imu['linear_acceleration_cov']},
                               'twist': {'dxyz': velocity,
                                         'dquaternions': imu['angular_velocity'],
                                         'cov': imu['angular_velocity_cov']}}

        elif pose_source == 'mavros_imu':
            name = [e for e in flight_files if re.search('.*mavros_imu.*.pkl', e) is not None][0]
            imu = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_info_left.*.pkl', e) is not None][0]
            self.camera_info_left = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_info_right.*.pkl', e) is not None][0]
            self.camera_info_right = pickle.load(open(name, 'rb'), encoding='latin1')

            name = [e for e in flight_files if re.search('.*tara_left.*.npy', e) is not None][0]
            self.left_frames = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_left_time.*.npy', e) is not None][0]
            self.left_frames_time = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_right.*.npy', e) is not None][0]
            self.right_frames = np.load(name)

            name = [e for e in flight_files if re.search('.*tara_right_time.*.npy', e) is not None][0]
            self.right_frames_time = np.load(name)

            quat = imu['orientation']
            raw_acc = imu['linear_acceleration'].copy()

            # linear acceleration correction
            # IMU orientated in drone = FLU = FLUi i in [0, T]
            mat_a = np.array([[1.0006205831082509, 0.0016298250067589282, -0.0008938592966822312],
                              [0.0016298250067589282, 0.999657268350121, 0.00157881502930865],
                              [-0.0008938592966822867, 0.0015788150293087333, 1.0001337579823208]])
            drone_rot_imu = np.array([[0.9997857431831232, 0.01288008771845976, -0.010557797456369658],
                                      [-0.012787554819759774, 0.9997241101787383, -0.012350706373651883],
                                      [0.01022487186656836, 0.012584111856585653, 0.9998202351036332]])
            b = np.array([-0.0036374953246893216, -0.030957561876831713, -0.007030521095834748])
            g = 9.81
            acc_g_in_global = np.array([0, 0, g])
            for i, acc in enumerate(imu['linear_acceleration']):
                acc = mat_a @ (imu['linear_acceleration'][i] - b)  # Remove noise
                flu0_rot_drone = Transformation().from_pose(np.zeros(3), quat[i]).inv().get_rot()
                imu['linear_acceleration'][i] = flu0_rot_drone @ drone_rot_imu @ acc
                imu['linear_acceleration'][i] = imu['linear_acceleration'][i] - acc_g_in_global
                raw_acc[i] = flu0_rot_drone @ drone_rot_imu @ raw_acc[i]
                raw_acc[i] = raw_acc[i] - acc_g_in_global

            # linear acceleration double integration to position
            velocity = np.zeros_like(imu['linear_acceleration'])
            xyz = np.zeros_like(imu['linear_acceleration'])
            raw_velocity = np.zeros_like(imu['linear_acceleration'])
            raw_xyz = np.zeros_like(imu['linear_acceleration'])
            for i in range(3):
                velocity[:, i] = integrate.cumtrapz(y=imu['linear_acceleration'][:, i], x=imu['time'], initial=0.)
                xyz[:, i] = integrate.cumtrapz(y=velocity[:, i], x=imu['time'], initial=0.)
                raw_velocity[:, i] = integrate.cumtrapz(y=raw_acc[:, i], x=imu['time'], initial=0.)
                raw_xyz[:, i] = integrate.cumtrapz(y=raw_velocity[:, i], x=imu['time'], initial=0.)

            self.poses_data = {'time': imu['time'],
                               'pose': {'xyz': xyz,
                                        'quaternions': quat,
                                        'cov': imu['linear_acceleration_cov']},
                               'twist': {'dxyz': velocity,
                                         'dquaternions': imu['angular_velocity'],
                                         'cov': imu['angular_velocity_cov']}}

            # fig = plt.figure(figsize=[2 * 6.4, 4.8])
            # # fig.suptitle(self.flight_name[:-1].replace("/", "_"), fontsize=14)
            # plt.subplot(121)
            # xx = self.poses_data['time'] - np.min(self.poses_data['time'])
            # plt.plot(xx, self.poses_data['pose']['xyz'][:, 0], label='x', color='tab:blue')
            # plt.plot(xx, raw_xyz[:, 0], label='x_raw', linestyle='dashed', color='tab:blue')
            # plt.plot(xx, self.poses_data['pose']['xyz'][:, 1], label='y', color='tab:orange')
            # plt.plot(xx, raw_xyz[:, 1], label='y_raw', linestyle='dashed', color='tab:orange')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Position (m)')
            # plt.legend(loc=1, framealpha=0.5)
            # plt.subplot(122)
            # plt.plot(xx, self.poses_data['pose']['xyz'][:, 2], label='z', color='tab:blue')
            # plt.plot(xx, raw_xyz[:, 2], label='z_raw', linestyle='dashed', color='tab:blue')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Position (m)')
            # plt.legend(loc=1, framealpha=0.5)
            # fig.tight_layout()
            # name = (ABSOLUTE_PATH[:-11] +
            #         f'plots/mavros_imu_denoising_comparison/{self.flight_name[:-1].replace("/", "_")}.png')
            # fig.savefig(name, dpi=fig.dpi)
            # plt.show()

        assert hasattr(self, 'camera_info_left')
        assert hasattr(self, 'camera_info_right')
        assert hasattr(self, 'left_frames')
        assert hasattr(self, 'left_frames_time')
        assert hasattr(self, 'right_frames')
        assert hasattr(self, 'right_frames_time')
        assert hasattr(self, 'poses_data')

        self.frames_height = self.left_frames.shape[1]
        self.frames_width = self.left_frames.shape[2]
        self.frames_time = None

        self.tac = time.time()
        print(f'Primary import done | Time since beginning : {self.tac - self.tic:.03f}s |'
              f' Time since last step : {self.tac - self.tic:.03f}s')
        self.toc = time.time()

    def time_analysis(self):
        poses_time = self.poses_data['time']
        frames_time = self.left_frames_time

        plt.plot(poses_time, np.zeros(len(poses_time)), label='pose')
        plt.plot(frames_time, np.zeros(len(frames_time)) + 1, label='frames')
        plt.legend()
        plt.title(self.flight_name)
        plt.show()
        print(self.flight_name)
        print(f'Nb of poses : {len(poses_time)}')
        print(f'Nb of frames : {len(frames_time)}')
        print()

    def fusion(self):

        # Nan verification FOLD
        poses_time = self.poses_data['time'].copy()
        x = np.sum(np.isnan(poses_time))
        assert x == 0

        poses_xyz = self.poses_data['pose']['xyz'].copy()
        x = np.sum(np.isnan(poses_xyz))
        assert x == 0

        poses_quaternions = self.poses_data['pose']['quaternions'].copy()
        # poses_quaternions[0, :] = [0, 0, 0, 0]  # Init first quaternions
        ind = [i for i, x in enumerate(np.isnan(poses_quaternions)) if not any(x)]
        poses_quaternions = poses_quaternions[ind]
        poses_xyz = poses_xyz[ind]
        poses_time = poses_time[ind]
        x = np.sum(np.isnan(poses_quaternions))
        assert x == 0
        self.poses_data['pose']['quaternions'] = poses_quaternions.copy()

        poses_cov = self.poses_data['pose']['cov'].copy()
        x = np.sum(np.isnan(poses_cov))
        assert x == 0

        assert len(self.left_frames_time) == len(np.unique(self.left_frames_time))
        x = np.sum(np.isnan(self.left_frames_time))
        assert x == 0

        assert len(self.right_frames_time) == len(np.unique(self.right_frames_time))
        x = np.sum(np.isnan(self.right_frames_time))
        assert x == 0

        x = np.sum(np.isnan(self.left_frames))
        assert x == 0
        assert self.frames_height < self.frames_width

        x = np.sum(np.isnan(self.right_frames))
        assert x == 0
        assert self.right_frames.shape[1] == self.frames_height
        assert self.right_frames.shape[2] == self.frames_width

        self.tac = time.time()
        print(f'Nan finished | Time since beginning : {self.tac - self.tic:.03f}s |'
              f' Time since last step : {self.tac - self.toc:.03f}s')
        self.toc = time.time()

        # Intersection between time arrays and sorting them in chronological order FOLD
        self.frames_time, left_ind, right_ind = np.intersect1d(self.left_frames_time, self.right_frames_time,
                                                               assume_unique=True, return_indices=True)
        self.left_frames = self.left_frames[left_ind]
        self.right_frames = self.right_frames[right_ind]

        # Chronological order
        poses_ind = np.argsort(poses_time)
        poses_time = poses_time[poses_ind]
        poses_xyz = poses_xyz[poses_ind]
        poses_quaternions = poses_quaternions[poses_ind]

        self.tac = time.time()
        print(f'Chronological order & Left right merge | Time since beginning : {self.tac - self.tic:.03f}s '
              f'| Time since last step : {self.tac - self.toc:.03f}s')
        self.toc = time.time()

        # Time normalization
        # assert np.min(self.frames_time) < np.min(poses_time)

        t0 = min(np.min(self.frames_time), np.min(poses_time))
        self.frames_time = self.frames_time - t0
        poses_time = poses_time - t0

        self.tac = time.time()
        print(f'Time normalization | Time since beginning : {self.tac - self.tic:.03f}s |'
              f' Time since last step : {self.tac - self.toc:.03f}s')
        self.toc = time.time()

        # Merge poses time and frames time FOLD
        new_poses_time = []
        new_xyz = []
        new_quaternions = []
        new_left_frames = []
        new_right_frames = []
        new_cov = []

        frames_ids = []
        poses_ids = []

        for i, p_time in enumerate(poses_time):
            mini_ind = 0
            mini = abs(self.frames_time[0] - p_time)
            for j, f_time in enumerate(self.frames_time):
                diff = abs(f_time - p_time)
                if mini > diff:
                    mini_ind = j
                    mini = diff
            if (mini_ind not in frames_ids) and (i not in poses_ids):
                new_poses_time.append(poses_time[i])
                new_xyz.append(poses_xyz[i])
                new_quaternions.append(poses_quaternions[i])
                new_left_frames.append(self.left_frames[mini_ind])
                new_right_frames.append(self.right_frames[mini_ind])
                new_cov.append(poses_cov[i])

                frames_ids.append(mini_ind)
                poses_ids.append(i)

        (self.poses_data['time'], self.poses_data['pose']['xyz'], self.poses_data['pose']['quaternions'],
         self.left_frames, self.right_frames, self.poses_data['pose']['cov']) = map(np.asarray, [new_poses_time,
                                                                                                 new_xyz,
                                                                                                 new_quaternions,
                                                                                                 new_left_frames,
                                                                                                 new_right_frames,
                                                                                                 new_cov])

        self.tac = time.time()
        print(f'Fusion left & right frames with poses |'
              f' Time since beginning : {self.tac - self.tic:.03f}s |'
              f' Time since last step : {self.tac - self.toc:.03f}s')
        self.toc = time.time()

        clean_data = {'time': self.poses_data['time'],
                      'xyz': self.poses_data['pose']['xyz'],
                      'quaternions': self.poses_data['pose']['quaternions'],
                      'cov': self.poses_data['pose']['cov'],
                      'left_frames': self.left_frames,
                      'right_frames': self.right_frames,
                      'camera_info_right': self.camera_info_right,
                      'camera_info_left': self.camera_info_left}

        # path = ABSOLUTE_PATH + self.flight_name + 'clean_data/'
        # Path(path).mkdir(parents=True, exist_ok=True)
        # pickle.dump(clean_data, open(path + 'data.pkl', 'wb'))
        print()

        # poses_time = self.poses_data['time']
        #
        # poses_time_before = poses_time[:-1]
        # poses_time_after = poses_time[1:]
        # diff = poses_time_after - poses_time_before
        # plt.plot(poses_time_after, diff, '.')
        # plt.title('Poses timestamp difference')
        # plt.xlabel('Timestamp')
        # plt.ylabel('Time difference')
        # plt.show()

        # frames_time_before = self.left_frames_time[:-1]
        # frames_time_after = self.left_frames_time[1:]
        # diff = frames_time_after - frames_time_before
        # plt.plot(frames_time_after, diff, '.')
        # plt.title('Frames timestamp difference')
        # plt.xlabel('Timestamp')
        # plt.ylabel('Time difference')
        # plt.show()
        #
        # right_frames_time_before = self.right_frames_time[:-1]
        # right_frames_time_after = self.right_frames_time[1:]
        # diff = right_frames_time_after - right_frames_time_before
        # plt.plot(right_frames_time_after, diff, '.')
        # plt.title('Right frames timestamp difference')
        # plt.xlabel('Timestamp')
        # plt.ylabel('Time difference')
        # plt.show()


if __name__ == '__main__':
    main()
