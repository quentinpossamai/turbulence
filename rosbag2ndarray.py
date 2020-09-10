from __future__ import print_function
import rosbag
import os
import glob
import re
import numpy as np
import sys
import pickle
from util2 import Progress, ABSOLUTE_PATH, FRAME_HEIGHT, FRAME_WIDTH
from scipy.spatial.transform import Rotation
from typing import List

major, _, _, _, _ = sys.version_info
assert major == 2


# This code must be executed with python 2.7 because of the rosbag package.


class Extractor(object):
    def __init__(self, bag_files_path: List[str]):
        # Frame size for the Tara camera
        self.tara_frame_width = FRAME_WIDTH
        self.tara_frame_height = FRAME_HEIGHT


        self.bags_n_flights = bags_n_flights

    def _print_topics_types(self, bags_paths, nb_msg=1):
        """
        Print topics and info of bags located in the bags_paths variable.
        :param bags_paths: List of string containing the paths of bags to be analyzed.
        """

        # For each bags and topics print info
        print('Number of bags analysed {}. Should be 12.'.format(len(bags_paths)))
        for i, path in enumerate(bags_paths):
            bag = rosbag.Bag(path)
            print('Nb {} | Bag analyzed : {}'.format(i, path))
            for topic, (msg_type, message_count, connections, frequency) in bag.get_type_and_topic_info()[1].items():
                print('Topic : {} | Type : {} | Msg count : {}'.format(topic, msg_type, message_count))
            print()
            bag.close()

        # For each different topic print first message to understand format
        topic_msg_analysed = []
        print_limit = 1000
        for path in bags_paths:
            bag = rosbag.Bag(path)
            for topic, _ in bag.get_type_and_topic_info()[1].items():
                if topic not in topic_msg_analysed:
                    to_print = [repr(e[1]) for i, e in enumerate(bag.read_messages(topics=topic)) if i < nb_msg]
                    for msg in to_print:
                        if len(msg) >= print_limit:
                            print('PRINT CROPPED')
                            print('Topic : {} | Msg : \n{}'.format(topic, msg[0:print_limit]))
                        else:
                            print('Topic : {} | Msg : \n{}'.format(topic, msg))
                        print()
                    topic_msg_analysed.append(topic)
            bag.close()

    def tara_extract(self, get_topics_types=False, topic='', nb_msg=1):
        """
        Extract topic or convert data
        :param get_topics_types: if get_topics_types=False converts .bag data to .npy format for tara bags.
        Else print topics and types of topics for all the bag found.
        :param topic: The topic to extract if get_topics_types=false
        :param nb_msg: Number of message to display per topic/
        :return: .npy file for all flights or a print

        Known topics : 'tara/left/image_raw', 'tara/right/image_raw', 'tara/right/camera_info', 'tara/left/camera_info'

        By ros convention, 'tara/right/camera_info' contain the information of the stereo camera which seems to be
        constant. 'tara/left/camera_info' is empty.
        """
        bags = [e for e in self.bags_n_flights if re.search('.*(/tara.*)', e[0]) is not None]
        # List of tuple (path to tara.bag file, path to tara.bag folder)

        if get_topics_types:
            self._print_topics_types([e[0] for e in bags], nb_msg)
        else:
            assert topic != '', 'Please insert a valid topic (see docstring)'
            for paths in bags:
                bag_file_path = paths[0]
                # bag_folder_path = paths[1]
                # break

                print(bag_file_path)

                bag = rosbag.Bag(bag_file_path)
                msg_n = bag.get_message_count(topic)
                p = Progress(max_iter=msg_n, end_print='\n')

                if topic in ['tara/left/image_raw', 'tara/right/image_raw']:
                    bag_data = []
                    time = np.zeros(msg_n)
                    for i, (topic, msg_raw, t) in enumerate(bag.read_messages(topics=topic)):
                        time[i] = msg_raw.header.stamp.secs + msg_raw.header.stamp.nsecs * 1e-9
                        # Shaping data as a ndarray
                        msg_encoded = repr(msg_raw)
                        match = re.search('data: .(.*).', msg_encoded)
                        data_string = match.group(1)
                        data = np.fromstring(data_string, dtype=np.uint8, sep=', ')

                        # Size of the data
                        img = data.reshape((self.tara_frame_height, self.tara_frame_width))
                        bag_data.append(img)

                        p.update_pgr()
                    array = np.stack(bag_data)
                    path = bag_file_path[:-4]

                    path_without_filename = path[:-25]
                    end_filename = path[-21:]

                    if topic == 'tara/right/image_raw':
                        np.save(path_without_filename + 'tara_right_' + end_filename + '.npy', array)
                        np.save(path_without_filename + 'tara_right_' + 'time' + end_filename + '.npy', time)
                    if topic == 'tara/left/image_raw':
                        np.save(path_without_filename + 'tara_left_' + end_filename + '.npy', array)
                        np.save(path_without_filename + 'tara_left_' + 'time' + end_filename + '.npy', time)

                elif topic in ['tara/right/camera_info', 'tara/left/camera_info']:
                    camera_info = {}
                    for i, (topic, msg_raw, t) in enumerate(bag.read_messages(topics=topic)):
                        a = np.array(msg_raw.D)
                        b = np.array(msg_raw.K)
                        c = np.array(msg_raw.P)
                        d = np.array(msg_raw.R)

                        if ((not a.shape == (0,) and not np.sum(np.abs(b)) == 0) and
                                (not np.sum(np.abs(c)) == 0 and not np.sum(np.abs(d)) == 0)):
                            camera_info['D'] = np.array(msg_raw.D)
                            camera_info['K'] = np.array(msg_raw.K).reshape((3, 3))
                            camera_info['P'] = np.array(msg_raw.P).reshape((3, 4))
                            camera_info['R'] = np.array(msg_raw.R).reshape((3, 3))
                            break
                    path = bag_file_path[:-4]
                    path_without_filename = path[:-25]
                    end_filename = path[-21:]

                    assert camera_info != {}

                    if topic == 'tara/right/camera_info':
                        pickle.dump(camera_info, open(path_without_filename + 'tara_info_right_'
                                                      + end_filename + '.pkl', 'wb'))
                    if topic == 'tara/left/camera_info':
                        pickle.dump(camera_info, open(path_without_filename + 'tara_info_left_'
                                                      + end_filename + '.pkl', 'wb'))

                bag.close()

    def tf_extract(self, get_topics_types=False, topic='', nb_msg=1):
        """
        Extract topic or convert data
        :param get_topics_types: if get_topics_types=False converts .bag data to .npy format for tf bags.
        Else print topics and types of topics for all the bag found.
        :param topic: The topic to extract if get_topics_types=false
        :param nb_msg: Number of message to display per topic
        :return: .npy file for all flights or a print

        Known topics : /tf, /tf_static
        """
        bags = [e for e in self.bags_n_flights if re.search('.*(/tf.*)', e[0]) is not None]
        # List of tuple (path to tara.bag file, path to tara.bag folder)

        if get_topics_types:
            text_file = open("tf.txt", "w")
            bags_paths = [e[0] for e in bags]
            self._print_topics_types(bags_paths, nb_msg)
            for i, path in enumerate(bags_paths):
                bag = rosbag.Bag(path)
                print('Nb {} | Bag analyzed : {}'.format(i, path), file=text_file)
                msg_n = bag.get_message_count(topic)
                p = Progress(max_iter=msg_n, end_print='\n')

                topics = ['/tf', '/tf_static']
                unique_tfs = {}
                for topic, msg_raw, t in bag.read_messages(topics=topics):
                    for e in msg_raw.transforms:
                        a = e.child_frame_id
                        b = e.header.frame_id
                        name = topic + '___' + a + '___' + b
                        if name not in unique_tfs.keys():
                            rotation = e.transform.rotation
                            rotation = np.array([rotation.x, rotation.y, rotation.z, rotation.w])
                            if np.sum((np.isnan(rotation))) == 4:
                                r = Rotation.from_quat([0, 0, 0, 1]).as_dcm()
                            else:
                                r = Rotation.from_quat(rotation).as_dcm()

                            translation = e.transform.translation
                            translation = np.array([translation.x, translation.y, translation.z])
                            if np.sum((np.isnan(translation))) == 3:
                                t = np.array([0, 0, 0])
                            else:
                                t = translation
                            tf = np.eye(4)
                            tf[:3, :3] = r
                            tf[:3, 3] = t
                            unique_tfs[name] = tf
                bag.close()

                with np.printoptions(precision=3, suppress=True):
                    for name, tf in unique_tfs.items():
                        print('{} \n{}\n'.format(name, tf), file=text_file)
                    print()
            text_file.close()
        else:
            pass

    def pose_extract(self, get_topics_types=False, topic='', nb_msg=1):
        """
        Extract topic or convert data
        :param get_topics_types: if get_topics_types=False converts .bag data to .npy format for tubex estimator bags.
        Else print topics and types of topics for all the bag found.
        :param topic: The topic to extract if get_topics_types=false
        :param nb_msg: Number of message to display per topic
        :return: .npy file for all flights or a print

        Known topics : tubex_estimator/odom
        """
        bags = [e for e in self.bags_n_flights if re.search('.*(/tubex_estimator.*)', e[0]) is not None]
        # List of tuple (path to tara.bag file, path to tara.bag folder)

        if get_topics_types:
            self._print_topics_types([e[0] for e in bags], nb_msg)
        else:
            for paths in bags:
                bag_file_path = paths[0]
                # bag_folder_path = paths[1]
                # break

                print(bag_file_path)

                bag = rosbag.Bag(bag_file_path)
                msg_n = bag.get_message_count(topic)
                p = Progress(max_iter=msg_n, end_print='\n')

                # Metadata
                time = np.zeros(msg_n)

                # Pose
                xyz = np.zeros((msg_n, 3))
                quaternions = np.zeros((msg_n, 4))
                covariance_pose = np.zeros((msg_n, 6, 6))

                # Twist
                dxyz = np.zeros((msg_n, 3))
                dquaternions = np.zeros((msg_n, 3))
                covariance_twist = np.zeros((msg_n, 6, 6))
                for i, (topic, msg, t) in enumerate(bag.read_messages(topics=topic)):
                    time[i] = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
                    xyz[i, :] = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
                    quaternions[i, :] = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                         msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
                    dxyz[i, :] = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
                    dquaternions[i, :] = [msg.twist.twist.angular.x, msg.twist.twist.angular.y,
                                          msg.twist.twist.angular.z]
                    for j in range(6):
                        for k in range(6):
                            covariance_pose[i, j, k] = msg.pose.covariance[j + k]
                            covariance_twist[i, j, k] = msg.twist.covariance[j + k]
                    # Progression
                    p.update_pgr()

                data = {'time': time, 'pose': {'xyz': xyz, 'quaternions': quaternions, 'cov': covariance_pose},
                        'twist': {'dxyz': dxyz, 'dquaternions': dquaternions, 'cov': covariance_twist}}
                # np.save(open(bag_file_path[:-4] + '.npy', 'wb'), data)  # Saving arrays
                pickle.dump(data, open(bag_file_path[:-4] + '.pkl', 'wb'))
                bag.close()

    def control_extract(self, get_topics_types=False, topic='', nb_msg=1):
        """
        Extract topic or convert data
        :param get_topics_types: if get_topics_types=False converts .bag data to .npy format for control bags.
        Else print topics and types of topics for all the bag found.
        :param topic: The topic to extract if get_topics_types=false
        :param nb_msg: Number of message to display per topic
        :return: .npy file for all flights or a print

        Known topics :
        """
        bags = [e for e in self.bags_n_flights if re.search('.*(/control.*)', e[0]) is not None]
        # List of tuple (path to tara.bag file, path to tara.bag folder)

        if get_topics_types:
            self._print_topics_types([e[0] for e in bags], nb_msg)
        else:
            pass

    def imu_extract(self, get_topics_types=False, topic='', nb_msg=1):
        """
         Extract topic or convert data
         :param get_topics_types: if get_topics_types=False converts .bag data to .npy format for board_imu and
          tara_imu bags.
         Else print topics and types of topics for all the bag found.
         :param topic: The topic to extract if get_topics_types=false
         :param nb_msg: Number of message to display per topic
         :return: .npy file for all flights or a print

         Known topics :
         """
        bags = [e for e in self.bags_n_flights if re.search('.*(/board_imu.*)', e[0]) is not None]
        # bags2 = [e for e in self.bags_n_flights if re.search('.*(/board_imu.*)', e[0]) is not None]
        # List of tuple (path to tara.bag file, path to tara.bag folder)

        if get_topics_types:
            self._print_topics_types([e[0] for e in bags], nb_msg)
        else:
            for paths in bags:
                bag_file_path = paths[0]
                print(bag_file_path)

                bag = rosbag.Bag(bag_file_path)
                msg_n = bag.get_message_count(topic)
                p = Progress(max_iter=msg_n, end_print='\n')

                time = np.zeros(msg_n)
                orientation = np.zeros((msg_n, 4))  # quaternions
                orientation_cov = np.zeros((msg_n, 3, 3))
                angular_velocity = np.zeros((msg_n, 3))
                angular_velocity_cov = np.zeros((msg_n, 3, 3))
                linear_acceleration = np.zeros((msg_n, 3))
                linear_acceleration_cov = np.zeros((msg_n, 3, 3))

                for i, (topic, msg_raw, t) in enumerate(bag.read_messages(topics=topic)):
                    time[i] = msg_raw.header.stamp.secs + msg_raw.header.stamp.nsecs * 1e-9

                    orientation[i, :] = [msg_raw.orientation.x, msg_raw.orientation.y, msg_raw.orientation.z,
                                         msg_raw.orientation.w]
                    orientation_cov[i, :, :] = np.array(msg_raw.orientation_covariance).reshape((3, 3))

                    angular_velocity[i, :] = [msg_raw.angular_velocity.x, msg_raw.angular_velocity.y,
                                              msg_raw.angular_velocity.z]
                    angular_velocity_cov[i, :, :] = np.array(msg_raw.angular_velocity_covariance).reshape((3, 3))

                    linear_acceleration[i, :] = [msg_raw.linear_acceleration.x, msg_raw.linear_acceleration.y,
                                                 msg_raw.linear_acceleration.z]
                    linear_acceleration_cov[i, :, :] = np.array(msg_raw.linear_acceleration_covariance).reshape(
                        (3, 3))

                data = {'time': time,
                        'orientation': orientation,
                        'orientation_cov': orientation_cov,
                        'angular_velocity': angular_velocity,
                        'angular_velocity_cov': angular_velocity_cov,
                        'linear_acceleration': linear_acceleration,
                        'linear_acceleration_cov': linear_acceleration_cov}

                # assert nans
                for name, e in data.items():
                    assert not np.any(np.isnan(e)), '{} contains nan'.format(name)
                assert np.all(np.sort(time) == time), 'Time array is not chronological'
                pickle.dump(data, open(bag_file_path[:-4] + '.pkl', 'wb'))
                bag.close()

    def mavros_extract(self, get_topics_types=False, topic='', nb_msg=1):
        """
        Extract topic or convert data
        :param get_topics_types: if get_topics_types=False converts .bag data to .npy format for mavros bags.
        Else print topics and types of topics for all the bag found.
        :param topic: The topic to extract if get_topics_types=false
        :param nb_msg: Number of message to display per topic
        :return: .npy file for all flights or a print

        Known topics : mavros/imu/data, mavros/global_position/raw/gps_vel, mavros/rc/out,
        mavros/global_position/compass_hdg, mavros/imu/temperature_imu, mavros/battery, mavros/global_position/rel_alt,
        mavros/extended_state, mavros/global_position/raw/fix, mavros/imu/mag, mavros/local_position/odom,
        mavros/imu/static_pressure, mavros/rc/in, mavros/state, mavros/imu/data_raw"""
        bags = [e for e in self.bags_n_flights if re.search('.*(/mavros.*)', e[0]) is not None]
        # List of tuple (path to tara.bag file, path to tara.bag folder)

        if get_topics_types:
            self._print_topics_types([e[0] for e in bags], nb_msg)
        else:
            for paths in bags:
                bag_file_path = paths[0]
                print(bag_file_path)

                bag = rosbag.Bag(bag_file_path)
                msg_n = bag.get_message_count(topic)
                p = Progress(max_iter=msg_n, end_print='\n')
                if topic=='mavros/imu/data':
                    time = np.zeros(msg_n)
                    orientation = np.zeros((msg_n, 4))  # quaternions
                    orientation_cov = np.zeros((msg_n, 3, 3))
                    angular_velocity = np.zeros((msg_n, 3))
                    angular_velocity_cov = np.zeros((msg_n, 3, 3))
                    linear_acceleration = np.zeros((msg_n, 3))
                    linear_acceleration_cov = np.zeros((msg_n, 3, 3))
                    for i, (topic, msg_raw, t) in enumerate(bag.read_messages(topics=topic)):
                        time[i] = msg_raw.header.stamp.secs + msg_raw.header.stamp.nsecs * 1e-9

                        orientation[i, :] = [msg_raw.orientation.x, msg_raw.orientation.y, msg_raw.orientation.z,
                                             msg_raw.orientation.w]
                        orientation_cov[i, :, :] = np.array(msg_raw.orientation_covariance).reshape((3, 3))

                        angular_velocity[i, :] = [msg_raw.angular_velocity.x, msg_raw.angular_velocity.y,
                                                  msg_raw.angular_velocity.z]
                        angular_velocity_cov[i, :, :] = np.array(msg_raw.angular_velocity_covariance).reshape(
                            (3, 3))

                        linear_acceleration[i, :] = [msg_raw.linear_acceleration.x, msg_raw.linear_acceleration.y,
                                                     msg_raw.linear_acceleration.z]
                        linear_acceleration_cov[i, :, :] = np.array(msg_raw.linear_acceleration_covariance).reshape(
                            (3, 3))
                        # Progression
                        p.update_pgr()
                    data = {'time': time,
                            'orientation': orientation,
                            'orientation_cov': orientation_cov,
                            'angular_velocity': angular_velocity,
                            'angular_velocity_cov': angular_velocity_cov,
                            'linear_acceleration': linear_acceleration,
                            'linear_acceleration_cov': linear_acceleration_cov}

                    # assert nans
                    for name, e in data.items():
                        assert not np.any(np.isnan(e)), '{} contains nan'.format(name)
                    assert np.all(np.sort(time) == time), 'Time array is not chronological'
                    name = bag_file_path[:-4].replace('mavros', topic.replace('/', '_')) + '.pkl'
                    pickle.dump(data, open(name, 'wb'))

                    # pickle.dump(data, open(bag_file_path[:-4] + '.pkl', 'wb'))
                bag.close()

    def get_freq(self):
        topic_freq = {}
        for path, _ in self.bags_n_flights:
            bag = rosbag.Bag(path)
            topics = bag.get_type_and_topic_info()[1].keys()
            for topic in topics:
                if topic not in topic_freq.keys():
                    count = bag.get_message_count(topic)
                    timestamps = np.zeros(count)
                    for i, (_, _, t) in enumerate(bag.read_messages(topic)):
                        timestamps[i] = t.secs + t.nsecs * 1e-9
                    timestamps = np.sort(timestamps)
                    period = timestamps[-1] - timestamps[0]
                    assert any(np.isnan(timestamps)) is False
                    if period == 0:
                        topic_freq[topic] = 0
                    else:
                        topic_freq[topic] = count / period
            bag.close()
        for topic, freq in topic_freq.items():
            print('Topic : {} | Frequency : {:0.03f}'.format(topic, freq))


if __name__ == '__main__':
    def main():
        bag_files_path = ''

        bags_n_flights = []
        for day in sorted(next(os.walk(ABSOLUTE_PATH))[1]):
            temp_path = ABSOLUTE_PATH + day + '/'
            for flight_name in sorted(next(os.walk(temp_path))[1]):
                flight_path = temp_path + flight_name + '/'
                for bag in sorted(glob.glob(flight_path + '*.bag')):
                    bags_n_flights.append((bag, flight_path))

        e = Extractor()
        e.mavros_extract(get_topics_types=False, topic='mavros/imu/data')

    main()
