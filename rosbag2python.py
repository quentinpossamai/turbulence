from __future__ import print_function
import rosbag
import os
import glob
import re
import numpy as np
import sys
import pickle
import time
from scipy.spatial.transform import Rotation

# This code must be executed with python 2.7 because of the rosbag package.
major, _, _, _, _ = sys.version_info
assert major == 2


def main():
    """
    Load rosbag file and save (pickle dump) it in python form. Adaptable to any message type.
    """
    # Data to extract parameters
    abs_path = '/Users/quentin/phd/turbulence/'
    data_folder = 'data_drone2'
    flight_number = 0
    pattern = '.*(/tara.*)'

    flights = []
    if data_folder == 'data_drone':
        flights = ['20200214_vol_1_Exterieur_MaintientPositionPX4',
                   '20200214_vol_2_Exterieur_MaintientPositionPX4',
                   '20200218_vol_1_Exterieur',
                   '20200218_vol_2_Exterieur',
                   '20200218_vol_3_Exterieur',
                   '20200218_vol_4_Exterieur',
                   '20200218_vol_5_Exterieur',
                   '20200218_vol_6_Exterieur',
                   '20200219_vol_1_Exterieur',
                   '20200219_vol_2_Exterieur',
                   '20200219_vol_3_Exterieur',
                   '20200220_vol_1_Exterieur']
    elif data_folder == 'data_drone2':
        flights = ['vol_1', 'vol_2', 'vol_3', 'vol_4_poubelle', 'vol_5_poubelle']

    # Locate the file to be extracted
    flight_path = abs_path + data_folder + '/raw/' + flights[flight_number] + '/'
    bags = []
    for bag in sorted(glob.glob(flight_path + '*.bag')):
        bags.append(bag)
    file_to_extract = [e for e in bags if re.search(pattern, e) is not None]
    assert len(file_to_extract) == 1
    file_to_extract = file_to_extract[0]

    # Create saving path
    filename = file_to_extract.split('/')[-1].split('.')[0]
    saving_path = abs_path + data_folder + '/raw_python/' + flights[flight_number] + '/' + filename + '.pkl'

    # Extract data
    e = Extractor(file_to_extract)
    data_extracted = e.extract()

    # Saving file
    print('Saving at : {}'.format(saving_path))
    pickle.dump(data_extracted, open(saving_path, 'wb'))


class Extractor(object):
    def __init__(self, file_to_extract):
        """
        From a .bag file, extract all its messages in to a python object.
        :param file_to_extract:
        """
        self.file_to_extract = file_to_extract

    def _print_topics_types(self, bags_paths, nb_msg=1):
        """
        Print topics and info of bags located in the bags_paths variable.
        :param bags_paths: List of string containing the paths of bags to be analyzed.
        :param nb_msg: The number of messages to print. Useful to not spam the python console.
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
                    p.update_pgr()
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
                if topic == 'mavros/imu/data':
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

    def extract(self):
        """
        Returns a dict of all the data from a rosbag object
        :return: Dict[topic][message_number] = dict of message data depending of its type.
        """
        bag = rosbag.Bag(self.file_to_extract)
        print('Extracting {}'.format(self.file_to_extract))
        print('List of topic to be extracted :')
        for topic, (msg_type, message_count, connections, frequency) in bag.get_type_and_topic_info()[1].items():
            print('    - {} | Message count : {}'.format(topic, message_count))
        print()

        extracted_data = {}
        for topic, (msg_type, message_count, connections, frequency) in bag.get_type_and_topic_info()[1].items():
            # Initialization
            print('Extracting : {}'.format(topic))
            extracted_data[topic] = {}
            p = Progress(max_iter=message_count, end_print='\n')  # Progress print

            # Choosing correct processing function according to data type
            processing_func = None
            if msg_type == 'sensor_msgs/Image':
                processing_func = _process_sensor_msgs_image
            elif msg_type == 'sensor_msgs/CameraInfo':
                processing_func = _process_sensor_msgs_camera_info
            assert processing_func is not None, 'Message type not not known.'

            # Processing
            for i, (_, msg_raw, t) in enumerate(bag.read_messages(topics=topic)):
                extracted_data[topic][i] = processing_func(msg_raw)
                extracted_data[topic][i]['t'] = t.to_sec()

                p.update_pgr()  # # Progress print
        bag.close()
        return extracted_data


def _process_sensor_msgs_image(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : 'sensor_msgs/Image'
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    # Getting image
    msg_clear = repr(msg)
    pixel_values = re.search('data: .(.*).', msg_clear).group(1)
    data = np.fromstring(pixel_values, dtype=np.uint8, sep=', ')

    # Size of the data
    img = data.reshape((msg.height, msg.width))
    return {'image': img}


def _process_sensor_msgs_camera_info(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : 'sensor_msgs/CameraInfo'
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    # Getting matrices
    return {'D': np.array(msg.D),
            'K': np.array(msg.K).reshape((3, 3)),
            'P': np.array(msg.P).reshape((3, 4)),
            'R': np.array(msg.R).reshape((3, 3))}


def object_analysis(obj):
    """
    Returns all attributes and methods of an object and classify them by
    (variable, method, public, private, protected).
    :param obj: object. The object to by analysed.
    :return: Dict[str, Dict[str, List]]
        Returns a dict of 2 dicts :
        -'variable'= dict
        -'method'= dict
        and for each of those dicts 3 lists:
        public, private, or protected.
    """
    res = {'variable': {'public': [], 'private': [], 'protected': []},
           'method': {'public': [], 'private': [], 'protected': []}}
    var_n_methods = sorted(dir(obj))
    for e in var_n_methods:
        try:
            if callable(getattr(obj, e)):
                attribute_or_method = 'method'
            else:
                attribute_or_method = 'variable'
            if len(e) > 1:
                if e[0] + e[1] == '__':
                    res[attribute_or_method]['private'].append(e)
                    continue
            if e[0] == '_':
                res[attribute_or_method]['protected'].append(e)
            else:
                res[attribute_or_method]['public'].append(e)
        except AttributeError:
            print('Attribute : {} of object {} is listed by dir() but cannot be accessed.'.format(e, type(obj)))
    return res


class Progress(object):
    def __init__(self, max_iter, end_print=''):
        """
        Easy way to progress print a for loop for example.
        :param max_iter: Maximum iteration of the loop.
        :param end_print: String that will be printed at the end of the file.
        """
        self.max_iter = float(max_iter)
        self.end_print = end_print
        self.iter = float(0)
        self.initial_time = time.time()

    def update_pgr(self, iteration=None):
        """
        Update the print.
        :param iteration: The actual iteration if there has been more iteration passed than 1.
        """
        if iteration is not None:
            self.iter = iteration
        # Progression
        print('\rProgression : {:0.02f}% | Time passed : {:.03f}s'.format((self.iter + 1) * 100 / self.max_iter,
                                                                          time.time() - self.initial_time), end='')
        sys.stdout.flush()
        if self.iter + 1 == self.max_iter:
            print(self.end_print)
        self.iter += 1


if __name__ == '__main__':
    main()
