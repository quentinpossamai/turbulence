from __future__ import print_function
import rosbag
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


def main_euroc_mav():
    """
    Load rosbag file and save (pickle dump) it in python form. Adaptable to any message type.
    """
    # Data to extract parameters
    abs_path = '/Users/quentin/phd/turbulence/'
    data_folder = 'euroc_mav'
    for flight_number, flight in enumerate(['V1_03_difficult']):
        pattern = '.*(/{}.*)'.format(flight)

        # Locate the file to be extracted
        flight_path = abs_path + data_folder + '/raw/' + flight + '/'
        bags = []
        for bag in sorted(glob.glob(flight_path + '*.bag')):
            bags.append(bag)
        file_to_extract = [e for e in bags if re.search(pattern, e) is not None]
        assert len(file_to_extract) == 1, FileNotFoundError
        file_to_extract = file_to_extract[0]

        # Create saving path
        filename = file_to_extract.split('/')[-1].split('.')[0]
        saving_path = abs_path + data_folder + '/raw_python/' + flight + '/' + filename + '.pkl'

        # Extract data
        e = Extractor(file_to_extract)
        data_extracted = e.extract()

        # Saving file
        print('Saving at : {}'.format(saving_path))
        pickle.dump(data_extracted, open(saving_path, 'wb'))
        if 'tara/left/image_raw' in data_extracted:
            del data_extracted['tara/right/image_raw']
            del data_extracted['tara/right/camera_info']
            pickle.dump(data_extracted, open(saving_path.replace('tara', 'tara_left'), 'wb'))


def main_data_drones():
    """
    Load rosbag file and save (pickle dump) it in python form. Adaptable to any message type.
    """
    # Data to extract parameters
    abs_path = '/Users/quentin/phd/turbulence/'
    data_folder = 'data_drone2'
    for flight_number in [3, 4]:
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
        assert len(file_to_extract) == 1, FileNotFoundError
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
        if 'tara/left/image_raw' in data_extracted:
            del data_extracted['tara/right/image_raw']
            del data_extracted['tara/right/camera_info']
            pickle.dump(data_extracted, open(saving_path.replace('tara', 'tara_left'), 'wb'))


class Extractor(object):
    def __init__(self, file_to_extract):
        """
        From a .bag file, extract all its messages in to a python object.
        :param file_to_extract:
        """
        self.file_to_extract = file_to_extract

    def get_freq(self):
        topic_freq = {}
        bag = rosbag.Bag(self.file_to_extract)
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
            print('    - {} | Type : {} | Message count : {}'.format(topic, msg_type, message_count))
        print()

        extracted_data = {}
        for topic, (msg_type, message_count, connections, frequency) in bag.get_type_and_topic_info()[1].items():
            # Initialization
            print('Extracting : {}'.format(topic))
            main_key = topic + '__' + msg_type
            extracted_data[main_key] = {}
            p = Progress(max_iter=message_count, end_print='\n')  # Progress print

            # Choosing correct processing function according to data type
            processing_func = None
            if msg_type == 'sensor_msgs/Image':
                processing_func = _process_sensor_msgs_image
            elif msg_type == 'sensor_msgs/CameraInfo':
                processing_func = _process_sensor_msgs_camera_info
            elif msg_type == 'geometry_msgs/TransformStamped':
                processing_func = _process_geometry_msgs_transform_stamped
            elif msg_type == 'asctec_hl_comm/MotorSpeed':
                processing_func = _process_asctec_hl_comm_motor_speed
            elif msg_type == 'sensor_msgs/Imu':
                processing_func = _process_sensor_msgs_imu
            assert processing_func is not None, 'Message type : {} not known.'.format(msg_type)

            # Processing
            for i, (_, msg_raw, t) in enumerate(bag.read_messages(topics=topic)):
                extracted_data[main_key][i] = processing_func(msg_raw)
                assert extracted_data[main_key][i].get('tan') is None, "'tan' key already used."
                extracted_data[main_key][i]['tan'] = t.to_sec()

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


def _process_geometry_msgs_transform_stamped(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : 'geometry_msgs/TransformStamped'.
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {'translation': np.array([msg.transform.translation.x,
                                     msg.transform.translation.y,
                                     msg.transform.translation.z]),
            'rotation': np.array([msg.transform.rotation.x,
                                  msg.transform.rotation.y,
                                  msg.transform.rotation.z,
                                  msg.transform.rotation.w])}


def _process_asctec_hl_comm_motor_speed(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : 'asctec_hl_comm/MotorSpeed'.
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {'motor_speed': np.array(msg.motor_speed)}


def _process_sensor_msgs_imu(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : 'sensor_msgs/Imu'.
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {'angular_velocity': np.array([msg.angular_velocity.x,
                                          msg.angular_velocity.y,
                                          msg.angular_velocity.z]),
            'angular_velocity_covariance': np.array(msg.angular_velocity_covariance).reshape((3, 3)),
            'linear_acceleration': np.array([msg.linear_acceleration.x,
                                             msg.linear_acceleration.y,
                                             msg.linear_acceleration.z]),
            'linear_acceleration_covariance': np.array(msg.linear_acceleration_covariance).reshape((3, 3)),
            'orientation': np.array([msg.orientation.x,
                                     msg.orientation.y,
                                     msg.orientation.z]),
            'orientation_covariance': np.array(msg.orientation_covariance).reshape((3, 3))}


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
    # main_data_drones()
    main_euroc_mav()
