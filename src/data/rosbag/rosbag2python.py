from __future__ import print_function
import numpy as np
import re
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr
from tqdm import tqdm


class Extractor(object):
    def __init__(self, file_to_extract):
        """
        From a .bag file, extract all its messages in to a python object.
        :param file_to_extract:
        """
        self.file_to_extract = file_to_extract

    def get_freq(self):
        self.show_topics()
        freq = {}
        timestamps = {}
        with Reader(self.file_to_extract) as reader:
            for topic, topic_info in reader.topics.items():
                timestamps[topic] = {'t': np.zeros(topic_info.msgcount), 'idx': 0}
            for i, (connection, timestamp, raw_data) in enumerate(tqdm(reader.messages())):
                topic = connection.topic
                timestamps[topic]['t'][timestamps[topic]['idx']] = timestamp
                timestamps[topic]['idx'] += 1

            for topic, topic_info in reader.topics.items():
                assert any(np.isnan(timestamps[topic]['t'])) is False
                assert (timestamps[topic]['t'] == np.sort(timestamps[topic]['t'])).all()
                assert (timestamps[topic]['t'] != 0).any()
                period = (timestamps[topic]['t'][-1] - timestamps[topic]['t'][0]) * 1e-9
                if period == 0:
                    freq[topic] = 0
                else:
                    freq[topic] = topic_info.msgcount / period
        for topic, freq in freq.items():
            print(f"Topic : {topic} | Frequency : {freq:0.03f}")

    def show_topics(self):
        with Reader(self.file_to_extract) as reader:
            print(f"Extracting {self.file_to_extract}")
            print("List of topic to be extracted :")
            for topic, topic_info in reader.topics.items():
                print(f"    - {topic} | Type : {topic_info.msgtype} | Message count : {topic_info.msgcount}")

    def extract(self):
        """
        Returns a dict of all the data from a rosbag object
        :return: Dict[topic][message_number] = dict of message data depending of its type.
        """
        self.show_topics()

        with Reader(self.file_to_extract) as reader:
            extracted_data = {}
            # Initialization
            for topic, topic_info in reader.topics.items():
                print("Extracting : {}".format(topic))
                main_key = topic + "__" + topic_info.msgtype
                extracted_data[main_key] = {}
            # Processing
            # connections = [x for x in reader.connections.values() if x.topic == topic]
            for i, (connection, timestamp, raw_data) in enumerate(tqdm(reader.messages())):
                msg_type = topic_info.msgtype
                main_key = topic + "__" + topic_info.msgtype

                # Choosing correct processing function according to data type
                processing_func = None
                if msg_type == "sensor_msgs/Image":
                    processing_func = _process_sensor_msgs_image
                elif msg_type == "sensor_msgs/CameraInfo":
                    processing_func = _process_sensor_msgs_camera_info
                elif msg_type == "geometry_msgs/TransformStamped":
                    processing_func = _process_geometry_msgs_transform_stamped
                elif msg_type == "asctec_hl_comm/MotorSpeed":
                    processing_func = _process_asctec_hl_comm_motor_speed
                elif msg_type == "sensor_msgs/Imu":
                    processing_func = _process_sensor_msgs_imu
                elif msg_type == "blackbird/MotorPWM":
                    processing_func = _process_blackbird_motorpwm
                elif msg_type == "tf/tfMessage":
                    processing_func = _process_tf_tfmessage
                elif msg_type == "geometry_msgs/PoseStamped":
                    processing_func = _process_geometry_msgs_posestamped
                elif msg_type == "blackbird/MotorRPM":
                    processing_func = _process_blackbird_motorrpm
                elif msg_type == "tf2_msgs/TFMessage":
                    processing_func = _process_tf_tfmessage
                elif msg_type == "_process_geometry_msgs_point_stamped":
                    processing_func = _process_geometry_msgs_point_stamped
                assert processing_func is not None, "Message type : {} not known.".format(msg_type)

                msg = deserialize_cdr(ros1_to_cdr(raw_data, connection.msgtype), connection.msgtype)
                extracted_data[main_key][i] = processing_func(msg)

                assert extracted_data[main_key][i].get("secs") is None, "'secs' key already used."
                assert extracted_data[main_key][i].get("nsecs") is None, "'nsecs' key already used."
                extracted_data[main_key][i]["secs"] = msg.header.stamp.sec
                extracted_data[main_key][i]["nsecs"] = msg.header.stamp.nanosec
        return extracted_data


def _process_sensor_msgs_image(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "sensor_msgs/Image"
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    # Getting image
    msg_clear = repr(msg)
    pixel_values = re.search("data: .(.*).", msg_clear).group(1)
    data = np.fromstring(pixel_values, dtype=np.uint8, sep=", ")

    # Size of the data
    img = data.reshape((msg.height, msg.width))
    return {"image": img,
            "height": msg.height,
            "is_bigendian": msg.is_bigendian,
            "step": msg.step,
            "width": msg.width}


def _process_sensor_msgs_camera_info(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "sensor_msgs/CameraInfo"
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    # Getting matrices
    return {"D": np.array(msg.D),
            "K": np.array(msg.K).reshape((3, 3)),
            "P": np.array(msg.P).reshape((3, 4)),
            "R": np.array(msg.R).reshape((3, 3)),
            "binning_x": msg.binning_x,
            "binning_y": msg.binning_y,
            "distortion_model": msg.distortion_model,
            "height": msg.height,
            "width": msg.width,
            "roi_do_rectify": msg.roi.do_rectify,
            "roi_height": msg.roi.height,
            "roi_width": msg.roi.width,
            "roi_x_offset": msg.roi.x_offset,
            "roi_y_offset": msg.roi.y_offset,
            "frame_id": msg.header.frame_id}


def _process_sensor_msgs_imu(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "sensor_msgs/Imu".
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {"angular_velocity": np.array([msg.angular_velocity.x,
                                          msg.angular_velocity.y,
                                          msg.angular_velocity.z]),
            "angular_velocity_covariance": np.array(msg.angular_velocity_covariance).reshape((3, 3)),
            "linear_acceleration": np.array([msg.linear_acceleration.x,
                                             msg.linear_acceleration.y,
                                             msg.linear_acceleration.z]),
            "linear_acceleration_covariance": np.array(msg.linear_acceleration_covariance).reshape((3, 3)),
            "orientation": np.array([msg.orientation.x,
                                     msg.orientation.y,
                                     msg.orientation.z]),
            "orientation_covariance": np.array(msg.orientation_covariance).reshape((3, 3)),
            "frame_id": msg.header.frame_id}


def _process_geometry_msgs_transform_stamped(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "geometry_msgs/TransformStamped".
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {"translation": np.array([msg.transform.translation.x,
                                     msg.transform.translation.y,
                                     msg.transform.translation.z]),
            "rotation": np.array([msg.transform.rotation.x,
                                  msg.transform.rotation.y,
                                  msg.transform.rotation.z,
                                  msg.transform.rotation.w]),
            "child_frame_id": msg.child_frame_id,
            "frame_id": msg.header.frame_id}


def _process_tf_tfmessage(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "tf/tfMessage".
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    transform_object = msg.transforms[0].transform
    return {"translation": np.array([transform_object.translation.x,
                                     transform_object.translation.y,
                                     transform_object.translation.z]),
            "rotation": np.array([transform_object.rotation.x,
                                  transform_object.rotation.y,
                                  transform_object.rotation.z,
                                  transform_object.rotation.w]),
            "child_frame_id": msg.transforms[0].child_frame_id,
            "frame_id": msg.transforms[0].header.frame_id}


def _process_geometry_msgs_posestamped(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "geometry_msgs/PoseStamped".
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    transform_object = msg.pose
    return {"translation": np.array([transform_object.position.x,
                                     transform_object.position.y,
                                     transform_object.position.z]),
            "rotation": np.array([transform_object.orientation.x,
                                  transform_object.orientation.y,
                                  transform_object.orientation.z,
                                  transform_object.orientation.w]),
            "frame_id": msg.header.frame_id}


def _process_asctec_hl_comm_motor_speed(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "asctec_hl_comm/MotorSpeed".
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {"motor_speed": np.array(msg.motor_speed),
            "frame_id": msg.header.frame_id}


def _process_blackbird_motorpwm(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "blackbird/MotorPWM".
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {"motor_pwm": np.array(msg.pwm)}  # Motor"s Pulse Width Modulation


def _process_blackbird_motorrpm(msg):
    """
    Processing function to extract data from rosbag message to built-in python object.
    This function must be used for message of type : "blackbird/MotorRPM".
    :param msg: The rosbag message to extract.
    :return: Dict containing the useful data.
    """
    return {0: (msg.sample_stamp[0], msg.rpm[0]),
            1: (msg.sample_stamp[1], msg.rpm[1]),
            2: (msg.sample_stamp[2], msg.rpm[2]),
            3: (msg.sample_stamp[3], msg.rpm[3])}


def _process_geometry_msgs_point_stamped(msg):
    """
    geometry_msgs/PointStamped
    """
    return msg
