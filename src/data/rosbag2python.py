from __future__ import print_function
import glob
import numpy as np
import pickle
import os
import re
import rosbag
import sys
import time

# This code must be executed with python 2.7 because of the rosbag package.
major, _, _, _, _ = sys.version_info
assert major == 2


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
            print("Topic : {} | Frequency : {:0.03f}".format(topic, freq))

    def show_topics(self):
        bag = rosbag.Bag(self.file_to_extract)
        print("Extracting {}".format(self.file_to_extract))
        print("List of topic to be extracted :")
        for topic, (msg_type, message_count, connections, frequency) in bag.get_type_and_topic_info()[1].items():
            print("    - {} | Type : {} | Message count : {}".format(topic, msg_type, message_count))

    def extract(self):
        """
        Returns a dict of all the data from a rosbag object
        :return: Dict[topic][message_number] = dict of message data depending of its type.
        """
        self.show_topics()

        bag = rosbag.Bag(self.file_to_extract)
        extracted_data = {}
        for topic, (msg_type, message_count, connections, frequency) in bag.get_type_and_topic_info()[1].items():
            # Initialization
            print("Extracting : {}".format(topic))
            main_key = topic + "__" + msg_type
            extracted_data[main_key] = {}
            p = Progress(max_iter=message_count, end_print="\n")  # Progress print

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

            # Processing
            for i, (_, msg_raw, t) in enumerate(bag.read_messages(topics=topic)):
                extracted_data[main_key][i] = processing_func(msg_raw)

                # assert extracted_data[main_key][i].get("tan") is None, ""tan" key already used."
                # extracted_data[main_key][i]["tan"] = t.to_sec()

                assert extracted_data[main_key][i].get("secs") is None, "'secs' key already used."
                assert extracted_data[main_key][i].get("nsecs") is None, "'nsecs' key already used."
                extracted_data[main_key][i]["secs"] = t.secs
                extracted_data[main_key][i]["nsecs"] = t.nsecs
                p.update_pgr()  # # Progress print
        bag.close()
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
    return 0


def main_data_drones():
    """
    Load rosbag file and save (pickle dump) it in python form. Adaptable to any message type.
    """
    # Data to extract parameters
    abs_path = "//"
    data_folder = "data_drone2"
    for flight_number in [3, 4]:
        pattern = ".*(/tara.*)"

        flights = []
        if data_folder == "data_drone":
            flights = ["20200214_vol_1_Exterieur_MaintientPositionPX4",
                       "20200214_vol_2_Exterieur_MaintientPositionPX4",
                       "20200218_vol_1_Exterieur",
                       "20200218_vol_2_Exterieur",
                       "20200218_vol_3_Exterieur",
                       "20200218_vol_4_Exterieur",
                       "20200218_vol_5_Exterieur",
                       "20200218_vol_6_Exterieur",
                       "20200219_vol_1_Exterieur",
                       "20200219_vol_2_Exterieur",
                       "20200219_vol_3_Exterieur",
                       "20200220_vol_1_Exterieur"]
        elif data_folder == "data_drone2":
            flights = ["vol_1", "vol_2", "vol_3", "vol_4_poubelle", "vol_5_poubelle"]

        # Locate the file to be extracted
        flight_path = abs_path + data_folder + "/raw/" + flights[flight_number] + "/"
        bags = []
        for bag in sorted(glob.glob(flight_path + "*.bag")):
            bags.append(bag)
        file_to_extract = [e for e in bags if re.search(pattern, e) is not None]
        assert len(file_to_extract) == 1, FileNotFoundError
        file_to_extract = file_to_extract[0]

        # Create saving path
        filename = file_to_extract.split("/")[-1].split(".")[0]
        saving_path = abs_path + data_folder + "/raw_python/" + flights[flight_number] + "/" + filename + ".pkl"

        # Extract data
        e = Extractor(file_to_extract)
        data_extracted = e.extract()

        # Saving file
        print("Saving at : {}".format(saving_path))
        pickle.dump(data_extracted, open(saving_path, "wb"))
        if "tara/left/image_raw" in data_extracted:
            del data_extracted["tara/right/image_raw"]
            del data_extracted["tara/right/camera_info"]
            pickle.dump(data_extracted, open(saving_path.replace("tara", "tara_left"), "wb"))


def main_euroc_mav():
    """
    Load rosbag file and save (pickle dump) it in python form. Adaptable to any message type.
    """
    # Data to extract parameters
    loading_path = "/Users/quentin/phd/turbulence/data/euroc_mav/raw/"
    saving_folder = "/Users/quentin/phd/turbulence/data/euroc_mav/raw_python/"
    flight_list = ["V1_01_easy",
                   "V1_02_medium",
                   "V1_03_difficult",
                   "V2_01_easy",
                   "V2_02_medium",
                   "V2_03_difficult"]
    for flight in flight_list:
        # Create saving path
        path = loading_path + flight + "/" + flight + ".bag"
        saving_path = saving_folder + flight + "/" + "data.pkl"

        # Extract data
        e = Extractor(path)
        # e.show_topics()
        data_extracted = e.extract()

        # Saving file
        print("Saving at : {}".format(saving_path))
        pickle.dump(data_extracted, open(saving_path, "wb"))


def main_blackbird():
    """
    Load rosbag file and save (pickle dump) it in python form. Adaptable to any message type.
    """
    dataset_path = os.getcwd() + "/data/blackbird/"

    flight_combinations = {"3dFigure8": {"yawConstant": [0.5, 1, 2, 3, 4, 5],
                                         "yawForward": []},
                           "ampersand": {"yawConstant": [1, 2, 3],
                                         "yawForward": [1, 2]},
                           "bentDice": {"yawConstant": [1, 2, 3, 4],
                                        "yawForward": [0.5, 1, 2, 3]},
                           "clover": {"yawConstant": [1, 2, 3, 4, 5, 6],
                                      "yawForward": [0.5, 1, 2, 3, 4, 5]},
                           "dice": {"yawConstant": [2, 3, 4],
                                    "yawForward": [1, 2, 3]},
                           "egg": {"yawConstant": [13.8],
                                   "yawForward": [3, 4, 5, 6, 7, 8, 9]},
                           "figure8": {"yawConstant": [0.5, 1, 2, 3.25, 5],
                                       "yawForward": []},
                           "halfMoon": {"yawConstant": [1, 2, 3, 4],
                                        "yawForward": [1, 1.5, 2, 3, 4, "maxSpeed1p0_run_2"]},
                           "mouse": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                     "yawForward": [0.5, 1, 2, 3, 4, 5, 6, 7, "maxSpeed0p5_run_2"]},
                           "oval": {"yawConstant": [2, 3, 4],
                                    "yawForward": [1, 2, 3, 4]},
                           "patrick": {"yawConstant": [1, 2, 3, 4, 5],
                                       "yawForward": [0.5, 1, 2, 3, 4]},
                           "picasso": {"yawConstant": [0.5, 1, 2, 3, 4, 5, 6],
                                       "yawForward": [0.5, 1, 3, 4, 5]},
                           "sid": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                   "yawForward": [0.5, 1, 2, 3, 4, 5]},
                           "sphinx": {"yawConstant": [1, 2, 3, 4],
                                      "yawForward": [1, 2, 3, 4]},
                           "star": {"yawConstant": [1, 2, 3, 4, 5, "maxSpeed4p0_run_2"],
                                    "yawForward": [0.5, 1, 2, 3, 4, 5]},
                           "thrice": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                      "yawForward": [0.5, 1, 2, 3, 4, 5, 6]},
                           "tiltedThrice": {"yawConstant": [1, 2, 3, 4, 5, 6, 7],
                                            "yawForward": [0.5, 1, 2, 3, 4, 5, 6]},
                           "winter": {"yawConstant": [1, 2, 3, 4, 5],
                                      "yawForward": [0.5, 2, 3, 4, "maxSpeed3p0_run_2"]}
                           }

    flight_list = []
    for flight_name, traj in flight_combinations.items():
        for flight_yaw, speeds in traj.items():
            for speed in speeds:
                if type(speed) == int:
                    flight_speed = "maxSpeed{}p0".format(speed)
                elif type(speed) == float:
                    unit, dec = str(speed).split(".")
                    flight_speed = "maxSpeed{}p{}".format(int(unit), int(dec))
                else:
                    # speed must be str
                    flight_speed = speed
                flight_list.append("{}/{}/{}/".format(flight_name, flight_yaw, flight_speed))
    # ["clover/yawForward/maxSpeed0p5/",
    #  "clover/yawForward/maxSpeed1p0/",
    #  "clover/yawForward/maxSpeed2p0/",
    #  "clover/yawForward/maxSpeed3p0/",
    #  "clover/yawForward/maxSpeed4p0/",
    #  "clover/yawForward/maxSpeed5p0/",
    #  "clover/yawConstant/maxSpeed1p0/",
    #  "clover/yawConstant/maxSpeed2p0/",
    #  "clover/yawConstant/maxSpeed3p0/",
    #  "clover/yawConstant/maxSpeed4p0/",
    #  "clover/yawConstant/maxSpeed5p0/",
    #  "clover/yawConstant/maxSpeed6p0/",
    #  "halfMoon/yawForward/maxSpeed1p0/",
    #  "halfMoon/yawForward/maxSpeed1p5/",
    #  "halfMoon/yawForward/maxSpeed2p0/",
    #  "halfMoon/yawForward/maxSpeed3p0/",
    #  "halfMoon/yawForward/maxSpeed4p0/",
    #  "halfMoon/yawForward/maxSpeed1p0_run_2/",
    #  "halfMoon/yawConstant/maxSpeed1p0/",
    #  "halfMoon/yawConstant/maxSpeed2p0/",
    #  "halfMoon/yawConstant/maxSpeed3p0/",
    #  "halfMoon/yawConstant/maxSpeed4p0/",
    #  "dice/yawForward/maxSpeed1p0/",
    #  "dice/yawForward/maxSpeed2p0/",
    #  "dice/yawForward/maxSpeed3p0/",
    #  "dice/yawConstant/maxSpeed2p0/",
    #  "dice/yawConstant/maxSpeed3p0/",
    #  "dice/yawConstant/maxSpeed4p0/",
    #  "bentDice/yawForward/maxSpeed0p5/",
    #  "bentDice/yawForward/maxSpeed1p0/",
    #  "bentDice/yawForward/maxSpeed2p0/",
    #  "bentDice/yawForward/maxSpeed3p0/",  14h12
    #  "bentDice/yawConstant/maxSpeed1p0/",
    #  "bentDice/yawConstant/maxSpeed2p0/",
    #  "bentDice/yawConstant/maxSpeed3p0/",
    #  "bentDice/yawConstant/maxSpeed4p0/",
    #  "mouse/yawForward/maxSpeed0p5/",
    #  "mouse/yawForward/maxSpeed1p0/",
    #  "mouse/yawForward/maxSpeed2p0/",
    #  "mouse/yawForward/maxSpeed3p0/",
    #  "mouse/yawForward/maxSpeed4p0/",
    #  "mouse/yawForward/maxSpeed5p0/",
    #  "mouse/yawForward/maxSpeed6p0/",
    #  "mouse/yawForward/maxSpeed7p0/",
    #  "mouse/yawForward/maxSpeed0p5_run_2/",
    #  "mouse/yawConstant/maxSpeed1p0/",
    #  "mouse/yawConstant/maxSpeed2p0/",
    #  "mouse/yawConstant/maxSpeed3p0/",
    #  "mouse/yawConstant/maxSpeed4p0/",
    #  "mouse/yawConstant/maxSpeed5p0/",
    #  "mouse/yawConstant/maxSpeed6p0/",
    #  "mouse/yawConstant/maxSpeed7p0/",
    #  "sid/yawForward/maxSpeed0p5/",
    #  "sid/yawForward/maxSpeed1p0/",
    #  "sid/yawForward/maxSpeed2p0/",
    #  "sid/yawForward/maxSpeed3p0/",
    #  "sid/yawForward/maxSpeed4p0/",
    #  "sid/yawForward/maxSpeed5p0/",
    #  "sid/yawConstant/maxSpeed1p0/",
    #  "sid/yawConstant/maxSpeed2p0/",
    #  "sid/yawConstant/maxSpeed3p0/",
    #  "sid/yawConstant/maxSpeed4p0/",
    #  "sid/yawConstant/maxSpeed5p0/",
    #  "sid/yawConstant/maxSpeed6p0/",
    #  "sid/yawConstant/maxSpeed7p0/",
    #  "star/yawForward/maxSpeed0p5/",
    #  "star/yawForward/maxSpeed1p0/",
    #  "star/yawForward/maxSpeed2p0/",
    #  "star/yawForward/maxSpeed3p0/",
    #  "star/yawForward/maxSpeed4p0/",
    #  "star/yawForward/maxSpeed5p0/",
    #  "star/yawConstant/maxSpeed1p0/",
    #  "star/yawConstant/maxSpeed2p0/",
    #  "star/yawConstant/maxSpeed3p0/",
    #  "star/yawConstant/maxSpeed4p0/",
    #  "star/yawConstant/maxSpeed5p0/",
    #  "star/yawConstant/maxSpeed4p0_run_2/",
    #  "thrice/yawForward/maxSpeed0p5/",
    #  "thrice/yawForward/maxSpeed1p0/",
    #  "thrice/yawForward/maxSpeed2p0/",
    #  "thrice/yawForward/maxSpeed3p0/",
    #  "thrice/yawForward/maxSpeed4p0/",
    #  "thrice/yawForward/maxSpeed5p0/",
    #  "thrice/yawForward/maxSpeed6p0/",
    #  "thrice/yawConstant/maxSpeed1p0/",
    #  "thrice/yawConstant/maxSpeed2p0/",
    #  "thrice/yawConstant/maxSpeed3p0/",
    #  "thrice/yawConstant/maxSpeed4p0/",
    #  "thrice/yawConstant/maxSpeed5p0/",
    #  "thrice/yawConstant/maxSpeed6p0/",
    #  "thrice/yawConstant/maxSpeed7p0/",
    #  "winter/yawForward/maxSpeed0p5/",
    #  "winter/yawForward/maxSpeed2p0/",
    #  "winter/yawForward/maxSpeed3p0/",
    #  "winter/yawForward/maxSpeed4p0/",
    #  "winter/yawForward/maxSpeed3p0_run_2/",
    #  "winter/yawConstant/maxSpeed1p0/",
    #  "winter/yawConstant/maxSpeed2p0/",
    #  "winter/yawConstant/maxSpeed3p0/",
    #  "winter/yawConstant/maxSpeed4p0/",
    #  "winter/yawConstant/maxSpeed5p0/",
    #  "tiltedThrice/yawForward/maxSpeed0p5/",
    #  "tiltedThrice/yawForward/maxSpeed1p0/",
    #  "tiltedThrice/yawForward/maxSpeed2p0/",
    #  "tiltedThrice/yawForward/maxSpeed3p0/",
    #  "tiltedThrice/yawForward/maxSpeed4p0/",
    #  "tiltedThrice/yawForward/maxSpeed5p0/",
    #  "tiltedThrice/yawForward/maxSpeed6p0/",
    #  "tiltedThrice/yawConstant/maxSpeed1p0/",
    #  "tiltedThrice/yawConstant/maxSpeed2p0/",
    #  "tiltedThrice/yawConstant/maxSpeed3p0/",
    #  "tiltedThrice/yawConstant/maxSpeed4p0/",
    #  "tiltedThrice/yawConstant/maxSpeed5p0/",
    #  "tiltedThrice/yawConstant/maxSpeed6p0/",
    #  "tiltedThrice/yawConstant/maxSpeed7p0/",
    #  "ampersand/yawForward/maxSpeed1p0/",
    #  "ampersand/yawForward/maxSpeed2p0/",
    #  "ampersand/yawConstant/maxSpeed1p0/",
    #  "ampersand/yawConstant/maxSpeed2p0/",
    #  "ampersand/yawConstant/maxSpeed3p0/",
    #  "3dFigure8/yawConstant/maxSpeed0p5/",
    #  "3dFigure8/yawConstant/maxSpeed1p0/",
    #  "3dFigure8/yawConstant/maxSpeed2p0/",
    #  "3dFigure8/yawConstant/maxSpeed3p0/",
    #  "3dFigure8/yawConstant/maxSpeed4p0/",
    #  "3dFigure8/yawConstant/maxSpeed5p0/",
    #  "patrick/yawForward/maxSpeed0p5/",
    #  "patrick/yawForward/maxSpeed1p0/",
    #  "patrick/yawForward/maxSpeed2p0/",
    #  "patrick/yawForward/maxSpeed3p0/",
    #  "patrick/yawForward/maxSpeed4p0/",
    #  "patrick/yawConstant/maxSpeed1p0/",
    #  "patrick/yawConstant/maxSpeed2p0/",
    #  "patrick/yawConstant/maxSpeed3p0/",
    #  "patrick/yawConstant/maxSpeed4p0/",
    #  "patrick/yawConstant/maxSpeed5p0/",
    #  "oval/yawForward/maxSpeed1p0/",
    #  "oval/yawForward/maxSpeed2p0/",
    #  "oval/yawForward/maxSpeed3p0/",
    #  "oval/yawForward/maxSpeed4p0/",
    #  "oval/yawConstant/maxSpeed2p0/",
    #  "oval/yawConstant/maxSpeed3p0/",
    #  "oval/yawConstant/maxSpeed4p0/",
    #  "picasso/yawForward/maxSpeed0p5/",
    #  "picasso/yawForward/maxSpeed1p0/",
    #  "picasso/yawForward/maxSpeed3p0/",
    #  "picasso/yawForward/maxSpeed4p0/",
    #  "picasso/yawForward/maxSpeed5p0/",
    #  "picasso/yawConstant/maxSpeed0p5/",
    #  "picasso/yawConstant/maxSpeed1p0/",
    #  "picasso/yawConstant/maxSpeed2p0/",
    #  "picasso/yawConstant/maxSpeed3p0/",
    #  "picasso/yawConstant/maxSpeed4p0/",
    #  "picasso/yawConstant/maxSpeed5p0/",
    #  "picasso/yawConstant/maxSpeed6p0/",
    #  "egg/yawForward/maxSpeed3p0/",
    #  "egg/yawForward/maxSpeed4p0/",
    #  "egg/yawForward/maxSpeed5p0/",
    #  "egg/yawForward/maxSpeed6p0/",
    #  "egg/yawForward/maxSpeed7p0/",
    #  "egg/yawForward/maxSpeed8p0/",
    #  "egg/yawForward/maxSpeed9p0/",
    #  "egg/yawConstant/maxSpeed13p8/",
    #  "sphinx/yawForward/maxSpeed1p0/",
    #  "sphinx/yawForward/maxSpeed2p0/",
    #  "sphinx/yawForward/maxSpeed3p0/",
    #  "sphinx/yawForward/maxSpeed4p0/",
    #  "sphinx/yawConstant/maxSpeed1p0/",
    #  "sphinx/yawConstant/maxSpeed2p0/",
    #  "sphinx/yawConstant/maxSpeed3p0/",
    #  "sphinx/yawConstant/maxSpeed4p0/",
    #  "figure8/yawConstant/maxSpeed0p5/",
    #  "figure8/yawConstant/maxSpeed1p0/",
    #  "figure8/yawConstant/maxSpeed2p0/",
    #  "figure8/yawConstant/maxSpeed3p25/",
    #  "figure8/yawConstant/maxSpeed5p0/"]

    # Flight loading
    # Data loading path
    # flight_name = "mouse"  # "ampersand"
    # flight_yaw = "yawForward"
    # flight_speed = "maxSpeed7p0"  # "maxSpeed1p0"

    # raw_path = dataset_path + "{}/{}/{}/{}/".format("raw", flight_name, flight_yaw, flight_speed)
    # intermediate_path = dataset_path + "{}/{}/{}/{}/".format("intermediate", flight_name,  # data saving folder
    #                                                          flight_yaw, flight_speed)

    for flight in flight_list:
        raw_path = dataset_path + "raw/" + flight
        intermediate_path = dataset_path + "intermediate/" + flight
        rosbag_path = raw_path + "rosbag.bag"
        saving_path = intermediate_path + "0_rosbag.pkl"

        # # Calibration loading
        # rosbag_path = dataset_path + "raw/cameraCalibration/imuCameraCalibration.bag"
        # intermediate_path = dataset_path + "intermediate/cameraCalibration/"
        # saving_path = intermediate_path + "imuCameraCalibration.pkl"

        if not os.path.exists(intermediate_path):
            os.makedirs(intermediate_path)

        # Extract data
        e = Extractor(rosbag_path)
        data_extracted = e.extract()

        # Saving processed data
        print("Saving at : {}".format(saving_path))
        pickle.dump(data_extracted, open(saving_path, "wb"))
        print()


def object_analysis(obj):
    """
    Returns all attributes and methods of an object and classify them by
    (variable, method, public, private, protected).
    :param obj: object. The object to by analysed.
    :return: Dict[str, Dict[str, List]]
        Returns a dict of 2 dicts :
        -"variable"= dict
        -"method"= dict
        and for each of those dicts 3 lists:
        public, private, or protected.
    """
    res = {"variable": {"public": [], "private": [], "protected": []},
           "method": {"public": [], "private": [], "protected": []}}
    var_n_methods = sorted(dir(obj))
    for e in var_n_methods:
        try:
            if callable(getattr(obj, e)):
                attribute_or_method = "method"
            else:
                attribute_or_method = "variable"
            if len(e) > 1:
                if e[0] + e[1] == "__":
                    res[attribute_or_method]["private"].append(e)
                    continue
            if e[0] == "_":
                res[attribute_or_method]["protected"].append(e)
            else:
                res[attribute_or_method]["public"].append(e)
        except AttributeError:
            print("Attribute : {} of object {} is listed by dir() but cannot be accessed.".format(e, type(obj)))
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
        print("\rProgression : {:0.02f}% | Time passed : {:.03f}s".format((self.iter + 1) * 100 / self.max_iter,
                                                                          time.time() - self.initial_time), end='')
        sys.stdout.flush()
        if self.iter + 1 == self.max_iter:
            print(self.end_print)
        self.iter += 1


if __name__ == "__main__":
    # main_data_drones()
    main_euroc_mav()
    # main_blackbird()
