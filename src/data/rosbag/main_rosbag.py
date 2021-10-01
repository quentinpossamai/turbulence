from __future__ import print_function
import re
import glob
import pickle
import os

from src.data.rosbag.rosbag2python import Extractor


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


def main_hilti():
    extractor = Extractor('/Users/quentin/Downloads/Office_Mitte_1.bag')
    extractor.get_freq()
    # extractor.extract()


if __name__ == "__main__":
    # main_data_drones()
    # main_euroc_mav()
    # main_blackbird()
    main_hilti()
