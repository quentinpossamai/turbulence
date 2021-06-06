import csv
import numpy as np

from src.trace_trajectory import trace_trajectory
from src.utils.paths import WD_PATH
from src.utils.transform import Transform

DATASET_PATH = WD_PATH + "data/blackbird/"

FLIGHT_COMBINATIONS = {"3dFigure8": {"yawConstant": [0.5, 1, 2, 3, 4, 5],
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


def _get_flight_list():
    flight_list = []
    for flight_name, traj in FLIGHT_COMBINATIONS.items():
        for flight_yaw, speeds in traj.items():
            for speed in speeds:
                if type(speed) == int:
                    flight_speed = f"maxSpeed{speed}p0"
                elif type(speed) == float:
                    unit, dec = str(speed).split(".")
                    flight_speed = f"maxSpeed{int(unit)}p{int(dec)}"
                else:
                    # speed must be str
                    flight_speed = speed
                flight_list.append(f"{flight_name}/{flight_yaw}/{flight_speed}/")
        return flight_list


FLIGHT_LIST = _get_flight_list()

CSV_FILE = ["blackbird_slash_imu.csv",
            "blackbird_slash_pose_ref.csv",
            "blackbird_slash_rotor_rpm.csv",
            "blackbird_slash_state.csv",
            "tf.csv"]


def load_clean_data(raw_flight_path, intermediate_flight_path):
    # name = "blackbird_slash_imu"
    # with open(raw_flight_path + f"{name}.csv", newline='') as csv_file:
    #     # TODO
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     header = next(csv_reader)
    #     print(f"{name}: {header}")
    #     data = [e for e in csv_reader]
    #     assert len(data[0]) == 19
    #     time = np.array([float(e[0]) * 1e-9 for e in data])
    #     trans = np.array([np.array([float(e[12]), float(e[13]), float(e[14])]) for e in data])
    #     quat = np.array([np.array([float(e[16]), float(e[17]), float(e[18]), float(e[19])]) for e in data])
    #     tf = Transform.from_quat_trans(quat, trans).as_matrix()
    #     blackbird_slash_imu = {"t": time, "tf": tf}

    # name = "blackbird_slash_pose_ref"
    # with open(raw_flight_path + f"{name}.csv", newline='') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     header = next(csv_reader)
    #     print(f"{name}: {header}")
    #     data = [e for e in csv_reader]
    #     assert len(data[0]) == 17
    #     time = np.array([float(e[0]) * 1e-9 for e in data])
    #     trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
    #     quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
    #     np.array([quat, trans])
    #     tf = Transform.from_quat_trans(quat, trans).as_matrix()
    #     blackbird_slash_pose_ref = {"t": time, "tf": tf}

    # name = "blackbird_slash_rotor_rpm"
    # with open(raw_flight_path + f"{name}.csv", newline='') as csv_file:
    #     # TODO
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     header = next(csv_reader)
    #     print(f"{name}: {header}")
    #     data = [e for e in csv_reader]
    #     assert len(data[0]) == 19
    #     time = np.array([float(e[0]) * 1e-9 for e in data])
    #     trans = np.array([np.array([float(e[12]), float(e[13]), float(e[14])]) for e in data])
    #     quat = np.array([np.array([float(e[16]), float(e[17]), float(e[18]), float(e[19])]) for e in data])
    #     tf = Transform.from_quat_trans(quat, trans).as_matrix()
    #     blackbird_slash_rotor_rpm = {"t": time, "tf": tf}

    name = "blackbird_slash_state"
    with open(raw_flight_path + f"{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        print(f"{name}: {header}")
        data = [e for e in csv_reader]
        assert len(data[0]) == 17
        time = np.array([float(e[0]) * 1e-9 for e in data])
        trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
        quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
        tf = Transform.from_quat_trans(quat, trans).inv().inv().as_matrix()
        blackbird_slash_state = {"t": time, "tf": tf}

    # name = "tf"
    # with open(raw_flight_path + f"{name}.csv", newline='') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     header = next(csv_reader)
    #     print(f"{name}: {header}")
    #     data = [e for e in csv_reader]
    #     assert len(data[0]) == 19
    #     time = np.array([float(e[0]) * 1e-9 for e in data])
    #     trans = np.array([np.array([float(e[11]), float(e[12]), float(e[13])]) for e in data])
    #     quat = np.array([np.array([float(e[15]), float(e[16]), float(e[17]), float(e[18])]) for e in data])
    #     tf = Transform.from_quat_trans(quat, trans).as_matrix()
    #     blackbird_tf = {"t": time, "tf": tf}

    # blackbird_slash_imu
    # blackbird_slash_pose_ref
    # blackbird_slash_rotor_rpm
    # blackbird_slash_state
    # blackbird_tf
    trace_trajectory(position=trans, quaternion=quat)


def main():
    flight_name = "mouse"
    flight_yaw = "yawForward"
    flight_speed = "maxSpeed7p0"
    raw_flight_path = DATASET_PATH + f"raw/{flight_name}/{flight_yaw}/{flight_speed}/csv/"
    intermediate_flight_path = DATASET_PATH + f"intermediate/{flight_name}/{flight_yaw}/{flight_speed}/"
    print(f"/{flight_name}/{flight_yaw}/{flight_speed}/")
    load_clean_data(raw_flight_path, intermediate_flight_path)


if __name__ == '__main__':
    main()
