import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import random
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    # name = "blackbird_slash_pose_ref"
    # with open(raw_flight_path + f"csv/{name}.csv", newline='') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     header = next(csv_reader)
    #     print(f"{name}: {header}")
    #     data = [e for e in csv_reader]
    #     assert len(data[0]) == 17
    #     time = np.array([float(e[0]) * 1e-9 for e in data])
    #     trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
    #     quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
    #     state_blackbird_slash_pose_ref = np.concatenate([trans, quat], axis=-1)
    #     tf = Transform.from_quat_trans(quat, trans).as_matrix()
    #     blackbird_slash_pose_ref = {"t": time, "tf": tf}

    # name = "tf"
    # with open(raw_flight_path + f"csv/{name}.csv", newline='') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     header = next(csv_reader)
    #     print(f"{name}: {header}")
    #     data = [e for e in csv_reader]
    #     assert len(data[0]) == 19
    #     time = np.array([float(e[0]) * 1e-9 for e in data])
    #     trans = np.array([np.array([float(e[11]), float(e[12]), float(e[13])]) for e in data])
    #     quat = np.array([np.array([float(e[15]), float(e[16]), float(e[17]), float(e[18])]) for e in data])
    #     state_blackbird_tf = np.concatenate([trans, quat], axis=-1)
    #     tf = Transform.from_quat_trans(quat, trans).as_matrix()
    #     blackbird_tf = {"t": time, "tf": tf}

    # name = "groundTruthPoses"
    # with open(raw_flight_path + f"{name}.csv", newline='') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     data = [e for e in csv_reader]
    #     assert len(data[0]) == 8
    #     time = np.array([float(e[0]) * 1e-9 for e in data])
    #     trans = np.array([np.array([float(e[1]), float(e[2]), float(e[3])]) for e in data])
    #     quat = np.array([np.array([float(e[5]), float(e[6]), float(e[7]), float(e[4])]) for e in data])
    #     state_ground_truth_poses = np.concatenate([trans, quat], axis=-1)
    #     tf = Transform.from_quat_trans(quat, trans).as_matrix()
    #     ground_truth_poses = {"t": time, "tf": tf}

    # is_equal1 = (state_blackbird_slash_pose_ref == state_blackbird_slash_state).all()
    # is_equal2 = (state_blackbird_slash_state == state_blackbird_tf).all()
    # print(is_equal1)
    # print(is_equal2)

    # blackbird_slash_pose_ref is the pose_ref given to command the drone at a high level.
    # blackbird_slash_state is the pose of the drone measured by the motion capture.
    # tf is the pose of the drone measured by the motion capture, exactly the same as blackbird_slash_state.
    # groundTruthPoses is the same as tf and blackbird_slash_state but with less accuracy

    name = "blackbird_slash_imu"
    with open(raw_flight_path + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        # print(f"{name}: {header}")
        data = [e for e in csv_reader]
        assert len(data[0]) == 23
        time = np.array([float(e[0]) * 1e-9 for e in data])
        angular_velocity = np.array([np.array([float(e[14]), float(e[15]), float(e[16])]) for e in data])
        linear_acceleration = np.array([np.array([float(e[19]), float(e[20]), float(e[21])]) for e in data])
        orientation = np.array([np.array([float(e[8]), float(e[9]), float(e[10]), float(e[11])]) for e in data])
        assert (np.linalg.norm(orientation, axis=-1) == 0).all()  # Orientation is empty and not normed.
        ang_acc = np.concatenate([angular_velocity, linear_acceleration], axis=1)
        imu_dict = {"t": time, "ang_acc": ang_acc}

    name = "blackbird_slash_rotor_rpm"
    with open(raw_flight_path + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        # print(f"{name}: {header}")
        data = [e for e in csv_reader]
        assert len(data[0]) == 17
        time0 = np.array([float(e[8]) + float(e[9]) * 1e-9 for e in data])
        time1 = np.array([float(e[10]) + float(e[11]) * 1e-9 for e in data])
        time2 = np.array([float(e[12]) + float(e[13]) * 1e-9 for e in data])
        time3 = np.array([float(e[14]) + float(e[15]) * 1e-9 for e in data])
        motors = np.array([np.array(e[16].strip("[]").split(","), dtype=float) for e in data]) * np.pi / 30
        motors_dict = {"m0": np.stack([time0, motors[:, 0]], 1),
                       "m1": np.stack([time0, motors[:, 1]], 1),
                       "m2": np.stack([time0, motors[:, 2]], 1),
                       "m3": np.stack([time0, motors[:, 3]], 1)}

    name = "blackbird_slash_state"
    with open(raw_flight_path + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        # print(f"{name}: {header}")
        data = [e for e in csv_reader]
        assert len(data[0]) == 17
        time = np.array([float(e[0]) * 1e-9 for e in data])
        trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
        trans = trans  # reset initial position
        quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
        quat_trans = np.concatenate([quat, trans], axis=1)
        pose_dict = {"t": time, "quat_trans": quat_trans}

    # Computing state
    angles = Transform.from_quat_trans(quat, trans).as_rot_euler("zyx", False)[:, ::-1]

    dt = (pose_dict["t"][1:] - pose_dict["t"][:-1]).reshape((-1, 1))

    # Computing angular velocity
    dq_dt = (quat[1:] - quat[:-1]) / dt
    dq_dt = np.concatenate([np.zeros((1, 4)), dq_dt], axis=0)
    dq_dt_convention = dq_dt[:, [3, 0, 1, 2]]
    q0, q1, q2, q3 = quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]
    g_matrix = np.array([[-q1, q0, q3, -q2],
                         [-q2, -q3, q0, q1],
                         [-q3, q2, -q1, q0]]).transpose((2, 0, 1))

    angular_velocity = (2 * g_matrix @ dq_dt_convention.reshape((-1, 4, 1))).squeeze()

    # Computing linear speed
    linear_speed = (trans[1:] - trans[:-1]) / dt
    linear_speed = np.concatenate([np.zeros((1, 3)), linear_speed], axis=0)
    linear_speed = (Rotation.from_quat(quat).inv().as_matrix() @ linear_speed.reshape((-1, 3, 1))).squeeze()

    state = np.concatenate([angles, angular_velocity, linear_speed, trans], axis=1)

    # State Euler derivation
    dstate = (state[1:] - state[:-1]) / dt
    dstate = np.concatenate([np.zeros((1, 12)), dstate], axis=0)

    # Synchronisation
    imu_sync = np.stack([np.interp(x=pose_dict["t"],
                                   xp=imu_dict["t"],
                                   fp=imu_dict["ang_acc"][:, i]) for i in range(6)], axis=1)
    motor_sync = np.stack([np.interp(x=pose_dict["t"],
                                     xp=motors_dict[f"m{i}"][:, 0],
                                     fp=motors_dict[f"m{i}"][:, 1]) for i in range(4)], axis=1)
    synchronisation = {"t": pose_dict["t"], "imu_ang_vel": imu_sync, "motor": motor_sync,
                       "pose_quat_trans": pose_dict["quat_trans"], "state": state, "dstate": dstate}

    # Take data only when drone is flying
    start, stop = 0, 0
    has_started = False
    for i, ms in enumerate(synchronisation["motor"]):
        if i < 6000:
            continue
        if (ms > 1100).any() and not has_started:
            start = i
            has_started = True
        if has_started and (ms < 500).any():
            stop = i
            break
    assert start != stop
    [plt.plot(synchronisation["motor"][:, i]) for i in range(4)]

    # return start, stop
    cut_data = {}
    for key in synchronisation:
        cut_data[key] = synchronisation[key][start:stop]
    cut_data["t"] = cut_data["t"] - cut_data["t"][0]

    # Flight data keys:
    # ['t', 'imu_ang_vel', 'motor', 'pose_quat_trans', 'state', 'dstate']
    # state:12, motor:4, imu_ang_vel:6, dstate:12, next_state:12 = 46
    res = np.concatenate((cut_data["state"][:-1],
                          cut_data["motor"][:-1],
                          cut_data["imu_ang_vel"][:-1],
                          cut_data["dstate"][1:],
                          cut_data["state"][1:]),
                         axis=1)

    save_path = intermediate_flight_path + "data.pkl"
    p = pathlib.Path(intermediate_flight_path)
    p.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(save_path, "wb"))


def building_data_and_splitting():
    torch_dataset_path = DATASET_PATH + f"intermediate_torch_dataset/"

    idx = list(range(len(FLIGHT_LIST)))
    random.shuffle(idx)
    idx = np.array(idx)
    train_validation_idx = round(len(idx) * 0.6)
    validation_test_idx = round(len(idx) * 0.8)

    train = []
    for flight in FLIGHT_LIST[:train_validation_idx]:
        intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
        train.append(pickle.load(open(intermediate_flight_path + "data.pkl", "rb")))
    train = torch.from_numpy(np.concatenate(train, axis=0)).float()
    torch.save(train, torch_dataset_path + "train.pt")

    validation = []
    for flight in FLIGHT_LIST[train_validation_idx:validation_test_idx]:
        intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
        validation.append(pickle.load(open(intermediate_flight_path + "data.pkl", "rb")))
    validation = torch.from_numpy(np.concatenate(validation, axis=0)).float()
    torch.save(validation, torch_dataset_path + "validation.pt")

    test = []
    for flight in FLIGHT_LIST[validation_test_idx:]:
        intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
        test.append(pickle.load(open(intermediate_flight_path + "data.pkl", "rb")))
    test = torch.from_numpy(np.concatenate(test, axis=0)).float()
    torch.save(test, torch_dataset_path + "test.pt")

    pickle.dump({"train_flights": FLIGHT_LIST[:train_validation_idx],
                 "validation_flights": FLIGHT_LIST[train_validation_idx:validation_test_idx],
                 "test_flights": FLIGHT_LIST[validation_test_idx:]},
                open(torch_dataset_path + "flight_split.pkl", "wb"))


class BlackbirdTrain(Dataset):
    def __init__(self, train_validation_test: str):
        torch_dataset_path = DATASET_PATH + f"intermediate_torch_dataset/"
        if train_validation_test not in ["train", "validation", "test"]:
            raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
        self.data = torch.load(torch_dataset_path + f"{train_validation_test}.pt")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        One sample is state:12, motor:4, imu_ang_vel:6, dstate:12, next_state:12 = 46
        """
        data = self.data[idx]
        # state = data[:12]
        # motor = data[12:16]
        # imu = data[16:22]
        # dstate = data[22:34]
        # next_state = data[34:]
        return data[:16], data[22:34]


def main():
    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    # flight_name, flight_yaw, flight_speed = 'egg', 'yawConstant', 'maxSpeed13p8'
    # flight_name, flight_yaw, flight_speed = 'halfMoon', 'yawForward', 'maxSpeed4p0'
    # raw_flight_path = DATASET_PATH + f"raw/{flight_name}/{flight_yaw}/{flight_speed}/"
    # intermediate_flight_path = DATASET_PATH + f"intermediate/{flight_name}/{flight_yaw}/{flight_speed}/"
    # print(f"/{flight_name}/{flight_yaw}/{flight_speed}/")
    # load_clean_data(raw_flight_path, intermediate_flight_path)

    # for i in tqdm(range(len(FLIGHT_LIST))):
    #     flight = FLIGHT_LIST[i]
    #     raw_flight_path = DATASET_PATH + f"raw/{flight}"
    #     intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
    #     load_clean_data(raw_flight_path, intermediate_flight_path)
    #
    # building_data_and_splitting()


if __name__ == '__main__':
    main()
