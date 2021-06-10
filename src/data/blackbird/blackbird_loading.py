import csv
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle
import random

from scipy.spatial.transform import Rotation
from scipy.interpolate import splrep, splev
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.trace_trajectory import trace_trajectory
from src.utils.paths import WD_PATH, ANALYSIS_PATH
from src.utils.transform import Transform

DATASET_PATH = WD_PATH + "data/blackbird/"
ANALYSIS_SECTION_PATH = ANALYSIS_PATH + "blackbird_processing_data/"

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
        imu_angular_velocity = np.array([np.array([float(e[14]), float(e[15]), float(e[16])]) for e in data])
        imu_linear_acceleration = np.array([np.array([float(e[19]), float(e[20]), float(e[21])]) for e in data])
        imu_orientation = np.array([np.array([float(e[8]), float(e[9]), float(e[10]), float(e[11])]) for e in data])
        assert (np.linalg.norm(imu_orientation, axis=-1) == 0).all()  # Orientation is empty and not normed.
        ang_acc = np.concatenate([imu_angular_velocity, imu_linear_acceleration], axis=1)
        imu_dict = {"t": time, "ang_acc": ang_acc}

    name = "blackbird_slash_rotor_rpm"
    with open(raw_flight_path + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        # print(f"{name}: {header}")
        data = [e for e in csv_reader]
        assert len(data[0]) == 17
        # TODO if problem with data maybe main timestamp is the good one instead of each motor's timestamp.
        time0 = np.array([float(e[8]) + float(e[9]) * 1e-9 for e in data])
        time1 = np.array([float(e[10]) + float(e[11]) * 1e-9 for e in data])
        time2 = np.array([float(e[12]) + float(e[13]) * 1e-9 for e in data])
        time3 = np.array([float(e[14]) + float(e[15]) * 1e-9 for e in data])
        motors = np.array([np.array(e[16].strip("[]").split(","), dtype=float) for e in data]) * np.pi / 30
        motors_dict = {"m0": np.stack([time0, motors[:, 0]], 1),
                       "m1": np.stack([time1, motors[:, 1]], 1),
                       "m2": np.stack([time2, motors[:, 2]], 1),
                       "m3": np.stack([time3, motors[:, 3]], 1)}

    name = "blackbird_slash_state"
    with open(raw_flight_path + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        header = next(csv_reader)
        # print(f"{name}: {header}")
        data = [e for e in csv_reader]
        assert len(data[0]) == 17
        time = np.array([float(e[0]) * 1e-9 for e in data])
        trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
        quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
        quat_trans = np.concatenate([quat, trans], axis=1)
        pose_dict = {"t": time, "quat_trans": quat_trans}

    # Computing state.
    # Rot is rot_fixed_body
    rot = Rotation.from_quat(quat).as_matrix().reshape((-1, 9))
    rot_reps = [splrep(pose_dict["t"], rot[:, i], k=5, s=5e-2) for i in range(9)]
    # rot = rot.reshape((-1, 3, 3))
    den_rot = np.stack([splev(pose_dict["t"], rot_reps[i]) for i in range(9)], axis=1).reshape((-1, 3, 3))
    den_drot = np.stack([splev(pose_dict["t"], rot_reps[i], der=1) for i in range(9)], axis=1).reshape((-1, 3, 3))
    den_d2rot = np.stack([splev(pose_dict["t"], rot_reps[i], der=2) for i in range(9)], axis=1).reshape((-1, 3, 3))

    # angles = Rotation.from_quat(quat).as_euler("zyx", False)[:, ::-1]
    den_angles = Rotation.from_matrix(den_rot).as_euler("zyx", False)[:, ::-1]
    den_dangles = Rotation.from_matrix(den_drot).as_euler("zyx", False)[:, ::-1]
    # plt.plot(angles)
    # plt.plot(den_angles)
    # plt.plot(den_d2rot.reshape((-1, 9)))

    # Computing angular velocity
    omega = Rotation.from_matrix(den_rot).inv().as_matrix() @ den_drot
    omega = omega.transpose((1, 2, 0))
    wz1 = -omega[0, 1, :]
    wz2 = omega[1, 0, :]
    wy1 = omega[0, 2, :]
    wy2 = -omega[2, 0, :]
    wx1 = -omega[1, 2, :]
    wx2 = omega[2, 1, :]
    den_angular_velocity_matrix = np.stack([(wx1 + wx2) / 2, (wy1 + wy2) / 2, (wz1 + wz2) / 2], axis=1)

    # Compute angular acceleration
    domega = Rotation.from_matrix(den_rot).inv().as_matrix() @ den_d2rot
    domega = domega.transpose((1, 2, 0))
    dwz1 = -domega[0, 1, :]
    dwz2 = domega[1, 0, :]
    dwy1 = domega[0, 2, :]
    dwy2 = -domega[2, 0, :]
    dwx1 = -domega[1, 2, :]
    dwx2 = domega[2, 1, :]
    den_angular_acc_matrix = np.stack([(dwx1 + dwx2) / 2, (dwy1 + dwy2) / 2, (dwz1 + dwz2) / 2], axis=1)

    # quat_reps = [splrep(pose_dict["t"], quat[:, i], k=3, s=5e-3) for i in range(4)]
    # den_quat = np.stack([splev(pose_dict["t"], quat_reps[i]) for i in range(4)], axis=1)
    # den_dquat = np.stack([splev(pose_dict["t"], quat_reps[i], der=1) for i in range(4)], axis=1)

    # # Quaternion's way to compute the angular velocity. However it needs to derive the quaternion and it is not
    # trivial: Euler differentiator and smoothed spline interpolation differentiator leads to a bad quaternion
    # derivative estimate. It has spikes where the quaternion is not differentiable.
    # dt = (pose_dict["t"][1:] - pose_dict["t"][:-1]).reshape((-1, 1))
    # dq_dt = (quat[1:] - quat[:-1]) / dt
    # dq_dt = np.concatenate([np.zeros((1, 4)), dq_dt], axis=0)
    # dq_dt_convention = dq_dt[:, [3, 0, 1, 2]]
    # q0, q1, q2, q3 = quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]
    # g_matrix = np.array([[-q1, q0, q3, -q2],
    #                      [-q2, -q3, q0, q1],
    #                      [-q3, q2, -q1, q0]]).transpose((2, 0, 1))
    # angular_velocity = (2 * g_matrix @ dq_dt_convention.reshape((-1, 4, 1))).squeeze()

    # Computing linear speed
    # dt = pose_dict["t"][1:] - pose_dict["t"][:-1]
    # euler_linear_speed = (trans[1:] - trans[:-1]) / dt.reshape((-1, 1))
    # euler_linear_speed = np.concatenate([np.zeros((1, 3)), euler_linear_speed])
    # euler_linear_speed = Rotation.from_quat(quat).inv().as_matrix() @ euler_linear_speed.reshape((-1, 3, 1))
    trans_rep = [splrep(pose_dict["t"], trans[:, i], k=5, s=1e-3) for i in range(3)]
    # den_trans = np.stack([splev(pose_dict["t"], trans_rep[i]) for i in range(3)], axis=1)
    den_dtrans = np.stack([splev(pose_dict["t"], trans_rep[i], der=1) for i in range(3)], axis=1)
    den_d2trans = np.stack([splev(pose_dict["t"], trans_rep[i], der=2) for i in range(3)], axis=1)
    den_linear_speed = (Rotation.from_quat(quat).inv().as_matrix() @ den_dtrans.reshape((-1, 3, 1))).squeeze()
    den_linear_acc = (Rotation.from_matrix(den_rot).inv().as_matrix() @ den_d2trans.reshape((-1, 3, 1))).squeeze()
    pass

    # # PLOT Linear speed derivation denoising
    # fig, ax = plt.subplots(1, 1)
    # for i, label in zip(range(3), ["x", "y", "z"]):
    #     # ax.plot(pose_dict["t"] - pose_dict["t"][0], trans[:, i], label=f"Position {label}[m]")
    #     # ax.plot(pose_dict["t"] - pose_dict["t"][0], den_trans[:, i], label=f"Denoised position {label}[m]")
    #     ax.plot(pose_dict["t"] - pose_dict["t"][0], euler_linear_speed[:, i], label=f"Euler derivative {label}[m/s]")
    #     ax.plot(pose_dict["t"] - pose_dict["t"][0], den_linear_speed[:, i],
    #             label=f"Denoised linear speed {label}[m/s]")
    # ax.set_title(intermediate_flight_path.split("/")[-4:-1])
    # ax.set_xlabel("Time[s]")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # plt.show()

    # # PLOT Quaternion denoising
    # fig, ax = plt.subplots(1, 1)
    # for i in range(4):
    #     # ax.plot(pose_dict["t"] - pose_dict["t"][0], quat[:, i], label=f"q{i}")
    #     ax.plot(pose_dict["t"] - pose_dict["t"][0], den_dquat[:, i], label=f"q{i}")
    # ax.set_title(intermediate_flight_path.split("/")[-4:-1])
    # ax.set_xlabel("Time[s]")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # plt.show()

    # #  PLOT Rotation matrix derivation denoising
    # fig, ax = plt.subplots(1, 1)
    # for i in range(9):
    #     ax.plot(pose_dict["t"] - pose_dict["t"][0], den_drot.reshape((-1, 9))[:, i], label=f"coef{i}")
    # ax.set_title(intermediate_flight_path.split("/")[-4:-1])
    # ax.set_xlabel("Time[s]")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # plt.show()

    # PLOT Angular velocity denoising
    # fig, ax = plt.subplots(1, 1)
    # for i, label in zip(range(3), ("x", "y", "z")):
    #     ax.plot(pose_dict["t"] - pose_dict["t"][0], angular_velocity[:, i], label=f"Euler quat derivative {label}")
    #     ax.plot(pose_dict["t"] - pose_dict["t"][0], den_angular_velocity_matrix[:, i],
    #             label=f"Denoised matrix derivative {label}")
    # ax.set_title(intermediate_flight_path.split("/")[-4:-1])
    # ax.set_xlabel("Time[s]")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # plt.show()

    state = np.concatenate([den_angles, den_angular_velocity_matrix, den_linear_speed, trans], axis=1)

    dstate = np.concatenate([den_dangles, den_angular_acc_matrix, den_linear_acc, den_dtrans], axis=1)

    # Synchronisation
    imu_sync = np.stack([np.interp(x=pose_dict["t"],
                                   xp=imu_dict["t"],
                                   fp=imu_dict["ang_acc"][:, i]) for i in range(6)], axis=1)
    motor_sync = np.stack([np.interp(x=pose_dict["t"],
                                     xp=motors_dict[f"m{i}"][:, 0],
                                     fp=motors_dict[f"m{i}"][:, 1]) for i in range(4)], axis=1)

    # den_motor = np.stack([splev(pose_dict["t"],
    #                             splrep(pose_dict["t"], motor_sync[:, i], k=3, s=1e-3)) for i in range(4)], axis=1)
    # # Plot motors
    # fig, ax = plt.subplots(1, 1)
    # [ax.plot(pose_dict["t"] - pose_dict["t"][0], motor_sync[:, i], label=f"Motor {i}") for i in range(4)]
    # [ax.plot(pose_dict["t"] - pose_dict["t"][0], den_motor[:, i], label=f"Denoised motor {i}") for i in range(4)]
    # ax.set_title(intermediate_flight_path.split("/")[-4:-1])
    # ax.set_xlabel("Time[s]")
    # ax.legend()
    # ax.grid()
    # plt.tight_layout()
    # plt.show()

    synchronisation = {"t": pose_dict["t"], "imu_ang_acc": imu_sync, "motor": motor_sync,
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
    stop = min(len(synchronisation["motor"]) - 4000, stop)
    assert start != stop

    # return start, stop
    cut_data = {}
    for key in synchronisation:
        cut_data[key] = synchronisation[key][start:stop]

    # Setting initial parameters
    cut_data["t"] = cut_data["t"] - cut_data["t"][0]

    # Flight data keys:
    # ['t', 'imu_ang_acc', 'motor', 'pose_quat_trans', 'state', 'dstate']
    # time:1, state:12, motor:4, imu_ang_acc:6, time_next_state:1, dstate:12, next_state:12 = 48
    res = np.concatenate((cut_data["t"][:-1].reshape((-1, 1)),
                          cut_data["state"][:-1],
                          cut_data["motor"][:-1],
                          cut_data["imu_ang_acc"][:-1],
                          cut_data["t"][1:].reshape((-1, 1)),
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


class BlackbirdDataset(Dataset):
    def __init__(self, train_validation_test: str):
        torch_dataset_path = DATASET_PATH + f"intermediate_torch_dataset/"
        if train_validation_test not in ["train", "validation", "test"]:
            raise ValueError("train_validation_test must be 'train', 'validation' or 'test'.")
        self.data = torch.load(torch_dataset_path + f"{train_validation_test}.pt")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        One sample is time:1, state:12, motor:4, imu_ang_acc:6, time_next_state:1, dstate:12, next_state:12 = 48

        """
        data = self.data[idx]
        return {"t": data[0:1], "state": data[1:13], "motor": data[13:17], "imu": data[17:23], "next_t": data[23:24],
                "dstate": data[24:36], "next_state": data[36:]}


def madiha_laurent_csv():
    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    flight = f"/{flight_name}/{flight_yaw}/{flight_speed}/"
    intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
    data = pickle.load(open(intermediate_flight_path + "data.pkl", "rb"))
    # return {"t": data[0:1], "state": data[1:13], "motor": data[13:17], "imu": data[17:23], "next_t": data[23:24],
    #         "dstate": data[24:36], "next_state": data[36:]}

    t = data[:, 0]
    state = data[:, 1:13]
    motor = data[:, 13:17]
    imu_ang_acc = data[:, 17:23]
    header = ["t[s]",
              "phi[rad]", "theta[rad]", "psi[rad]",
              "x[m]", "y[m]", "z[m]",
              "motor0[rad/s]", "motor1[rad/s]", "motor2[rad/s]", "motor3[rad/s]",
              "imu_ang_vel_x[rad/s]", "imu_ang_vel_y[rad/s]", "imu_ang_vel_z[rad/s]",
              "imu_acc_x[m/s2]", "imu_acc_y[m/s2]", "imu_acc_z[m/s2]"]
    ml_data = np.concatenate([t.reshape((-1, 1)),
                              state[:, 0:3],
                              state[:, 9:12],
                              motor,
                              imu_ang_acc], axis=1)
    assert len(header) == ml_data.shape[1]
    with open(DATASET_PATH + "intermediate/madiha_laurent.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in ml_data:
            writer.writerow(row)


def main():
    # flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    # flight_name, flight_yaw, flight_speed = 'egg', 'yawConstant', 'maxSpeed13p8'
    # flight_name, flight_yaw, flight_speed = 'halfMoon', 'yawForward', 'maxSpeed4p0'
    # raw_flight_path = DATASET_PATH + f"raw/{flight_name}/{flight_yaw}/{flight_speed}/"
    # intermediate_flight_path = DATASET_PATH + f"intermediate/{flight_name}/{flight_yaw}/{flight_speed}/"
    # print(f"/{flight_name}/{flight_yaw}/{flight_speed}/")
    # load_clean_data(raw_flight_path, intermediate_flight_path)

    for i in tqdm(range(len(FLIGHT_LIST))):
        flight = FLIGHT_LIST[i]
        raw_flight_path = DATASET_PATH + f"raw/{flight}"
        intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
        load_clean_data(raw_flight_path, intermediate_flight_path)
    #
    # building_data_and_splitting()

    # madiha_laurent_csv()


def plot_final_data():
    flight_name, flight_yaw, flight_speed = 'mouse', 'yawForward', 'maxSpeed7p0'
    flight = f"/{flight_name}/{flight_yaw}/{flight_speed}/"

    intermediate_flight_path = DATASET_PATH + f"intermediate/{flight}"
    data = pickle.load(open(intermediate_flight_path + "data.pkl", "rb"))
    # return {"t": data[0:1], "state": data[1:13], "motor": data[13:17], "imu": data[17:23], "next_t": data[23:24],
    #         "dstate": data[24:36], "next_state": data[36:]}

    # State orientation
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    dt = 1 / 360
    dphi = (data[1:, 0] - data[:-1, 0]) / dt
    dtheta = (data[1:, 1] - data[:-1, 1]) / dt
    dpsi = (data[1:, 2] - data[:-1, 2]) / dt

    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("phi", "theta", "psi")):
        ax.plot(range(len(data)), data[:, 0 + i], label=name)
    ax.set_xlabel("Nb d'échantillon")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # Autre derivee
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for array, name in zip([dphi, dtheta, dpsi], ("phi", "theta", "psi")):
        ax.plot(list(range(len(data)))[1:], array, label=name)
    ax.set_xlabel("Nb d'échantillon")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # State orientation
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("phi", "theta", "psi")):
        ax.plot(range(len(data)), data[:, 0 + i], label=name)
    ax.set_xlabel("Nb d'échantillon")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    # plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # State angular velocity
    # fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"Flight: {flight_name} - {flight_yaw} - {flight_speed}")
    for i, name in zip(range(3), ("phi", "theta", "psi")):
        ax.plot(range(len(data)), data[:, 3 + i], label=name)
    ax.set_xlabel("Nb d'échantillon")
    ax.set_ylabel("State orientation [rad]")
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
    # plot_final_data()
