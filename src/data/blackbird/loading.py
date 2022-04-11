import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import scipy.integrate
import scipy.signal as signal
from scipy.spatial.transform import Rotation
from scipy import interpolate
from scipy.interpolate import splev, splrep
from tqdm import tqdm

import src.data.datavis as datavis
from src.utils import quaternion
from src.utils.transform import Transform as Tf
from src.utils.path import WD_PATH

# plt.style.use('seaborn')

if os.environ.get("USER") == "sjanny":
    DATASET_PATH = "/Volumes/Samsung_T5/BlackBird/blackbird/"
else:
    DATASET_PATH = "/Users/quentin/phd/turbulence/data/blackbird/"
VICE_PATH = f'{WD_PATH}data/blackbird/raw/patrick/yawConstant/maxSpeed3p0/csv/'

FLIGHT = {"3dFigure8": {"yawConstant": [0.5, 1, 2, 3, 4, 5],
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


def get_flight_list():
    flight_list = []
    flight_dict = {}
    raw_path = f'{DATASET_PATH}raw/'
    for trajectory in next(os.walk(raw_path))[1]:
        flight_dict[trajectory] = {}
        for orientation in next(os.walk(f'{raw_path}{trajectory}/'))[1]:
            flight_dict[trajectory][orientation] = []
            all_speed = next(os.walk(f'{raw_path}{trajectory}/{orientation}/'))[1]
            for speed in all_speed:
                for e in FLIGHT[trajectory][orientation]:
                    if not isinstance(e, str):
                        e = str(e)
                        if '.' in e:
                            a, b = e.split('.')
                            assert f'maxSpeed{a}p{b}' in all_speed
                        else:
                            assert f'maxSpeed{e}p0' in all_speed
                    else:
                        assert e in all_speed
                flight_list.append(f"{raw_path}{trajectory}/{orientation}/{speed}/")
                flight_dict[trajectory][orientation].append(speed)
    return flight_list, flight_dict


def load_from_csv(flight):
    name = "blackbird_slash_imu"
    with open(flight + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)
        data = [e for e in csv_reader]
        time = np.array([int(e[0]) for e in data], dtype=np.int)
        imu_angular_velocity = np.array([np.array([float(e[14]), float(e[15]), float(e[16])]) for e in data])
        imu_linear_acceleration = np.array([np.array([float(e[19]), float(e[20]), float(e[21])]) for e in data])
        imu_dict = {"t": time, "ang_vel": imu_angular_velocity, "lin_acc": imu_linear_acceleration}

    name = "blackbird_slash_rotor_rpm"
    with open(flight + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)
        data = [e for e in csv_reader]
        time = np.array([int(e[0]) for e in data])
        time0 = np.array([int(e[8]) * int(1e9) + int(e[9]) for e in data])
        time1 = np.array([int(e[10]) * int(1e9) + int(e[11]) for e in data])
        time2 = np.array([int(e[12]) * int(1e9) + int(e[13]) for e in data])
        time3 = np.array([int(e[14]) * int(1e9) + int(e[15]) for e in data])
        motors = np.array([np.array(e[16].strip("[]").split(","), dtype=float) for e in data]) * np.pi / 30
        motor_dict = {"m0": motors[:, 0], 't0': time0,
                      "m1": motors[:, 1], 't1': time1,
                      "m2": motors[:, 2], 't2': time2,
                      "m3": motors[:, 3], 't3': time3,
                      't': time}

    name = "blackbird_slash_state"
    with open(flight + f"csv/{name}.csv", newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)
        data = [e for e in csv_reader]
        time = np.array([int(e[0]) for e in data], dtype=np.int)
        trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
        quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
        quat = Rotation.from_quat(quat).as_quat()
        pose_dict = {"t": time, "trans": trans, "quat": quat}
    return imu_dict, pose_dict, motor_dict


def normalize(x):
    return x / np.linalg.norm(x, axis=-1)[..., np.newaxis]


def process_raw_data(imu_dict, pose_dict, motor_dict, freq):
    # CUT
    begin = [np.nonzero(motor_dict[f'm{i}'] > 1000)[0][0] for i in range(4)]
    start_motor = np.max(begin) + 1000
    start_pose = np.argmin(np.abs(motor_dict[f't{np.argmax(begin)}'][start_motor] - pose_dict['t']))  # for pose
    start_imu = np.argmin(np.abs(motor_dict[f't{np.argmax(begin)}'][start_motor] - imu_dict['t']))  # for imu

    end = [np.nonzero(motor_dict[f'm{i}'] > 1000)[0][-1] for i in range(4)]
    stop_motor = np.min(end) - 1000
    stop_pose = np.argmin(np.abs(motor_dict[f't{np.argmin(end)}'][stop_motor] - pose_dict['t']))  # for pose
    stop_imu = np.argmin(np.abs(motor_dict[f't{np.argmin(end)}'][stop_motor] - imu_dict['t']))  # for imu

    for dictionary, start, stop in zip([motor_dict, pose_dict, imu_dict],
                                       [start_motor, start_pose, start_imu],
                                       [stop_motor, stop_pose, stop_imu]):
        for key in dictionary:
            dictionary[key] = dictionary[key][start:stop]

    def low_pass_butter(x, period, order, wn):
        sos = signal.butter(N=order, Wn=wn, btype='lowpass', fs=1 / period, analog=False, output='sos')
        return signal.sosfiltfilt(sos, x, axis=0)

    # DENOISING STATE
    quat = pose_dict['quat']
    trans = pose_dict['trans']
    time = pose_dict['t']
    # plt.hist(time * 1e-9)
    dt = np.mean(time[1:] - time[:-1], dtype=np.int) * 1e-9
    std = np.std(time[1:] - time[:-1], dtype=np.int) * 1e-9
    assert (np.sort(time) == time).all()
    assert (np.unique(time) == time).all()
    assert std / dt < 0.20

    assert np.isclose(np.linalg.norm(quat, axis=1).reshape((-1, 1)), 1).all()
    quat = quaternion.qfix(quat)  # Fix quaternion swap in continuity
    # datavis.plot_butterworth_quat(quat, dt, 0, (time-time[0]) * 1e-9)
    # datavis.plot_butterworth_comparison(quat, dt, True, 5, 11, (time-time[0]) * 1e-9)
    quat = low_pass_butter(quat, dt, order=5, wn=11)
    quat = normalize(quat)

    # datavis.plot_butterworth_comparison(trans, dt, False, 2, 12, (time-time[0]) * 1e-9)
    trans = low_pass_butter(trans, dt, order=2, wn=12)

    pose_dict['quat'] = quat
    pose_dict['trans'] = trans
    pose_dict['t'] = time

    # DENOISING MOTOR
    for i in range(4):
        time = motor_dict[f"t{i}"]
        motor = motor_dict[f"m{i}"]
        argsort = np.argsort(time)
        time = time[argsort]
        time, index = np.unique(time, True)
        motor = motor[argsort][index]
        assert (np.sort(time) == time).all()
        assert (np.unique(time) == time).all()
        dt = np.mean(time[1:] - time[:-1]) * 1e-9
        std = np.std(time[1:] - time[:-1]) * 1e-9
        assert std / dt < 140  # Huge !!!
        # plt.scatter(time, motor)

        # datavis.plot_butterworth_comparison(motor, dt, False, 3, 6, None)
        motor = low_pass_butter(motor, dt, order=3, wn=6)
        # plt.plot((time - time[0]) * 1e-9, motor)

        motor_dict[f"t{i}"] = time
        motor_dict[f"m{i}"] = motor

    # DENOISING IMU
    ang_vel = imu_dict['ang_vel']
    lin_acc = imu_dict['lin_acc']
    time = imu_dict['t']
    dt = np.mean(time[1:] - time[:-1], dtype=np.int) * 1e-9
    std = np.std(time[1:] - time[:-1], dtype=np.int) * 1e-9
    assert (np.sort(time) == time).all()
    assert (np.unique(time) == time).all()
    assert std / dt < 0.20

    # datavis.plot_butterworth_comparison(ang_vel, dt, False, 5, 11, None)
    ang_vel = low_pass_butter(ang_vel, dt, order=5, wn=11)

    # datavis.plot_butterworth_comparison(lin_acc, dt, False, 1, 2, None)
    lin_acc = low_pass_butter(lin_acc, dt, order=1, wn=2)

    imu_dict['ang_vel'] = ang_vel
    imu_dict['lin_acc'] = lin_acc
    imu_dict['t'] = time

    # STATE AND IMU FIXED ORIENTATION
    pose = Tf.from_quat_trans(pose_dict['quat'], pose_dict['trans']).as_matrix()
    pi_rotation = Tf.from_euler_trans("zyx", np.array([0, 0, np.pi]), degrees=False, trans=np.zeros(3)).as_matrix()
    pose = pi_rotation @ pose @ pi_rotation
    pose_dict['quat'], pose_dict['trans'] = Tf.from_matrix(pose).as_quat_trans()

    pi_rotation = Rotation.from_euler("zyx", np.array([np.pi / 2, 0, np.pi]), degrees=False)
    imu_dict['ang_vel'] = pi_rotation.apply(imu_dict['ang_vel'])
    imu_dict['lin_acc'] = pi_rotation.apply(imu_dict['lin_acc'])

    # SYNCHRONISATION
    time0 = pose_dict['t'][0]
    time1 = pose_dict['t'][-1]
    for time in [pose_dict['t'], imu_dict['t']] + [motor_dict[f't{i}'] for i in range(4)]:
        assert (sorted(time) == time).all()
        time0 = max(time0, time[0])
        time1 = min(time1, time[-1])
    # num = np.round((time1 - time0) * 1e-9 * freq).astype(int)
    # sync_time = np.linspace(time0, time1, num, dtype=np.int)
    sync_time = np.arange(time0, time1, np.int(1 / 100 * 1e9))
    dt = np.mean(sync_time[1:] - sync_time[:-1], dtype=np.int) * 1e-9
    std = np.std(sync_time[1:] - sync_time[:-1], dtype=np.int) * 1e-9
    assert (np.sort(sync_time) == sync_time).all()
    assert (np.unique(sync_time) == sync_time).all()
    assert std / dt == 0

    pose = np.concatenate([pose_dict['trans'], pose_dict['quat']], axis=-1)
    pose_sync = np.stack([interpolate.interp1d(pose_dict['t'], pose[:, i], bounds_error=True)(sync_time)
                          for i in range(7)], axis=-1)

    y = np.concatenate([imu_dict['ang_vel'], imu_dict['lin_acc']], axis=-1)
    imu_sync = np.stack([interpolate.interp1d(imu_dict["t"], y[:, i], bounds_error=True)(sync_time)
                         for i in range(6)], axis=-1)
    motor_sync = np.stack([interpolate.interp1d(motor_dict[f't{i}'], motor_dict[f"m{i}"], bounds_error=True)(sync_time)
                           for i in range(4)], axis=-1)

    sync = {"t": sync_time, "imu_ang_acc": imu_sync, "motor": motor_sync, "pose": pose_sync}

    # COMPUTE STATE DERIVATIVE
    trans = sync['pose'][:, :3]
    quat = sync['pose'][:, 3:7]
    time = (sync['t'] - sync['t'][0]) * 1e-9

    dquat = np.stack([np.gradient(quat[:, i], time) for i in range(4)], axis=-1)
    ang_vel = 2. * quaternion.qmul_np(quat * np.array([[-1., -1., -1., 1.]]), dquat)[:, :3]

    rep = [splrep(time, trans[:, i], k=5, s=0.01) for i in range(3)]
    dtrans = np.stack([splev(time, rep[i], der=1) for i in range(3)], -1)
    state = np.concatenate([trans, quat, dtrans, ang_vel], axis=1)

    # state_acc = np.stack([splev(time, rep[i], der=2) for i in range(3)], -1)
    data = {'t': sync['t'], 'imu_ang_acc': sync['imu_ang_acc'], 'motor': sync['motor'], 'state': state}
    return data


import torch


def torch_integrate_imu(dt: float, state0: torch.Tensor, imu: torch.Tensor, state, device):
    """
    Quaternion integration is not working, tested on (batch, t, val) tensors.
    So the rest is not tested as quaternion integration is the first step.
    Last arg 'state' is here only for debugging purposes
    """
    # IMU tensor has shape (*, t, val)
    # state0 tensor has shape (*, val)
    position0 = state0[..., :3]
    quaternion0 = state0[..., 3:7]
    lin_speed0 = state0[..., 7:10]
    angular_speed0 = state0[..., 10:]

    sensor_ang_vel = imu[..., :3]  # in body
    sensor_lin_acc = imu[..., 3:]  # in body with g to be removed

    length = imu.shape[:-2]

    # # Imu integrating quat
    sensor_quat = [quaternion0]
    zero = torch.zeros(length + (1,), device=device)
    ang_vel4 = torch.cat([sensor_ang_vel[..., 0, :], zero], dim=-1)
    previous_dot_quat = 0.5 * qop.qmul(sensor_quat[-1], ang_vel4)
    for i in range(1, imu.shape[-2]):
        ang_vel4 = torch.cat([sensor_ang_vel[..., i, :], zero], dim=-1)
        dot_quat = 0.5 * qop.qmul(sensor_quat[-1], ang_vel4)
        next_quat = (dot_quat + previous_dot_quat) / 2 * dt + sensor_quat[-1]
        next_quat = next_quat / torch.norm(next_quat, dim=-1).unsqueeze(-1)
        previous_dot_quat = dot_quat
        sensor_quat.append(next_quat)
    sensor_quat = torch.stack(sensor_quat, dim=-2)

    # Removing g
    sensor_lin_acc_fixed = qop.qrot(sensor_quat, sensor_lin_acc)
    sensor_lin_acc_fixed[..., 2] = sensor_lin_acc_fixed[..., 2] - torch.tensor(9.81, device=device)

    plt.plot(state[0, :, 10:])
    plt.plot(sensor_ang_vel[0])

    plt.plot(state[0, :, 3:7])
    plt.plot(sensor_quat[0])
    # Integrating acc
    # integrator = scipy.integrate.cumtrapz
    # dtrans = sync['state'][:, 7:10]
    # dtrans_imu = np.stack([integrator(sync['imu_ang_acc'][:, 3 + i], sync['t'] * 1e-9,
    #                                   initial=dtrans[0, i]) for i in range(3)], -1)
    # trans = sync['state'][:, :3]
    # trans_imu = np.stack([integrator(dtrans_imu[:, i], sync['t'] * 1e-9,
    #                                  initial=trans[0, i]) for i in range(3)], -1)

    sensor_lin_speed = torch.cat([lin_speed0.unsqueeze(-2),
                                  torch.cumulative_trapezoid(sensor_lin_acc, dx=dt, dim=-2) + lin_speed0.unsqueeze(
                                      -2)],
                                 dim=-2)

    sensor_position = torch.cat([position0.unsqueeze(-2),
                                 torch.cumulative_trapezoid(sensor_lin_speed, dx=dt, dim=-2) + position0.unsqueeze(
                                     -2)],
                                dim=-2)

    next_state = torch.cat([sensor_position, sensor_quat, sensor_lin_speed, sensor_ang_vel], dim=-1)

    integrate_imu_numpy()

    return next_state


def integrate_imu_numpy(dt: float, state0: np.ndarray, imu: np.ndarray):
    """
    Integration is working
    """
    imu_ang_vel = imu[:, :3]
    imu_lin_acc = imu[:, 3:]

    imu_quat = [state0[3:7]]
    previous_dot_quat = 0.5 * quaternion.qmul_np(imu_quat[-1], np.concatenate([imu_ang_vel[0], np.zeros(1)]))
    for i in range(1, len(imu)):
        dot_quat = 0.5 * quaternion.qmul_np(imu_quat[-1], np.concatenate([imu_ang_vel[i], np.zeros(1)]))
        next_quat = (dot_quat + previous_dot_quat) / 2 * dt + imu_quat[-1]
        next_quat = normalize(next_quat)
        previous_dot_quat = dot_quat
        imu_quat.append(next_quat)
    imu_quat = np.stack(imu_quat, 0)

    # Rotating lin_acc and  Removing g
    rot_fixed_body = Rotation.from_quat(imu_quat).as_matrix()
    imu_lin_acc = (rot_fixed_body @ imu_lin_acc.reshape((-1, 3, 1))).reshape((-1, 3))
    imu_lin_acc[:, 2] = imu_lin_acc[:, 2] - 9.81

    # Integrating acc
    imu_dtrans = state0[7:10] + np.concatenate([np.zeros((1, 3)),
                                                np.cumsum((imu_lin_acc[1:] + imu_lin_acc[:-1]) / 2 * dt, axis=0)])
    imu_trans = state0[:3] + np.concatenate([np.zeros((1, 3)),
                                             np.cumsum((imu_dtrans[1:] + imu_dtrans[:-1]) / 2 * dt, axis=0)])
    return np.concatenate([imu_trans, imu_quat, imu_dtrans, imu_ang_vel], axis=-1)


def main():
    flight_list, flight_dict = get_flight_list()
    all_data = []
    prog = tqdm(desc='Processing flights', total=len(flight_list))
    for trajectory, dictio in flight_dict.items():
        for orientation, speed_list in dictio.items():
            for speed in speed_list:
                flight = f'{DATASET_PATH}raw/{trajectory}/{orientation}/{speed}/'
                imu_dict, pose_dict, motor_dict = load_from_csv(flight)
                data = process_raw_data(imu_dict, pose_dict, motor_dict, freq=100)
                data['trajectory'] = trajectory
                data['orientation'] = orientation
                data['speed'] = speed
                save_location = f'{DATASET_PATH}intermediate/{trajectory}/{orientation}/{speed}/data.pkl'
                pickle.dump(data, open(f'{save_location}', 'wb'))
                all_data.append(data)
                prog.update()
    pickle.dump(all_data, open(f'{DATASET_PATH}intermediate/data.pkl', 'wb'))
    pickle.dump(all_data, open(f'/Users/quentin/phd/CDC_Observer/Data/Drone/data.pkl', 'wb'))
    cmd = f'''rsync -rtvP {DATASET_PATH}intermediate/data.pkl 
              remsto:/storage/Datasets/BlackBird/blackbird/intermediate/data.pkl'''
    os.system(cmd)


if __name__ == '__main__':
    main()
