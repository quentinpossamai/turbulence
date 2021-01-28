"""
This file contains a drone models to compute the turbulence forces and torques labels.
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint  # https://github.com/rtqichen/torchdiffeq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

from typing import Iterable, Callable

import utils


# 05 June 2020
# DRONE MODEL


# noinspection PyAbstractClass
class AscTecFireflyDroneModel(nn.Module):
    """
    AscTec Firefly drone model used in EuRoC MAV data implementing "f" the derivative of the state over time, "forward"
    to compute the error between ground truth and model estimation.

    Every equations in this code are described in:
    -Sabatino, Francesco. ‘Quadrotor Control: Modeling, Nonlinear Control Design, and Simulation’, 2015, 67.

    -Akkinapalli, Venkata Sravan, Guillermo Falconí, and Florian Holzapfel. ‘Attitude Control of a Multicopter Using L1
     Augmented Quaternion Based Backstepping’, 170–78, 2014. https://doi.org/10.1109/ICARES.2014.7024376.

    -Achtelik, Michael, Klaus-Michael Doth, Daniel Gurdan, and Jan Stumpf. ‘Design of a Multi Rotor MAV with Regard to
     Efficiency, Dynamics and Redundancy’. In AIAA Guidance, Navigation, and Control Conference. Guidance, Navigation,
      and Control and Co-Located Conferences. American Institute of Aeronautics and Astronautics, 2012.
      https://doi.org/10.2514/6.2012-4779.

    """

    def __init__(self, m: float, g: float, kt: float, km: float, l_arm: float, inertia_matrix: np.ndarray):
        """
        Drone model base of Sabatino Francesco thesis adapted to an hexarotor.

        :param m: mass of the drone (scalar) (kg).
        :param g: gravity constant (scalar) (9.81m/s2).
        :param kt: Thrust factor (scalar) (kg.m/rad2).
        :param km: Moment factor (scalar) (kg.m2/rad2).
        :param l_arm: Drone's arm distance (scalar) (m).
        :param inertia_matrix: (3x3 matrix) (kg.m2).
        """
        super().__init__()
        # Drone model parameters.
        self.m = torch.tensor(m, dtype=torch.float)
        self.g = torch.tensor(g, dtype=torch.float)
        self.kt = torch.tensor(kt, dtype=torch.float)
        self.km = torch.tensor(km, dtype=torch.float)
        self.l_arm = torch.tensor(l_arm, dtype=torch.float)
        self.inertia_matrix = torch.from_numpy(inertia_matrix)

        self.fa = nn.parameter.Parameter(torch.zeros(12, dtype=torch.float, requires_grad=True))

    def f(self, time: torch.Tensor, state: torch.Tensor, w_func: Callable, fa: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the state. x_dot = f(x, u, fa).
        :param time: scalar in seconds (Tensor of shape (1,)).
        :param state: state of the drone torch.Tensor[phi, theta, psi, p, q, r, u, v, w, x, y, z].
        :param w_func: Command: 6 rotors angular velocity in rad/s.
        :param fa: External perturbation in N=kg.m/s2.
        :return: Return the derivative of the state.
        """
        assert len(fa) == 12, "fa = _, _, _, tau_x, tau_y, tau_z, fx, fy, fz, _, _, _"
        # # Quadrotor
        # w1 = torch.tensor(0., dtype=torch.float)
        # w2 = torch.tensor(0., dtype=torch.float)
        # w3 = torch.tensor(0., dtype=torch.float)
        # w4 = torch.tensor(0., dtype=torch.float)
        #
        # f_cmd_t = self.b * (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2)
        # tau_cmd_x = self.b * self.l_arm * (w3 ** 2 - w1 ** 2)
        # tau_cmd_y = self.b * self.l_arm * (w4 ** 2 - w2 ** 2)
        # tau_cmd_z = self.d * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)

        # constant = 1 / torch.mean(w_func(44.)) * (self.g * self.m / (6 * self.kt)) ** 0.5
        # w = constant * w_func(time.item())

        # Hexarotor
        km = self.km
        la = self.l_arm
        c = 3 ** 0.5 / 2
        b = torch.tensor([[-0.5 * la, -la, -0.5 * la, 0.5 * la, la, 0.5 * la],
                          [c * la, 0, -c * la, -c * la, 0, c * la],
                          [-km, km, -km, km, -km, km],
                          [1, 1, 1, 1, 1, 1]])

        # Theoretic hovering command
        body = torch.tensor([0, 0, 0, self.m * self.g]).reshape((4, 1))
        b_inv = torch.pinverse(b)
        w_hovering_model = (b_inv @ body / self.kt) ** 0.5

        w_hovering_cmd = w_func(time.item())
        print()
        w = None

        # Command to forces and torques
        u = self.kt * w ** 2
        drone_torsor = b @ u
        tau_cmd_x = drone_torsor[0]
        tau_cmd_y = drone_torsor[1]
        tau_cmd_z = drone_torsor[2]
        f_cmd_t = drone_torsor[3]

        # -----------------------------------------------------------

        cos = torch.cos
        sin = torch.sin
        tan = torch.tan

        phi, theta, psi, p, q, r, u, v, w, x, y, z = state

        ixx = self.inertia_matrix[0, 0]
        iyy = self.inertia_matrix[1, 1]
        izz = self.inertia_matrix[2, 2]

        dphi = p + r * cos(phi) * tan(theta) + q * sin(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = r * cos(phi) / cos(theta) + q * sin(phi) / cos(theta)

        dp = (iyy - izz) / ixx * r * q + tau_cmd_x / ixx
        dq = (iyy - izz) / ixx * r * q + tau_cmd_y / iyy
        dr = (iyy - izz) / ixx * r * q + tau_cmd_z / izz

        du = r * v - q * w + self.g * sin(theta)
        dv = p * w - r * u - self.g * sin(phi) * cos(theta)
        dw = q * u - p * v - self.g * cos(theta) * cos(phi) + f_cmd_t / self.m

        dx = (w * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
              - v * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)) + u * cos(psi) * cos(theta))
        dy = (v * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta))
              - w * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)) + u * cos(theta) * sin(psi))
        dz = w * cos(phi) * cos(theta) - u * sin(theta) + v * cos(theta) * sin(phi)

        dstate = torch.tensor([dphi, dtheta, dpsi, dp, dq, dr, du, dv, dw, dx, dy, dz], dtype=torch.float)

        c = torch.tensor([0., 0., 0., 1 / ixx, 1 / iyy, 1 / izz,
                          1 / self.m, 1 / self.m, 1 / self.m, 0., 0., 0.], dtype=torch.float)

        dstate = dstate + c * fa
        return dstate

    def forward(self, xn1: torch.Tensor, xn: torch.Tensor, wn: torch.Tensor,
                dt: torch.Tensor, tn: torch.Tensor) -> torch.Tensor:
        """
        Compute the norm of the difference between the state and a state estimate. fa derivable in pytorch.
        :param xn1: state at time tn+1.
        :param xn: state at time tn.
        :param wn: Rotors speed at time tn.
        :param dt: n+1 time step dt_n+1 = tn+1 - tn.
        :param tn: time of the measure in s.
        :return: torch.Tensor of dimension (1,).
        """
        return torch.norm(xn1 - (xn + dt * self.f(tn, xn, wn, self.fa)))

    def forward_fa_direct_subtract(self, xn1: torch.Tensor, xn: torch.Tensor, wn: torch.Tensor,
                                   dt: torch.Tensor, tn: torch.Tensor) -> torch.Tensor:
        """
        A different forward to not used pytorch optimizers because computing fa* is just a subtraction.
        :param xn1: state a step n+1.
        :param xn: state a step n.
        :param wn: command at time n.
        :param dt: time step.
        :param tn: (s).
        :return: fa, ground truth at step n+1, estimated state at step n+1.
        """
        ixx = self.inertia_matrix[0, 0]
        iyy = self.inertia_matrix[1, 1]
        izz = self.inertia_matrix[2, 2]
        c = torch.tensor([0., 0., 0., 1 / ixx, 1 / iyy, 1 / izz,
                          1 / self.m, 1 / self.m, 1 / self.m, 0., 0., 0.], dtype=torch.float)

        xn1_hat = xn + dt * self.f(tn, xn, wn, fa=torch.zeros(12))
        fa = (xn1 - xn1_hat) / (dt * c)
        return fa, xn1, xn1_hat


def compute_fa():
    """
    Compute fa* using FSDroneModel integrated with Euler as an estimator. EuRoC MAV data are used (AscTec Firefly).

    AscTec parameters:
    -Achtelik, Michael, Klaus-Michael Doth, Daniel Gurdan, and Jan Stumpf. ‘Design of a Multi Rotor MAV with Regard to
    Efficiency, Dynamics and Redundancy’. In AIAA Guidance, Navigation, and Control Conference. Guidance, Navigation,
    and Control and Co-Located Conferences. American Institute of Aeronautics and Astronautics, 2012.
    https://doi.org/10.2514/6.2012-4779.
    -Akkinapalli, Venkata Sravan, Guillermo Falconí, and Florian Holzapfel. ‘Attitude Control of a Multicopter Using L1
    Augmented Quaternion Based Backstepping’, 170–78, 2014. https://doi.org/10.1109/ICARES.2014.7024376.
    """
    # Loading data
    f = utils.DataFolder("euroc_mav")
    flight_number = 0
    data_path = f.get_unique_file_path(".pkl", f.folders["intermediate"][flight_number], "sensors_synchronised")
    print(f"Loading: {data_path}")
    data = pd.read_pickle(data_path)
    data: pd.DataFrame

    # Data cleaning
    data["time"] = (data["time"] - data["time"][0])

    # data["vicon_pose"] is tf_drone_origin
    data["vicon_pose"] = data["vicon_pose"].apply(lambda _x: _x.inv())
    # Now data["vicon_pose"] is tf_origin_drone

    data["trans"] = data["vicon_pose"].apply(lambda _x: _x.get_trans())
    data["quat"] = data["vicon_pose"].apply(lambda _x: _x.get_pose()[1])
    data.motor_speed = data.motor_speed.apply(lambda _x: torch.from_numpy(_x).float())

    dt = pd.Series(data["time"][1:].values - data["time"][:len(data) - 1].values)
    dt.index = range(1, len(dt) + 1)

    # Euler derivative compute
    # Linear speed in origin
    linear_speed = (data["trans"][1:].values - data["trans"][:len(data) - 1].values) / dt
    # Linear speed in drone
    linear_speed = linear_speed.to_frame(0) \
        .apply(lambda _x: data["vicon_pose"][_x.name].get_rot().T @ _x.values[0], axis=1)

    def angular_velocity_euler(_x: pd.Series) -> np.ndarray:
        """
        :return: The angular velocity computed form 2 quaternions and a time difference. Used with DataFrame.apply().
        """
        qn1: np.ndarray = data["quat"][_x.name]
        qn: np.ndarray = data["quat"][_x.name - 1]
        delta_t: float = _x.values[0]

        dq_dt = (qn1 - qn) / delta_t  # Euler differentiator
        q0, q1, q2, q3 = qn1
        g_matrix = np.array([[-q1, q0, q3, -q2],
                             [-q2, -q3, q0, q1],
                             [-q3, q2, -q1, q0]])

        return (2 * g_matrix @ dq_dt.reshape((4, 1))).reshape((3,))

    angular_velocity = dt.to_frame(0).apply(angular_velocity_euler, axis=1)

    data["linear_speed"] = linear_speed
    data["angular_velocity"] = angular_velocity
    data["dt"] = dt

    data["state"] = np.zeros(len(data))
    data[["state"]] = data[["state"]].apply(
        lambda _x: torch.from_numpy(np.hstack([data["vicon_pose"][_x.name].get_rot_euler("xyz", True),
                                               data["angular_velocity"][_x.name],
                                               data["linear_speed"][_x.name],
                                               data["trans"][_x.name]]).astype(np.float)), axis=1)

    drone_model = AscTecFireflyDroneModel(m=0.64, g=9.81, kt=6.546e-6, km=1.2864e-7, l_arm=0.215,
                                          inertia_matrix=np.array([[10.007e-3, 0., 0.],
                                                                   [0., 10.2335e-3, 0],
                                                                   [0., 0., 8.1e-3]]))
    # Fa computation and plots
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{f.workspace_path}tensorboard/drone_model/{now}/")
    # Run in terminal: tensorboard --logdir /Users/quentin/phd/turbulence/tensorboard/drone_model/
    fa_dict = {}
    for index in tqdm(data.index):
        if index < 2:
            fa_dict[index] = np.nan
            continue
        fa, xn1, xn1_hat = drone_model.forward_fa_direct_subtract(xn1=data["state"][index],
                                                                  xn=data["state"][index - 1],
                                                                  wn=data,
                                                                  dt=torch.tensor(data["dt"][index], dtype=float),
                                                                  tn=torch.tensor(data["time"][index - 1],
                                                                                  dtype=float))
        _, _, _, tau_x, tau_y, tau_z, fx, fy, fz, _, _, _ = fa
        phi, theta, psi, p, q, r, u, v, w, x, y, z = xn1
        hphi, htheta, hpsi, hp, hq, hr, hu, hv, hw, hx, hy, hz = xn1_hat
        fa_dict[index] = torch.tensor([fx, fy, fz, tau_x, tau_y, tau_z], dtype=torch.float)

        writer.add_scalars("fa's forces", {"fx": fx, "fy": fy, "fz": fz}, walltime=data.loc[index, "time"])
        writer.add_scalars("fa's moments", {"tau_x": tau_x, "tau_y": tau_y, "tau_z": tau_z},
                           walltime=data.loc[index, "time"])
        for var in ["phi", "theta", "psi", "p", "q", "r", "u", "v", "w", "x", "y", "z"]:
            eval(f"""writer.add_scalars("{var}", {{"ground truth": {var},
                                                   "estimator": h{var}}}, walltime=data.loc[index, "time"])""")
    data["fa"] = pd.Series(fa_dict)

    # Saving data
    new_data_path = utils.get_folder_path(data_path) + "fa.pkl"
    print(f"Saving: {new_data_path}")
    data.to_pickle(new_data_path)

    # # Adam optimization
    # optimizer = optim.Adam(drone_model.parameters(), lr=0.01)
    # progress = util.Progress(len(data) - 2, "Computing fa", "fa computed")
    # epochs = 5
    # fas = []
    # for i in range(1, len(data) - 1):
    #     if i == 1:
    #         loss_tmp = np.zeros(epochs)
    #     for epoch in range(epochs):
    #         optimizer.zero_grad()  # zero the gradient buffers
    #         loss = drone_model(xn1=data["state"][i + 1],
    #                            xn=data["state"][i],
    #                            wn=data["motor_speed"][i],
    #                            dt=data["dt"][i + 1],
    #                            tn=data["time"][i])
    #         if i == 1:
    #             loss_tmp[epoch] = loss
    #         loss.backward()
    #         optimizer.step()  # Does the update
    #     fas.append(drone_model.fa)
    #     progress.update()
    # plt.figure()
    # plt.plot(loss_tmp)
    # plt.show()

    def plot_compare_differentiators():
        """
        Plots the linear speed and the angular velocity over time for Euler and first order differentiation to compare
        them.
        """
        # Run in terminal: tensorboard --logdir /Users/quentin/phd/turbulence/tensorboard/drone_model_differentiators/
        _now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _writer = SummaryWriter(f"{f.workspace_path}tensorboard/drone_model_differentiators/{now}/")

        # Euler derivative compute
        _linear_speed = (data["trans"][1:].values - data["trans"][:len(data) - 1].values) / dt

        def _angular_velocity_euler(_x: pd.Series) -> np.ndarray:
            """
            :return: The angular velocity computed form 2 quaternions and a time difference.
            Used with DataFrame.apply().
            """
            qn1: np.ndarray = data["quat"][_x.name]
            qn: np.ndarray = data["quat"][_x.name - 1]
            delta_t: float = _x.values[0]

            dq_dt = (qn1 - qn) / delta_t  # Euler differentiator
            q0, q1, q2, q3 = qn1
            g_matrix = np.array([[-q1, q0, q3, -q2],
                                 [-q2, -q3, q0, q1],
                                 [-q3, q2, -q1, q0]])

            return (2 * g_matrix @ dq_dt.reshape((4, 1))).reshape((3,))

        _angular_velocity = data["dt"][1:].to_frame(0).apply(_angular_velocity_euler, axis=1)

        [_writer.add_scalars("Linear speed", {"Euler x": _x[0], "Euler y": _x[1], "Euler z": _x[2]},
                             walltime=data.loc[_i, "time"]) for _x, _i in zip(_linear_speed, _linear_speed.index)]
        [_writer.add_scalars("Angular velocity", {"Euler x": _x[0], "Euler y": _x[1], "Euler z": _x[2]},
                             walltime=data.loc[_i, "time"]) for _x, _i in zip(_angular_velocity,
                                                                              _angular_velocity.index)]

        # First order(n=6 q=4) derivative plot
        def fo_differentiator_n6_q4_trans(row):
            """
            Differentiate the translation using a first order, 6 past measures, and 4 ?. Used with DataFrame.apply().
            """
            ind = row.name
            if ind == 0:
                return np.nan
            elif 0 < ind < 5:
                return (data["trans"][ind] - data["trans"][ind - 1]) / dt[ind]
            else:
                return 6 * (1. / 3. * data["trans"][ind - 5]
                            - 17. / 12. * data["trans"][ind - 4]
                            + 2. * data["trans"][ind - 3]
                            - 1. / 3. * data["trans"][ind - 2]
                            - 7. / 3. * data["trans"][ind - 1]
                            + 7. / 4. * data["trans"][ind]) / (data.loc[ind, "time"] - data.loc[ind - 5, "time"])

        _linear_speed = data["trans"][1:].to_frame(0).apply(fo_differentiator_n6_q4_trans, axis=1)

        def fo_differentiator_n6_q4_quat(row):
            """
            Differentiate the rotation using a first order, 6 past measures, and 4 ?. Used with DataFrame.apply().
            """
            ind = row.name
            if ind == 0:
                return np.nan
            qn1 = data["quat"][ind]
            if 0 < ind < 5:
                qn = data["quat"][ind - 1]
                dq_dt = (qn1 - qn) / dt[ind]  # Euler differentiator
            else:
                dq_dt = 6 * (1. / 3. * data["quat"][ind - 5]
                             - 17. / 12. * data["quat"][ind - 4]
                             + 2. * data["quat"][ind - 3]
                             - 1. / 3. * data["quat"][ind - 2]
                             - 7. / 3. * data["quat"][ind - 1]
                             + 7. / 4. * data["quat"][ind]) / (data.loc[ind, "time"] - data.loc[ind - 5, "time"])
            q0, q1, q2, q3 = qn1
            g_matrix = np.array([[-q1, q0, q3, -q2],
                                 [-q2, -q3, q0, q1],
                                 [-q3, q2, -q1, q0]])

            return (2 * g_matrix @ dq_dt.reshape((4, 1))).reshape((3,))

        _angular_velocity = data["dt"][1:].to_frame(0).apply(fo_differentiator_n6_q4_quat, axis=1)

        # First order(n=6 q=4) derivative plot
        [_writer.add_scalars("Linear speed", {"FO n=6 q=4 x": _x[0], "FO n=6 q=4 y": _x[1], "FO n=6 q=4 z": _x[2]},
                             walltime=data.loc[_i, "time"]) for _x, _i in zip(_linear_speed, _linear_speed.index)]
        [_writer.add_scalars("Angular velocity", {"FO n=6 q=4 x": _x[0], "FO n=6 q=4 y": _x[1], "FO n=6 q=4 z": _x[2]},
                             walltime=data.loc[_i, "time"]) for _x, _i in zip(_angular_velocity,
                                                                              _angular_velocity.index)]

    # plot_compare_differentiators()


def ode_solving():
    """
    Solve the FSDroneModel dot_x = FSDroneModel.f(x, u) using pytorch ODE solver.
    """
    f = utils.DataFolder("euroc_mav")
    flight_number = 0
    data_path = f.get_unique_file_path(".pkl", f.folders["intermediate"][flight_number], "fa")
    data = pd.read_pickle(data_path)

    drone_model = AscTecFireflyDroneModel(m=0.64, g=9.81, kt=6.546e-6, km=1.2864e-7, l_arm=0.215,
                                          inertia_matrix=np.array([[10.007e-3, 0., 0.],
                                                                   [0., 10.2335e-3, 0],
                                                                   [0., 0., 8.1e-3]]))

    # phi, theta, psi, p, q, r, u, v, w, x, y, z
    initial_state = data["state"][1]
    initial_state = torch.zeros(12).float()
    t_tab = torch.tensor(list(data["time"][1:])).float()[500:2500]  # only when the drone is flying,
    # no ground reaction
    t_tab = t_tab - t_tab[0]

    # Motor speed
    def motor_speed(t: float) -> Iterable:
        ms = torch.stack(list(data["motor_speed"])).numpy()
        time_array = t_tab.numpy()
        return torch.tensor([np.interp(t, time_array, ms[:, _i]) for _i in range(6)]).float()

    # array solution(phi, theta, psi, p, q, r, u, v, wn, x, y, z)
    states = odeint(lambda t, x: drone_model.f(time=t, state=x, w_func=motor_speed,
                                               fa=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()),
                    initial_state, t_tab)

    def plot_simulated_drone_position(x_start, x_end):
        fig, axs = plt.subplots(3)
        inds = [i for i, e in enumerate(t_tab) if x_start < e < x_end]
        axs[0].plot(t_tab[inds], states[:, 9][inds], c='r')
        axs[0].set_xlabel("temps")
        axs[0].set_ylabel("x")
        axs[1].plot(t_tab[inds], states[:, 10][inds], c='b')
        axs[1].set_xlabel("temps")
        axs[1].set_ylabel("y")
        axs[2].plot(t_tab[inds], states[:, 11][inds], c='g')
        axs[2].set_xlabel("temps")
        axs[2].set_ylabel("z")
        plt.tight_layout()
        plt.show()

    def plot_motor_speed():
        fig, ax = plt.subplots(1, 1)
        for i in range(6):
            ax.plot(data["time"], [e[i] for e in data["motor_speed"]], label=f"{i}")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()

    print()
    # plot_simulated_drone_position(0, 20)
    # plot_motor_speed()


def estimate_euroc_mav_parameters():
    f = utils.DataFolder("euroc_mav")
    data_path = f.get_unique_file_path(".pkl", f.folders["intermediate"][flight_number], "fa")
    data = pd.read_pickle(data_path)

    drone_model = AscTecFireflyDroneModel(m=0.64, g=9.81, kt=6.546e-6, km=1.2864e-7, l_arm=0.215,
                                          inertia_matrix=np.array([[10.007e-3, 0., 0.],
                                                                   [0., 10.2335e-3, 0],
                                                                   [0., 0., 8.1e-3]]))

    # phi, theta, psi, p, q, r, u, v, w, x, y, z
    initial_state = data["state"][1]
    initial_state = torch.zeros(12).float()
    t_tab = torch.tensor(list(data["time"][1:])).float()[500:2500]  # only when the drone is flying,
    # no ground reaction
    t_tab = t_tab - t_tab[0]

    # Motor speed
    def motor_speed(t: float) -> Iterable:
        ms = torch.stack(list(data["motor_speed"])).numpy()
        time_array = t_tab.numpy()
        return torch.tensor([np.interp(t, time_array, ms[:, _i]) for _i in range(6)]).float()

    # array solution(phi, theta, psi, p, q, r, u, v, wn, x, y, z)
    states = odeint(lambda t, x: drone_model.f(time=t, state=x, w_func=motor_speed,
                                               fa=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()),
                    initial_state, t_tab)


if __name__ == '__main__':
    # compute_fa()
    # ode_solving()
    estimate_euroc_mav_parameters()
