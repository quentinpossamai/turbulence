"""
This file contains a drone models to compute the turbulence forces and torques labels.
"""
import pickle
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint  # https://github.com/rtqichen/torchdiffeq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import os
from scipy.signal import savgol_filter
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

    def __init__(self,
                 g: torch.tensor,
                 m: torch.tensor,
                 kt: torch.tensor,
                 km: torch.tensor,
                 l_arm: torch.tensor,
                 ixx: torch.tensor,
                 iyy: torch.tensor,
                 izz: torch.tensor,

                 coef_m: torch.tensor = torch.tensor(1.),
                 coef_kt: torch.tensor = torch.tensor(1.),
                 coef_km: torch.tensor = torch.tensor(1.),
                 coef_l_arm: torch.tensor = torch.tensor(1.),
                 coef_ixx: torch.tensor = torch.tensor(1.),
                 coef_iyy: torch.tensor = torch.tensor(1.),
                 coef_izz: torch.tensor = torch.tensor(1.)):
        """
        Drone model base of Sabatino Francesco thesis adapted to an hexarotor.

        :param coef_m: mass of the drone (scalar) (kg).
        :param g: gravity constant (scalar) (9.81m/s2).
        :param coef_kt: Thrust factor (scalar) (kg.m/rad2).
        :param coef_km: Moment factor (scalar) (kg.m2/rad2).
        :param coef_l_arm: Drone's arm distance (scalar) (m).
        :param coef_ixx: x-axis component of the inertia matrix (kg.m2).
        :param coef_iyy: x-axis component of the inertia matrix (kg.m2).
        :param coef_izz: x-axis component of the inertia matrix (kg.m2).

        """
        super().__init__()
        # Drone model parameters.

        self.coef_m = coef_m
        self.coef_kt = coef_kt
        self.coef_km = coef_km
        self.coef_l_arm = coef_l_arm
        self.coef_ixx = coef_ixx
        self.coef_iyy = coef_iyy
        self.coef_izz = coef_izz

        self.m = m
        self.kt = kt
        self.km = km
        self.l_arm = l_arm
        self.ixx = ixx
        self.iyy = iyy
        self.izz = izz

        self.g = g
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

        m = self.m * self.coef_m
        kt = self.kt * self.coef_kt
        km = self.km * self.coef_km
        l_arm = self.l_arm * self.coef_l_arm
        ixx = self.ixx * self.coef_ixx
        iyy = self.iyy * self.coef_iyy
        izz = self.izz * self.coef_izz

        assert len(fa) == 12, "fa len > 12 fa supposed to be = _, _, _, tau_x, tau_y, tau_z, fx, fy, fz, _, _, _"
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

        # Hexarotor force generation -----------------------------------------------------------------------------------
        # b = torch.tensor([[-0.5 * la, -la, -0.5 * la, 0.5 * la, la, 0.5 * la],
        #                   [3. ** 0.5 / 2. * la, 0., -3. ** 0.5 / 2. * la,
        #                   -3. ** 0.5 / 2. * la, 0., 3. ** 0.5 / 2. * la],
        #                   [-km, km, -km, km, -km, km],
        #                   [1., 1., 1., 1., 1., 1.]], dtype=torch.float)

        # make derivable to parameters
        l_arm_mat = torch.tensor([[-0.5, -1., -0.5, 0.5, 1., 0.5],
                                  [3. ** 0.5 / 2., 0., -3. ** 0.5 / 2., -3. ** 0.5 / 2., 0., 3. ** 0.5 / 2.],
                                  [0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0.]]) * l_arm
        km_mat = torch.tensor([[0., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0.],
                               [-1., 1., -1., 1., -1., 1.],
                               [0., 0., 0., 0., 0., 0.]]) * km
        const_mat = torch.tensor([[0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0.],
                                  [0., 0., 0., 0., 0., 0.],
                                  [1., 1., 1., 1., 1., 1.]])
        b = l_arm_mat + km_mat + const_mat

        # Theoretic hovering command
        # body = torch.tensor([0, 0, 0, m * self.g]).reshape((4, 1))
        # b_inv = torch.pinverse(b)
        # w_hovering_model = (b_inv @ body / kt) ** 0.5

        w = w_func(time.item())

        # Command converted to forces and torques
        cmd = kt * w ** 2
        drone_torsor = b @ cmd

        # Drone reaction -----------------------------------------------------------------------------------------------

        cos = torch.cos
        sin = torch.sin
        tan = torch.tan

        phi, theta, psi, p, q, r, u, v, w, x, y, z = state

        dphi = p + r * cos(phi) * tan(theta) + q * sin(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = r * cos(phi) / cos(theta) + q * sin(phi) / cos(theta)

        dp = 0.
        dq = 0.
        dr = 0.

        du = r * v - q * w + self.g * sin(theta)
        dv = p * w - r * u - self.g * sin(phi) * cos(theta)
        dw = q * u - p * v - self.g * cos(theta) * cos(phi)

        dx = (w * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
              - v * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)) + u * cos(psi) * cos(theta))
        dy = (v * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta))
              - w * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)) + u * cos(theta) * sin(psi))
        dz = w * cos(phi) * cos(theta) - u * sin(theta) + v * cos(theta) * sin(phi)

        dstate = torch.tensor([dphi, dtheta, dpsi, dp, dq, dr, du, dv, dw, dx, dy, dz], dtype=torch.float)

        # Tensor writing -----------------------------------------------------------------------------------------------

        # forces and torques
        f_t_factor = (torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]) / ixx +
                      torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]) / iyy +
                      torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]) / izz +
                      torch.tensor([0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.]) / m)

        drone_torsor_mat = torch.tensor([[0., 0., 0., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.],
                                         [1., 0., 0., 0.],
                                         [0., 1., 0., 0.],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 1.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.]])
        dstate = dstate + f_t_factor * (fa + drone_torsor_mat @ drone_torsor)

        # Angular velocity
        dp = (iyy - izz) / ixx * r * q
        dq = (izz - ixx) / iyy * p * r
        dr = (ixx - iyy) / izz * p * q
        angular_velocity_mat = (torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]) * dp +
                                torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]) * dq +
                                torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]) * dr)
        dstate = dstate + angular_velocity_mat
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
        m = self.m * self.coef_m
        ixx = self.ixx * self.coef_ixx
        iyy = self.iyy * self.coef_iyy
        izz = self.izz * self.coef_izz
        c = torch.tensor([0., 0., 0., 1 / ixx, 1 / iyy, 1 / izz,
                          1 / m, 1 / m, 1 / m, 0., 0., 0.], dtype=torch.float)

        xn1_hat = xn + dt * self.f(tn, xn, wn, fa=torch.zeros(12))
        fa = (xn1 - xn1_hat) / (dt * c)
        return fa, xn1, xn1_hat


def euroc_mav_compute_fa():
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

    drone_model = AscTecFireflyDroneModel(coef_m=0.64, g=9.81, coef_kt=6.546e-6, coef_km=1.2864e-7, coef_l_arm=0.215,
                                          coef_ixx=10.007e-3, coef_iyy=10.2335e-3, coef_izz=8.1e-3)
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
            exec(f"""writer.add_scalars("{var}", {{"ground truth": {var},
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


def euroc_mav_ode_solving(load_previous=True):
    """
    Solve the FSDroneModel dot_x = FSDroneModel.f(x, u) using pytorch ODE solver.
    """
    f = utils.DataFolder("euroc_mav")
    flight_number = 0
    data_path = f.get_unique_file_path(".pkl", f.folders["intermediate"][flight_number], "fa")
    data = pd.read_pickle(data_path)
    result_folder = f.folders["results"][flight_number] + "drone_parameters_estimation/"

    # Parameter's scaling coefficients initialization
    prm_name = ["m", "kt", "km", "l_arm", "ixx", "iyy", "izz"]  # Parameters name

    coef_m = torch.tensor(1., dtype=torch.float, requires_grad=True)
    coef_kt = torch.tensor(1., dtype=torch.float, requires_grad=True)
    coef_km = torch.tensor(1., dtype=torch.float, requires_grad=True)
    coef_l_arm = torch.tensor(1., dtype=torch.float, requires_grad=True)
    coef_ixx = torch.tensor(1., dtype=torch.float, requires_grad=True)
    coef_iyy = torch.tensor(1., dtype=torch.float, requires_grad=True)
    coef_izz = torch.tensor(1., dtype=torch.float, requires_grad=True)

    if load_previous is not True:
        # First initialization of the parameters
        m = torch.tensor(0.64)
        kt = torch.tensor(6.546e-6)
        km = torch.tensor(1.2864e-7)
        l_arm = torch.tensor(0.215)
        ixx = torch.tensor(10.007e-3)
        iyy = torch.tensor(10.2335e-3)
        izz = torch.tensor(8.1e-3)
    else:
        # Loading parameters from previous training
        # If the list is empty then there is no previous training file found.
        last_filename = sorted([e for e in next(os.walk(result_folder))[2] if (e[0] != ".") and (e[-1] == "l")])[-1]
        to_plot = pickle.load(open(result_folder + last_filename, "rb"))

        m = torch.tensor(to_plot["m"][-1])
        kt = torch.tensor(to_plot["kt"][-1])
        km = torch.tensor(to_plot["km"][-1])
        l_arm = torch.tensor(to_plot["l_arm"][-1])
        ixx = torch.tensor(to_plot["ixx"][-1])
        iyy = torch.tensor(to_plot["iyy"][-1])
        izz = torch.tensor(to_plot["izz"][-1])
        print(f"Previous training loaded at {result_folder + last_filename}.")

    drone_model = AscTecFireflyDroneModel(coef_m=coef_m, g=9.81, coef_kt=coef_kt, coef_km=coef_km, coef_l_arm=coef_l_arm,
                                          coef_ixx=coef_ixx, coef_iyy=coef_iyy, coef_izz=coef_izz,
                                          m=m, kt=kt, km=km, l_arm=l_arm, ixx=ixx, iyy=iyy, izz=izz)

    t_tab = torch.tensor(list(data["time"][1:])).float()[500:2500]  # only when the drone is flying, no ground reaction
    t_tab = t_tab - t_tab[0]
    initial_state = data["state"][500]

    # Motor speed
    def motor_speed(t: float) -> Iterable:
        ms = torch.stack(list(data["motor_speed"])).numpy()[500:2500]
        time_array = t_tab.numpy()
        return torch.tensor([np.interp(t, time_array, ms[:, i]) for i in range(6)]).float()

    # array solution(phi, theta, psi, p, q, r, u, v, wn, x, y, z)
    states = odeint(lambda t, x: drone_model.f(time=t, state=x, w_func=motor_speed,
                                               fa=drone_model.fa),
                    initial_state, t_tab)

    def plot_simulated_drone_position(x_start, x_end):
        fig, axs = plt.subplots(3, figsize=(19.2, 10.8), dpi=200)
        inds = [i for i, e in enumerate(t_tab) if x_start < e < x_end]
        axs[0].plot(t_tab[inds], states[:, 9][inds].detach(), c='r')
        axs[0].set_xlabel("temps")
        axs[0].set_ylabel("x")
        axs[1].plot(t_tab[inds], states[:, 10][inds].detach(), c='b')
        axs[1].set_xlabel("temps")
        axs[1].set_ylabel("y")
        axs[2].plot(t_tab[inds], states[:, 11][inds].detach(), c='g')
        axs[2].set_xlabel("temps")
        axs[2].set_ylabel("z")
        plt.tight_layout()
        plt.show()

    def plot_motor_speed():
        fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
        for i in range(6):
            ax.plot(data["time"], [e[i] for e in data["motor_speed"]], label=f"{i}")
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()

    print()
    plot_simulated_drone_position(0, 200)
    plot_motor_speed()


def estimate_euroc_mav_parameters(epochs: int, batch_size: int, lr: float, load_previous: bool = None):
    """
    The purpose of this function is to estimate the drone parameters according to the EuRoC MAV data.
    :param epochs: Number of epoch for training.
    :param batch_size: Size of batch for training.
    :param lr: Learning rate of Adam.
    :param load_previous: If True load most recent computed drone parameters.
    :return: The parameters of the drone m, kt, km, l_arm, ixx, iyy, izz.
    """
    # Loading dataset
    f = utils.DataFolder("euroc_mav")
    flight_number = 0
    data_path = f.get_unique_file_path(".pkl", f.folders["intermediate"][flight_number], "fa")
    data = pd.read_pickle(data_path)
    result_folder = f.folders["results"][flight_number] + "drone_parameters_estimation/"

    # Plotting tools
    # tensorboard --logdir /Users/quentin/phd/turbulence/euroc_mav/results/V1_01_easy/drone_parameters_estimation/
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(result_folder + now + "/")

    # Parameter's scaling coefficients initialization
    prm_name = ["m", "kt", "km", "l_arm", "ixx", "iyy", "izz"]  # Parameters name

    coef_m = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_kt = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_km = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_l_arm = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_ixx = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_iyy = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))
    coef_izz = nn.parameter.Parameter(torch.tensor(1., dtype=torch.float, requires_grad=True))

    if load_previous is not True:
        # First initialization of the parameters
        m = torch.tensor(0.64)
        kt = torch.tensor(6.546e-6)
        km = torch.tensor(1.2864e-7)
        l_arm = torch.tensor(0.215)
        ixx = torch.tensor(10.007e-3)
        iyy = torch.tensor(10.2335e-3)
        izz = torch.tensor(8.1e-3)
        to_plot = {"loss": [], "iteration": []}
        for e in prm_name:
            exec(f"""to_plot["{e}"] = [(coef_{e} * {e}).item()]""")
            exec(f"""writer.add_scalars("{e}", {{"{e}": (coef_{e} * {e}).item()}}, global_step=0)""")
        global_iteration = 1
        print(f"New training.")
    else:
        # Loading parameters from previous training
        # If the list is empty then there is no previous training file found.
        last_filename = sorted([e for e in next(os.walk(result_folder))[2] if (e[0] != ".") and (e[-1] == "l")])[-1]
        to_plot = pickle.load(open(result_folder + last_filename, "rb"))

        m = torch.tensor(to_plot["m"][-1])
        kt = torch.tensor(to_plot["kt"][-1])
        km = torch.tensor(to_plot["km"][-1])
        l_arm = torch.tensor(to_plot["l_arm"][-1])
        ixx = torch.tensor(to_plot["ixx"][-1])
        iyy = torch.tensor(to_plot["iyy"][-1])
        izz = torch.tensor(to_plot["izz"][-1])
        global_iteration = to_plot["iteration"][-1]
        print(f"Previous training loaded at {result_folder + last_filename}.")

    drone_model = AscTecFireflyDroneModel(coef_m=coef_m, g=9.81, coef_kt=coef_kt, coef_km=coef_km,
                                          coef_l_arm=coef_l_arm, coef_ixx=coef_ixx, coef_iyy=coef_iyy,
                                          coef_izz=coef_izz, m=m, kt=kt, km=km, l_arm=l_arm, ixx=ixx, iyy=iyy, izz=izz)

    for e in prm_name:
        exec(f"""print(f"scaled: {{drone_model.coef_{e}}} | k: {{drone_model.{e}}}")""")

    dataset_start, dateset_end = 500, 2500  # dataset_start > 0 because of nan angular and linear speed
    mc_states_train = torch.stack(list(data["state"][dataset_start:dateset_end]))
    time_train = torch.tensor(list(data["time"])).float()[dataset_start:dateset_end]  # only when the drone is
    # flying, no ground reaction
    time_train = time_train - time_train[0]

    # Continuous motor speed
    def motor_speed(t: float) -> Iterable:
        ms = torch.stack(list(data["motor_speed"])).numpy()[dataset_start:dateset_end]
        return torch.tensor([np.interp(t, time_train.numpy(),
                                       ms[:, i_motor_speed]) for i_motor_speed in range(6)]).float()

    # Adam optimization
    optimizer = torch.optim.Adam(drone_model.parameters(), lr=lr)

    batch_nb = len(time_train) // batch_size
    for epoch in range(epochs):
        losses = []
        for batch_i in tqdm(range(batch_nb + 1), desc=f"Epoch nº{epoch + 1}/{epochs}"):
            # Dividing data by batch
            if batch_nb == batch_i:
                time_batch = time_train[batch_i * batch_size:]
                mc_states_batch = mc_states_train[batch_i * batch_size:]
            else:
                time_batch = time_train[batch_i * batch_size:(batch_i + 1) * batch_size]
                mc_states_batch = mc_states_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            initial_state = mc_states_train[batch_i * batch_size]

            # Forward
            states = odeint(lambda t, x: drone_model.f(time=t, state=x, w_func=motor_speed,
                                                       fa=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).float()),
                            initial_state, time_batch, method="rk4")
            loss = torch.sqrt((mc_states_batch - states) ** 2).mean()
            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()
            optimizer.step()  # Does the update

            # Saving data
            writer.add_scalars("Loss", {"Training loss": loss.item()}, global_step=global_iteration)
            to_plot["loss"].append(loss.item())
            for e in prm_name:
                exec(f"""to_plot["{e}"].append((coef_{e} * {e}).item())""")
                exec(f"""writer.add_scalars("{e}", {{"{e}": (coef_{e} * {e}).item()}},
                         global_step=global_iteration)""")
            global_iteration += 1

    for e in prm_name:
        exec(f"""print(f"scaled: {{drone_model.coef_{e}}} | k: {{drone_model.{e}}}")""")

    def plot_loss_ov_epochs():
        fig, ax = plt.subplots(1, 1)
        ax.plot(to_plot["loss"])
        plt.tight_layout()
        plt.show()

    def plot_parameters_ov_epochs(window, degree, saving_path_name: str = None):
        fig, axs = plt.subplots(2, 4, figsize=[19.20, 10.80], dpi=200)
        i = 0
        for ax_row in axs:
            for ax in ax_row:
                name = list(to_plot.keys())[i]
                if name == "loss":
                    # ax.plot(np.arange(len(to_plot[name])) + 1,to_plot[name])  # Loss starts at 1
                    # Loss starts at 1
                    ax.plot(np.arange(len(to_plot[name])) + 1, savgol_filter(to_plot["loss"], window, degree))
                else:
                    ax.plot(range(len(to_plot[name])), to_plot[name])  # Parameters start at 0
                ax.set_ylabel(name)
                ax.set_xlabel("Iterations")
                ax.grid()
                [ax.axvline(run_iteration - 1, c="r") for run_iteration in to_plot["iteration"]]
                i += 1
        plt.tight_layout()
        if saving_path_name is not None:
            plt.savefig(saving_path_name)
        plt.show()

    # plot_loss_ov_epochs()
    # plot_parameters_ov_epochs(101, 3)
    # plot_parameters_ov_epochs(window=101, 3, saving_path_name=result_folder + f"{now}.png")

    to_plot["iteration"].append(len(to_plot["m"]))
    pickle.dump(to_plot, open(result_folder + f"{now}.pkl", "wb"))
    print()


if __name__ == '__main__':
    # euroc_mav_compute_fa()
    euroc_mav_ode_solving(load_previous=True)
    # estimate_euroc_mav_parameters(epochs=200, batch_size=32, lr=1e-2, load_previous=True)
