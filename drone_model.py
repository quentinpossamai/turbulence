"""
This file contains a drone models to compute the turbulence forces and torques labels.
# Run in terminal: tensorboard --logdir /Users/quentin/phd/turbulence/tensorboard/drone_model/
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

    def f(self, time: torch.Tensor, state: torch.Tensor, w: torch.Tensor, fa: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the state. x_dot = f(x, u, fa)
        :param time: scalar in seconds (Tensor of shape (1,)).
        :param state: state of the drone torch.Tensor[phi, theta, psi, p, q, r, u, v, w, x, y, z].
        :param w: Command: 6 rotors angular velocity in rad/s.
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

        # Hexarotor
        u = w ** 2

        kt = self.kt
        km = self.km
        la = self.l_arm
        c = 3 ** 0.5 / 2

        f_cmd_t = self.kt * torch.sum(u)
        b = torch.tensor([[-0.5 * la * kt, -la * kt, -0.5 * la * kt, 0.5 * la * kt, la * kt, 0.5 * la * kt],
                          [-c * la * kt, 0, c * la * kt, c * la * kt, 0, -c * la * kt],
                          [-km, km, -km, km, -km, km]])
        moments = b @ u
        tau_cmd_x = moments[0]
        tau_cmd_y = moments[1]
        tau_cmd_z = moments[2]

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
    data["vicon_pose"] = data["vicon_pose"].apply(lambda x_: x_.inv())
    # Now data["vicon_pose"] is tf_origin_drone

    data["trans"] = data["vicon_pose"].apply(lambda x_: x_.get_trans())
    data["quat"] = data["vicon_pose"].apply(lambda x_: x_.get_pose()[1])

    dt = pd.Series(data["time"][1:].values - data["time"][:len(data) - 1].values)
    dt.index = range(1, len(dt) + 1)

    # Euler derivative compute
    # Linear speed in origin
    linear_speed = (data["trans"][1:].values - data["trans"][:len(data) - 1].values) / dt
    # Linear speed in drone
    linear_speed = linear_speed.to_frame(0) \
        .apply(lambda x_: data["vicon_pose"][x_.name].get_rot().T @ x_.values[0], axis=1)

    def angular_velocity_calc(x_: pd.Series) -> np.ndarray:
        """
        :return: The angular velocity computed form 2 quaternions and a time difference. Used with DataFrame.apply().
        """
        qn1: np.ndarray = data["quat"][x_.name]
        qn: np.ndarray = data["quat"][x_.name - 1]
        delta_t: float = x_.values[0]

        dq_dt = (qn1 - qn) / delta_t
        q0, q1, q2, q3 = qn1
        g_matrix = np.array([[-q1, q0, q3, -q2],
                             [-q2, -q3, q0, q1],
                             [-q3, q2, -q1, q0]])

        return (2 * g_matrix @ dq_dt.reshape((4, 1))).reshape((3,))

    angular_velocity = dt.to_frame(0).apply(angular_velocity_calc, axis=1)

    data["linear_speed"] = linear_speed
    data["angular_velocity"] = angular_velocity
    data["dt"] = dt

    data["state"] = np.zeros(len(data))
    data[["state"]] = data[["state"]].apply(
        lambda x_: torch.from_numpy(np.hstack([data["vicon_pose"][x_.name].get_rot_euler("xyz", True),
                                               data["angular_velocity"][x_.name],
                                               data["linear_speed"][x_.name],
                                               data["trans"][x_.name]]).astype(np.float)), axis=1)

    drone_model = AscTecFireflyDroneModel(m=0.64, g=9.81, kt=6.546e-6, km=1.2864e-7, l_arm=0.215,
                                          inertia_matrix=np.array([[10.007e-3, 0., 0.],
                                                                   [0., 10.2335e-3, 0],
                                                                   [0., 0., 8.1e-3]]))
    # Fa computation and plots
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{f.workspace_path}tensorboard/drone_model/{now}/")
    fa_dict = {}
    for index in tqdm(data.index):
        if index < 2:
            fa_dict[index] = np.nan
            continue
        fa, xn1, xn1_hat = drone_model.forward_fa_direct_subtract(xn1=data["state"][index],
                                                                  xn=data["state"][index - 1],
                                                                  wn=torch.from_numpy(
                                                                      data["motor_speed"][index - 1]).float(),
                                                                  dt=torch.tensor(data["dt"][index], dtype=float),
                                                                  tn=torch.tensor(data["time"][index - 1],
                                                                                  dtype=float))
        _, _, _, tau_x, tau_y, tau_z, fx, fy, fz, _, _, _ = fa
        phi, theta, psi, p, q, r, u, v, w, x, y, z = xn1
        hphi, htheta, hpsi, hp, hq, hr, hu, hv, hw, hx, hy, hz = xn1_hat
        fa_dict[index] = torch.tensor([fx, fy, fz, tau_x, tau_y, tau_z], dtype=torch.float)

        writer.add_scalars("fa's forces", {"fx": fx, "fy": fy, "fz": fz}, data.loc[index, "time"])
        writer.add_scalars("fa's moments", {"tau_x": tau_x, "tau_y": tau_y, "tau_z": tau_z}, data.loc[index, "time"])

        for var in ["phi", "theta", "psi", "p", "q", "r", "u", "v", "w", "x", "y", "z"]:
            eval(f"""writer.add_scalars("{var}", {{"ground truth": {var},
                                                   "estimator": h{var}}}, data.loc[index, "time"])""")
    data["fa"] = pd.Series(fa_dict)

    # Saving data
    new_data_path = utils.get_folder_path(data_path) + "fa.pkl"
    print(f"Saving: {new_data_path}")
    data.to_pickle(new_data_path)
    print()

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

    # Plots
    def plot_position_6d():
        """
        Plots 6D position of data over time.
        """
        fig, axs = plt.subplots(3, 2)
        euler_angles = data["vicon_pose"].apply(lambda x: x.get_rot_euler(seq="xyz", degrees=True))
        axs[0, 0].plot(data["time"], [x[0] for x in data["trans"]], label="Position x")
        axs[1, 0].plot(data["time"], [x[1] for x in data["trans"]], label="Position y")
        axs[2, 0].plot(data["time"], [x[2] for x in data["trans"]], label="Position z")
        axs[0, 1].plot(data["time"], [x[0] for x in euler_angles], label="Euler Phi")
        axs[1, 1].plot(data["time"], [x[1] for x in euler_angles], label="Euler Theta")
        axs[2, 1].plot(data["time"], [x[2] for x in euler_angles], label="Euler Psi")
        for i in range(len(axs)):
            for j in range(len(axs[0])):
                axs[i, j].legend()
        plt.show()

    def plot_compare_differentiators():
        """
        Plots the linear speed and the angular velocity over time for Euler and first order differentiation to compare
        them.
        """
        # Euler derivative compute
        _linear_speed = (data["trans"][1:].values - data["trans"][:len(data) - 1].values) / dt
        _angular_velocity = ((data["rot"][1:].values - data["rot"][:len(data) - 1].values) / dt).to_frame(0) \
            .apply(lambda x: x.values[0] @ data["rot"][1:][x.name].T, axis=1) \
            .apply(lambda x: np.array([(x[1, 2] - x[2, 1]) / 2,
                                       (x[2, 0] - x[0, 2]) / 2,
                                       (x[0, 1] - x[1, 0]) / 2]))

        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(data["time"][1:], [x[0] for x in _linear_speed], label="Euler")
        axs[1, 0].plot(data["time"][1:], [x[1] for x in _linear_speed], label="Euler")
        axs[2, 0].plot(data["time"][1:], [x[2] for x in _linear_speed], label="Euler")
        axs[0, 1].plot(data["time"][1:], [x[0] for x in _angular_velocity], label="Euler")
        axs[1, 1].plot(data["time"][1:], [x[1] for x in _angular_velocity], label="Euler")
        axs[2, 1].plot(data["time"][1:], [x[2] for x in _angular_velocity], label="Euler")

        # First order(n=6 q=4) derivative plot
        def fo_differentiator_n6_q4_trans(x):
            """
            Differentiate the translation using a first order, 6 past measures, and 4 ?. Used with DataFrame.apply().
            """
            array = x.values[0]
            ind = x.name
            if ind == 0:
                return np.nan
            elif 0 < ind < 5:
                return (array - data["trans"][ind - 1]) / dt[ind]
            else:
                return (1. / 3. * data["trans"][ind - 5]
                        - 17. / 12. * data["trans"][ind - 4]
                        + 2. * data["trans"][ind - 3]
                        - 1. / 3. * data["trans"][ind - 2]
                        - 7. / 3. * data["trans"][ind - 1]
                        + 7. / 4. * data["trans"][ind]) / dt[ind]

        _linear_speed = data["trans"][1:].to_frame(0).apply(fo_differentiator_n6_q4_trans, axis=1)

        def fo_differentiator_n6_q4_rot(x):
            """
            Differentiate the rotation using a first order, 6 past measures, and 4 ?. Used with DataFrame.apply().
            """
            array = x.values[0]
            ind = x.name
            if ind == 0:
                return np.nan
            elif 0 < ind < 5:
                skew_sym_w = ((array - data["rot"][ind - 1]) / dt[ind]) @ data["rot"][x.name].T
            else:
                skew_sym_w = (1. / 3. * data["rot"][ind - 5]
                              - 17. / 12. * data["rot"][ind - 4]
                              + 2. * data["rot"][ind - 3]
                              - 1. / 3. * data["rot"][ind - 2]
                              - 7. / 3. * data["rot"][ind - 1]
                              + 7. / 4. * data["rot"][ind]) / dt[ind] @ data["rot"][x.name].T

            return np.array([(skew_sym_w[1, 2] - skew_sym_w[2, 1]) / 2,
                             (skew_sym_w[2, 0] - skew_sym_w[0, 2]) / 2,
                             (skew_sym_w[0, 1] - skew_sym_w[1, 0]) / 2])

        _angular_velocity = data["rot"][1:].to_frame(0).apply(fo_differentiator_n6_q4_rot, axis=1)

        # First order(n=6 q=4) derivative plot
        axs[0, 0].plot(data["time"][1:], [x[0] for x in _linear_speed], label="FO n=6 q=4")
        axs[1, 0].plot(data["time"][1:], [x[1] for x in _linear_speed], label="FO n=6 q=4")
        axs[2, 0].plot(data["time"][1:], [x[2] for x in _linear_speed], label="FO n=6 q=4")
        axs[0, 1].plot(data["time"][1:], [x[0] for x in _angular_velocity], label="FO n=6 q=4")
        axs[1, 1].plot(data["time"][1:], [x[1] for x in _angular_velocity], label="FO n=6 q=4")
        axs[2, 1].plot(data["time"][1:], [x[2] for x in _angular_velocity], label="FO n=6 q=4")

        for i in range(len(axs)):
            for j in range(len(axs[0])):
                axs[i, j].legend()
        plt.show()


def ode_solving():
    """
    Solve the FSDroneModel dot_x = FSDroneModel.f(x, u) using pytorch ODE solver.
    """
    drone_model = AscTecFireflyDroneModel(g=9.81, kt=0.01, km=0.01, l_arm=0.215,
                                          inertia_matrix=np.array([[10.007e-3, 0., 0.],
                                                                   [0., 10.2335e-3, 0.],
                                                                   [0., 0., 8.1e-3]]))

    # phi, theta, psi, p, q, r, u, v, w, x, y, z
    initial_state = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=torch.float)
    t_tab = torch.arange(0, 10, 0.1)  # temps

    # array solution(phi, theta, psi, p, q, r, u, v, wn, x, y, z)
    states = odeint(lambda t, x: drone_model.f(time=t, state=x, w=torch.tensor([0, 0, 0, 0, 0, 0]),
                                               fa=torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
                    initial_state, t_tab)

    fig, axs = plt.subplots(3)
    axs[0].plot(t_tab, states[:, 9], 'r')
    axs[0].set_xlabel("temps")
    axs[0].set_ylabel("x")

    axs[1].plot(t_tab, states[:, 10], 'b')
    axs[1].set_xlabel("temps")
    axs[1].set_ylabel("y")

    axs[2].plot(t_tab, states[:, 11], 'g')
    axs[2].set_xlabel("temps")
    axs[2].set_ylabel("z")

    plt.show()
    pass


if __name__ == '__main__':
    compute_fa()
