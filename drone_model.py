import torch
import torch.nn as nn
# import torch.optim as optim
from torchdiffeq import odeint  # https://github.com/rtqichen/torchdiffeq
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import util


# 05 June 2020
# DRONE MODEL

# Every equations in this code are described in Francesco Sabatino's Thesis
# Sabatino, Francesco. ‘Quadrotor Control: Modeling, Nonlinear Control Design, and Simulation’, 2015, 67.

# noinspection PyAbstractClass
class FSDroneModel(nn.Module):
    def __init__(self, m: float, g: float, b: float, d: float, l_arm: float):
        """
        Drone model base of Sabatino Francesco thesis adapted to an hexarotor.

        :param m: mass of the drone (scalar) (kg).
        :param g: gravity constant (scalar) (9.81m/s2).
        :param b: Thrust factor (kg.m/rad2).
        :param d: Drag factor (kg.m2/rad2).
        :param l_arm: Drone's arm distance (scalar) (m).
        """
        super().__init__()
        # Parameters
        self.m = torch.tensor(m, dtype=torch.float)  # masse
        self.g = torch.tensor(g, dtype=torch.float)  # Newton's constant
        self.b = torch.tensor(b, dtype=torch.float)  # Thrust factor
        self.d = torch.tensor(d, dtype=torch.float)  # Drag factor
        self.l_arm = torch.tensor(l_arm, dtype=torch.float)  # Drone's arm distance

        self.inertia_matrix = torch.tensor([[1 / 6 * self.m * self.l_arm ** 2, 0, 0],
                                            [0, 1 / 6 * self.m * self.l_arm ** 2, 0],
                                            [0, 0, 1 / 6 * self.m * self.l_arm ** 2]], dtype=torch.float)

        self.fa = nn.parameter.Parameter(torch.zeros(12, dtype=torch.float, requires_grad=True))

    # Differential equation
    def f(self, time: torch.tensor, state: torch.tensor, w: torch.tensor, fa: torch.tensor) -> torch.tensor:
        """
        Compute the derivative of the state. x_dot = f(x, u, fa)

        :param time: scalar in seconds (tensor of shape (1,).

        :param state: state of the drone torch.tensor[phi, theta, psi, p, q, r, u, v, w, x, y, z].

        :param w: Command: 6 rotors angular velocity in rad/s.

        :param fa: External perturbation in N=kg.m/s2.

        :return: Return the derivative of the state.
        """
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
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]
        w4 = w[3]
        w5 = w[4]
        w6 = w[5]

        f_cmd_t = self.b * (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2 + w5 ** 2 + w6 ** 2)
        tau_cmd_x = self.b * self.l_arm * (- w2 ** 2 + w5 ** 2 + 1 / 2 * (- w1 ** 2 - w3 ** 2 + w4 ** 2 + w6 ** 2))
        tau_cmd_y = 3. ** 0.5 / 2. * self.b * self.l_arm * (- w1 ** 2 + w3 ** 2 + w4 ** 2 - w6 ** 2)
        tau_cmd_z = self.d * (- w1 ** 2 + w2 ** 2 - w3 ** 2 + w4 ** 2 - w5 ** 2 + w6 ** 2)

        # -----------------------------------------------------------

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

    def forward(self, xn1: torch.tensor, xn: torch.tensor, wn: torch.tensor,
                dt: torch.tensor, tn: torch.tensor) -> torch.tensor:
        """
        Compute the norm of the difference between the state and a state estimate. fa derivable in pytorch.

        :param xn1: state at time tn+1.

        :param xn: state at time tn.

        :param wn: Rotors speed at time tn.

        :param dt: n+1 time step dt_n+1 = tn+1 - tn.

        :param tn: time of the measure in s.

        :return: torch.tensor of dimension (1,).
        """
        return torch.norm(xn1 - (xn + dt * self.f(tn, xn, wn, self.fa)))

    def fa_direct_subtract(self, xn1: torch.tensor, xn: torch.tensor, wn: torch.tensor,
                           dt: torch.tensor, tn: torch.tensor) -> torch.tensor:
        """
        A different forward to not used pytorch optimizers because computing fa is just a subtraction.

        :param xn1:

        :param xn:

        :param wn:

        :param dt:

        :param tn: (s)

        :return:
        """
        ixx = self.inertia_matrix[0, 0]
        iyy = self.inertia_matrix[1, 1]
        izz = self.inertia_matrix[2, 2]
        c = torch.tensor([0., 0., 0., 1 / ixx, 1 / iyy, 1 / izz,
                          1 / self.m, 1 / self.m, 1 / self.m, 0., 0., 0.], dtype=torch.float)

        return (xn1 - (xn + dt * self.f(tn, xn, wn, torch.tensor(12, dtype=torch.float)))) / (dt * c)


def cos(var):
    return torch.cos(var)


def sin(var):
    return torch.sin(var)


def tan(var):
    return torch.tan(var)


def compute_fa():
    f = util.DataFolder("euroc_mav")
    flight_number = 2
    data_path = f.get_unique_file_path(".pkl", f.folders["intermediate"][flight_number], "sensors_synchronised")
    print(f"Loading: {data_path}")
    data = pd.read_pickle(data_path)
    data: pd.DataFrame

    data["time"] = (data["time"] - data["time"][0])

    # data["vicon_pose"] is tf_drone_origin
    data["vicon_pose"] = data["vicon_pose"].apply(lambda x: x.inv())
    # Now data["vicon_pose"] is tf_origin_drone

    data["trans"] = data["vicon_pose"].apply(lambda x: x.get_trans())
    data["rot"] = data["vicon_pose"].apply(lambda x: x.get_rot())

    dt = pd.Series(data["time"][1:].values - data["time"][:len(data) - 1].values)
    dt.index = range(1, len(dt) + 1)

    # Euler derivative compute
    # Linear speed in origin
    linear_speed = (data["trans"][1:].values - data["trans"][:len(data) - 1].values) / dt
    # Linear speed in drone
    linear_speed = linear_speed.to_frame(0)\
        .apply(lambda x: data["vicon_pose"][x.name].get_rot().T @ x.values[0], axis=1)

    angular_velocity = ((data["rot"][1:].values - data["rot"][:len(data) - 1].values) / dt).to_frame(0) \
        .apply(lambda x: data["rot"][1:][x.name].T @ x.values[0], axis=1) \
        .apply(lambda x: np.array([(x[2, 1] - x[1, 2]) / 2,
                                   (x[0, 2] - x[2, 0]) / 2,
                                   (x[1, 0] - x[0, 1]) / 2]))

    data["linear_speed"] = linear_speed
    data["angular_velocity"] = angular_velocity
    data["dt"] = dt

    data["state"] = np.zeros(len(data))
    data[["state"]] = data[["state"]].apply(
        lambda x: torch.from_numpy(np.hstack([data["vicon_pose"][x.name].get_rot_euler("xyz", True),
                                              data["angular_velocity"][x.name],
                                              data["linear_speed"][x.name],
                                              data["trans"][x.name]]).astype(np.float)), axis=1)

    drone_model = FSDroneModel(m=3., g=9.81, b=0.01, d=0.01, l_arm=0.2)

    # Direct computation
    data["fa"] = np.nan

    def extract_fa(x):
        fa = drone_model.fa_direct_subtract(xn1=data["state"][x.name],
                                            xn=data["state"][x.name - 1],
                                            wn=data["motor_speed"][x.name - 1],
                                            dt=data["dt"][x.name],
                                            tn=data["time"][x.name - 1])
        _, _, _, tau_x, tau_y, tau_z, f_x, f_y, f_z, _, _, _ = fa
        return torch.tensor([f_x, f_y, f_z, tau_x, tau_y, tau_z], dtype=torch.float)

    data[["fa"]] = data[["fa"]][2:].apply(extract_fa, axis=1)
    new_data_path = util.get_folder_path(data_path) + "fa.pkl"
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
    drone_model = FSDroneModel(m=3., g=9.81, b=0.01, d=0.01, l_arm=0.2)

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
