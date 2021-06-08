import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import torch
import torch.nn as nn
from torchdiffeq import odeint
from typing import Callable, Iterable

from src.data.blackbird.blackbird_processing import BlackbirdDataset


class Quadrotor(nn.Module):
    """
    This class implements the quadrotor described in:
    -Sabatino, Francesco. ‘Quadrotor Control: Modeling, Nonlinear Control Design, and Simulation’, 2015, 67.
    Fixed frame and body frame are FRD (Forward Right Down).
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

    def f(self, t: torch.Tensor, state: torch.Tensor, ms_func: Callable,
          fa: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the state. x_dot = f(x, u, fa).
        :param t: scalar in seconds (Tensor of shape (1,)).
        :param state: state of the drone torch.Tensor[phi, theta, psi, p, q, r, u, v, w, x, y, z].
        :param ms_func: Command: 6 rotors angular velocity in rad/s.
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

        if callable(ms_func):
            motor_speed = ms_func(t.item())
        else:
            motor_speed = ms_func

        assert len(fa) == 12, "fa len > 12 fa supposed to be = _, _, _, tau_x, tau_y, tau_z, fx, fy, fz, _, _, _"
        assert len(motor_speed) == 4

        motor_speed_squared = motor_speed ** 2

        # # Quadrotor assuming the drone's body frame is in O_ABC from FIG 2.2:
        # Sabatino, Francesco. ‘Quadrotor Control: Modeling, Nonlinear Control Design, and Simulation’, 2015, 67.
        # tau_cmd_x = kt * l_arm * (w3 ** 2 - w1 ** 2)
        # tau_cmd_y = kt * l_arm * (w4 ** 2 - w2 ** 2)
        # tau_cmd_z = km * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
        # f_cmd_t = kt * (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2)

        # But here the model is using a Forward Right Down body frame's like in the blackbird dataset FIG1:
        # Antonini, Amado, Winter Guerra, Varun Murali, Thomas Sayre-McCord, and Sertac Karaman.
        # ‘The Blackbird Dataset: A Large-Scale Dataset for UAV Perception in Aggressive Flight’.
        # ArXiv:1810.01987 [Cs], 3 October 2018. http://arxiv.org/abs/1810.01987.
        # So the model becomes:
        # tau_cmd_x = 1 / sqrt(2) * kt * l_arm * (w3 ** 2 + w4 ** 2 - w1 ** 2 - w2 ** 2)
        # tau_cmd_y = 1 / sqrt(2) * kt * l_arm * (w1 ** 2 + w4 ** 2 - w2 ** 2 - w3 ** 2)
        # tau_cmd_z = km * (w2 ** 2 + w4 ** 2 - w1 ** 2 - w3 ** 2)
        # f_cmd_t = kt * (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2)

        # # Motor model derivable with pytorch
        # kt_l_arm_matrix = 1 * torch.sqrt(torch.tensor(2.)) * torch.tensor([[-1, -1, 1, 1],
        #                                                                    [1, -1, -1, 1],
        #                                                                    [0, 0, 0, 0],
        #                                                                    [0, 0, 0, 0]]) * kt * l_arm
        # km_matrix = torch.tensor([[0, 0, 0, 0],
        #                           [0, 0, 0, 0],
        #                           [-1, 1, -1, 1],
        #                           [0, 0, 0, 0]]) * km
        #
        # kt_matrix = torch.tensor([[0, 0, 0, 0],
        #                           [0, 0, 0, 0],
        #                           [0, 0, 0, 0],
        #                           [1, 1, 1, 1]]) * kt
        #
        # b = kt_l_arm_matrix + km_matrix + kt_matrix

        b = torch.tensor([[1.1929e-06, -6.6763e-07, -1.0711e-06, 5.8162e-07],
                          [1.1559e-06, 4.2571e-07, -1.0062e-06, -4.0577e-07],
                          [-1.0810e-07, 3.2904e-08, -2.3441e-07, 2.6588e-07],
                          [1.3094e-06, 1.1255e-06, 2.9530e-06, 2.4706e-06]])

        # Command converted to forces and torques
        drone_torsor = b @ motor_speed_squared

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

        du = r * v - q * w - self.g * sin(theta)
        dv = p * w - r * u + self.g * sin(phi) * cos(theta)
        dw = q * u - p * v + self.g * cos(theta) * cos(phi)

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

        # Derivative angular velocity
        dp = (iyy - izz) / ixx * r * q
        dq = (izz - ixx) / iyy * p * r
        dr = (ixx - iyy) / izz * p * q
        angular_velocity_mat = (torch.tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]) * dp +
                                torch.tensor([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]) * dq +
                                torch.tensor([0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]) * dr)
        dstate = dstate + angular_velocity_mat
        return dstate


def _test():
    # Physics parameters
    g = 9.81

    # Drone's parameters
    m = torch.tensor(1.56779)  # Drone's mass [kg]
    kt = torch.tensor(8.54858e-6)  # Thrust coefficient [kg.m/rad2] = [N.s2.rad-2]
    km = torch.tensor(1.6e-2)  # Torque coefficient [kg.m2/rad2] = [N.m.s2.rad-2]
    l_arm = torch.tensor(0.215)  # Length of drone's arm  [m]
    ixx = torch.tensor(0.0347563)  # Inertia matrix diagonal first parameter [kg.m2]
    iyy = torch.tensor(0.0458929)  # Inertia matrix diagonal first parameter [kg.m2]
    izz = torch.tensor(0.0977)  # Inertia matrix diagonal first parameter [kg.m2]

    coef_m = torch.tensor(1., requires_grad=True)
    coef_kt = torch.tensor(1., requires_grad=True)
    coef_km = torch.tensor(1., requires_grad=True)
    coef_l_arm = torch.tensor(1., requires_grad=True)
    coef_ixx = torch.tensor(1., requires_grad=True)
    coef_iyy = torch.tensor(1., requires_grad=True)
    coef_izz = torch.tensor(1., requires_grad=True)

    drone = Quadrotor(g=g, m=m, kt=kt, km=km, l_arm=l_arm, ixx=ixx, iyy=iyy, izz=izz,
                      coef_m=coef_m, coef_kt=coef_kt, coef_km=coef_km, coef_l_arm=coef_l_arm, coef_ixx=coef_ixx,
                      coef_iyy=coef_iyy, coef_izz=coef_izz)
    x_dot = drone.f(t=torch.zeros(1), state=torch.zeros(12),
                    ms_func=torch.tensor([1200., 1200., 1200., 1200.]),
                    fa=torch.tensor([0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0.], requires_grad=True))

    x_dot.sum().backward()
    for e in [coef_m, coef_kt, coef_km, coef_l_arm, coef_ixx, coef_iyy, coef_izz]:
        print(e.grad)

    time = torch.linspace(0, 10, 100)
    initial_state = torch.zeros(12)

    def ms_cmd(t: float) -> Iterable:
        ms_start = torch.tensor([0., 0., 0., 0.])
        ms_final = torch.tensor([1200., 1200., 1200., 1200.])
        ms = torch.stack((ms_start, ms_final))
        time_array = torch.tensor([0., 10.])
        ms_interp = torch.tensor([np.interp(t, time_array, ms[:, j]) for j in range(4)]).float()
        return ms_interp

    states = odeint(lambda t, x: drone.f(t=t, state=x, ms_func=ms_cmd, fa=torch.zeros(12)),
                    initial_state, time, method="rk4")

    # State position
    fig_path = os.getcwd() + "/reports/quadrotor_tests/"
    p = pathlib.Path(fig_path)
    p.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Drone take-off")
    for i, name in zip(range(3), ("x", "y", "z")):
        ax.plot(time, states[:, (9 + i)].detach().numpy(), label=name)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State position[m]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_position.png")
    plt.show()

    pass


def _test_blackbird_motor_speed_matrix():
    fig_path = os.getcwd() + "/reports/quadrotor_tests/"
    path_manager = pathlib.Path(fig_path)
    path_manager.mkdir(parents=True, exist_ok=True)

    # Physics parameters
    g = 9.81

    # Blackbird's parameters
    m = torch.tensor(0.873)  # Drone's mass [kg]
    kt = torch.tensor(1.91e-06)  # Thrust coefficient [kg.m/rad2] = [N.s2.rad-2]
    km = torch.tensor(2.6e-7)  # Torque coefficient [kg.m2/rad2] = [N.m.s2.rad-2]
    l_arm = torch.tensor(0.08)  # Length of drone's arm  [m]
    ixx = torch.tensor(4.9e-3)  # Inertia matrix diagonal first parameter [kg.m2]
    iyy = torch.tensor(4.9e-3)  # Inertia matrix diagonal first parameter [kg.m2]
    izz = torch.tensor(6.9e-3)  # Inertia matrix diagonal first parameter [kg.m2]

    data = blackbird.BlackbirdDataset(flight_name="mouse", flight_yaw="yawForward", flight_speed="maxSpeed7p0")

    all_motor_permutations = itertools.permutations([0, 1, 2, 3], 4)
    # actuator_dynamics = [{"motor_permutation": [], "b": np.ndarray(4, 4)}]
    permuted_motors_dynamic = []

    # Identifying when the drone start to fly and stops
    start, stop = 0, 0
    start_done = False
    for i, (sample, label) in enumerate(data):
        if sample["motor_speed"][0] > 1000 and not start_done:
            start = i
            start_done = True
        elif start_done and sample["motor_speed"][0] < 900:
            stop = i
            break

    # Computing actuator dynamic matrix for each motor permutations
    for motor_permutation in all_motor_permutations:
        # Identifying motor's matrix
        left, right = [], []
        # Compute angular and linear acceleration of the state
        for i, e in enumerate(data):
            if i < start or i > stop:
                continue
            sample, label = e
            xn = sample["state"]
            xn1 = label["state"]
            dt = (label["t"] - sample["t"])
            p_dot, q_dot, r_dot, _, _, w_dot = (xn1 - xn)[3:9] / dt
            phi, theta, psi, p, q, r, u, v, w, x, y, z = xn
            taux = p_dot * ixx + (izz - iyy) * r * q
            tauy = q_dot * iyy + (ixx - izz) * p * r
            tauz = r_dot * izz + (iyy - ixx) * p * q
            ft = m * (q * u - p * v) + g * torch.cos(phi) * torch.cos(theta) - w_dot
            right.append(sample["motor_speed"][list(motor_permutation)] ** 2)
            left.append(torch.tensor([taux, tauy, tauz, ft]))
        right = torch.stack(right).T
        left = torch.stack(left).T
        b = left @ right.T @ (right @ right.T).pinverse()
        print(b)
        permuted_motors_dynamic.append({"motor_permutation": motor_permutation, "actuator_dynamic": b})
    pass



def _test_blackbird():
    fig_path = os.getcwd() + "/reports/quadrotor_tests/"
    p = pathlib.Path(fig_path)
    p.mkdir(parents=True, exist_ok=True)

    # Physics parameters
    g = 9.81

    # Blackbird's parameters
    m = torch.tensor(0.915)  # Drone's mass [kg]
    kt = torch.tensor(2.0699917818129613e-06)  # Thrust coefficient [kg.m/rad2] = [N.s2.rad-2]
    km = torch.tensor(5.11e-8)  # Torque coefficient [kg.m2/rad2] = [N.m.s2.rad-2]
    l_arm = torch.tensor(0.13)  # Length of drone's arm  [m]
    ixx = torch.tensor(4.9e-2)  # Inertia matrix diagonal first parameter [kg.m2]
    iyy = torch.tensor(4.9e-2)  # Inertia matrix diagonal first parameter [kg.m2]
    izz = torch.tensor(6.9e-2)  # Inertia matrix diagonal first parameter [kg.m2]

    # data = blackbird.BlackbirdDataset(flight_name="ampersand", flight_yaw="yawForward",
    # flight_speed="maxSpeed1p0")
    data = blackbird.BlackbirdDataset(flight_name="mouse", flight_yaw="yawForward", flight_speed="maxSpeed7p0")

    # Identifying when the drone start to fly and stops
    start, stop = 0, 0
    start_done = False
    for i, (sample, label) in enumerate(data):
        if sample["motor_speed"][0] > 1000 and not start_done:
            start = i
            start_done = True
        elif start_done and sample["motor_speed"][0] < 900:
            stop = i
            break
    if stop - start > 100:
        stop = start + 100

    with torch.no_grad():
        drone = Quadrotor(g=g, m=m, kt=kt, km=km, l_arm=l_arm, ixx=ixx, iyy=iyy, izz=izz)

        time = torch.tensor([e[0]["t"] for e in data])[start:stop]
        states_gt = torch.stack([e[0]["state"] for e in data])[start:stop, :]
        initial_state = data[start][0]["state"]

        def ms_cmd(t: float) -> Iterable:
            ms = torch.stack([e[0]["motor_speed"] for e in data])[start:stop, :]
            ms_interp = torch.tensor([np.interp(t, time, ms[:, j]) for j in range(4)]).float()
            return ms_interp

        states = odeint(lambda t, x: drone.f(t=t, state=x, ms_func=ms_cmd, fa=torch.zeros(12)),
                        initial_state, time, method="euler")
    # State orientation
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Blackbird data with quadrotor model")
    for i, name, color in zip(range(3), ("phi", "theta", "psi"), ("blue", "green", "purple")):
        ax.plot(time, states[:, (0 + i)].detach().numpy(), label=name, color=color)
        ax.plot(time, states_gt[:, (0 + i)], label=name + "ground_truth", linestyle="--", color=color)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State orientation[rad]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_orientation.png")
    plt.show()

    # State angular velocity
    p = pathlib.Path(fig_path)
    p.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Blackbird data with quadrotor model")
    for i, name, color in zip(range(3), ("x", "y", "z"), ("blue", "green", "purple")):
        ax.plot(time, states[:, (3 + i)].detach().numpy(), label=name, color=color)
        ax.plot(time, states_gt[:, (3 + i)], label=name + "ground_truth", linestyle="--", color=color)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State angular velocity [rad/s]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_angular_velocity.png")
    plt.show()

    # State linear speed
    p = pathlib.Path(fig_path)
    p.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Blackbird data with quadrotor model")
    for i, name, color in zip(range(3), ("x", "y", "z"), ("blue", "green", "purple")):
        ax.plot(time, states[:, (6 + i)].detach().numpy(), label=name, color=color)
        ax.plot(time, states_gt[:, (6 + i)], label=name + "ground_truth", linestyle="--", color=color)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State linear speed [m/s]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_linear_speed.png")
    plt.show()

    # State position
    p = pathlib.Path(fig_path)
    p.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=200)
    ax.set_title(f"Blackbird data with quadrotor model")
    for i, name, color in zip(range(3), ("x", "y", "z"), ("blue", "green", "purple")):
        ax.plot(time, states[:, (9 + i)].detach().numpy(), label=name, color=color)
        ax.plot(time, states_gt[:, (9 + i)], label=name + "ground_truth", linestyle="--", color=color)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State position [m]")
    plt.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(fig_path + "state_position.png")
    plt.show()

    pass


if __name__ == "__main__":
    # _test()
    _test_blackbird_motor_speed_matrix()
    # _test_blackbird()
