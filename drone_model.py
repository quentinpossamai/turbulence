import torch
import torch.nn as nn
from torchdiffeq import odeint  # https://github.com/rtqichen/torchdiffeq
import matplotlib.pyplot as plt

import util


# 05 June 2020
# DRONE MODEL

# Every equations in this code are described in Francesco Sabatino's Thesis
# Sabatino, Francesco. ‘Quadrotor Control: Modeling, Nonlinear Control Design, and Simulation’, 2015, 67.

class FSDroneModel(nn.Module):
    def __init__(self, m, g, b, d, l_arm):
        super().__init__()
        # Parameters
        self.m = torch.tensor(m, dtype=torch.float)  # masse
        self.g = torch.tensor(g, dtype=torch.float)  # Newton's constant
        self.b = torch.tensor(b, dtype=torch.float)  # Thrust factor
        self.d = torch.tensor(d, dtype=torch.float)  # Drag factor
        self.l_arm = torch.tensor(l_arm, dtype=torch.float)  # Drone's arm distance

        self.inertia_matrix = torch.tensor([[1/6 * self.m * self.l_arm ** 2, 0, 0],
                                           [0, 1/6 * self.m * self.l_arm ** 2, 0],
                                           [0, 0, 1/6 * self.m * self.l_arm ** 2]], dtype=torch.float)

        self.init_state = torch.empty(12)

    def state_initialization(self, phi, theta, psi, p, q, r, u, v, w, x, y, z):
        phi = torch.tensor(phi, dtype=torch.float)
        theta = torch.tensor(theta, dtype=torch.float)
        psi = torch.tensor(psi, dtype=torch.float)
        p = torch.tensor(p, dtype=torch.float)
        q = torch.tensor(q, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float)
        u = torch.tensor(u, dtype=torch.float)
        v = torch.tensor(v, dtype=torch.float)
        w = torch.tensor(w, dtype=torch.float)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        z = torch.tensor(z, dtype=torch.float)
        self.init_state = torch.tensor([phi, theta, psi, p, q, r, u, v, w, x, y, z], dtype=torch.float)

    # # Differential equation
    # noinspection PyUnusedLocal
    def f(self, time, state: torch.tensor, w: torch.tensor, fa: torch.tensor):

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
        constant_thrust = torch.sqrt(self.m * self.g / (6. * self.b)).numpy() + 1
        w1 = torch.tensor(constant_thrust, dtype=torch.float)
        w2 = torch.tensor(constant_thrust, dtype=torch.float)
        w3 = torch.tensor(constant_thrust, dtype=torch.float)
        w4 = torch.tensor(constant_thrust, dtype=torch.float)
        w5 = torch.tensor(constant_thrust, dtype=torch.float)
        w6 = torch.tensor(constant_thrust, dtype=torch.float)

        f_cmd_t = self.b * (w1 ** 2 + w2 ** 2 + w3 ** 2 + w4 ** 2 + w5 ** 2 + w6 ** 2)
        tau_cmd_x = self.b * self.l_arm * (- w2 ** 2 + w5 ** 2 + 1 / 2 * (- w1 ** 2 - w3 ** 2 + w4 ** 2 + w6 ** 2))
        tau_cmd_y = 3. ** 0.5 / 2. * self.b * self.l_arm * (- w1 ** 2 + w3 ** 2 + w4 ** 2 - w6 ** 2)
        tau_cmd_z = self.d * (- w1 ** 2 + w2 ** 2 - w3 ** 2 + w4 ** 2 - w5 ** 2 + w6 ** 2)

        # Wind's torque
        wind_effect = torch.zeros(6)
        f_wind_x = wind_effect[0]
        f_wind_y = wind_effect[1]
        f_wind_z = wind_effect[2]
        tau_wind_x = wind_effect[3]
        tau_wind_y = wind_effect[4]
        tau_wind_z = wind_effect[5]

        # -----------------------------------------------------------

        phi, theta, psi, p, q, r, u, v, w, x, y, z = state

        ixx = self.inertia_matrix[0, 0]
        iyy = self.inertia_matrix[1, 1]
        izz = self.inertia_matrix[2, 2]

        dphi = p + r * cos(phi) * tan(theta) + q * sin(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = r * cos(phi) / cos(theta) + q * sin(phi) / cos(theta)

        dp = (iyy - izz) / ixx * r * q + (tau_cmd_x + tau_wind_x) / ixx
        dq = (iyy - izz) / ixx * r * q + (tau_cmd_y + tau_wind_y) / iyy
        dr = (iyy - izz) / ixx * r * q + (tau_cmd_z + tau_wind_z) / izz

        du = r * v - q * w + self.g * sin(theta) + f_wind_x / self.m
        dv = p * w - r * u - self.g * sin(phi) * cos(theta) + f_wind_y / self.m
        dw = q * u - p * v - self.g * cos(theta) * cos(phi) + (f_wind_z + f_cmd_t) / self.m

        dx = (w * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
              - v * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)) + u * cos(psi) * cos(theta))
        dy = (v * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta))
              - w * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)) + u * cos(theta) * sin(psi))
        dz = w * cos(phi) * cos(theta) - u * sin(theta) + v * cos(theta) * sin(phi)

        return torch.tensor([dphi, dtheta, dpsi, dp, dq, dr, du, dv, dw, dx, dy, dz], dtype=torch.float)




def cos(var):
    return torch.cos(var)


def sin(var):
    return torch.sin(var)


def tan(var):
    return torch.tan(var)


def main():
    drone_model = FSDroneModel(m=3., g=9.81, b=0.01, d=0.01, l_arm=0.2)
    drone_model.state_initialization(phi=0, theta=0, psi=0, p=0, q=0, r=0, u=0, v=0, w=0, x=0, y=0, z=0)
    t_tab = torch.arange(0, 10, 0.1)  # temps

    # array solution(phi, theta, psi, p, q, r, u, v, w, x, y, z)
    states = odeint(drone_model.f, drone_model.init_state, t_tab)

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
    main()
