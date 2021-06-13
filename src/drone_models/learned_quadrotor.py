import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from src.data.blackbird.blackbird_loading import BlackbirdDataset


class Quadrotor(nn.Module):
    def __init__(self):
        super(Quadrotor, self).__init__()
        self.a = nn.Parameter(torch.randn(3, dtype=torch.float64) * 0.01 / np.sqrt(3), requires_grad=True)
        self.b = nn.Parameter(torch.randn(4, 4, dtype=torch.float64) * 0.01 / 4, requires_grad=True)

    def forward(self, state, motor_speed):
        cmd = self.compute_cmd(motor_speed)
        dstate = self.compute_derivative(state, cmd)
        return dstate

    def compute_cmd(self, motor_speed):
        motor_squared = motor_speed ** 2
        b = self.b.unsqueeze(0).repeat(motor_speed.shape[0], 1, 1)
        command = b @ motor_squared.unsqueeze(-1)
        return command.squeeze(-1)

    def compute_derivative(self, state, cmd):
        phi, psi, theta = state[:, 0], state[:, 1], state[:, 2]
        p, q, r = state[:, 3], state[:, 4], state[:, 5]
        u, v, w = state[:, 6], state[:, 7], state[:, 8]
        cos, sin, tan = torch.cos, torch.sin, torch.tan
        dphi = p + r * cos(phi) * tan(theta) + q * sin(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = r * cos(phi) / cos(theta) + q * sin(phi) / cos(theta)

        dp = self.a[0] * r * q + cmd[:, 0]
        dq = self.a[1] * p * r + cmd[:, 1]
        dr = self.a[2] * p * q + cmd[:, 2]

        g = 9.81
        du = r * v - q * w - g * sin(theta)
        dv = p * w - r * u + g * sin(phi) * cos(theta)
        dw = q * u - p * v + g * cos(theta) * cos(phi) - cmd[:, 3]

        dx = (w * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
              - v * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)) + u * cos(psi) * cos(theta))
        dy = (v * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta))
              - w * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)) + u * cos(theta) * sin(psi))
        dz = w * cos(phi) * cos(theta) - u * sin(theta) + v * cos(theta) * sin(phi)

        dstate = torch.stack([dphi, dtheta, dpsi, dp, dq, dr, du, dv, dw, dx, dy, dz], dim=-1)
        return dstate


def _test():
    pass


if __name__ == '__main__':
    _test()
