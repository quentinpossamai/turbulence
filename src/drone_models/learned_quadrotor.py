import torch
import torch.nn as nn
import numpy as np


class Quadrotor(nn.Module):
    def __init__(self):
        super(Quadrotor, self).__init__()
        self.a = nn.Parameter(torch.randn(3) * 0.01 / np.sqrt(3), requires_grad=True)
        self.b = nn.Parameter(torch.randn(4, 4) * 0.01 / 4, requires_grad=True)
        self.motor = None
        self.time = None

    def forward(self, state, motor_speed):
        cmd = self.compute_cmd(motor_speed)
        dstate = self.compute_derivative(state, cmd)

        next_state = state + (1 / 36) * dstate  # Reduce 360 to 36 since dataloader only keep 1pts every 10pts
        # next_state[:, 3] = next_state[:, 3] % (2 * np.pi)
        # next_state[:, 4] = next_state[:, 4] % (np.pi)
        # next_state[:, 5] = next_state[:, 5] % (np.pi)
        return next_state

    def compute_cmd(self, motor_speed):
        B = motor_speed.shape[0]

        rpm_squarred = motor_speed ** 2
        b = self.b.unsqueeze(0).repeat(B, 1, 1)
        command = b @ rpm_squarred.unsqueeze(-1)
        return command.squeeze(-1)

    def compute_derivative(self, state, cmd):
        psi, theta, phi = state[:, 3], state[:, 4], state[:, 5]
        u, v, w = state[:, 6], state[:, 7], state[:, 8]
        p, q, r = state[:, 9], state[:, 10], state[:, 11]
        cos, sin, tan = torch.cos, torch.sin, torch.tan
        dphi = p + r * cos(phi) * tan(theta) + q * sin(phi) * tan(theta)
        dtheta = q * cos(phi) - r * sin(phi)
        dpsi = r * cos(phi) / cos(theta) + q * sin(phi) / cos(theta)

        dp = self.a[0] * r * q + cmd[:, 0]
        dq = self.a[1] * p * r + cmd[:, 1]
        dr = self.a[2] * p * q + cmd[:, 2]

        G = 9.81
        du = r * v - q * w - G * sin(theta)
        dv = p * w - r * u + G * sin(phi) * cos(theta)
        dw = q * u - p * v + G * cos(theta) * cos(phi) - cmd[:, 3]

        dx = (w * (sin(phi) * sin(psi) + cos(phi) * cos(psi) * sin(theta))
              - v * (cos(phi) * sin(psi) - cos(psi) * sin(phi) * sin(theta)) + u * cos(psi) * cos(theta))
        dy = (v * (cos(phi) * cos(psi) + sin(phi) * sin(psi) * sin(theta))
              - w * (cos(psi) * sin(phi) - cos(phi) * sin(psi) * sin(theta)) + u * cos(theta) * sin(psi))
        dz = w * cos(phi) * cos(theta) - u * sin(theta) + v * cos(theta) * sin(phi)

        dstate = torch.stack([dx, dy, dz, dpsi, dtheta, dphi, du, dv, dw, dp, dq, dr], dim=-1)
        return dstate


if __name__ == '__main__':
    model = Quadrotor()
    state = torch.randn(16, 12)
    motor_rpm = torch.randn(16, 4)

    y = model(None, state, motor_rpm)
    print(y.shape)