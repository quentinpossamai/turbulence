import torch
import torch.nn as nn
from typing import Callable

m = torch.tensor(1.56779)  # kg
kt = torch.tensor(8.54858e-6)  # N / (rad/s)
km = torch.tensor(1.6e-2)
l_arm = torch.tensor(0.215)  # m
ixx = torch.tensor(0.0347563)  # kg m-2
iyy = torch.tensor(0.0458929)  # kg m-2
izz = torch.tensor(0.0977)  # kg m-2


# 05 June 2020
# DRONE MODEL
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

    def f(self, t_time: torch.Tensor, state: torch.Tensor, w_func: Callable, fa: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the state. x_dot = f(x, u, fa).
        :param t_time: scalar in seconds (Tensor of shape (1,)).
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

        # Hexarotor force generation -----------------------------------------------------------------------------------
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

        if callable(w_func):
            w = w_func(t_time.item())
        else:
            w = w_func

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
        # noinspection PyUnboundLocalVariable
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
