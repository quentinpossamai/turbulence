import numpy as np
import torch
import torch.nn as nn
from typing import Union, Dict, Any, List

import src.utils.quaternion as quaternion


# theta = [kt / m, kt * l / ix, km / iz, iz / ix]

# Parameter from blackbird paper
THETA_BLACKBIRD_PAPER = {'theta': np.array([2.08e-6, 2 ** 0.5 * 3.58e-6, 3.77e-8, 1.40]),
                         'permutation': [1, 0, 3, 2]}
# Parameter from blackbird identification
THETA_BLACKBIRD_IDENTIFICATION = {'theta': np.array([2.1019e-06, 2 ** 0.5 * 4.2478e-06, 2.1783e-06, 2.3617e-01]),
                                  'permutation': [1, 0, 3, 2]}


class DynamicalModel:
    def __init__(self, param: Dict[str, Any], dtype, g: float, device=None):
        """
        Model used for the dynamics of the quadcopter.

        Args:
            param: dict["permutation", "theta"] with theta = np.array([[kt / m], [kt * l / ix], [km / iz], [iz / ix]])
                   and permutation = [0, 1, 2, 3] for example.
            dtype: Datatype of the model. np.float32, np.float64, torch.float32, or torch.float64.
            g: gravity used by the model, usually 9.81.
            device: If you choose a torch dtype, the device of the tensor.
        """
        super().__init__()
        self.motor_size = 4

        # Definition of functions depending on the packaged used.
        if dtype in [torch.float32, torch.float64]:
            self.torch_mode = True
            self.farray = lambda x: torch.tensor(x, dtype=self.dtype, device=device)
            self.fzeros = lambda x: torch.zeros(x, dtype=dtype, device=device)
            self.fcat = lambda x, dim: torch.cat(x, dim=dim)
            self.fqrot = lambda q, v: quaternion.qrot(q, v)
            self.fqmul = lambda q, r: quaternion.qmul(q, r)
            self.fcross = lambda a, b, dim: torch.cross(a, b, dim=dim)
            self.fnorm = lambda x, dim: torch.norm(x, dim=dim)
        else:
            self.torch_mode = False
            self.farray = lambda x: np.array(x, dtype=self.dtype)
            self.fzeros = lambda x: np.zeros(x, dtype=dtype)
            self.fcat = lambda x, dim: np.concatenate(x, axis=dim)
            self.fqrot = lambda q, v: quaternion.qrot_np(q, v)
            self.fqmul = lambda q, r: quaternion.qmul_np(q, r)
            self.fcross = lambda a, b, dim: np.cross(a, b, axis=dim)
            self.fnorm = lambda x, dim: np.linalg.norm(x, axis=dim)

        self.dtype = dtype
        self.g = self.farray([0, 0, g])

        msg = '''dict["permutation", "theta"]'''
        assert all([key in ["permutation", "theta"] for key in list(param.keys())]), msg

        self.theta = param["theta"]
        assert self.theta.dtype == self.dtype

        self.permutation: List = param["permutation"]

        self.t3_tensor = self.farray([1, 1, self.theta[3]])
        self.t12_tensor = self.farray([self.theta[1], self.theta[1], self.theta[2]])

        c = 1 / 2 ** 0.5
        self.motor_matrix = self.farray([[-c, +c, +c, -c],
                                         [-c, -c, +c, +c],
                                         [-1, +1, -1, +1],
                                         [+1, +1, +1, +1]])

    def forward(self, **kwargs) -> Union[torch.tensor, np.ndarray]:
        r"""
        Compute the quadrotor dynamics with the acceleration.

        Args:
            **kwargs: motor, shape (self.motor_size) t+1.
                      state, shape (13) t.
                      dt: shape (1) t+1 - t.
                      optional: residual shape (6) t+1.
        Returns:
            next_state t+1, shape (19).
        """

        motor = kwargs['motor']
        state = kwargs['state']
        dt = kwargs['dt']

        pos = state[..., 0:3]
        quat = state[..., 3:7]
        lin_vel = state[..., 7:10]
        ang_vel = state[..., 10:13]

        residual = kwargs['residual'] if 'residual' in kwargs else None  # optional kwarg
        if residual is None:
            kfa = self.fzeros(pos.shape)
            ktaua = self.fzeros(pos.shape)
        else:
            kfa = residual[..., :3]
            ktaua = residual[..., 3:]

        motor_torsor = self.motor_dynamic(motor)  # shape = (4)

        # Linear PFD
        thrust = self.fcat([self.fzeros(2), motor_torsor[3:]], dim=-1)
        next_lin_acc = (self.fqrot(quat, self.theta[0] * thrust + kfa) - self.g)  # in fixed
        next_lin_vel = lin_vel + dt * next_lin_acc  # in fixed

        # Position
        next_pos = pos + dt * next_lin_vel

        # Angular PFD
        next_ang_acc = (self.fcross(self.t3_tensor * ang_vel, ang_vel, dim=-1)  # in body
                        + self.t12_tensor * motor_torsor[:3]
                        + ktaua)
        next_ang_vel = ang_vel + dt * next_ang_acc  # in body

        # Quat
        next_quat = quat + dt * 0.5 * self.fqmul(quat, self.fcat([next_ang_vel, self.fzeros(1)], dim=-1))
        next_quat = next_quat / self.fnorm(next_quat, dim=-1)

        return self.fcat([next_pos, next_quat, next_lin_vel, next_ang_vel], dim=-1)

    def motor_dynamic(self, motor: np.ndarray):
        """

        Args:
            motor: at time t (self.motor_size)

        Returns:
            motor_torsor at time t (4)
        """

        motor = motor[self.permutation]
        motor_torsor = self.motor_matrix @ motor ** 2  # [tx, ty, tz, T]
        return motor_torsor


class DynamicalModelTrapez:
    def __init__(self, param: Dict[str, Any], dtype, g: float, device=None):
        """
        Model used for the dynamics of the quadcopter. Trapezoidal integration need acceleration in the state.

        Args:
            param: dict["permutation", "theta"] with theta = np.array([[kt / m], [kt * l / ix], [km / iz], [iz / ix]])
                   and permutation = [0, 1, 2, 3] for example.
            dtype: Datatype of the model. np.float32, np.float64, torch.float32, or torch.float64.
            g: gravity used by the model, usually 9.81.
            device: If you choose a torch dtype, the device of the tensor.
        """
        super().__init__()
        self.motor_size = 4

        # Definition of functions depending on the packaged used.
        if dtype in [torch.float32, torch.float64]:
            self.torch_mode = True
            self.farray = lambda x: torch.tensor(x, dtype=self.dtype, device=device)
            self.fzeros = lambda x: torch.zeros(x, dtype=dtype, device=device)
            self.fcat = lambda x, dim: torch.cat(x, dim=dim)
            self.fqrot = lambda q, v: quaternion.qrot(q, v)
            self.fqmul = lambda q, r: quaternion.qmul(q, r)
            self.fcross = lambda a, b, dim: torch.cross(a, b, dim=dim)
            self.fnorm = lambda x, dim: torch.norm(x, dim=dim)
        else:
            self.torch_mode = False
            self.farray = lambda x: np.array(x, dtype=self.dtype)
            self.fzeros = lambda x: np.zeros(x, dtype=dtype)
            self.fcat = lambda x, dim: np.concatenate(x, axis=dim)
            self.fqrot = lambda q, v: quaternion.qrot_np(q, v)
            self.fqmul = lambda q, r: quaternion.qmul_np(q, r)
            self.fcross = lambda a, b, dim: np.cross(a, b, axis=dim)
            self.fnorm = lambda x, dim: np.linalg.norm(x, axis=dim)

        self.dtype = dtype
        self.g = self.farray([0, 0, g])

        msg = '''dict["permutation", "theta"]'''
        assert all([key in ["permutation", "theta"] for key in list(param.keys())]), msg

        self.theta = param["theta"]
        assert self.theta.dtype == self.dtype

        self.permutation: List = param["permutation"]

        self.t3_tensor = self.farray([1, 1, self.theta[3]])
        self.t12_tensor = self.farray([self.theta[1], self.theta[1], self.theta[2]])

        c = 1 / 2 ** 0.5
        self.motor_matrix = self.farray([[-c, +c, +c, -c],
                                         [-c, -c, +c, +c],
                                         [-1, +1, -1, +1],
                                         [+1, +1, +1, +1]])

    def forward(self, **kwargs) -> Union[torch.tensor, np.ndarray]:
        r"""
        Compute the quadrotor dynamics with the acceleration.

        Args:
            **kwargs: motor, shape (self.motor_size) t+1.
                      state, shape (19) t.
                      dt: shape (1) t+1 - t.
                      optional: residual shape (6) t+1.
        Returns:
            next_state t+1, shape (19).
        """

        motor = kwargs['motor']
        state = kwargs['state']
        dt = kwargs['dt']

        pos = state[..., 0:3]
        quat = state[..., 3:7]
        lin_vel = state[..., 7:10]
        ang_vel = state[..., 10:13]
        lin_acc = state[..., 13:16]
        ang_acc = state[..., 16:19]

        residual = kwargs['residual'] if 'residual' in kwargs else None  # optional kwarg
        if residual is None:
            kfa = self.fzeros(pos.shape)
            ktaua = self.fzeros(pos.shape)
        else:
            kfa = residual[..., :3]
            ktaua = residual[..., 3:]

        motor_torsor = self.motor_dynamic(motor)  # shape = (4)

        # Linear PFD
        thrust = self.fcat([self.fzeros(2), motor_torsor[3:]], dim=-1)
        next_lin_acc = (self.fqrot(quat, self.theta[0] * thrust + kfa) - self.g)  # in fixed
        next_lin_vel = lin_vel + dt * (lin_acc + next_lin_acc) / 2  # in fixed

        # Position
        next_pos = pos + dt * (lin_vel + next_lin_vel) / 2

        # Angular PFD
        next_ang_acc = (self.fcross(self.t3_tensor * ang_vel, ang_vel, dim=-1)  # in body
                        + self.t12_tensor * motor_torsor[:3]
                        + ktaua)
        next_ang_vel = ang_vel + dt * (ang_acc + next_ang_acc) / 2  # in body

        # Quat
        next_quat = quat + dt * 0.5 * self.fqmul(quat, self.fcat([(ang_vel + next_ang_vel) / 2,
                                                                  self.fzeros(1)], dim=-1))
        next_quat = next_quat / self.fnorm(next_quat, dim=-1)

        return self.fcat([next_pos, next_quat, next_lin_vel, next_ang_vel, next_lin_acc, next_ang_acc], dim=-1)

    def motor_dynamic(self, motor: np.ndarray):
        """

        Args:
            motor: at time t (self.motor_size)

        Returns:
            motor_torsor at time t (4)
        """

        motor = motor[self.permutation]
        motor_torsor = self.motor_matrix @ motor ** 2  # [tx, ty, tz, T]
        return motor_torsor


def _test1():
    """
    Testing shape.
    """
    dtype = torch.float64
    batch_size = 1000
    time_size = 60
    motor = torch.randn(batch_size, time_size, 4, dtype=dtype)
    state = torch.randn(batch_size, time_size, 13, dtype=dtype)
    state[..., 3:7] = state[..., 3:7] / torch.norm(state[..., 3:7], dim=-1).unsqueeze(-1)
    dt = torch.tensor(1 / 20, dtype=dtype).repeat(batch_size, time_size, 1)
    next_state = torch.randn(batch_size, time_size, 13, dtype=dtype)
    next_state[..., 3:7] = next_state[..., 3:7] / torch.norm(state[..., 3:7], dim=-1).unsqueeze(-1)
    residual = torch.randn(batch_size, time_size, 6, dtype=dtype)

    model = TorchDynamicalModel(param='identification', dtype=dtype)
    res = model.forward(motor=motor, state=state, dt=dt, residual=residual)
    assert res.shape == (batch_size, time_size, 13)

    res = model.motor_dynamic(motor=motor)
    assert res.shape == (batch_size, time_size, 4)

    res = model.residual_dynamic(motor=motor, state=state, dt=dt, next_state=next_state)
    assert res.shape == (batch_size, time_size, 6)

    error, error_tab, res = model.rollout(motor=motor, state=state, dt=dt, horizon=59)
    assert res.shape == (batch_size, time_size - 1, 13)


def _test2():
    """
    Testing hovering stability.
    """
    import src.data.habitat.loading as habitat
    device = 'cpu'
    for dtype in [np.float32, np.float64, torch.float32, torch.float64]:
        print(f'Testing {dtype}')
        batch_size = 1000

        # Initializing model
        if dtype in [torch.float32, torch.float64]:
            theta = torch.from_numpy(habitat.THETA).type(dtype)
            state = [torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)]
            motor = torch.tensor([150, 150, 150, 150], dtype=dtype)
        else:
            theta = habitat.THETA.astype(dtype)
            state = [np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)]
            motor = np.array([150, 150, 150, 150], dtype=dtype)
        param = {'theta': theta, 'permutation': [0, 1, 2, 3]}
        model = DynamicalModel(param=param, dtype=dtype, g=habitat.g, device=device)
        dt = 1 / 20

        # Hovering test
        for i in range(batch_size):
            # Test forward
            next_state = model.forward(motor=motor, state=state[-1], dt=dt)
            state.append(next_state)
        assert (state[-1] == state[0]).all()
        state_np = np.stack(state, axis=0)

        # from src.utils.trajectory import trace_trajectory3d
        # trace_trajectory3d(state_np[:, :3], state_np[:, 3:7], 1e-4, 1)


def main():
    _test2()


if __name__ == '__main__':
    main()
