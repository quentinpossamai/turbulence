import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import src.utils.quaternion as quaternion


class DynamicalModel(nn.Module):
    def __init__(self, param, dtype):
        """
        :param param: "paper", "identification", or dict["permutation", "theta"]
        :param dtype: Datatype of the model.
        """
        super().__init__()
        self.motor_size = 6
        assert dtype in [torch.float32, torch.float64]
        self.dtype = dtype

        msg = '''"paper", "identification", or dict["permutation", "theta"]'''
        assert param in ["paper", "identification"] or list(param.keys()) == ["permutation", "theta"], msg

        # Point: 6
        self.permutation = [0, 1, 2, 3, 4, 5]
        if param == "paper":
            # # Paper parameters from
            # # Achtelik, M. et al. (2012) ‘Design of a Multi Rotor MAV with regard to Efficiency, Dynamics and
            # Redundancy’
            # m = 0.64
            # # From Akkinapalli et al. ‘Attitude Control of a Multicopter Using L1 Augmented Quaternion Based
            # Backstepping’
            # ix = 10.007e-3  # [kg.m2]
            # iy = 10.2335e-3  # [kg.m2]  # Hypothesis ix == iy nearly confirmed
            # iz = 8.1e-3  # [kg.m2]
            # kt = 6.546e-6  # [N.s2.rad-1]
            # km = 1.2864e-7  # [N.m2.rad-1]
            # l = 0.215  # [m]
            # theta = torch.tensor([[kt / m], [kt * l / ix], [km / iz], [iz / ix]])
            self.theta = torch.tensor([[1.0228e-05],
                                       [1.4064e-04],
                                       [1.5881e-05],
                                       [8.0943e-01]], dtype=self.dtype)
        elif param == "identification":
            self.theta = torch.tensor([[1.14061868e-04],
                                       [2.35845582e-04],
                                       [2.90530320e-05],
                                       [2.52134583e+00]], dtype=self.dtype)  # MSE
        else:
            self.theta = param["theta"]
            self.permutation = param["permutation"]

    def forward(self, motor, state, dt, residual=None):
        """
        :param motor: (*, self.motor_size). t+1
        :param state: (*, 13). t
        :param dt: (*, 1). t+1 - t
        :param residual: (*, 6). t+1
        :return: (*, 6). residual t+1
        """
        assert motor.shape[-1] == self.motor_size,  f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.view(-1, self.motor_size, 1)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.view(-1, 13, 1)

        assert dt.shape[-1] == 1,  "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.view(-1, 1, 1)

        if residual is None:
            residual_shape = list(state_shape)
            residual_shape[-1] = 6
            residual_shape = tuple(residual_shape)
            residual = torch.zeros(residual_shape).view(-1, 6, 1)
        else:
            assert residual.shape[-1] == 6, "residual len is not 6"
            residual_shape = residual.shape
            residual = residual.view(-1, 6, 1)
        kfa = residual[:, :3]
        ktaua = residual[:, 3:]

        size = len(motor)
        assert motor_shape[:-1] == state_shape[:-1]
        assert motor_shape[:-1] == dt_shape[:-1]
        assert motor_shape[:-1] == residual_shape[:-1]

        pos = state[:, :3]
        quat = state[:, 3:7]
        lin_vel = state[:, 7:10]
        ang_vel = state[:, 10:]

        motor_torsor = self.motor_dynamic(motor.view(-1, self.motor_size)).view(-1, 4, 1)

        # Linear PFD
        g = torch.tensor([0, 0, 9.81], dtype=self.dtype).view(1, 3, 1)
        thrust = torch.cat([torch.zeros(size, 2, 1), motor_torsor[:, 3:]], dim=1)
        next_lin_vel = lin_vel + dt * (quaternion.qrot(quat.view(-1, 4), self.theta[0] * thrust + kfa) - g)

        # Position
        next_pos = pos + dt * next_lin_vel

        # Angular PFD
        t3_tensor = torch.tensor([1, 1, self.theta[3]], dtype=self.dtype).view(1, 3, 1)
        t12_tensor = torch.tensor([self.theta[1], self.theta[1], self.theta[2]], dtype=self.dtype).view(1, 3, 1)
        next_ang_vel = ang_vel + dt * (torch.cross(t3_tensor * ang_vel, ang_vel, dim=1)
                                       + t12_tensor * motor_torsor[:, :3]
                                       + ktaua)

        # Quat
        next_quat = quat + dt * 0.5 * quaternion.qmul(quat, torch.cat([next_ang_vel,
                                                                       torch.zeros(size, 1, 1, dtype=self.dtype)],
                                                                      dim=1))
        next_quat = next_quat / torch.norm(next_quat, dim=1, keepdim=True)

        return torch.cat([next_pos, next_quat, next_lin_vel, next_ang_vel], dim=1).view(state_shape)

    def motor_dynamic(self, motor):
        """
        :param motor: (*, self.motor_size) t
        :return: (*, 4) t
        """
        assert motor.shape[-1] == self.motor_size,  f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.view(-1, self.motor_size, 1)
        motor_torsor_shape = list(motor_shape)
        motor_torsor_shape[-1] = 4

        a = 0.5
        b = 3 ** 0.5 / 2
        motor_matrix = torch.tensor([[-a, +a, +1, +a, -a, -1],
                                     [-b, -b, +0, +b, +b, +0],
                                     [-1, +1, -1, +1, -1, +1],
                                     [+1, +1, +1, +1, +1, +1]], dtype=self.dtype).view(1, 4, 6)

        motor = motor[:, self.permutation]
        motor_torsor = motor_matrix @ motor ** 2  # [tx, ty, tz, T]
        return motor_torsor.view(motor_torsor_shape)

    def residual_dynamic(self, motor, state, dt, next_state):
        """
        :param motor: (*, self.motor_size). t+1
        :param state: (*, 13). t
        :param dt: (*, 1). t+1 - t
        :param next_state: (*, 13). state t+1
        :return: (*, 6). residual t+1
        """
        assert motor.shape[-1] == self.motor_size,  f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.view(-1, self.motor_size, 1)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.view(-1, 13, 1)

        assert dt.shape[-1] == 1,  "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.view(-1, 1, 1)

        assert next_state.shape[-1] == 13, "next_state len is not 13"
        next_state_shape = next_state.shape
        next_state = next_state.view(-1, 13, 1)

        size = len(motor)
        assert motor_shape[:-1] == state_shape[:-1]
        assert motor_shape[:-1] == dt_shape[:-1]
        assert motor_shape[:-1] == next_state_shape[:-1]

        residual_shape = list(motor_shape)
        residual_shape[-1] = 6

        quat = state[:, 3:7]
        ang_vel = state[:, 10:]
        lin_acc = (next_state[:, 7:10] - state[:, 7:10]) / dt
        ang_acc = (next_state[:, 10:] - state[:, 10:]) / dt
        motor_torsor = self.motor_dynamic(motor.view(-1, self.motor_size)).view(-1, 4, 1)

        quat_bar = quat * torch.tensor([-1, -1, -1, 1], dtype=self.dtype).view(1, 4, 1)
        g = torch.tensor([0, 0, 9.81], dtype=self.dtype).view(1, 3, 1)

        thrust = torch.cat([torch.zeros((size, 2, 1)), self.theta[0] * motor_torsor[:, 3:]], dim=1)
        kfa = quaternion.qrot(quat_bar, lin_acc + g) - thrust

        t3_tensor = torch.tensor([1, 1, self.theta[3]], dtype=self.dtype).view(1, 3, 1)
        t12_tensor = torch.tensor([self.theta[1], self.theta[1], self.theta[2]], dtype=self.dtype).view(1, 3, 1)
        ktaua = (ang_acc - torch.cross(t3_tensor * ang_vel, ang_vel, dim=1)
                 - t12_tensor * motor_torsor[:, :3])
        return torch.cat([kfa, ktaua], dim=1).view(residual_shape)

    def rollout(self, motor, state, horizon, dt):
        """
        :param motor: (*, horizon+1, motor_size). t+1
        :param state: (*, horizon+1, 13). t
        :param horizon: int
        :param dt: (*, horizon+1, 1). t+1 - t
        :return: (*, horizon, 13). state t+1
        """
        assert motor.shape[-1] == self.motor_size,  f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.view(-1, horizon + 1, self.motor_size)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.view(-1, horizon + 1, 13)

        assert dt.shape[-1] == 1,  "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.view(-1, horizon + 1, 1)

        pred_shape = list(state_shape)
        pred_shape[-2] = horizon

        size = len(motor)
        assert motor_shape[:-1] == state_shape[:-1]
        assert motor_shape[:-1] == dt_shape[:-1]
        assert horizon + 1 == motor_shape[-2]

        next_state_gt = state[:, 1:]

        with torch.no_grad():
            self.eval()
            # Prediction rollout
            prediction = [state[:, :1]]
            for i in range(horizon):
                next_state = self.forward(motor=motor[:, i+1:i+2], state=prediction[-1], dt=dt[:, i+1:i+2])
                prediction.append(next_state)

        prediction = torch.cat(prediction, dim=1)[:, 1:]  # First is not a prediction
        error = ((prediction - next_state_gt) ** 2).mean(dim=0)
        return error.mean(), error, prediction.view(pred_shape)


def _test():
    dtype = torch.float64
    batch_size = 1000
    time_size = 60
    motor = torch.randn(batch_size, time_size, 6, dtype=dtype)
    state = torch.randn(batch_size, time_size, 13, dtype=dtype)
    dt = torch.tensor(1 / 200, dtype=dtype).repeat(batch_size, time_size, 1)
    next_state = torch.randn(batch_size, time_size, 13, dtype=dtype)
    residual = torch.randn(batch_size, time_size, 6, dtype=dtype)

    model = DynamicalModel(param="identification", dtype=dtype)
    res = model.forward(motor=motor, state=state, dt=dt, residual=residual)
    assert res.shape == (batch_size, time_size, 13)

    res = model.motor_dynamic(motor=motor)
    assert res.shape == (batch_size, time_size, 4)

    res = model.residual_dynamic(motor=motor, state=state, dt=dt, next_state=next_state)
    assert res.shape == (batch_size, time_size, 6)

    error, error_tab, res = model.rollout(motor=motor, state=state, horizon=59, dt=dt)
    assert res.shape == (batch_size, time_size - 1, 13)


def main():
    _test()


if __name__ == '__main__':
    main()
