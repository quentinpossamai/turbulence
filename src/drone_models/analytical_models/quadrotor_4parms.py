import torch
import torch.nn as nn

import src.utils.quaternion as quaternion
from src.drone_models.model_structure import Run


class DynamicalModel(nn.Module, Run):
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

        self.permutation = [1, 0, 3, 2]
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
            self.theta =  torch.tensor([[2.08e-6],
                                       [2 ** 0.5 * 3.58e-6],
                                       [3.77e-8],
                                       [1.40]], dtype=self.dtype)  # Parameter from blackbird paper
        elif param == "identification":
            # self.theta = torch.tensor([[1.14061868e-04],
            #                            [2.35845582e-04],
            #                            [2.90530320e-05],
            #                            [2.52134583e+00]], dtype=self.dtype)  # old MSE
            self.theta = torch.tensor([[2.1019e-06],  # kt / m
                                       [2 ** 0.5 * 4.2478e-06],  # kt l / Ix
                                       [2.1783e-06],  # km / Iz
                                       [2.3617e-01]], dtype=self.dtype)  # Iz / Ix MSE results
        else:
            self.theta = param["theta"]
            self.permutation = param["permutation"]

    def forward(self, **kwargs):
        """
        kwargs: motor: (*, self.motor_size). t+1

        kwargs: state: (*, 13). t

        kwargs: dt: (*, 1). t+1 - t

        kwargs optional: residual: (*, 6). t+1

        :return: (*, 6). residual t+1
        """
        # 1.-Input processing
        motor = kwargs['motor']
        state = kwargs['state']
        residual = kwargs['residual'] if 'residual' in kwargs else None
        dt = kwargs['dt']

        assert motor.shape[-1] == self.motor_size, f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.reshape(-1, self.motor_size, 1)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.reshape(-1, 13, 1)

        assert dt.shape[-1] == 1, "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.reshape(-1, 1, 1)

        if residual is None:
            residual_shape = list(state_shape)
            residual_shape[-1] = 6
            residual_shape = tuple(residual_shape)
            residual = torch.zeros(residual_shape).reshape(-1, 6, 1)
        else:
            assert residual.shape[-1] == 6, "residual len is not 6"
            residual_shape = residual.shape
            residual = residual.reshape(-1, 6, 1)
        kfa = residual[:, :3]
        ktaua = residual[:, 3:]

        size = len(motor)
        assert motor_shape[:-1] == state_shape[:-1]
        assert motor_shape[:-1] == dt_shape[:-1]
        assert motor_shape[:-1] == residual_shape[:-1]
        # Input processing end

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

        return torch.cat([next_pos, next_quat, next_lin_vel, next_ang_vel], dim=1).reshape(state_shape)

    def motor_dynamic(self, motor):
        """
        :param motor: (*, self.motor_size) t

        :return: (*, 4) t
        """
        assert motor.shape[-1] == self.motor_size, f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.reshape(-1, self.motor_size, 1)
        motor_torsor_shape = list(motor_shape)
        motor_torsor_shape[-1] = 4

        a = 1 / 2 ** 0.5
        motor_matrix = torch.tensor([[-a, +a, +a, -a],
                                     [-a, -a, +a, +a],
                                     [-1, +1, -1, +1],
                                     [+1, +1, +1, +1]], dtype=self.dtype).view(1, 4, 4)

        motor = motor[:, self.permutation]
        motor_torsor = motor_matrix @ motor ** 2  # [tx, ty, tz, T]
        return motor_torsor.reshape(motor_torsor_shape)

    def residual_dynamic(self, motor, state, dt, next_state):
        """
        :param motor: (*, self.motor_size). t+1

        :param state: (*, 13). t

        :param dt: (*, 1). t+1 - t

        :param next_state: (*, 13). state t+1

        :return: (*, 6). residual t+1
        """
        assert motor.shape[-1] == self.motor_size, f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.reshape(-1, self.motor_size, 1)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.reshape(-1, 13, 1)

        assert dt.shape[-1] == 1, "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.reshape(-1, 1, 1)

        assert next_state.shape[-1] == 13, "next_state len is not 13"
        next_state_shape = next_state.shape
        next_state = next_state.reshape(-1, 13, 1)

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
        return torch.cat([kfa, ktaua], dim=1).reshape(residual_shape)

    def rollout(self, **kwargs):
        """
        kwargs: motor: (*, horizon+1, motor_size). t+1

        kwargs: state: (*, horizon+1, 13). t

        kwargs: horizon: int

        kwargs: dt: (*, horizon+1, 1). t+1 - t

        :return: (*, horizon, 13). state t+1
        """
        # Input processing
        motor = kwargs['motor']
        state = kwargs['state']
        dt = kwargs['dt']
        horizon = kwargs['horizon']

        time_size = horizon + 1
        assert motor.shape[-1] == self.motor_size, f"motor len is not {self.motor_size}"
        motor_shape = motor.shape
        motor = motor.reshape(-1, time_size, self.motor_size)

        assert state.shape[-1] == 13, "state len is not 13"
        state_shape = state.shape
        state = state.reshape(-1, time_size, 13)

        assert dt.shape[-1] == 1, "dt len is not 1"
        dt_shape = dt.shape
        dt = dt.reshape(-1, time_size, 1)

        pred_shape = list(state_shape)
        pred_shape[-2] = horizon

        assert motor_shape[:-1] == state_shape[:-1]
        assert motor_shape[:-1] == dt_shape[:-1]
        assert time_size == motor_shape[-2]
        # Input processing end

        next_state_gt = state[:, 1:]

        with torch.no_grad():
            self.eval()
            # Prediction rollout
            prediction = [state[:, 0:1]]
            for i in range(horizon):
                next_state = self.forward(motor=motor[:, i + 1:i + 2], state=prediction[-1], dt=dt[:, i + 1:i + 2])
                prediction.append(next_state)

        prediction = torch.cat(prediction, dim=1)[:, 1:]  # First is not a prediction
        error = ((prediction - next_state_gt) ** 2)
        return error.mean(), error.reshape(pred_shape), prediction.reshape(pred_shape)


def _test():
    dtype = torch.float64
    batch_size = 1000
    time_size = 60
    motor = torch.randn(batch_size, time_size, 4, dtype=dtype)
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

    error, error_tab, res = model.rollout(motor=motor, state=state, dt=dt, horizon=59)
    assert res.shape == (batch_size, time_size - 1, 13)


def main():
    _test()


if __name__ == '__main__':
    main()
