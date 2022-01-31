import numpy as np
import scipy.optimize
import scipy.interpolate
from typing import Callable
from src.utils.transform import Transform, Rotation


# class LossLog:
#     def __init__(self):
#         self.position_loss = []
#         self.yaw_loss = []
#         self.angular_speed_loss = []
#         self.yaw_ref = []
#
#     def log(self, position_loss, yaw_loss, angular_speed_loss, yaw_ref):
#         self.position_loss.append(position_loss)
#         self.yaw_loss.append(yaw_loss)
#         self.angular_speed_loss.append(angular_speed_loss)
#         self.yaw_ref.append(yaw_ref)
#
#
# LOSS_LOGGER = LossLog()
# MODEL: Callable = None


def mpc_controller(model: Callable,
                   state_ref: np.ndarray,
                   state0: np.ndarray,
                   u0: np.ndarray,
                   horizon_cmd: int,
                   horizon_pred: int,
                   gamma: float,
                   verbose: bool):
    """

    param model:

    param state_ref:

    param state0:

    param u0: Shape (horizon, 4)

    param horizon_cmd:

    param horizon_pred:

    param gamma:

    param verbose:

    return:

    """
    assert horizon_pred % horizon_cmd == 0

    def _mpc(_u, _state, _state_ref, _horizon_cmd, _horizon_pred, _gamma):
        # Option 1
        _u = _u.reshape(_horizon_cmd, 4)
        if _horizon_pred - _horizon_cmd > 0:
            _u = np.concatenate([_u, np.stack([_u[-1] for _ in range(_horizon_pred - _horizon_cmd)])])

        # # Option 2
        # num_repeat = horizon_pred // horizon_cmd
        # u_tiled = np.zeros((_horizon_pred, 4))
        # _u = _u.reshape(_horizon_cmd, 4)
        # for i, cmd in enumerate(_u):
        #     u_tiled[i*num_repeat:(i+1)*num_repeat] = np.tile(cmd, num_repeat).reshape((num_repeat, 4))
        # _u = u_tiled

        state_hat = [_state]
        for t in range(_horizon_pred):
            next_state = model(state_hat[-1], _u[t])
            state_hat.append(next_state)
        state_hat = np.stack(state_hat[1:])
        return loss(state_ref=_state_ref, state_hat=state_hat, horizon=_horizon_pred, gamma=_gamma)

    # Initialize command
    if verbose:
        print('MPC starting')
    out: scipy.optimize.OptimizeResult = scipy.optimize.minimize(fun=_mpc, x0=u0.flatten(),
                                                                 args=(state0, state_ref, horizon_cmd,
                                                                       horizon_pred, gamma),
                                                                 method='bfgs',
                                                                 jac=None,
                                                                 tol=None,
                                                                 options={
                                                                     "disp": False,  # True
                                                                     "maxiter": 10000,  # None
                                                                     "gtol": 1e-2,  # 1e-5
                                                                     # "norm": 2,  # np.inf
                                                                     "eps": 1e-2,  # 1.4901161193847656e-08
                                                                     "return_all": True,  # False
                                                                     "finite_diff_rel_step": None}  # None
                                                                 )
    u, error_mpc = out.x, out.fun
    u = u.reshape((-1, 4))
    if verbose:
        print(f'Finished | mpc_error: {error_mpc} | u: {u} | message: {out.message} \n')
    # u = np.abs(u)
    return u, error_mpc


def loss(state_ref: np.ndarray, state_hat: np.ndarray, gamma: float, horizon: int) -> float:
    coefs = [1]
    for i in range(horizon - 1):
        coefs.append(gamma * coefs[-1])
    coefs = np.stack(coefs)

    position_ref, quat_ref, lin_vel_ref, angular_speed_ref = (state_ref[:, :3], state_ref[:, 3:7],
                                                              state_ref[:, 7:10], state_ref[:, 10:])
    position_hat, quat_hat, lin_vel_hat, angular_speed_hat = (state_hat[:, :3], state_hat[:, 3:7],
                                                              state_hat[:, 7:10], state_hat[:, 10:])

    # Position loss, shape=(horizon_pred, 3)
    position_loss = np.mean((position_ref - position_hat) ** 2, axis=1)
    position_loss = coefs @ position_loss

    # Yaw loss
    # METHOD 1
    # rot_loss = np.arccos(np.clip(np.sum((rot_ref * rot_hat).reshape(-1, 9), axis=1) / 3, -1, 1))
    # METHOD 2
    # yaw_ref = tf.as_rot_euler(seq='zyx', degrees=False)[:, 0]
    # yaw_hat = Rotation.from_quat(quat_hat).as_euler(seq='zyx', degrees=False)[:, 0]
    # yaw_loss = (yaw_ref - yaw_hat) ** 2
    # yaw_loss = coefs @ yaw_loss
    # METHOD 3
    rot_global_drone_hat = Transform.from_quat_trans(quat_hat, position_hat).as_rot_mat()
    vect_hat = rot_global_drone_hat @ np.array([1, 0, 0])
    vect_hat[:, 2] = 0
    vect_hat / np.linalg.norm(vect_hat, axis=1).reshape((-1, 1))
    vect_ref = Rotation.from_quat(quat_ref).as_matrix() @ np.array([1, 0, 0])
    vect_ref[:, 2] = 0
    vect_ref / np.linalg.norm(vect_ref, axis=1).reshape((-1, 1))
    yaw_loss = -np.sum(vect_ref * vect_hat, axis=1)
    yaw_loss = coefs @ yaw_loss

    # Linear velocity loss
    lin_vel_loss = coefs @ np.mean((lin_vel_ref - lin_vel_hat) ** 2, axis=1)
    return 1. * position_loss + 0.05 * yaw_loss + 0. * lin_vel_loss
