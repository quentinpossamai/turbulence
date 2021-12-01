import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.interpolate
from typing import Callable

from src.utils.trajectory import trace_trajectory3d
from src.drone_models.analytical_models.quadrotor_4parms import DynamicalModel
from src.utils.paths import DATA_HABITAT_PATH
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
                   ref_tf: np.ndarray,
                   ref_lin_vel: np.ndarray,
                   state0: np.ndarray,
                   u0: np.ndarray,
                   horizon_cmd: int,
                   horizon_pred: int,
                   gamma: float):
    """

    :param model:
    :param ref_tf:
    :param ref_lin_vel:
    :param state0:
    :param u0: Shape (horizon, 4)
    :param horizon_cmd:
    :param horizon_pred:
    :param gamma:
    :return:
    """

    def _mpc(_u, _state, _ref_tf, _ref_lin_vel, _horizon_cmd, _horizon_pred, _gamma):
        _u = _u.reshape(_horizon_cmd, 4)
        if _horizon_pred - _horizon_cmd > 0:
            _u = np.concatenate([_u, np.stack([_u[-1] for _ in range(_horizon_pred - _horizon_cmd)])])
        state_hat = [_state]
        for t in range(_horizon_pred):
            next_state = model(state_hat[-1], _u[t])
            state_hat.append(next_state)
        state_hat = np.stack(state_hat[1:])
        return loss(tf_ref=_ref_tf, ref_lin_vel=_ref_lin_vel, state_hat=state_hat, horizon=_horizon_pred, gamma=_gamma)

    # Initialize command
    print('MPC starting')
    out: scipy.optimize.OptimizeResult = scipy.optimize.minimize(fun=_mpc, x0=u0.flatten(),
                                                                 args=(state0, ref_tf, ref_lin_vel, horizon_cmd,
                                                                       horizon_pred, gamma),
                                                                 method='bfgs',
                                                                 jac=None,
                                                                 tol=None,
                                                                 options={
                                                                     "disp": False,  # True
                                                                     "maxiter": 10000,  # None
                                                                     "gtol": 1e-5,  # 1e-5
                                                                     # "norm": 2,  # np.inf
                                                                     "eps": 1e-4,  # 1.4901161193847656e-08
                                                                     "return_all": True,  # False
                                                                     "finite_diff_rel_step": None}  # None
                                                                 )
    u, error_mpc = out.x, out.fun
    u = u.reshape((-1, 4))
    print(f'Finished | mpc_error: {error_mpc} | u: {u} | message: {out.message} \n')
    # u = np.abs(u)
    return u, error_mpc


def loss(tf_ref: np.ndarray, ref_lin_vel: np.ndarray, state_hat: np.ndarray, gamma: float, horizon: int) -> float:
    coefs = [1]
    for i in range(horizon - 1):
        coefs.append(gamma * coefs[-1])
    coefs = np.stack(coefs)

    tf = Transform.from_matrix(tf_ref)
    position_hat, quat_hat, lin_vel_hat, angular_speed_hat = (state_hat[:, :3], state_hat[:, 3:7],
                                                              state_hat[:, 7:10], state_hat[:, 10:])

    # Position loss, shape=(horizon_pred, 3)
    position_ref = tf.as_trans()
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
    vect_ref = tf.as_rot_mat() @ np.array([1, 0, 0])
    vect_ref[:, 2] = 0
    vect_ref / np.linalg.norm(vect_ref, axis=1).reshape((-1, 1))
    yaw_loss = -np.sum(vect_ref * vect_hat, axis=1)
    yaw_loss = coefs @ yaw_loss

    # Linear velocity loss
    lin_vel_loss = coefs @ np.mean((ref_lin_vel - lin_vel_hat) ** 2, axis=1)
    return 1. * position_loss + 0.05 * yaw_loss + 0. * lin_vel_loss


def main():
    import pickle
    import torch
    from tqdm import tqdm
    ref_tf = pickle.load(open(f'{DATA_HABITAT_PATH}ref_tf_global_agent.pkl', 'rb'))
    TF_HABITAT__HABITAT_FLU = Transform.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
                                                         trans=np.zeros(3)).as_matrix()
    TF_HABITAT_FLU__HABITAT = Transform.from_matrix(TF_HABITAT__HABITAT_FLU).inv().as_matrix()

    ref_tf = TF_HABITAT_FLU__HABITAT @ ref_tf @ TF_HABITAT__HABITAT_FLU
    quat, trans = Transform.from_matrix(ref_tf).as_quat_trans()
    # trace_trajectory3d(position=trans, quaternion=quat, pause=0.001, step=1)

    a, b, c = 0.1, 0.1, 0.025  # drone's length along x, y and z axis
    volumetric_mass = 200
    g = 9.81
    m = volumetric_mass * a * b * c
    drone_prms = {'m': m,  # [kg]
                  'kt': m * g / (4 * 150 ** 2),  #
                  'km': m * g / (4 * 150 ** 2) * 2 ** 0.5 * a / 2,  #
                  'l': 2 ** 0.5 * a / 2,  # [m]
                  'ix': 1 / 3 * (b ** 2 + c ** 2) * m,  # [kg.m2]
                  'iy': 1 / 3 * (a ** 2 + c ** 2) * m,  # [kg.m2]
                  'iz': 1 / 3 * (a ** 2 + b ** 2) * m,  # [kg.m2]
                  'prop_radius': a * 0.1}  # [m]
    step = 1 / 20
    horizon_cmd = 1
    horizon_pred = 20
    gamma = 1.0
    dtype = torch.float32
    theta = torch.tensor([drone_prms['kt'] / drone_prms['m'],
                          drone_prms['kt'] * drone_prms['l'] / drone_prms['ix'],
                          drone_prms['km'] / drone_prms['iz'],
                          drone_prms['iz'] / drone_prms['ix']], dtype=dtype)
    quadrotor = DynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=dtype)

    def mpc_model(_state, _u):
        _next_state = quadrotor.forward(state=torch.tensor(_state, dtype=dtype),
                                        motor=torch.tensor(_u, dtype=dtype),
                                        dt=torch.tensor([step]), dtype=dtype).numpy()
        return _next_state

    # Setting the drone at the first reference
    quat0, trans0 = Transform.from_matrix(ref_tf[0]).as_quat_trans()
    state = [np.concatenate([trans0, quat0, np.zeros(6)])]
    hover_cmd = np.sqrt(drone_prms['m'] * 9.81 / (4 * drone_prms['kt']))
    motor_horizon = np.repeat(hover_cmd, horizon_cmd * 4).reshape((horizon_cmd, 4))

    motor = []
    mpc_error = []
    time_end = 10
    for i in tqdm(range(1, time_end)):
        # Using MPC
        motor_horizon, err = mpc_controller(model=mpc_model,
                                            ref_tf=ref_tf[i:i + horizon_pred],
                                            state0=state[-1],
                                            u0=motor_horizon,
                                            horizon_cmd=horizon_cmd,
                                            horizon_pred=horizon_pred,
                                            gamma=gamma)
        # _mpc(motor_horizon, state[-1], ref_tf[i:i + horizon_pred], horizon_cmd, horizon_pred, gamma, 'log_loss')
        motor.append(motor_horizon[0])
        mpc_error.append(err)
        next_state = quadrotor.forward(state=torch.tensor(state[-1], dtype=dtype),
                                       motor=torch.tensor(motor[-1], dtype=dtype),
                                       dt=torch.tensor([step], dtype=dtype)).numpy()
        state.append(next_state)

    # plt.plot(np.stack(LOSS_LOGGER.position_loss))
    # plt.plot(np.stack(LOSS_LOGGER.yaw_loss))
    # plt.plot(np.stack(LOSS_LOGGER.angular_speed_loss))
    # plt.plot(np.stack(LOSS_LOGGER.yaw_ref))

    state_tab, motor_tab, mpc_error_tab = (np.stack(e) for e in (state, motor, mpc_error))
    pose_ref = Transform.from_matrix(ref_tf).as_trans()
    quat_ref = Transform.from_matrix(ref_tf).as_rot_quat()
    pose_real = np.stack(state_tab[:, :3])
    quat_real = np.stack(state_tab[:, 3:7])
    trace_trajectory3d(pose_real, quat_real, pause=0.0001, step=1, position_ref=pose_ref[:time_end])
    # trace_trajectory3d(pose_ref, quat_ref, pause=0.0001, step=1, position_ref=None)


if __name__ == '__main__':
    main()
