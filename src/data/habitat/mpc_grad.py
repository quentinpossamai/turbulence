import numpy as np
import torch
from typing import Dict
import matplotlib.pyplot as plt

import src.learning.residual.predict as predict
from src.drone_model.quadrotor_4parms import DynamicalModel
import src.data.habitat.loading as loading
import src.data.habitat.data_gen as data_gen
import src.utils.quaternion as quaternion

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
z_estimator, _, _, _ = predict.get_trained_model_and_dataset('norm_gru_finetune', False, 'habitat')
z_estimator.to(DEVICE)
z_estimator.eval()

drone_prms = loading.define_drone()
theta = torch.FloatTensor([drone_prms['kt'] / drone_prms['m'],
                           drone_prms['kt'] * drone_prms['l'] / drone_prms['ix'],
                           drone_prms['km'] / drone_prms['iz'],
                           drone_prms['iz'] / drone_prms['ix']])
quadrotor = DynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=torch.float32,
                           g=9.81)


def mpc_controller(sim: data_gen.Simulation,
                   state_ref: np.ndarray,
                   state0: np.ndarray,
                   u0: np.ndarray,
                   depth: np.ndarray,
                   horizon_cmd: int,
                   horizon_pred: int,
                   gamma: float,
                   estimate_fa: bool,
                   db_verbose: bool,
                   db_analysis: bool,
                   ):
    """
    MPC controller for the drone.

    Args:
        sim:
        state_ref:
        state0:
        u0:
        depth
        horizon_cmd:
        horizon_pred:
        gamma:
        estimate_fa:
        db_verbose:
        db_analysis:
    """
    assert horizon_pred % horizon_cmd == 0
    state_ref = torch.from_numpy(state_ref).to(DEVICE).float()
    state0 = torch.from_numpy(state0).to(DEVICE).float()
    u0 = torch.from_numpy(u0).to(DEVICE).float()

    def to_minimize(_u):
        dt = 1 / 20
        if horizon_pred - horizon_cmd > 0:
            _u = torch.cat([_u, torch.stack([_u[-1] for _ in range(horizon_pred - horizon_cmd)])])

        state_hat = [state0]
        for t in range(horizon_pred):
            if estimate_fa and len(depth) == 20:
                zt = z_estimator(depth)
                fa = sim.compute_ground_effect(_u, zt, mu=1, cap=0.1, coef=1)
                next_state = quadrotor.forward(state=state_hat[-1], motor=_u[t], dt=dt, residual=fa)
            else:
                next_state = quadrotor.forward(state=state_hat[-1], motor=_u[t], dt=dt)
            state_hat.append(next_state)
        state_hat = torch.stack(state_hat[1:], dim=0)
        return loss(state_ref=state_ref, state_hat=state_hat, u=u, horizon=horizon_pred, gamma=gamma)

    # Initialize command
    if db_verbose:
        print('MPC starting', flush=True)
    # Backprop parameters
    lr = 1
    betas = (0.9, 0.999)
    niter = 100

    u = u0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam(params=[u], lr=lr, betas=betas)
    if db_analysis:
        log_mpc_loss = {}
        log_u = [u.clone().detach().cpu().numpy()]
    for i in range(niter):
        mpc_loss = to_minimize(u)
        mpc_loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()
        if db_analysis:
            if i == 0:
                for key in mpc_loss.keys():
                    log_mpc_loss[key] = [mpc_loss[key].item()]
            else:
                for key in mpc_loss.keys():
                    log_mpc_loss[key].append(mpc_loss[key].item())
            log_u.append(u.clone().detach().cpu().numpy())
    u_out = u.clone().detach().cpu().numpy()
    if db_analysis:
        for key in log_mpc_loss:
            log_mpc_loss[key] = np.stack(log_mpc_loss[key])  # (niter)
        log_u = np.stack(log_u)  # (niter + 1, horizon_cmd, 4)
        if False:
            fig, axs = plt.subplots(4, 2, figsize=(10, 10))
            axs[0, 0].plot(log_u[:, 0, 0])
            axs[0, 1].plot(log_u[:, 0, 1])
            axs[1, 0].plot(log_u[:, 0, 2])
            axs[1, 1].plot(log_u[:, 0, 3])

            axs[2, 0].plot(log_mpc_loss['loss'])
            axs[2, 0].set_ylabel('MPC loss')
            # axs[2, 0].set_yscale('log')

            axs[2, 1].plot(log_mpc_loss['position_loss'])
            axs[2, 1].set_ylabel('Position loss')
            axs[2, 1].set_yscale('log')

            axs[3, 0].plot(log_mpc_loss['yaw_loss'])
            axs[3, 0].set_xlabel('Iteration')

            axs[3, 1].plot(log_mpc_loss['u_loss'])
            axs[3, 1].set_xlabel('Iteration')
            plt.tight_layout()
            plt.show()
    if db_verbose:
        print(f'''Finished | mpc_error: {mpc_loss['loss'].item()} | u: {u_out}''')
    return u_out, mpc_loss['position_loss'].item(), mpc_loss['yaw_loss'].item(), mpc_loss['u_loss'].item()


def loss(state_ref: torch.Tensor, state_hat: torch.Tensor, u, gamma: float, horizon: int) -> Dict['str', torch.Tensor]:
    # Global minimum is -1 * 0.05
    # yaw loss impact is at most +-0.05
    coefs = [1]
    for i in range(horizon - 1):
        coefs.append(gamma * coefs[-1])
    coefs = torch.FloatTensor(coefs)

    position_ref, quat_ref, lin_vel_ref, ang_vel_ref = (state_ref[:, :3], state_ref[:, 3:7],
                                                        state_ref[:, 7:10], state_ref[:, 10:13])
    position_hat, quat_hat, lin_vel_hat, ang_vel_hat = (state_hat[:, :3], state_hat[:, 3:7],
                                                        state_hat[:, 7:10], state_hat[:, 10:13])

    # Position loss, shape=(horizon_pred, 3)
    position_loss = torch.mean((position_ref - position_hat) ** 2, dim=-1)
    position_loss = torch.mean(coefs * position_loss)

    # Yaw loss
    # METHOD 1
    # rot_loss = np.arccos(np.clip(np.sum((rot_ref * rot_hat).reshape(-1, 9), axis=1) / 3, -1, 1))
    # METHOD 2
    # yaw_ref = tf.as_rot_euler(seq='zyx', degrees=False)[:, 0]
    # yaw_hat = Rotation.from_quat(quat_hat).as_euler(seq='zyx', degrees=False)[:, 0]
    # yaw_loss = (yaw_ref - yaw_hat) ** 2
    # yaw_loss = coefs @ yaw_loss
    # METHOD 3
    vect_hat = quaternion.qrot(quat_hat, torch.FloatTensor([1, 0, 0]).repeat(horizon, 1))
    vect_hat[:, 2] = 0
    vect_hat = vect_hat / torch.norm(vect_hat, dim=-1).reshape((-1, 1))

    vect_ref = quaternion.qrot(quat_ref, torch.FloatTensor([1, 0, 0]).repeat(horizon, 1))
    vect_ref[:, 2] = 0
    vect_ref = vect_ref / torch.norm(vect_ref, dim=-1).reshape((-1, 1))
    yaw_loss = -torch.sum(vect_ref * vect_hat, dim=-1)
    yaw_loss = torch.mean(coefs * yaw_loss)

    # Command loss
    u_loss = torch.mean((u - torch.FloatTensor([150, 150, 150, 150])) ** 2)

    # Linear velocity loss
    lin_vel_loss = coefs @ torch.mean((lin_vel_ref - lin_vel_hat) ** 2, dim=-1)
    return {'loss': 1. * position_loss + 0.5 * yaw_loss + 0. * lin_vel_loss + 1e-2 * u_loss,
            'position_loss': position_loss,
            'yaw_loss': yaw_loss,
            'lin_vel_loss': lin_vel_loss,
            'u_loss': u_loss}
