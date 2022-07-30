import numpy as np
import os
import pickle
import torch
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.functional as f
import argparse

from src.utils.trajectory import trace_trajectory3d
import src.learning.residual.predict as predict
import src.data.habitat.data_gen as data_gen
from src.data.habitat.simulation import Simulation
import src.data.habitat.loading as habitat
import src.data.habitat.mpc_grad as mpc_grad
import src.data.habitat.mpc as mpc
from src.utils.transform import Transform as Tf
from src.data.habitat.hatbitat2flu import Habitat2FLU
import src.utils.path as path


def control_test():
    """
    Test the control module.
    """
    parser = argparse.ArgumentParser(description='Simulation and control parameter.')
    parser.add_argument('--estimate_fa', default=True, type=bool, help="If we should use the model to estimate fa")
    parser.add_argument('--test_number', default=12, type=int, help='The scene number to fly in.')
    parser.add_argument('--hold_fa', default=False, type=bool, help='If we should hold fa in the MPC prediction '
                                                                    'horizon.')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))

    scene_number = habitat.TEST[args.test_number]  # 73 = habitat.TEST[12]
    for flight_number in labels[scene_number]:
        print(f'TRAJECTORY {scene_number}_{flight_number}', flush=True)
        scene_id = habitat.MP3D_SCENE_LIST[scene_number]
        scene_path = f'{habitat.DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/'
        # Initialise drone
        drone_prms = habitat.DRONE_PRMS
        gradient_descent = True
        if gradient_descent:
            horizon_cmd = 20
            horizon_pred = 20
        else:
            horizon_cmd = 1
            horizon_pred = 20
        dt = 1 / 20
        gamma = 1.
        camera_resolution = np.array([480, 640])

        source_folder = f'{scene_path}{habitat.FLIGHT}/{flight_number}/'

        # Loading scene and simulation
        sim = Simulation(f'{scene_path}{scene_id}.glb', camera_resolution, drone_prms, gravity=9.81,
                         dtype=np.float64)
        # Generate reference
        if not os.path.exists(f'{source_folder}state_ref.npy'):
            print(f'Generating reference for {scene_id} {flight_number}')
            sim.generate_reference(uniform_speed=1, manual_ref=True, dt=dt, seed=flight_number,
                                   viz_navmesh=False, folder=source_folder,
                                   navmesh_path=f'{scene_path}.navmesh')

            np.save(f'{source_folder}state_ref.npy', sim.state_ref)
        else:
            print(f'Loading reference for {scene_id} {flight_number}', flush=True)
            sim.state_ref = np.load(f'{source_folder}state_ref.npy')
        # sim.interactive_sim(True, False,
        #                     tf_follow=Tf.from_quat_trans(sim.state_ref[:, 3:7],
        #                                                  sim.state_ref[:, :3]).as_matrix())

        sim.agent.transformation = Tf.from_quat_trans(sim.state_ref[0, 3:7], sim.state_ref[0, :3]).as_matrix()
        sim.agent.linear_velocity = np.zeros(3)
        sim.agent.angular_velocity = np.zeros(3)
        state_ref_flu = Habitat2FLU.state_habitat_2_state_flu(sim.state_ref)
        obs = sim.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        data_collection = {'state': [state_ref_flu[0]],  # flu
                           'state_ref': [state_ref_flu[0]],
                           'motor': [],
                           'mpc_position_error': [],
                           'mpc_yaw_error': [],
                           'mpc_u_error': [],
                           'depth': [obs['left_depth_camera']],
                           'surface_dist': [obs["surface_dist"][0]],
                           'ground_force': []}
        time_end = len(sim.state_ref) - horizon_pred + 1

        # First mpc command after it take the previous one for initialisation
        hover_cmd = np.sqrt(drone_prms['m'] * habitat.g / (4 * drone_prms['kt']))
        motor_horizon = np.repeat(hover_cmd, horizon_cmd * 4).reshape((horizon_cmd, 4))
        u_old = motor_horizon[0]

        def mpc_model(_state, _u):
            _next_state = sim.quadrotor.forward(state=_state, motor=_u, dt=dt)
            return _next_state

        has_collided = True
        for i in range(1, time_end):
            if i < 40:
                sim.mu = 1
                sim.cap = 0.1
                sim.coef = 1
            else:
                sim.mu = 100  # 1
                sim.cap = 50  # 0.1
                sim.coef = 10000  # 1
            # Computing command with the controller
            # try:
            if gradient_descent:
                out = mpc_grad.mpc_controller(sim=sim,
                                              state_ref=state_ref_flu[i:i + horizon_pred],
                                              state0=data_collection['state'][-1],
                                              u0=motor_horizon,
                                              depth=np.stack(data_collection['depth'][-40:]),
                                              u_old=u_old,
                                              horizon_cmd=horizon_cmd,
                                              horizon_pred=horizon_pred,
                                              gamma=gamma,
                                              estimate_fa=args.estimate_fa,
                                              hold_fa=args.hold_fa,
                                              db_verbose=False,
                                              db_analysis=False)
                motor_horizon, pos_err, yaw_err, u_err, u_dyn_err = out
            else:
                motor_horizon, err = mpc.mpc_controller(model=mpc_model,
                                                        state_ref=state_ref_flu[i:i + horizon_pred],
                                                        state0=data_collection['state'][-1],
                                                        u0=motor_horizon,
                                                        horizon_cmd=horizon_cmd,
                                                        horizon_pred=horizon_pred,
                                                        gamma=gamma,
                                                        db_verbose=False)
                pos_err = 0
                yaw_err = 0
                u_err = 0
                u_dyn_err = 0
            print(f'i: {i} | '
                  f'mpc position error: {pos_err:3.4f} | yaw error: {yaw_err:3.3f} | u error: {u_err:3.1f} | '
                  f'u dyn error: {u_dyn_err:3.4f} | '
                  f'min {np.min(motor_horizon):3.1f} | max {np.max(motor_horizon):3.1f}')
            # except ValueError:
            # print(f'MPC diverged', file=sys.stderr, flush=True)
            # sim.sim.close(True)
            # # pickle.dump({'done': None, 'info': 'diverged'}, open(f'{gen_folder}stats.pkl', 'wb'))
            # break

            out = sim.one_step_sim(dt=dt, state=data_collection['state'][-1], motor=motor_horizon[0],
                                   z=data_collection['surface_dist'][-1])
            u_old = motor_horizon[0]
            next_state_flu, obs, has_collided, ground_effect = out
            if (not np.isfinite(next_state_flu).all() or not np.isfinite(ground_effect).all() or
                    not np.isfinite(obs['surface_dist']).all()):
                print(f'Incorrect values from habitat', file=sys.stderr, flush=True)
                sim.sim.close(True)
                # pickle.dump({'done': None, 'info': 'nan'}, open(f'{gen_folder}stats.pkl', 'wb'))
                break
            if has_collided:
                print('Drone has collided', file=sys.stderr, flush=True)
                break
            if obs['surface_dist'][0] == 0:
                print('Drone has zeros height', file=sys.stderr, flush=True)
                break
            data_collection['motor'].append(motor_horizon[0])  # t to compute t+1
            data_collection['mpc_position_error'].append(pos_err)  # t to compute t+1
            data_collection['mpc_yaw_error'].append(yaw_err)  # t to compute t+1
            data_collection['mpc_u_error'].append(u_err)  # t to compute t+1
            data_collection['ground_force'].append(ground_effect)  # t used to compute t+1
            data_collection['depth'].append(obs['left_depth_camera'])  # t+1
            data_collection['surface_dist'].append(obs['surface_dist'][0])  # t+1
            data_collection['state'].append(next_state_flu)  # t+1
            data_collection['state_ref'].append(state_ref_flu[i])  # t+1

        # Observe drone trajectory
        state_hab = Habitat2FLU.state_flu_2_state_habitat(np.stack(data_collection['state'], axis=0))
        data_generated = {}
        for key, val in data_collection.items():
            if len(val) != 0:
                data_generated[key] = np.stack(val, axis=0)

        # sim.interactive_sim(True, False,
        #                     tf_follow=Tf.from_quat_trans(sim.state_ref[:, 3:7], sim.state_ref[:, :3]).as_matrix())
        # sim.interactive_sim(True, False,
        #                     tf_follow=Tf.from_quat_trans(state_hab[:, 3:7], state_hab[:, :3]).as_matrix())
        # state_ref_flu = Habitat2FLU.state_habitat_2_state_flu(sim.state_ref)
        # trace_trajectory3d(np.stack(data_collection['state'])[:, :3],
        #                    np.stack(data_collection['state'])[:, 3:7], pause=1e-4, step=1,
        #                    position_ref=state_ref_flu[:, :3])

        # sim.sim.close(True)
        # print()
        # plt.plot(data_generated['surface_dist'])
        # plt.plot(data_generated['ground_force'][:, 2])
        # plt.plot(data_generated['ground_force'])
        # Saving
        # dict_keys(['state', 'state_ref', 'motor', 'mpc_position_error', 'mpc_yaw_error', 'mpc_u_error', 'depth',
        # 'surface_dist', 'ground_force'])
        # if not sys.platform == "darwin":  # then saving

        t1 = f'control_fa_strong_'
        t2 = f'no_' if args.estimate_fa else f''
        t3 = f'pred_u_dyn_'
        t4 = f'hold_fa' if args.hold_fa else f''

        folder_name = f'{t1}{t2}{t3}{t4}'

        save_folder = f'{scene_path}{habitat.FLIGHT}/{flight_number}/{folder_name}/'
        path.mkdir([save_folder])
        import time
        tic = time.time()
        print(f'Saving', file=sys.stderr, flush=True)
        if len(data_generated['motor']) == 0:
            pickle.dump({'done': None, 'info': 'Not enough data'}, open(f'{save_folder}stats.pkl', 'wb'))
        for name, array in data_generated.items():
            np.save(f'{save_folder}{name}.npy', array.astype(np.float32))
        info = 'collided' if has_collided else None
        pickle.dump({'done': i + 1, 'length': time_end, 'info': f'{info}'},
                    open(f'{save_folder}stats.pkl', 'wb'))
        print(f'Progress: {i + 1} | Max_length: {time_end} | has_collided: {has_collided}', file=sys.stderr,
              flush=True)
        print(f'Time without compression: {time.time() - tic}s', file=sys.stderr, flush=True)
        break


def compare_control():
    plt.switch_backend('macosx')
    plt.style.use('seaborn-deep')
    scene_number = 73
    flight_number = 0
    scene_id = habitat.MP3D_SCENE_LIST[scene_number]
    scene_path = f'{habitat.DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/'
    save_folder_no_pred = f'{scene_path}{habitat.FLIGHT}/{flight_number}/control_fa_strong_no_pred/'
    save_folder_pred = f'{scene_path}{habitat.FLIGHT}/{flight_number}/control_fa_strong_pred/'

    # Load data
    no_pred = {
        'state': np.load(f'{save_folder_no_pred}state.npy'),
        'state_ref': np.load(f'{save_folder_no_pred}state_ref.npy'),
        'motor': np.load(f'{save_folder_no_pred}motor.npy'),
        'mpc_position_error': np.load(f'{save_folder_no_pred}mpc_position_error.npy'),
        'mpc_yaw_error': np.load(f'{save_folder_no_pred}mpc_yaw_error.npy'),
        'mpc_u_error': np.load(f'{save_folder_no_pred}mpc_u_error.npy'),
        'fa_gt': np.load(f'{save_folder_no_pred}ground_force.npy'),
    }
    pred = {
        'state': np.load(f'{save_folder_pred}state.npy'),
        'state_ref': np.load(f'{save_folder_pred}state_ref.npy'),
        'motor': np.load(f'{save_folder_pred}motor.npy'),
        'mpc_position_error': np.load(f'{save_folder_pred}mpc_position_error.npy'),
        'mpc_yaw_error': np.load(f'{save_folder_pred}mpc_yaw_error.npy'),
        'mpc_u_error': np.load(f'{save_folder_pred}mpc_u_error.npy'),
        'depth': np.load(f'{save_folder_pred}depth.npy'),
        # 'fa_hat': np.load(f'{save_folder_pred}fa_hat.npy'),
        'fa_gt': np.load(f'{save_folder_pred}ground_force.npy'),

    }

    no_pred['real_pos_loss'] = np.sqrt((no_pred['state'][:, :3] - no_pred['state_ref'][:, :3]) ** 2)
    pred['real_pos_loss'] = np.sqrt((pred['state'][:, :3] - pred['state_ref'][:, :3]) ** 2)
    xx_fa = range(40, len(pred['state']))
    plot_results = True
    if plot_results:
        # Plotting RMSE position loss for x, y, z
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i, ax_name in zip(range(3), ['x', 'y', 'z']):
            ax: plt.Axes = axs[i]
            ax.plot(no_pred['real_pos_loss'][:, i], label='MPC')
            ax.plot(xx_fa, pred['real_pos_loss'][40:, i], label='MPC using fa model')
            ax.set_ylabel(f'RMSE {ax_name} loss')
            ax.set_xlabel('Time (# frames)')
            ax.grid()
            ax.legend()
            ax.set_ylim([-0.05, 0.8])
            ax.axvline(x=40, c='black')
            plt.tight_layout()
        plt.savefig(f'{path.WD_PATH}reports/compare_control_error.png')

        # Plotting RMSE position loss for x, y, z
        no_pred['rmse_per_dim'] = np.sqrt(
            np.mean((no_pred['state'][40:, :3] - no_pred['state_ref'][40:, :3]) ** 2, axis=0))
        pred['rmse_per_dim'] = np.sqrt(np.mean((pred['state'][40:, :3] - pred['state_ref'][40:, :3]) ** 2, axis=0))
        print(f'RMSE x: {10000 * no_pred["rmse_per_dim"][0]:.0f}')
        print(f'RMSE x with fa(depth, theta): {10000 * pred["rmse_per_dim"][0]:.0f}')
        print(f'RMSE y: {10000 * no_pred["rmse_per_dim"][1]:.0f}')
        print(f'RMSE y with fa(depth, theta): {10000 * pred["rmse_per_dim"][1]:.0f}')
        print(f'RMSE z: {10000 * no_pred["rmse_per_dim"][2]:.0f}')
        print(f'RMSE z with fa(depth, theta): {10000 * pred["rmse_per_dim"][2]:.0f}')

        no_pred['rmse_total'] = np.sqrt(np.mean((no_pred['state'][40:, :3] - no_pred['state_ref'][40:, :3]) ** 2))
        pred['rmse_total'] = np.sqrt(np.mean((pred['state'][40:, :3] - pred['state_ref'][40:, :3]) ** 2))
        print(f'RMSE total: {10000 * no_pred["rmse_total"]:.0f}')
        print(f'RMSE total with fa(depth, theta): {10000 * pred["rmse_total"]:.0f}')

        # Plotting height of the drone
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax: plt.Axes = ax
        ax.plot(no_pred['state_ref'][:, 2], label='Reference')
        ax.plot(no_pred['state'][:, 2], label='MPC')
        ax.plot(pred['state'][:, 2], label='MPC + fa(depth, theta)')
        ax.set_ylabel('z (m)')
        ax.set_xlabel('Time (# frames)')
        ax.grid()
        ax.legend()
        ax.axvline(x=40, c='black')
        plt.tight_layout()
        plt.savefig(f'{path.WD_PATH}reports/compare_motor.png')

        # Plotting height of the drone
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax: plt.Axes = ax
        ax.plot(no_pred['state_ref'][:, 2], label='Reference')
        ax.plot(no_pred['state'][:, 2], label='MPC')
        ax.plot(xx_fa, pred['state'][40:, 2], label='MPC + fa(depth, theta)')
        ax.set_ylabel('z (m)')
        ax.set_xlabel('Time (# frames)')
        ax.grid()
        ax.legend()
        ax.axvline(x=40, c='black')
        plt.tight_layout()
        plt.savefig(f'{path.WD_PATH}reports/compare_motor.png')
        plt.show()

        # Plotting real fa and fa
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        ax: plt.Axes = axs[0]
        ax.plot(pred['fa_gt'][:, 2], label='Reference')
        ax.plot(range(39, len(pred['state'])), pred['fa_hat'], label='Ours')
        ax.set_ylabel('fa (N)')
        ax.set_xlabel('Time (# frames)')
        ax.grid()
        ax.legend()
        ax.axvline(x=39, c='black')

        ax: plt.Axes = axs[1]
        ax.plot(no_pred['fa_gt'][:, 2], label='Reference')
        ax.set_ylabel('fa (N)')
        ax.set_xlabel('Time (# frames)')
        ax.grid()
        ax.legend()
        ax.axvline(x=39, c='black')
        plt.tight_layout()
        plt.savefig(f'{path.WD_PATH}reports/compare_fa.png')
        plt.show()

    predict_fa = False
    if predict_fa:
        # Predicting fa
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z_estimator, _, _, _ = predict.get_trained_model_and_dataset('irs_normal_finetune', True, 'habitat')
        z_estimator.to(DEVICE)
        z_estimator.eval()
        depth = torch.from_numpy(pred['depth']).to(DEVICE).float()
        depth = f.resized_crop(depth, (480 - 408) // 2, 0, 408, 640, [240, 376])
        depth = torch.clip(depth / 20, 0, 1) * 2 - 1
        reshaped_depth = torch.empty(len(depth) - 39, 40, 240, 376) * np.nan
        for i in range(39, len(depth)):
            reshaped_depth[i - 39] = depth[i - 39:i + 1]
        assert reshaped_depth.isnan().sum() == 0
        reshaped_depth = reshaped_depth.to(DEVICE).float()
        fa_hat = torch.empty(len(reshaped_depth)) * np.nan
        for i in tqdm(range(len(reshaped_depth))):
            with torch.no_grad():
                fa_hat[i] = z_estimator(reshaped_depth[i])
        np.save(f'{save_folder_pred}fa_hat.npy', fa_hat.numpy())


def main():
    # control_test()
    compare_control()


if __name__ == '__main__':
    main()
