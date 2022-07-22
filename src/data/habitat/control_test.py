import numpy as np
import os
import pickle
import torch
import sys
from tqdm import tqdm

from src.utils.trajectory import trace_trajectory3d
import src.learning.residual.predict as predict
import src.data.habitat.data_gen as data_gen
from src.data.habitat.simulation import Simulation
import src.data.habitat.loading as habitat
import src.data.habitat.mpc_grad as mpc_grad
import src.data.habitat.mpc as mpc
from src.utils.transform import Transform as Tf
from src.data.habitat.hatbitat2flu import Habitat2FLU


def control_test():
    """
    Test the control module.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = pickle.load(open(f'{habitat.DATA_HABITAT_PATH}labels.pkl', 'rb'))
    # for scene_number in habitat.TEST:
    #     print(f'scene_number={scene_number} | labels={labels[scene_number].keys()}')

    for scene_number in habitat.TEST:
        scene_number = 73
        for flight_number in labels[scene_number]:
            scene_id = habitat.MP3D_SCENE_LIST[scene_number]
            scene_path = f'{habitat.DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/'
            # Initialise drone
            drone_prms = habitat.DRONE_PRMS
            gradient_descent = False
            if gradient_descent:
                horizon_cmd = 40
                horizon_pred = 40
            else:
                horizon_cmd = 1
                horizon_pred = 20
            dt = 1 / 20
            gamma = 1.
            camera_resolution = np.array([480, 640])

            source_folder = f'{scene_path}{habitat.FLIGHT}/{flight_number}/'
            save_folder = f'{scene_path}control_easy/{flight_number}/'

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

            def mpc_model(_state, _u):
                _next_state = sim.quadrotor.forward(state=_state, motor=_u, dt=dt)
                return _next_state

            has_collided = True
            for i in range(1, time_end):
                # Computing command with the controller
                # try:
                if gradient_descent:
                    motor_horizon, pos_err, yaw_err, u_err = mpc_grad.mpc_controller(sim=sim,
                                                                                     state_ref=state_ref_flu[
                                                                                               i:i + horizon_pred],
                                                                                     state0=data_collection['state'][-1],
                                                                                     u0=motor_horizon,
                                                                                     depth=np.stack(
                                                                                         data_collection['depth'][-20:]),
                                                                                     horizon_cmd=horizon_cmd,
                                                                                     horizon_pred=horizon_pred,
                                                                                     gamma=gamma,
                                                                                     estimate_fa=False,
                                                                                     db_verbose=False,
                                                                                     db_analysis=True)
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
                print(f'mpc position error: {pos_err:.4f} | yaw error: {yaw_err:.3f} | u error: {u_err:.1f} | '
                      f'min {np.min(motor_horizon):.1f} | max {np.max(motor_horizon):.1f}')
                # except ValueError:
                # print(f'MPC diverged', file=sys.stderr, flush=True)
                # sim.sim.close(True)
                # # pickle.dump({'done': None, 'info': 'diverged'}, open(f'{gen_folder}stats.pkl', 'wb'))
                # break

                out = sim.one_step_sim(dt=dt, state=data_collection['state'][-1], motor=motor_horizon[0],
                                       z=data_collection['surface_dist'][-1])
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
                # data_collection['depth'].append(obs['left_depth_camera'])  # t+1
                data_collection['surface_dist'].append(obs['surface_dist'][0])  # t+1
                data_collection['state'].append(next_state_flu)  # t+1
                data_collection['state_ref'].append(state_ref_flu[i])  # t+1

            # Observe drone trajectory
            state_hab = Habitat2FLU.state_flu_2_state_habitat(np.stack(data_collection['state'], axis=0))

            # TODO Get a working reference
            sim.interactive_sim(True, False,
                                tf_follow=Tf.from_quat_trans(sim.state_ref[:, 3:7], sim.state_ref[:, :3]).as_matrix())
            sim.interactive_sim(True, False,
                                tf_follow=Tf.from_quat_trans(state_hab[:, 3:7], state_hab[:, :3]).as_matrix())
            state_ref_flu = Habitat2FLU.state_habitat_2_state_flu(sim.state_ref)
            trace_trajectory3d(np.stack(data_collection['state'])[:, :3],
                               np.stack(data_collection['state'])[:, 3:7], pause=1e-4, step=1,
                               position_ref=state_ref_flu[:, :3])

            sim.sim.close(True)
            data_generated = {}
            for key, val in data_collection.items():
                if len(val) != 0:
                    data_generated[key] = np.stack(val, axis=0)
            print()
            # Saving
            # dict_keys(['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force'])
            if not sys.platform == "darwin":  # then saving
                import time
                tic = time.time()
                print(f'Saving', file=sys.stderr, flush=True)
                if len(data_generated['motor']) == 0:
                    pickle.dump({'done': None, 'info': 'Not enough data'}, open(f'{source_folder}stats.pkl', 'wb'))
                for name, array in data_generated.items():
                    np.save(f'{source_folder}{name}.npy', array.astype(np.float32))
                info = 'collided' if has_collided else None
                pickle.dump({'done': i + 1, 'length': time_end, 'info': f'{info}'},
                            open(f'{source_folder}stats.pkl', 'wb'))
                print(f'Progress: {i + 1} | Max_length: {time_end} | has_collided: {has_collided}', file=sys.stderr,
                      flush=True)
                print(f'Time without compression: {time.time() - tic}s', file=sys.stderr, flush=True)


def main():
    control_test()


if __name__ == '__main__':
    main()
