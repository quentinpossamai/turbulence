import argparse
import os
import pickle
import sys
import numpy as np
from tqdm import tqdm

import src.data.habitat.loading as habitat
from src.data.habitat.hatbitat2flu import Habitat2FLU
from src.data.habitat.mpc import mpc_controller
from src.utils.path import WD_PATH, mkdir
from src.utils.trajectory import trace_trajectory3d
from src.utils.transform import Transform as Tf
from src.data.habitat.simulation import Simulation

DTYPE = np.float32
SCENE_NUMBER_MISSING_DATA = [7, 8, 18, 21, 22, 23, 30, 37, 39, 41, 50, 52, 53, 61, 62, 71, 73, 76, 77, 79, 81, 82]

# NOTATION
# AGENT is the drone body in habitat convention
# AGENT_FLU is AGENT but turned from HABITAT to FLU forward left up
# GLOBAL_FLU is the earth reference frame but turned from HABITAT to FLU
# GLOBAL is the habitat earth reference frame

DATA_HABITAT_PATH = f'{WD_PATH}data/habitat/'
MP3_SCENE_LIST = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_scene_list.pkl', 'rb'))
GEN_DATA_KEY = ['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force']


def gen_data(scene_number: int, flight_number: int,
             db_manual: bool,
             db_same_env: bool = False,
             db_plot_ref: bool = False):
    """
    Generate drone data in habitat with ground effect.

    :param scene_number: path to a habitat scene.

    :param flight_number: The seed to generate the trajectory.


    Debug features:

    :param db_manual: Use to generate reference manually.

    :param db_same_env: Boolean if False add ground effect physics. If True uses the quadrotor model which is the same
    as the model use for MPC. Useful for debugging.

    :param db_plot_ref: Boolean to plot the reference for debugging.
    """
    # Initialize the scene
    scene_id = MP3_SCENE_LIST[scene_number]
    gen_folder = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight2/{flight_number}/'
    mkdir([gen_folder])

    # Initialise drone
    drone_prms = habitat.DRONE_PRMS
    dt = 1 / 20
    horizon_cmd = 1
    horizon_pred = 20
    gamma = 1.
    camera_resolution = np.array([480, 640])

    # Loading the scene in habitat
    cluster_scene_path = f"/storage/Datasets/3D-Simulator-Datasets/scene_datasets/mp3d/{scene_id}/{scene_id}"
    mac_scene_path = f"{WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}"
    if sys.platform != 'darwin':  # local on the macbook
        scene_path = cluster_scene_path
    else:  # on the cluster
        scene_path = mac_scene_path
        command = f'rsync -Ptv remsto:{cluster_scene_path}.glb {mac_scene_path}.glb'
        if not os.path.isfile(f'{scene_path}.glb'):  # If the scene does not exist, download it
            print(command)
            raise (FileNotFoundError('Scene is not in local'))
    simulation = Simulation(f'{scene_path}.glb', camera_resolution, drone_prms, habitat.g)

    # Generate reference
    has_worked = simulation.generate_reference(uniform_speed=1, manual_ref=db_manual, dt=dt, seed=flight_number,
                                               viz_navmesh=False, folder=gen_folder,
                                               navmesh_path=f'{scene_path}.navmesh')
    if not has_worked:
        simulation.sim.close(True)
        pickle.dump({'done': None, 'info': 'no_trajectory'}, open(f'{gen_folder}stats.pkl', 'wb'))
        return False

    def mpc_model(_state, _u):
        _next_state = simulation.quadrotor.forward(state=_state, motor=_u, dt=dt)
        return _next_state

    # state are always [trans 3, quat 4, lin_vel 3, ang_vel 3, acc_lin 3, acc_ang 3]
    # Setting the drone at the first reference
    simulation.agent.transformation = Tf.from_quat_trans(simulation.state_ref[0, 3:7],
                                                         simulation.state_ref[0, :3]).as_matrix()
    simulation.agent.linear_velocity = np.zeros(3)
    simulation.agent.angular_velocity = np.zeros(3)

    state_ref_flu = Habitat2FLU.state_habitat_2_state_flu(simulation.state_ref)
    if db_plot_ref:
        tf_follow = Tf.from_quat_trans(simulation.state_ref[:, 3:7], simulation.state_ref[:, 0:3]).as_matrix()
        simulation.interactive_sim(True, False, tf_follow)
        trace_trajectory3d(state_ref_flu[:, :3], state_ref_flu[:, 3:7], pause=1e-4, step=1)
        # os.remove(f'{gen_folder}initial_point_ref.pkl')

    # Initializing flight in habitat
    # Sensing
    obs = simulation.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
    data_collection = {'state': [state_ref_flu[0]],  # flu
                       'motor': [],
                       'mpc_error': [],
                       'rgb': [obs['left_camera']],
                       'depth': [obs['left_depth_camera']],
                       'surface_dist': [obs["surface_dist"][0]],
                       'ground_force': []}
    print(f'state_ref_flu[0] : {state_ref_flu[0]}')
    time_end = len(simulation.state_ref) - horizon_pred + 1

    # First mpc command after it take the previous one for initialisation
    hover_cmd = np.sqrt(drone_prms['m'] * habitat.g / (4 * drone_prms['kt']))
    motor_horizon = np.repeat(hover_cmd, horizon_cmd * 4).reshape((horizon_cmd, 4))
    has_collided = False
    for i in tqdm(range(1, time_end)):
        # Computing command with the controller
        try:
            motor_horizon, err = mpc_controller(model=mpc_model,
                                                state_ref=state_ref_flu[i:i + horizon_pred],
                                                state0=data_collection['state'][-1],
                                                u0=motor_horizon,
                                                horizon_cmd=horizon_cmd,
                                                horizon_pred=horizon_pred,
                                                gamma=gamma,
                                                db_verbose=False)
            print(f'mpc error: {err}')
            print(f'motor horizon: {motor_horizon}\n\n')
        except ValueError:
            print(f'Drone diverged', file=sys.stderr, flush=True)
            simulation.sim.close(True)
            pickle.dump({'done': None, 'info': 'diverged'}, open(f'{gen_folder}stats.pkl', 'wb'))
            return False

        # Simulation result
        if not db_same_env:
            out = simulation.one_step_sim(dt=dt, state=data_collection['state'][-1], motor=motor_horizon[0],
                                          z=data_collection['surface_dist'][-1])
            next_state_flu, obs, has_collided, ground_effect = out
            if (not np.isfinite(next_state_flu).all() or not np.isfinite(ground_effect).all() or
                    not np.isfinite(obs['surface_dist']).all()):
                print(f'Incorrect values from habitat', file=sys.stderr, flush=True)
                simulation.sim.close(True)
                pickle.dump({'done': None, 'info': 'nan'}, open(f'{gen_folder}stats.pkl', 'wb'))
                return False
            if has_collided:
                print('Drone has collided', file=sys.stderr, flush=True)
                break
            if obs['surface_dist'][0] == 0:
                print('Drone has zeros height', file=sys.stderr, flush=True)
                # state = np.stack(data_collection['state'])
                # state_habitat = Habitat2FLU.state_flu_2_state_habitat(state)
                # tf_follow = Tf.from_quat_trans(state_habitat[:, 3:7], state_habitat[:, :3]).as_matrix()
                # simulation.interactive_sim(camera_rgb=True, camera_depth=False, step=dt, tf_follow=tf_follow)
                break
            data_collection['motor'].append(motor_horizon[0])  # t to compute t+1
            data_collection['mpc_error'].append(err)  # t to compute t+1
            data_collection['ground_force'].append(ground_effect)  # t used to compute t+1
            data_collection['rgb'].append(obs['left_camera'])  # t+1
            data_collection['depth'].append(obs['left_depth_camera'])  # t+1
            data_collection['surface_dist'].append(obs['surface_dist'][0])  # t+1
            data_collection['state'].append(next_state_flu)  # t+1
        else:
            next_state = simulation.quadrotor.forward(dt=dt, state=data_collection['state'][-1],
                                                      motor=data_collection['motor'][-1])
            data_collection['state'].append(next_state)
    state_hab = Habitat2FLU.state_flu_2_state_habitat(np.stack(data_collection['state'], axis=0))

    simulation.interactive_sim(True, False,
                               tf_follow=Tf.from_quat_trans(state_hab[:, 3:7], state_hab[:, :3]).as_matrix())
    simulation.sim.close(True)
    data_generated = {}
    for key, val in data_collection.items():
        if len(val) != 0:
            data_generated[key] = np.stack(val, axis=0)

    # # Saving
    # # dict_keys(['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force'])
    # if not sys.platform == "darwin":  # then saving
    #     import time
    #     tic = time.time()
    #     print(f'Saving', file=sys.stderr, flush=True)
    #     if len(data_generated['motor']) == 0:
    #         pickle.dump({'done': None, 'info': 'Not enough data'}, open(f'{gen_folder}stats.pkl', 'wb'))
    #     for name, array in data_generated.items():
    #         np.save(f'{gen_folder}{name}.npy', array.astype(np.float32))
    #     info = 'collided' if has_collided else None
    #     pickle.dump({'done': i + 1, 'length': time_end, 'info': f'{info}'}, open(f'{gen_folder}stats.pkl', 'wb'))
    #     print(f'Progress: {i + 1} | Max_length: {time_end} | has_collided: {has_collided}', file=sys.stderr, flush=True)
    #     print(f'Time without compression: {time.time() - tic}s', file=sys.stderr, flush=True)
    return True


def main1():
    """
    Generate data with habitat using Matterport3D dataset with a flying UAV following procedurally generated references.
    """
    scene_number = 0
    flight_number = 0
    parser = argparse.ArgumentParser(description="Generate drone flight data.")
    parser.add_argument('--scene_start', default=scene_number, type=int, help='Scene number range start [0, 89]')
    parser.add_argument('--scene_stop', default=scene_number + 1, type=int, help='Scene number range stop [1, 90]')
    # fn = flight_number correspond to a specific trajectory.
    parser.add_argument('--fn_start', default=flight_number, type=int, help='Flight number range start')
    parser.add_argument('--fn_stop', default=flight_number + 1, type=int, help='Flight number range stop')
    args = parser.parse_args()

    for scene_number in range(args.scene_start, args.scene_stop):
        for flight_number in range(args.fn_start, args.fn_stop):
            print(f'SCENE_NUMBER={scene_number} | FLIGHT_NUMBER={flight_number}', file=sys.stderr, flush=True)
            gen_data(scene_number, flight_number,
                     db_manual=False)
            print(f'\n' * 20, file=sys.stderr, flush=True)


def main2():
    """
    Generate data with habitat using Matterport3D dataset with a flying UAV following manually defined reference.
    """
    for scene_number in SCENE_NUMBER_MISSING_DATA:
        # scene_number = SCENE_NUMBER_MISSING_DATA[21]
        for flight_number in range(0, 1):
            print(f'Generating data for scene {scene_number} and flight {flight_number}', file=sys.stderr, flush=True)
            gen_data(scene_number, flight_number, db_manual=False)
            print(f'\n' * 20, file=sys.stderr, flush=True)
            break


if __name__ == "__main__":
    main2()
