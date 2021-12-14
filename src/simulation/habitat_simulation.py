import cv2
import habitat_sim
import magnum
import numpy as np
import pickle
from tqdm import tqdm
import torch
from typing import Dict, Tuple, Any, Union

from src.drone_models.analytical_models.quadrotor_4parms import NumpyDynamicalModel
from src.simulation.generate_reference import generate_reference
from src.simulation.mpc import mpc_controller
from src.simulation.hatbitat2flu import Habitat2FLU
from src.utils.paths import WD_PATH, DATA_HABITAT_PATH
from src.utils.trajectory import trace_trajectory3d
from src.utils.transform import Transform

DTYPE = float

# NOTATION
# AGENT is the drone body in habitat convention
# AGENT_FLU is AGENT but turned from HABITAT to FLU forward left up
# GLOBAL_FLU is the earth reference frame but turned from HABITAT to FLU
# GLOBAL is the habitat earth reference frame

MP3_SCENE_LIST = pickle.load(open(f'{DATA_HABITAT_PATH}mp3_scene_list.pkl', 'rb'))


class Simulation:
    def __init__(self, scene_path: str,
                 camera_resolution: np.ndarray,
                 drone_prms: Dict[str, float]):
        """
        Habitat simulation of a quadrotor with ground effect.
        
        :param scene_path: The path to the .glb scene to load.

        :param camera_resolution: The resolution of the camera [height, width].

        :param drone_prms: Parameters of the drone.
        """
        # 1-Initializing the simulation and loading the scene
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.enable_physics = True
        backend_cfg.scene_id = scene_path

        # 2-Adding a stereo RGB-D sensors to the drone
        # 2.1-left rgb camera
        left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
        left_rgb_sensor.uuid = "left_camera"  # uuid is used to retrieve the observation
        left_rgb_sensor.resolution = camera_resolution
        left_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.LEFT

        # 2.2-right rgb camera
        right_rgb_sensor = habitat_sim.CameraSensorSpec()
        right_rgb_sensor.uuid = "right_camera"
        right_rgb_sensor.resolution = camera_resolution
        right_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT + 1.1 * habitat_sim.geo.BACK

        # 2.3-left depth camera
        left_depth_sensor = habitat_sim.CameraSensorSpec()
        left_depth_sensor.uuid = "left_depth_camera"
        left_depth_sensor.resolution = camera_resolution
        left_depth_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.LEFT
        left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # 2.4-right depth camera
        right_depth_sensor = habitat_sim.CameraSensorSpec()
        right_depth_sensor.uuid = "right_depth_camera"
        right_depth_sensor.resolution = camera_resolution
        right_depth_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT
        right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # 2.5-depth camera
        surface_distance_sensor = habitat_sim.CameraSensorSpec()
        surface_distance_sensor.uuid = "surface_dist"
        surface_distance_sensor.resolution = [1, 1]
        surface_distance_sensor.orientation = [-90 / 180 * np.pi, 0, 0]
        surface_distance_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT
        surface_distance_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # 3-Instantiating drone
        # 3.1-Connecting the drone to its sensors
        agent_config = habitat_sim.AgentConfiguration()
        agent_config.sensor_specifications = [surface_distance_sensor, left_rgb_sensor, right_rgb_sensor,
                                              left_depth_sensor, right_depth_sensor]
        # 3.2-Adding the drone to the simulation
        sim: habitat_sim.physics.ManagedRigidObject = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg,
                                                                                                      [agent_config]))
        # 3.3-Adding a rigid cube as a body to the drone
        # prim_templates_mgr = sim.get_asset_template_manager()
        obj_templates_mgr: habitat_sim.attributes_managers.ObjectAttributesManager = sim.get_object_template_manager()
        rigid_obj_mgr: habitat_sim.physics.RigidObjectManager = sim.get_rigid_object_manager()
        obj_templates_mgr.get_file_template_handles()

        # OPTION1: Drone has a cube shape
        cube_template_handle = obj_templates_mgr.get_template_handles('cubeSolid')[0]
        cube_attributes: habitat_sim.attributes.ObjectAttributes
        cube_attributes = obj_templates_mgr.get_template_by_handle(handle=cube_template_handle)
        cube_dim = [drone_prms['width'], drone_prms['height'], drone_prms['width']]
        cube_attributes.scale = cube_dim
        cube_attributes.collision_asset_size = cube_dim
        obj_templates_mgr.register_template(cube_attributes, specified_handle='my_cube')
        obj: habitat_sim.physics.ManagedRigidObject
        obj = rigid_obj_mgr.add_object_by_template_handle('my_cube', sim.agents[0].scene_node, light_setup_key='0')

        # # OPTION2: Drone has a quadrotor shape
        # # The loaded drone model is (0.1, 0.02, 0.1) meters of dimension
        # tmpl_id, = obj_templates_mgr.load_configs(f'{DATA_HABITAT_PATH}drone.object_config.json')
        # uav_attr: habitat_sim.attributes.ObjectAttributes = obj_templates_mgr.get_template_by_id(tmpl_id)
        # # uav_dim = [1., 1., 1.]
        # # uav_attr.scale = uav_dim
        # # uav_attr.collision_asset_size = uav_dim
        # obj_templates_mgr.register_template(uav_attr, specified_handle='uav')
        # obj = rigid_obj_mgr.add_object_by_template_handle('uav', sim.agents[0].scene_node, light_setup_key='0')

        # 4-Drone parameters
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        obj.mass = drone_prms['m']
        obj.intertia_diagonal = [drone_prms['ix'], drone_prms['iy'], drone_prms['iz']]
        self.sim: habitat_sim.simulator.Simulator = sim
        self.agent: habitat_sim.physics.ManagedRigidObject = obj
        self.drone_prms: Dict[str, float] = drone_prms
        self.ref_tf_global_agent = None
        self.ref_lin_vel_global = None

        # 5-simulation initialisation
        # 5.1-Drone model init
        assert self.drone_prms['ix'] == self.drone_prms['iy']
        theta = np.array([self.drone_prms['kt'] / self.drone_prms['m'],
                          self.drone_prms['kt'] * self.drone_prms['l'] / self.drone_prms['ix'],
                          self.drone_prms['km'] / self.drone_prms['iz'],
                          self.drone_prms['iz'] / self.drone_prms['ix']], dtype=DTYPE)
        self.quadrotor = NumpyDynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=DTYPE)

    def generate_reference_procedurally(self, uniform_speed: float, plot_topdown: bool, dt: float, seed: int,
                                        viz_navmesh: bool):
        """

        param uniform_speed: The number of step to cut the trajectory.

        param plot_topdown: If true will plot topdown view.

        param dt: dt to use for position derivation.

        param seed: Seed of trajectory generation.

        param viz_navmesh: If true rgb camera see the navmesh wireframe.
        """
        # Defining the drone's volume
        navmesh_settings: habitat_sim.nav.NavMeshSettings = habitat_sim.NavMeshSettings()
        navmesh_settings.agent_height = 0.2  # 1.5
        navmesh_settings.agent_max_climb = 2.0  # 0.2
        navmesh_settings.agent_max_slope = 87.0  # 45.0
        navmesh_settings.agent_radius = 0.2  # 0.1
        navmesh_settings.cell_height = 0.05  # 0.2
        navmesh_settings.cell_size = 0.10  # 0.05
        navmesh_settings.detail_sample_dist = 120.0  # 6.0
        navmesh_settings.detail_sample_max_error = 0.01  # 1.0
        navmesh_settings.edge_max_error = 0.01  # 1.3
        navmesh_settings.edge_max_len = 12.0  # 12.0
        navmesh_settings.filter_ledge_spans = True  # True
        navmesh_settings.filter_low_hanging_obstacles = False  # True
        navmesh_settings.filter_walkable_low_height_spans = True  # True
        navmesh_settings.region_merge_size = 20.0  # 20.0
        navmesh_settings.region_min_size = 20.0  # 20.0
        navmesh_settings.verts_per_poly = 6.0  # 6.0
        self.ref_tf_global_agent, self.ref_lin_vel_global = generate_reference(self.sim,
                                                                               uniform_speed,
                                                                               navmesh_settings,
                                                                               plot_topdown,
                                                                               flight_height=0.1,
                                                                               dt=dt,
                                                                               seed=seed,
                                                                               viz_navmesh=viz_navmesh)

    def one_step_sim(self, dt: float, state: np.ndarray, motor: np.ndarray) -> Tuple[np.ndarray, Any, bool]:
        # 1. Initializing agent
        # quat: x, y, z, w
        trans, quat, linear_velocity, angular_velocity = state[:3], state[3:7], state[7:10], state[10:]
        tf_global__agent = Transform.from_quat_trans(quat=quat, trans=trans).as_matrix()
        self.agent.transformation = tf_global__agent
        self.agent.linear_velocity = linear_velocity
        self.agent.angular_velocity = angular_velocity

        rot_global__agent_flu = Transform.from_matrix(tf_global__agent @ Habitat2FLU.tf_habitat__flu).as_rot_mat()

        # 2. Sensing
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        z = obs["surface_dist"][0]

        # 3. Motor and Ground Effect (gravity is done by habitat)
        motor_torsor = self.quadrotor.motor_dynamic(motor=motor)
        motor_torque = np.array([self.drone_prms['kt'] * self.drone_prms['l'],
                                 self.drone_prms['kt'] * self.drone_prms['l'],
                                 self.drone_prms['km']]) * motor_torsor[:3]

        motor_force = np.zeros_like(motor_torque)
        motor_force[2] = self.drone_prms['kt'] * motor_torsor[3]

        ground_force = np.zeros_like(motor_torque)
        # Todo enable ground_forces
        # ground_force[1] = np.clip((np.sum(motor) ** 2 * self.drone_prms['kt'] * 1
        #                            * (self.drone_prms['prop_radius'] / (4 * z)) ** 2), 0, 0.5)
        self.agent.apply_force(rot_global__agent_flu @ motor_force + ground_force,
                               relative_position=np.zeros(3))
        self.agent.apply_torque(rot_global__agent_flu @ motor_torque)

        # 4-Simulating for step seconds
        self.sim.step_physics(dt)
        quat, trans = Transform.from_matrix(np.array(self.agent.transformation)).as_quat_trans()
        next_state = np.concatenate([trans, quat, self.agent.linear_velocity, self.agent.angular_velocity], axis=0)
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        has_collided = self.agent.contact_test()
        obs['ground_force'] = ground_force
        return next_state, obs, has_collided

    def interactive_sim(self, controller, camera_rgb: bool, camera_depth: bool, step: float, tf_follow: np.ndarray):
        cv2.namedWindow("stereo_pair")
        # 1. Agent init
        simulation_iteration = 0
        self.agent.transformation = self.ref_tf_global_agent[0]
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone

        is_running = True
        while is_running:
            # 2. Sensors
            rgb_stereo_pair = np.concatenate([obs["left_camera"], obs["right_camera"]], axis=1)
            depth_stereo_pair = np.concatenate([obs["left_depth_camera"], obs["right_depth_camera"]], axis=1)
            z = obs["surface_dist"][0]
            tf_global_agent = np.array(self.agent.transformation)  # transform origin body
            rot_global_flu = Transform.from_matrix(tf_global_agent).as_rot_mat() @ Habitat2FLU.rot_habitat__flu
            if controller is not None:
                motor = controller
            else:
                motor = np.zeros((1000, 4))

            # Plotting sensors
            print(simulation_iteration)
            print(f'Height: {z}')
            has_collided = self.agent.contact_test()
            print(f'has_collided: {has_collided}')

            # 3-Motor and Ground Effect (gravity is done by habitat)
            motor_torsor = self.quadrotor.motor_dynamic(motor=motor[simulation_iteration])
            motor_torque = np.array([self.drone_prms['kt'] * self.drone_prms['l'],
                                     self.drone_prms['kt'] * self.drone_prms['l'],
                                     self.drone_prms['km']]) * motor_torsor[:3]

            motor_force = np.zeros_like(motor_torque)
            motor_force[2] = self.drone_prms['kt'] * motor_torsor[3]

            ground_force = np.zeros_like(motor_torque)
            ground_force[1] = (np.sum(motor[simulation_iteration]) ** 2 * self.drone_prms['kt'] * 1
                               * (self.drone_prms['prop_radius'] / (4 * z)) ** 2)

            self.agent.apply_force(rot_global_flu @ motor_force + ground_force,
                                   relative_position=magnum.Vector3([0, 0, 0]))
            self.agent.apply_torque(rot_global_flu @ motor_torque)
            # self.agent.apply_force(magnum.Vector3([0, self.drone_prms['m']*9.81, 0]),
            #                        relative_position=magnum.Vector3([0, 0, 0]))
            # self.agent.apply_torque(magnum.Vector3([0, 0, 0]))
            # 4-Display and interactivity
            if camera_rgb:
                cv2.imshow("rgb_stereo_pair", rgb_stereo_pair)
            if camera_depth:
                depth_stereo_pair = np.clip(depth_stereo_pair, 0, 10)
                depth_stereo_pair /= 10.0
                cv2.imshow("depth_stereo_pair", depth_stereo_pair)
            key_pressed = cv2.waitKey()

            # 5-Immobilizing the drone
            saved_velocity = (self.agent.linear_velocity, self.agent.angular_velocity)
            self.agent.linear_velocity = magnum.Vector3([0, 0, 0])
            self.agent.angular_velocity = magnum.Vector3([0, 0, 0])
            print(f"linear_velocity velocity: {self.agent.linear_velocity}")
            print(f"angular_velocity velocity: {self.agent.angular_velocity}\n\n")

            # print(f"angular_velocity velocity: {rot_global_flu @ np.array([+0, 0, 1])}\n\n")

            # 6-Exploring the scene
            # 6.1. Forward, Backward
            if key_pressed == ord("w"):  # Front
                self.agent.translate(rot_global_flu @ np.array([+0.1, 0, 0]))
            if key_pressed == ord("s"):  # Back
                self.agent.translate(rot_global_flu @ np.array([-0.1, 0, 0]))
            # 6.2 Thrust
            if key_pressed == ord("x"):  # Up
                self.agent.translate(rot_global_flu @ np.array([0, 0, +0.1]))
            if key_pressed == ord("z"):  # Down
                self.agent.translate(rot_global_flu @ np.array([0, 0, -0.1]))
            # 6.3. Roll
            if key_pressed == ord("q"):  # Left
                self.agent.rotate_z_local(magnum.Rad(+20 / 180 * np.pi))
            if key_pressed == ord("e"):  # Right
                self.agent.rotate_z_local(magnum.Rad(-20 / 180 * np.pi))
            # 6.4. Pitch
            if key_pressed == ord("r"):  # Up
                self.agent.rotate_x_local(magnum.Rad(+20 / 180 * np.pi))
            if key_pressed == ord("f"):  # Down
                self.agent.rotate_x_local(magnum.Rad(-20 / 180 * np.pi))
            # 6.5. Yaw
            if key_pressed == ord("a"):  # Left
                self.agent.rotate_y_local(magnum.Rad(+20 / 180 * np.pi))
            if key_pressed == ord("d"):  # Right
                self.agent.rotate_y_local(magnum.Rad(-20 / 180 * np.pi))
            # 7. Following the reference
            if key_pressed == ord("v"):
                simulation_iteration += 1
                # self.agent.transformation = self.ref_tf_global_agent[simulation_iteration]
                self.agent.transformation = tf_follow[simulation_iteration]
            if key_pressed == ord('c'):
                simulation_iteration -= 1
                self.agent.transformation = tf_follow[simulation_iteration]

            # 8. Resume the dynamic
            if key_pressed == ord(' '):
                self.agent.linear_velocity = saved_velocity[0]
                self.agent.angular_velocity = saved_velocity[1]
                simulation_iteration += 1

            if simulation_iteration == len(self.ref_tf_global_agent) - 1:
                break

            # 9-Simulating for step seconds
            self.sim.step_physics(step)
            s = self.sim.get_agent(0).state
            s.sensor_states['surface_dist'].rotation = np.quaternion(np.cos(0.25 * np.pi),
                                                                     -np.sin(0.25 * np.pi), 0, 0)
            self.sim.get_agent(0).set_state(s, False, False)
            obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone


def gen_data(scene_id: Union[str, None], flight_number: int, same_env: bool, plot_ref: bool, plot_topdown: bool):
    """
    Generate drone data in habitat with ground effect.

    param scene_id: path to a habitat scene. If None simulation.quadrotor will be used as a simulator.

    param flight_number: The seed to generate the trajectory.

    param plot_ref: Boolean to plot the reference for debugging.

    param same_env: Boolean if False use habitat sim for simulation. If True uses the quadrotor model which is the same
    as the model use for MPC. Useful for debugging.

    param plot_topdown: If true will plot topdown view when generating the trajectory.
    """
    # from src.utils.profiler import profiler
    # profiler()

    seed_to_flight = {0: 0, 1: 1}

    # Initialise drone
    drone_width, drone_height = 0.1, 0.025
    volumetric_mass = 200
    g = 9.8
    m = volumetric_mass * drone_width ** 2 * drone_height
    drone_prms = {'width': drone_width,  # [m]
                  'height': drone_height,  # [m]
                  'm': m,  # [kg]
                  'kt': m * g / (4 * 150 ** 2),  #
                  'km': m * g / (4 * 150 ** 2) * 2 ** 0.5 * drone_width / 2,  #
                  'l': 2 ** 0.5 * drone_width / 2,  # [m]
                  'ix': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iy': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
                  'iz': 1 / 3 * (drone_width ** 2 + drone_width ** 2) * m,  # [kg.m2]
                  'prop_radius': drone_width * 0.1}  # [m]
    dt = 1 / 20
    horizon_cmd = 1
    horizon_pred = 20
    gamma = 1.

    # Building the environment in habitat
    scene_path = f"{WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    camera_resolution = np.array([720, 1280])
    simulation = Simulation(scene_path=scene_path, camera_resolution=camera_resolution, drone_prms=drone_prms)
    simulation.generate_reference_procedurally(uniform_speed=1, plot_topdown=plot_topdown, dt=dt,
                                               seed=seed_to_flight[flight_number], viz_navmesh=True)

    # simulation.interactive_sim(controller=None, camera_rgb=True, camera_depth=False, step=step)

    def mpc_model(_state, _u):
        _next_state = simulation.quadrotor.forward(state=_state, motor=_u, dt=dt)
        return _next_state

    # Setting the drone at the first reference
    ref_tf_global_flu__agent_flu = (Habitat2FLU.tf_flu__habitat @ simulation.ref_tf_global_agent
                                    @ Habitat2FLU.tf_habitat__flu)
    ref_lin_vel_flu = (Habitat2FLU.rot_flu__habitat @ simulation.ref_lin_vel_global.T).T
    if plot_ref:
        quat, trans = Transform.from_matrix(ref_tf_global_flu__agent_flu).as_quat_trans()
        trace_trajectory3d(trans, quat, pause=1e-4, step=10)

    # Initializing flight in habitat
    quat0, trans0 = Transform.from_matrix(ref_tf_global_flu__agent_flu[0]).as_quat_trans()
    log = {'state': [np.concatenate([trans0, quat0, np.zeros(6)])],  # flu
           'motor': [],
           'mpc_error': [],
           'camera': [], 'depth': [], 'surface_dist': [], 'ground_force': []}  # sensors

    time_end = len(simulation.ref_tf_global_agent) - horizon_pred + 1

    # First mpc command after it take the previous one for initialisation
    hover_cmd = np.sqrt(drone_prms['m'] * g / (4 * drone_prms['kt']))
    motor_horizon = np.repeat(hover_cmd, horizon_cmd * 4).reshape((horizon_cmd, 4))
    i = 0
    for i in tqdm(range(1, time_end)):
        # Computing command with the controller
        motor_horizon, err = mpc_controller(model=mpc_model,
                                            ref_tf=ref_tf_global_flu__agent_flu[i:i + horizon_pred],
                                            ref_lin_vel=ref_lin_vel_flu[i:i + horizon_pred],
                                            state0=log['state'][-1],
                                            u0=motor_horizon,
                                            horizon_cmd=horizon_cmd,
                                            horizon_pred=horizon_pred,
                                            gamma=gamma)
        log['motor'].append(motor_horizon[0])
        log['mpc_error'].append(err)
        if err > 100:
            print()

        # Simulating result
        if not same_env:
            out = simulation.one_step_sim(dt=dt, state=Habitat2FLU.state_flu_2_state_habitat(log['state'][-1]),
                                          motor=log['motor'][-1])
            next_state_habitat, obs, has_collided = out
            log['camera'].append(obs['left_camera'])
            log['depth'].append(obs['left_depth_camera'])
            log['surface_dist'].append(obs['surface_dist'])
            log['ground_force'].append(obs['ground_force'])
            next_state = Habitat2FLU.state_habitat_2_state_flu(next_state_habitat)
            log['state'].append(next_state)
            if has_collided:
                print('Drone has collided')
                break
        else:
            next_state = simulation.quadrotor.forward(dt=dt, state=log['state'][-1], motor=log['motor'][-1])
            log['state'].append(next_state)

    for key, val in log.items():
        if len(val) != 0:
            log[key] = np.stack(val, axis=0)

    # Simulation vs reference in FLU for pose.
    position_ref_flu = Transform.from_matrix(ref_tf_global_flu__agent_flu).as_trans()
    pose_sim_flu, quat_sim_flu = log['state'][:, :3], log['state'][:, 3:7]
    trace_trajectory3d(pose_sim_flu, quat_sim_flu, pause=1e-4, step=1, position_ref=position_ref_flu[:time_end])

    # Error | MPC model vs Simulation
    state_mpc = [log['state'][0]]
    for i in range(len(log['motor'])):
        state_mpc.append(simulation.quadrotor.forward(dt=dt, motor=log['motor'][i], state=state_mpc[-1]))
    state_mpc = np.stack(state_mpc)
    position_error = ((log['state'][:, 0:3] - state_mpc[:, 0:3]) ** 2)
    quat_error = ((log['state'][:, 3:7] - state_mpc[:, 3:7]) ** 2)
    lin_vel_error = ((log['state'][:, 7:10] - state_mpc[:, 7:10]) ** 2)
    ang_vel_error = ((log['state'][:, 10:13] - state_mpc[:, 10:13]) ** 2)
    if same_env:
        assert position_error == 0 and quat_error == 0 and lin_vel_error == 0 and ang_vel_error == 0
    # MPC model FLU vs FLU reference | Does the mpc has a good model vs simulation environment ?
    trace_trajectory3d(state_mpc[:, 0:3], state_mpc[:, 3:7], 1e-4, step=1, position_ref=position_ref_flu[:time_end])

    # View camera's simulated trajectory
    tf_real = []
    for state in log['state']:
        state_habitat = Habitat2FLU.state_flu_2_state_habitat(state)
        tf_real.append(Transform.from_quat_trans(state_habitat[3:7], state_habitat[:3]).as_matrix())
    tf_real = np.stack(tf_real)
    simulation.interactive_sim(controller=None, camera_rgb=True, camera_depth=False, step=dt, tf_follow=tf_real)

    # # Motor variation
    # import matplotlib.pyplot as plt
    # time = np.linspace(0, (len(log['motor']) - 1) * 1 / 20, len(log['motor']))
    # plt.plot(time, log['motor'])
    # plt.grid()

    # Saving
    if i > 200:
        # dict_keys(['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force'])
        pickle.dump(log, open(f'{DATA_HABITAT_PATH}/{scene_id}_{flight_number}.pkl'))
        return True
    else:
        return False


def main():
    scene_id = ['17DRP5sb8fy']

    has_worked = gen_data(scene_id=MP3_SCENE_LIST[0],
                          flight_number=1,
                          same_env=False,
                          plot_ref=False,
                          plot_topdown=False)
    print()


if __name__ == "__main__":
    main()
