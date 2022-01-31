import argparse
import cv2
import habitat_sim
import magnum
import numpy as np
import pickle
import sys
from tqdm import tqdm
from typing import Dict, Tuple, Any, Union

from src.drone_models.analytical_models.quadrotor_4parms import NumpyDynamicalModel
from src.data.habitat.generate_reference import generate_reference
from src.data.habitat.mpc import mpc_controller
from src.data.habitat.hatbitat2flu import Habitat2FLU
from src.utils.paths import WD_PATH, mkdir_if_not_exist
from src.utils.trajectory import trace_trajectory3d
from src.utils.transform import Transform

DTYPE = np.float32

# NOTATION
# AGENT is the drone body in habitat convention
# AGENT_FLU is AGENT but turned from HABITAT to FLU forward left up
# GLOBAL_FLU is the earth reference frame but turned from HABITAT to FLU
# GLOBAL is the habitat earth reference frame

DATA_HABITAT_PATH = f'{WD_PATH}data/habitat/'
MP3_SCENE_LIST = pickle.load(open(f'{DATA_HABITAT_PATH}mp3_scene_list.pkl', 'rb'))
GEN_DATA_KEY = ['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force']


class Simulation:
    def __init__(self, scene_path: str,
                 camera_resolution: np.ndarray,
                 drone_prms: Dict[str, float],
                 gravity: float):
        """
        Habitat habitat of a quadrotor with ground effect.
        
        param scene_path: The path to the .glb scene to load.

        param camera_resolution: The resolution of the camera [height, width].

        param drone_prms: Parameters of the drone.

        param gravity: Value of the gravity for the habitat.
        """
        self.error_agent = None
        self.g = gravity
        self.drone_prms: Dict[str, float] = drone_prms

        # 1-Initializing the habitat and loading the scene
        backend_cfg = habitat_sim.SimulatorConfiguration()
        backend_cfg.enable_physics = True
        backend_cfg.scene_id = scene_path

        # 2-Adding a stereo RGB-D sensors to the drone
        # left rgb camera
        left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
        left_rgb_sensor.uuid = "left_camera"  # uuid is used to retrieve the observation
        left_rgb_sensor.resolution = camera_resolution
        left_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.LEFT

        # right rgb camera
        right_rgb_sensor = habitat_sim.CameraSensorSpec()
        right_rgb_sensor.uuid = "right_camera"
        right_rgb_sensor.resolution = camera_resolution
        right_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT + 1.1 * habitat_sim.geo.BACK

        # left depth camera
        left_depth_sensor = habitat_sim.CameraSensorSpec()
        left_depth_sensor.uuid = "left_depth_camera"
        left_depth_sensor.resolution = camera_resolution
        left_depth_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.LEFT
        left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # # right depth camera
        # right_depth_sensor = habitat_sim.CameraSensorSpec()
        # right_depth_sensor.uuid = "right_depth_camera"
        # right_depth_sensor.resolution = camera_resolution
        # right_depth_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT
        # right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # depth camera
        surface_distance_sensor = habitat_sim.CameraSensorSpec()
        surface_distance_sensor.uuid = "surface_dist"
        surface_distance_sensor.resolution = [1, 1]
        surface_distance_sensor.orientation = [-90 / 180 * np.pi, 0, 0]
        surface_distance_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT
        surface_distance_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # 3-Instantiating drone
        # Connecting the drone to its sensors
        agent_config = habitat_sim.AgentConfiguration()
        agent_config.sensor_specifications = [surface_distance_sensor,
                                              left_rgb_sensor,
                                              right_rgb_sensor,
                                              left_depth_sensor,
                                              # right_depth_sensor,
                                              ]

        # Error agent
        error_rgb_sensor = habitat_sim.CameraSensorSpec()
        error_rgb_sensor.uuid = "error_camera"
        error_rgb_sensor.resolution = camera_resolution
        error_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT + 1.1 * habitat_sim.geo.BACK

        dist_resolution = [1, 1]  # [1, 1]
        # Forward dist
        forward_dist = habitat_sim.CameraSensorSpec()
        forward_dist.uuid = "forward_dist"
        forward_dist.resolution = dist_resolution
        forward_dist.orientation = [0, 0, 0]
        forward_dist.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT
        forward_dist.sensor_type = habitat_sim.SensorType.DEPTH

        # Backward dist
        backward_dist = habitat_sim.CameraSensorSpec()
        backward_dist.uuid = "backward_dist"
        backward_dist.resolution = dist_resolution
        backward_dist.orientation = [0, 180 / 180 * np.pi, 0]
        backward_dist.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT
        backward_dist.sensor_type = habitat_sim.SensorType.DEPTH

        # Left dist
        left_dist = habitat_sim.CameraSensorSpec()
        left_dist.uuid = "left_dist"
        left_dist.resolution = dist_resolution
        left_dist.orientation = [0, 90 / 180 * np.pi, 0]
        left_dist.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT
        left_dist.sensor_type = habitat_sim.SensorType.DEPTH

        # Right dist
        right_dist = habitat_sim.CameraSensorSpec()
        right_dist.uuid = "right_dist"
        right_dist.resolution = dist_resolution
        right_dist.orientation = [0, -90 / 180 * np.pi, 0]
        right_dist.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT
        right_dist.sensor_type = habitat_sim.SensorType.DEPTH

        # Up dist
        up_dist = habitat_sim.CameraSensorSpec()
        up_dist.uuid = "up_dist"
        up_dist.resolution = dist_resolution
        up_dist.orientation = [90 / 180 * np.pi, 0, 0]
        up_dist.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT
        up_dist.sensor_type = habitat_sim.SensorType.DEPTH

        # depth camera
        down_dist = habitat_sim.CameraSensorSpec()
        down_dist.uuid = "down_dist"
        down_dist.resolution = dist_resolution
        down_dist.orientation = [-90 / 180 * np.pi, 0, 0]
        down_dist.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT
        down_dist.sensor_type = habitat_sim.SensorType.DEPTH

        error_agent_config = habitat_sim.AgentConfiguration()
        error_agent_config.sensor_specifications = [error_rgb_sensor, forward_dist, backward_dist, left_dist,
                                                    right_dist, up_dist, down_dist]

        # Adding the drone to the habitat
        sim = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config, error_agent_config]))
        sim.set_gravity(np.array([0, -gravity, 0]))

        # 5-Drone agent
        # Adding a rigid cube as a body to the drone
        # prim_templates_mgr = sim.get_asset_template_manager()
        self.obj_templates_mgr: habitat_sim.attributes_managers.ObjectAttributesManager
        self.obj_templates_mgr = sim.get_object_template_manager()
        self.rigid_obj_mgr: habitat_sim.physics.RigidObjectManager = sim.get_rigid_object_manager()
        # obj_templates_mgr.get_file_template_handles()

        # OPTION1: Drone has a cube shape
        cube_template_handle = self.obj_templates_mgr.get_template_handles('cubeSolid')[0]  # cubeSolid
        cube_attributes: habitat_sim.attributes.ObjectAttributes
        cube_attributes = self.obj_templates_mgr.get_template_by_handle(handle=cube_template_handle)
        cube_dim = [self.drone_prms['width'] / 2, self.drone_prms['height'] / 2, self.drone_prms['width'] / 2]
        cube_attributes.scale = cube_dim
        cube_attributes.collision_asset_size = [self.drone_prms['width'] / 4, self.drone_prms['height'] / 4,
                                                self.drone_prms['width'] / 4]
        # cube_attributes.bounding_box_collisions = True
        cube_attributes.linear_damping = 0  # drag
        cube_attributes.angular_damping = 0  # drag
        cube_attributes.friction_coefficient = 0
        cube_attributes.restitution_coefficient = 0
        self.obj_templates_mgr.register_template(cube_attributes, specified_handle='my_cube')
        obj: habitat_sim.physics.ManagedRigidObject
        obj = self.rigid_obj_mgr.add_object_by_template_handle('my_cube', sim.agents[0].scene_node, light_setup_key='0')

        # # OPTION2: Drone has a quadrotor shape
        # # The loaded drone model is (0.1, 0.02, 0.1) meters of dimension
        # tmpl_id, = obj_templates_mgr.load_configs(f'{DATA_HABITAT_PATH}drone.object_config.json')
        # uav_attr: habitat_sim.attributes.ObjectAttributes = obj_templates_mgr.get_template_by_id(tmpl_id)
        # # uav_dim = [1., 1., 1.]
        # # uav_attr.scale = uav_dim
        # # uav_attr.collision_asset_size = uav_dim
        # obj_templates_mgr.register_template(uav_attr, specified_handle='uav')
        # obj = rigid_obj_mgr.add_object_by_template_handle('uav', sim.agents[0].scene_node, light_setup_key='0')

        # Drone parameters
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        obj.mass = self.drone_prms['m']
        obj.intertia_diagonal = [self.drone_prms['ix'], self.drone_prms['iy'], self.drone_prms['iz']]
        self.agent: habitat_sim.physics.ManagedRigidObject = obj

        self.state_ref = None

        # 5-habitat initialisation
        # 5.1-Drone model init
        assert self.drone_prms['ix'] == self.drone_prms['iy']
        theta = np.array([self.drone_prms['kt'] / self.drone_prms['m'],
                          self.drone_prms['kt'] * self.drone_prms['l'] / self.drone_prms['ix'],
                          self.drone_prms['km'] / self.drone_prms['iz'],
                          self.drone_prms['iz'] / self.drone_prms['ix']], dtype=DTYPE)
        self.quadrotor = NumpyDynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=DTYPE,
                                             g=gravity)
        self.sim: habitat_sim.simulator.Simulator = sim

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
        navmesh_settings.agent_height = 0.4  # 1.5
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

        obj_handle = self.rigid_obj_mgr.get_object_handles()[0]  # my_cube_:0000
        self.rigid_obj_mgr.remove_object_by_handle(obj_handle, delete_object_node=False)

        # 6-Error agent
        # icosphereSolid_subdivs_1
        # Sphere body
        error_template_handle = self.obj_templates_mgr.get_template_handles('cubeSolid')[0]  # cubeSolid
        error_attributes: habitat_sim.attributes.ObjectAttributes
        error_attributes = self.obj_templates_mgr.get_template_by_handle(handle=error_template_handle)
        error_dim = [self.drone_prms['width'], self.drone_prms['width'] * 2, self.drone_prms['width']]
        error_attributes.scale = error_dim
        # sphere_attributes.collision_asset_size = cube_dim
        error_attributes.linear_damping = 0  # drag
        error_attributes.angular_damping = 0  # drag
        error_attributes.friction_coefficient = 0
        error_attributes.restitution_coefficient = 0
        self.obj_templates_mgr.register_template(error_attributes, specified_handle='error_body')
        error_obj: habitat_sim.physics.ManagedRigidObject
        error_obj = self.rigid_obj_mgr.add_object_by_template_handle('error_body', self.sim.agents[1].scene_node,
                                                                     light_setup_key='1')
        error_obj_handle = self.rigid_obj_mgr.get_object_handles()[0]  # error_body_:0000

        # Error agent parameters
        error_obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        error_obj.mass = self.drone_prms['m']
        error_obj.intertia_diagonal = [self.drone_prms['ix'], self.drone_prms['ix'], self.drone_prms['ix']]
        self.error_agent: habitat_sim.physics.ManagedRigidObject = error_obj

        self.state_ref = generate_reference(sim=self.sim,
                                            rigid_object=self.error_agent,
                                            uniform_speed=uniform_speed,
                                            navmesh_settings=navmesh_settings,
                                            plot_topdown=plot_topdown,
                                            flight_height=0.1,
                                            dt=dt,
                                            seed=seed,
                                            viz_navmesh=viz_navmesh)
        self.rigid_obj_mgr.remove_object_by_handle(error_obj_handle, delete_object_node=False)

        self.agent = self.rigid_obj_mgr.add_object_by_template_handle('my_cube', self.sim.agents[0].scene_node,
                                                                      light_setup_key='0')

        if type(self.state_ref) is bool:
            return False
        else:
            return True

    def one_step_sim(self, dt: float, state: np.ndarray, motor: np.ndarray,
                     z: float) -> Tuple[np.ndarray, Any, bool, np.ndarray]:
        """
        Take a drone state and a motor command at time 't_i' and return the new state and drone sensors' output at time
        't_i+1'.
        All inputs and outputs are in FLU.

        param dt: t_i+1 - t_i. The duration where all state and motor output will be considered constant (Euler
        integration scheme).

        param state: The initial state of the drone at time t_i.

        param motor: The command to apply on the drone at time t_i.

        param z: The distance of the drone to a surface below him at time t_i. Used to compute vertical ground effect
        force.

        return: next state at time t_i+1 (np.ndarray 13),
         obs a collection of sensors observations at time t_i+1 (dict[str, val]),
         if the drone has collided (bool),
         ground_effect the external perturbation (float).

        """
        # 1. Motor and Ground Effect (gravity is done by habitat)
        ground_effect = self.compute_ground_effect(motor, z, mu=1, cap=0.1)
        next_state = self.quadrotor.forward(state=state, motor=motor, dt=dt, residual=ground_effect)
        next_state_habitat = Habitat2FLU.state_flu_2_state_habitat(next_state)
        trans, quat, linear_velocity, angular_velocity = (next_state_habitat[:3], next_state_habitat[3:7],
                                                          next_state_habitat[7:10], next_state_habitat[10:])
        tf_global__agent = Transform.from_quat_trans(quat=quat, trans=trans).as_matrix()
        self.agent.transformation = tf_global__agent
        self.agent.linear_velocity = linear_velocity
        self.agent.angular_velocity = angular_velocity

        # 2-Sensing
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        has_collided = self.agent.contact_test()
        return next_state, obs, has_collided, ground_effect

    def interactive_sim(self, camera_rgb: bool, camera_depth: bool, step: float, tf_follow: np.ndarray):
        cv2.namedWindow("stereo_pair")
        # 1. Agent init
        simulation_iteration = 0
        # tf_ref = Transform.from_quat_trans(self.state_ref[:, 3:7], self.state_ref[:, :3]).as_matrix()
        self.agent.transformation = tf_follow[0]
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone

        is_running = True
        while is_running:
            # 2. Sensors
            rgb_stereo_pair = np.concatenate([obs["left_camera"],
                                              obs["right_camera"],
                                              ], axis=1)
            depth_stereo_pair = np.concatenate([obs["left_depth_camera"],
                                                # obs["right_depth_camera"]
                                                ], axis=1)
            z = obs["surface_dist"][0]
            tf_global_agent = np.array(self.agent.transformation)  # transform origin body
            rot_global_flu = Transform.from_matrix(tf_global_agent).as_rot_mat() @ Habitat2FLU.rot_hab__flu

            # Plotting sensors
            print(simulation_iteration)
            print(f'Height: {z}')
            has_collided = self.agent.contact_test()
            print(f'has_collided: {has_collided}')

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

            if simulation_iteration == len(self.state_ref) - 1:
                break

            # 9-Simulating for step seconds
            # self.sim.step_physics(step)
            s = self.sim.get_agent(0).state
            s.sensor_states['surface_dist'].rotation = np.quaternion(np.cos(0.25 * np.pi),
                                                                     -np.sin(0.25 * np.pi), 0, 0)
            self.sim.get_agent(0).set_state(s, False, False)
            obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone

    def compute_ground_effect(self, motor: np.ndarray, z: float, mu: float, cap: float) -> np.ndarray:
        ground_force = np.zeros(3)

        # ground_force[2] = np.clip((np.sum(motor) ** 2 * self.drone_prms['kt'] * 1
        #                            * (self.drone_prms['prop_radius'] / (4 * z)) ** 2), 0,
        #                           0.1 * self.drone_prms['m'] * self.g)
        if z < np.sqrt(mu) * self.drone_prms['prop_radius'] / 4:
            ground_force[2] = cap * self.drone_prms['m'] * self.g
        else:
            mu_r_z = mu * (self.drone_prms['prop_radius'] / (4 * z)) ** 2
            ground_force[2] = np.clip(np.sum(motor) ** 2 * self.drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                      a_min=0, a_max=0.1 * self.drone_prms['m'] * self.g)
        # # Debug plot
        # motor = np.array([150, 150, 150, 150])
        # z = np.linspace(1e-8, 0.1, 1000)
        # import matplotlib.pyplot as plt
        # plt.rcParams.update({"text.usetex": False,
        #                      "font.family": "serif",
        #                      "font.sans-serif": ["Computer Modern Roman"]})
        # for mu in [1, 2, 3, 4, 5]:
        #     mu_r_z = mu * (self.drone_prms['prop_radius'] / (4 * z)) ** 2
        #     to_plot = np.sum(motor) ** 2 * self.drone_prms['kt'] * mu_r_z / (1 - mu_r_z)
        #     to_plot[np.where(z < np.sqrt(mu) * self.drone_prms['prop_radius'] / 4)[0]] = 0.1 * self.drone_prms[
        #         'm'] * self.g
        #     to_plot = np.clip(to_plot, 0, 0.1 * self.drone_prms['m'] * self.g)
        #     plt.plot(z, to_plot, label=f'mu={mu}')
        # plt.title('Ground effect')
        # plt.grid()
        # plt.ylabel('Force [N]')
        # plt.xlabel('z (m)')
        # plt.legend()
        return np.concatenate([ground_force, np.zeros(3)])


def gen_data(scene_id: Union[str, None], flight_number: int, same_env: bool, plot_ref: bool, plot_topdown: bool,
             debug: bool, cluster: bool):
    """
    Generate drone data in habitat with ground effect.

    param scene_id: path to a habitat scene. If None habitat.quadrotor will be used as a simulator.

    param flight_number: The seed to generate the trajectory.

    param plot_ref: Boolean to plot the reference for debugging.

    param same_env: Boolean if False use habitat sim for habitat. If True uses the quadrotor model which is the same
    as the model use for MPC. Useful for debugging.

    param plot_topdown: If true will plot topdown view when generating the trajectory.

    param debug: Activate plot tools to debug data generation.

    param cluster: Change path if executing on cluster.
    """
    # from src.utils.profiler import profiler
    # profiler()
    # Initialise drone
    drone_width, drone_height = 0.1, 0.025
    volumetric_mass = 200
    g = 9.81
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
    if cluster:
        scene_path = f"/storage/Datasets/3D-Simulator-Datasets/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    else:
        scene_path = f"{WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    camera_resolution = np.array([480, 640])
    simulation = Simulation(scene_path=scene_path, camera_resolution=camera_resolution, drone_prms=drone_prms,
                            gravity=g)
    has_worked = simulation.generate_reference_procedurally(uniform_speed=1, plot_topdown=plot_topdown, dt=dt,
                                                            seed=flight_number, viz_navmesh=False)
    if not has_worked:
        simulation.sim.close(True)
        folder = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight/'
        mkdir_if_not_exist([folder])
        pickle.dump({'done': None}, open(f'{folder}{flight_number}_stats.pkl', 'wb'))
        return False

    # habitat.interactive_sim(controller=None, camera_rgb=True, camera_depth=False, step=step)

    def mpc_model(_state, _u):
        _next_state = simulation.quadrotor.forward(state=_state, motor=_u, dt=dt)
        return _next_state

    # state are always [trans 3, quat 4, lin_vel 3, ang_vel 3]
    # Setting the drone at the first reference
    simulation.agent.transformation = Transform.from_quat_trans(simulation.state_ref[0, 3:7],
                                                                simulation.state_ref[0, :3]).as_matrix()
    simulation.agent.linear_velocity = np.zeros(3)
    simulation.agent.angular_velocity = np.zeros(3)

    state_ref_flu = Habitat2FLU.state_habitat_2_state_flu(simulation.state_ref)
    if plot_ref:
        trace_trajectory3d(state_ref_flu[:, :3], state_ref_flu[:, 3:7], pause=1e-4, step=10)

    # Initializing flight in habitat
    # Sensing
    obs = simulation.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
    data_collected = {'state': [state_ref_flu[0]],  # flu
                      'motor': [],
                      'mpc_error': [],
                      'camera': [obs['left_camera']],
                      'depth': [obs['left_depth_camera']],
                      'surface_dist': [obs["surface_dist"][0]],
                      'ground_force': []}

    time_end = len(simulation.state_ref) - horizon_pred + 1

    # First mpc command after it take the previous one for initialisation
    hover_cmd = np.sqrt(drone_prms['m'] * g / (4 * drone_prms['kt']))
    motor_horizon = np.repeat(hover_cmd, horizon_cmd * 4).reshape((horizon_cmd, 4))
    has_collided = False
    i = 0
    for i in tqdm(range(1, time_end)):
        # Computing command with the controller
        try:
            motor_horizon, err = mpc_controller(model=mpc_model,
                                                state_ref=state_ref_flu[i:i + horizon_pred],
                                                state0=data_collected['state'][-1],
                                                u0=motor_horizon,
                                                horizon_cmd=horizon_cmd,
                                                horizon_pred=horizon_pred,
                                                gamma=gamma,
                                                verbose=False)
        except ValueError:
            print(f'Drone diverged', file=sys.stderr, flush=True)
            simulation.sim.close(True)
            folder = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight/'
            mkdir_if_not_exist([folder])
            pickle.dump({'done': None}, open(f'{folder}{flight_number}_stats.pkl', 'wb'))
            return False
        data_collected['motor'].append(motor_horizon[0])  # t to compute t+1
        data_collected['mpc_error'].append(err)  # t to compute t+1

        # Simulating result
        if not same_env:
            out = simulation.one_step_sim(dt=dt, state=data_collected['state'][-1], motor=data_collected['motor'][-1],
                                          z=data_collected['surface_dist'][-1])
            next_state_flu, obs, has_collided, ground_effect = out
            if (not np.isfinite(next_state_flu).all() or not np.isfinite(ground_effect).all() or
                    not np.isfinite(obs['surface_dist']).all()):
                print(f'Incorrect values from habitat', file=sys.stderr, flush=True)
                simulation.sim.close(True)
                folder = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight/'
                mkdir_if_not_exist([folder])
                pickle.dump({'done': None}, open(f'{folder}{flight_number}_stats.pkl', 'wb'))
                return False

            data_collected['ground_force'].append(ground_effect)  # t used to compute t+1
            # t+1
            data_collected['camera'].append(obs['left_camera'])  # t+1
            data_collected['depth'].append(obs['left_depth_camera'])  # t+1
            data_collected['surface_dist'].append(obs['surface_dist'][0])  # t+1
            data_collected['state'].append(next_state_flu)  # t+1

            if has_collided:
                print('Drone has collided', file=sys.stderr, flush=True)
                break
        else:
            next_state = simulation.quadrotor.forward(dt=dt, state=data_collected['state'][-1],
                                                      motor=data_collected['motor'][-1])
            data_collected['state'].append(next_state)
    simulation.sim.close(True)
    data_generated = {}
    for key, val in data_collected.items():
        if len(val) != 0:
            data_generated[key] = np.stack(val, axis=0)

    if debug:
        # Simulation vs reference in FLU for pose.
        pose_sim_flu, quat_sim_flu = data_generated['state'][:, :3], data_generated['state'][:, 3:7]
        # import matplotlib.pyplot as plt
        # plt.plot(state_ref_flu[:, 2])
        # plt.plot(pose_sim_flu[:, 2])
        trace_trajectory3d(pose_sim_flu, quat_sim_flu, pause=1e-4, step=1, position_ref=state_ref_flu[:time_end, :3])
        tf_ref_hab = Transform.from_quat_trans(simulation.state_ref[:, 3:7], simulation.state_ref[:, :3]).as_matrix()
        simulation.interactive_sim(camera_rgb=True, camera_depth=False, step=dt, tf_follow=tf_ref_hab)

        # Error | MPC model vs Simulation
        state_mpc = [data_generated['state'][0]]
        for i in range(len(data_generated['motor'])):
            state_mpc.append(simulation.quadrotor.forward(dt=dt, motor=data_generated['motor'][i], state=state_mpc[-1]))
        state_mpc = np.stack(state_mpc)
        position_error = ((data_generated['state'][:, 0:3] - state_mpc[:, 0:3]) ** 2)
        quat_error = ((data_generated['state'][:, 3:7] - state_mpc[:, 3:7]) ** 2)
        lin_vel_error = ((data_generated['state'][:, 7:10] - state_mpc[:, 7:10]) ** 2)
        ang_vel_error = ((data_generated['state'][:, 10:13] - state_mpc[:, 10:13]) ** 2)
        if same_env:
            assert position_error == 0 and quat_error == 0 and lin_vel_error == 0 and ang_vel_error == 0
        # MPC model FLU vs FLU reference | Does the mpc has a good model vs habitat environment ?
        trace_trajectory3d(state_mpc[:, 0:3], state_mpc[:, 3:7], 1e-4, step=1,
                           position_ref=state_ref_flu[:time_end, :3])

        # View camera's simulated trajectory
        state_habitat = Habitat2FLU.state_flu_2_state_habitat(data_generated['state'])
        tf_real = Transform.from_quat_trans(state_habitat[:, 3:7], state_habitat[:, :3]).as_matrix()
        simulation.interactive_sim(camera_rgb=True, camera_depth=False, step=dt, tf_follow=tf_real)

        # # Motor variation
        # import matplotlib.pyplot as plt
        # time = np.linspace(0, (len(log2['motor']) - 1) * 1 / 20, len(log2['motor']))
        # plt.plot(time, log2['motor'])
        # plt.grid()

        import matplotlib.pyplot as plt
        time = np.linspace(0, (len(data_generated['motor']) - 1) * 1 / 20, len(data_generated['motor']))
        plt.plot(time, data_generated['mpc_error'])
        plt.grid()

    # Saving
    # dict_keys(['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force'])
    if not debug:
        import time
        tic = time.time()
        print(f'Saving', file=sys.stderr, flush=True)
        folder = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight/{flight_number}/'
        mkdir_if_not_exist([folder])
        for name, array in data_generated.items():
            np.save(f'{folder}{name}.npy', array.astype(np.float32))
        pickle.dump({'done': i + 1, 'length': time_end, 'has_collided': has_collided}, open(f'{folder}stats.pkl', 'wb'))
        print(f'Progress: {i + 1} | Max_length: {time_end} | has_collided: {has_collided}', file=sys.stderr, flush=True)
        print(f'Time without compression: {time.time() - tic}s', file=sys.stderr, flush=True)
        return True


def main():
    parser = argparse.ArgumentParser(description="Generate drone flight data.")
    parser.add_argument('--scene_start', default=0, type=int, help='Scene number range start [0, 89]')
    parser.add_argument('--scene_stop', default=1, type=int, help='Scene number range stop [1, 90]')
    # fn = flight_number correspond to a specific trajectory.
    parser.add_argument('--fn_start', default=0, type=int, help='Flight number range start')
    parser.add_argument('--fn_stop', default=1, type=int, help='Flight number range stop')
    parser.add_argument('--cluster', default=False, type=bool, help='Execute on cluster.')
    args = parser.parse_args()

    for scene_number in range(args.scene_start, args.scene_stop):
        for flight_number in range(args.fn_start, args.fn_stop):
            print(f'SCENE_NUMBER={scene_number} | FLIGHT_NUMBER={flight_number}', file=sys.stderr, flush=True)
            has_worked = gen_data(scene_id=MP3_SCENE_LIST[scene_number],
                                  flight_number=flight_number,
                                  same_env=False,
                                  plot_ref=False,
                                  plot_topdown=False,
                                  debug=False,
                                  cluster=args.cluster)
            print(f'\n' * 20, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
