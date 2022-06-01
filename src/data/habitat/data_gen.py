import argparse
import os
import pickle
import sys
from typing import Dict, Tuple, Any

import cv2
import habitat_sim
import magnum
import numpy as np
from tqdm import tqdm

import src.data.habitat.loading as habitat
from src.data.habitat.generate_reference import generate_reference
from src.data.habitat.hatbitat2flu import Habitat2FLU
from src.data.habitat.mpc import mpc_controller
from src.drone_model.analytical_model.quadrotor_4parms import DynamicalModel
from src.utils.path import WD_PATH, mkdir
from src.utils.trajectory import trace_trajectory3d
from src.utils.transform import Transform as Tf

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


class Simulation:
    def __init__(self, scene_path: str,
                 camera_resolution: np.ndarray,
                 drone_prms: Dict[str, float],
                 gravity: float):
        """
        Habitat simulation of a quadrotor with ground effect.
        
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

        # # right rgb camera
        # right_rgb_sensor = habitat_sim.CameraSensorSpec()
        # right_rgb_sensor.uuid = "right_camera"
        # right_rgb_sensor.resolution = camera_resolution
        # right_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0. * habitat_sim.geo.RIGHT + 1.1 * habitat_sim.geo.BACK

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
                                              # right_rgb_sensor,
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
        self.quadrotor = DynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=DTYPE,
                                        g=self.g)
        self.sim: habitat_sim.simulator.Simulator = sim

        self.optimal_navmesh = {'agent_height': 0.4,  # 1.5
                                'agent_max_climb': 2.0,  # 0.2
                                'agent_max_slope': 90.0,  # 45.0
                                'agent_radius': 0.2,  # 0.1
                                'cell_height': 0.01,  # 0.2
                                'cell_size': 0.01,  # 0.05
                                'detail_sample_dist': 6.0,  # 6.0
                                'detail_sample_max_error': 1.,  # 1.0
                                'edge_max_error': 1.3,  # 1.3    0.01
                                'edge_max_len': 12.0,  # 12.0
                                'filter_ledge_spans': True,  # True,
                                'filter_low_hanging_obstacles': True,  # True,
                                'filter_walkable_low_height_spans': True,  # True,
                                'region_merge_size': 20.0,  # 20.0
                                'region_min_size': 20.0,  # 20.0
                                'verts_per_poly': 6.0}  # 6.0

    def generate_reference(self, uniform_speed: float,
                           manual_ref: bool,
                           dt: float,
                           seed: int,
                           viz_navmesh: bool,
                           folder: str,
                           navmesh_path: str):
        """
        Generate a reference trajectory for the drone.

        Args:
            uniform_speed: The number of step to cut the trajectory.
            manual_ref: If true will generate inital point manually else randomly.
            dt: dt to use for position derivation.
            seed: Seed of trajectory generation.
            viz_navmesh: If true rgb camera see the navmesh wireframe.
            folder: Folder to save / load the reference.

        Returns:
            Update self.state_ref with the generated reference.
        """
        # --- Defining a quality navmesh
        self.sim.seed(seed)
        # Recomputing the navmesh
        self.sim.navmesh_visualization = viz_navmesh

        if not manual_ref:
            navmesh_settings: habitat_sim.nav.NavMeshSettings = habitat_sim.NavMeshSettings()
            for key, val in self.optimal_navmesh.items():
                setattr(navmesh_settings, key, val)
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings=navmesh_settings)
            initial_point = np.stack([self.sim.pathfinder.get_random_navigable_point() for _ in range(10)])
            flight_height = 0.1
            navmesh_dict = self.optimal_navmesh
        else:
            # reload existing navmesh, set up my agent height to 3.12 meters
            # self.sim.pathfinder.load_nav_mesh(navmesh_path)
            navmesh_settings: habitat_sim.nav.NavMeshSettings = habitat_sim.NavMeshSettings()
            # for key, val in self.optimal_navmesh.items():
            #     setattr(navmesh_settings, key, val)
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings=navmesh_settings)
            # loop over list attributes of navmesh_settings and create a dict
            navmesh_dict = {k: getattr(navmesh_settings, k) for k in dir(navmesh_settings) if not k.startswith('_')}
            del navmesh_dict['set_defaults']

            try:
                initial_point = pickle.load(open(f'{folder}initial_point_ref.pkl', 'rb'))
            except FileNotFoundError:
                initial_point = self.interactive_sim(True, True, None, True)
                pickle.dump(initial_point, open(f'{folder}initial_point_ref.pkl', 'wb'))
            initial_point = Tf.from_matrix(initial_point).as_trans()
            flight_height = 0.

        # --- Defining an agent to check for errors
        # Habitat does not detect every collision, but filters a part of it.
        # Remove drone's agent body, called my_cube_:0000
        obj_handle = self.rigid_obj_mgr.get_object_handles()[0]  # my_cube_:0000
        self.rigid_obj_mgr.remove_object_by_handle(obj_handle, delete_object_node=False)
        # Cube body
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

        # Convert initially generated point to state reference
        self.state_ref = generate_reference(sim=self.sim,
                                            initial_point=initial_point,
                                            rigid_object=self.error_agent,
                                            uniform_speed=uniform_speed,
                                            navmesh_settings_dict=navmesh_dict,
                                            flight_height=flight_height,
                                            dt=dt,
                                            db_process_initial_point=manual_ref)
        # self.state_ref = np.concatenate([self.state_ref, np.zeros((len(self.state_ref), 6))], axis=-1) add acc
        self.rigid_obj_mgr.remove_object_by_handle(error_obj_handle, delete_object_node=False)

        # Redefining drone's agent body
        self.agent = self.rigid_obj_mgr.add_object_by_template_handle('my_cube', self.sim.agents[0].scene_node,
                                                                      light_setup_key='0')

        if type(self.state_ref) is bool:
            return False
        else:
            return True

    def interactive_sim(self, camera_rgb: bool, camera_depth: bool, tf_follow: np.ndarray = None,
                        return_point: bool = False,
                        return_label: bool = False):
        cv2.namedWindow("MP3D Scene")
        # 1. Agent init
        iteration = 0
        if tf_follow is not None:
            # tf_ref = Tf.from_quat_trans(self.state_ref[:, 3:7], self.state_ref[:, :3]).as_matrix()
            self.agent.transformation = tf_follow[0]
        if return_point:
            point = []
        if return_label and tf_follow is not None:
            last_iteration = 0
            label = np.zeros(len(tf_follow)) * np.nan


        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        inc = 0.1  # increment for the translation
        spd = 1  # speed for tf_follow
        z_vector = np.array([obs['surface_dist'][0]] * 10, dtype=float).reshape((-1, 1))
        forward = True

        is_running = True
        while is_running:
            # 2. Sensors
            rgb_stereo_pair = np.concatenate([obs["left_camera"],
                                              # obs["right_camera"]
                                              ], axis=1)[..., :3]
            depth_stereo_pair = np.concatenate([obs["left_depth_camera"],
                                                # obs["right_depth_camera"]
                                                ], axis=1)
            z = obs["surface_dist"][0]
            if forward:
                z_vector[:-1] = z_vector[1:]
                z_vector[-1] = z
            else:
                z_vector[1:] = z_vector[:-1]
                z_vector[0] = z
            tf_global_agent = np.array(self.agent.transformation)  # transform origin body
            rot_global_flu = Tf.from_matrix(tf_global_agent).as_rot_mat() @ Habitat2FLU.rot_hab__flu

            has_collided = self.agent.contact_test()
            # Printing and plotting sensors
            # print(iteration)
            # print(f'Height: {z}')
            # print(f'has_collided: {has_collided}')
            # print(f'\n\n')

            # 4-Display and interactivity
            if camera_rgb:
                cv2.imshow("rgb", rgb_stereo_pair[..., ::-1])
            if camera_depth:
                max_depth = 5
                depth_stereo_pair = np.clip(depth_stereo_pair, 0, max_depth)
                depth_stereo_pair /= max_depth
                depth_stereo_pair = np.uint8(depth_stereo_pair * 255)
                mask = [depth_stereo_pair == 0]
                depth_stereo_pair = cv2.cvtColor(depth_stereo_pair, cv2.COLOR_GRAY2RGB)
                depth_stereo_pair[mask] = (0, 0, 255)
                cv2.imshow("depth", depth_stereo_pair)
                cv2.imshow('z', z_vector)
            key_pressed = cv2.waitKey()

            # 5-Immobilizing the drone
            saved_velocity = (self.agent.linear_velocity, self.agent.angular_velocity)
            self.agent.linear_velocity = magnum.Vector3([0, 0, 0])
            self.agent.angular_velocity = magnum.Vector3([0, 0, 0])
            # print(f"linear_velocity velocity: {self.agent.linear_velocity}")
            # print(f"angular_velocity velocity: {self.agent.angular_velocity}\n\n")

            # print(f"angular_velocity velocity: {rot_global_flu @ np.array([+0, 0, 1])}\n\n")

            # 6-Exploring the scene
            # 6.1. Forward, Backward
            if key_pressed == ord("w"):  # Front
                self.agent.translate(rot_global_flu @ np.array([+inc, 0, 0]))
            if key_pressed == ord("s"):  # Back
                self.agent.translate(rot_global_flu @ np.array([-inc, 0, 0]))
            # 6.2 Thrust
            if key_pressed == ord("x"):  # Up
                self.agent.translate(rot_global_flu @ np.array([0, 0, +inc]))
            if key_pressed == ord("z"):  # Down
                self.agent.translate(rot_global_flu @ np.array([0, 0, -inc]))
            if key_pressed == ord("1"):  # Strafe left
                self.agent.translate(rot_global_flu @ np.array([0, +inc, 0]))
            if key_pressed == ord("2"):  # Strafe right
                self.agent.translate(rot_global_flu @ np.array([0, -inc, 0]))
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
            # try:
            if key_pressed == ord("v"):
                iteration = min(len(tf_follow) - 1, iteration + spd)
                self.agent.transformation = tf_follow[iteration]
                forward = True
            if key_pressed == ord('c'):
                iteration = max(0, iteration - spd)
                last_iteration = min(last_iteration, iteration)
                self.agent.transformation = tf_follow[iteration]
                forward = False
            if key_pressed == ord('7'):  # no label
                label[last_iteration:iteration + 1] = 0
                print(f'Label 0 for [{last_iteration}, {iteration}]')
                last_iteration = iteration
            if key_pressed == ord('8'):  # label drone_through_object
                label[last_iteration:iteration + 1] = 1
                print(f'Label 1 for [{last_iteration}, {iteration}]')
                last_iteration = iteration
            if key_pressed == ord('9'):  # label surface_discontinuity
                label[last_iteration:iteration + 1] = 2
                print(f'Label 2 for [{last_iteration}, {iteration}]')
                last_iteration = iteration
            if key_pressed == ord('0'):  # label surface_not_smooth
                label[last_iteration:iteration + 1] = 3
                print(f'Label 3 for [{last_iteration}, {iteration}]')
                last_iteration = iteration
            # except TypeError:
            #     print(f'No tf_follow')
            if key_pressed == ord('['):
                inc = max(inc - 0.1, 0.1)
                print(f'Now inc is {inc}')
            if key_pressed == ord(']'):
                inc = min(inc + 0.1, 2)
                print(f'Now inc is {inc}')

            if key_pressed == ord("u"):
                spd = max(spd - 1, 1)
                print(f'Now spd is {spd}')
            if key_pressed == ord("i"):
                spd = min(spd + 1, 10)
                print(f'Now spd is {spd}')

            if key_pressed == ord('n'):
                print(f'Last point will be deleted:')
                print(point)
            if key_pressed == ord('m'):
                tf = self.agent.transformation
                print(f'Point saved: {tf}')
                point.append(tf)
            if key_pressed == ord('l'):
                is_running = False

            # 9-Simulating for step seconds
            # self.sim.step_physics(step)
            s = self.sim.get_agent(0).state
            s.sensor_states['surface_dist'].rotation = np.quaternion(np.cos(0.25 * np.pi), -np.sin(0.25 * np.pi), 0, 0)
            self.sim.get_agent(0).set_state(s, False, False)
            obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        if return_label:
            assert not np.isnan(label).any()
            return label
        if return_point:
            return np.stack(point, axis=0)

    def one_step_sim(self, dt: float, state: np.ndarray, motor: np.ndarray,
                     z: float) -> Tuple[np.ndarray, Any, bool, np.ndarray]:
        """
        Take a drone state and a motor command at time t_i and return the new state and drone sensors' output at time
        t_i+1.
        All inputs and outputs are in FLU.

        Args:
            dt: t_i+1 - t_i. The duration where all state and motor output will be considered constant (Trapezoidal
                integration scheme).
            state: The initial state of the drone at time t_i.
            motor: The command to apply on the drone at time t_i to compute t_i+1.
            z: The distance of the drone to a surface below him at time t_i. Used to compute vertical ground effect
               force.

        Returns:
            next state at time t_i+1 (np.ndarray 19)
            obs a collection of sensors observations at time t_i+1 (dict[str, val])
            if the drone has collided (bool)
            ground_effect the external perturbation (float)
        """
        # 1. Motor and Ground Effect (gravity is done by habitat)
        # new fa mu=20, coef=10
        ground_effect = self.compute_ground_effect(motor, z, mu=1, cap=0.1, coef=1)
        quat = state[3:7]
        rot_body_fixed = Tf.from_quat_trans(quat, np.zeros(3)).as_rot_mat().T
        fa_body = rot_body_fixed @ ground_effect[:3]
        ta_body = rot_body_fixed @ ground_effect[3:]
        residual = np.concatenate([fa_body, ta_body])
        next_state = self.quadrotor.forward(state=state, motor=motor, dt=dt, residual=residual)
        next_state_hab = Habitat2FLU.state_flu_2_state_habitat(next_state)
        trans, quat, linear_velocity, angular_velocity = (next_state_hab[:3], next_state_hab[3:7],
                                                          next_state_hab[7:10], next_state_hab[10:13])
        tf_global__agent = Tf.from_quat_trans(quat=quat, trans=trans).as_matrix()
        self.agent.transformation = tf_global__agent
        self.agent.linear_velocity = linear_velocity
        self.agent.angular_velocity = angular_velocity

        # 2-Sensing
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        has_collided = self.agent.contact_test()
        return next_state, obs, has_collided, ground_effect

    def compute_ground_effect(self, motor: np.ndarray, z: float, mu: float, cap: float, coef: float) -> np.ndarray:
        ground_force = np.zeros(3)
        if z <= np.sqrt(mu) * self.drone_prms['prop_radius'] / 4:
            ground_force[2] = cap * self.drone_prms['m'] * self.g
        else:
            mu_r_z = mu * (self.drone_prms['prop_radius'] / (4 * z)) ** 2
            ground_force[2] = np.clip(coef * np.sum(motor ** 2) * self.drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                      a_min=0, a_max=cap * self.drone_prms['m'] * self.g)
        return np.concatenate([ground_force, np.zeros(3)])


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
