import cv2
import habitat_sim
import magnum
import numpy as np
import pickle
from tqdm import tqdm
import torch
from typing import Dict, Tuple, Any

from src.drone_models.analytical_models.quadrotor_4parms import DynamicalModel
from src.simulation.generate_reference import generate_reference
from src.simulation.mpc import mpc_controller
from src.utils.paths import WD_PATH, DATA_HABITAT_PATH
from src.utils.profiler import profiler
from src.utils.trajectory import trace_trajectory3d
from src.utils.transform import Transform

cv2.namedWindow("stereo_pair")

# habitat agent and global reference frame are oriented with x-left, y-up and z-backward.
TF_HABITAT__HABITAT_FLU = Transform.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
                                                     trans=np.zeros(3)).as_matrix()
TF_HABITAT_FLU__HABITAT = Transform.from_matrix(TF_HABITAT__HABITAT_FLU).inv().as_matrix()

DTYPE = torch.float64


# NOTATION
# AGENT is the drone body in habitat convention
# AGENT_FLU is AGENT but turned from HABITAT to FLU forward left up
# GLOBAL_FLU is the earth reference frame but turned from HABITAT to FLU
# GLOBAL is the habitat earth reference frame


class Simulation:
    def __init__(self, scene_path: str,
                 camera_resolution: np.ndarray,
                 drone_prms: Dict[str, float],
                 allow_gravity: bool):
        """
        Habitat simulation of a quadrotor with ground effect.
        
        :param scene_path: The path to the .glb scene to load.

        :param camera_resolution: The resolution of the camera [height, width].

        :param drone_prms: Parameters of the drone.

        :param allow_gravity: Boolean to allow gravity or not.
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
        left_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.LEFT

        # 2.2-right rgb camera
        right_rgb_sensor = habitat_sim.CameraSensorSpec()
        right_rgb_sensor.uuid = "right_camera"
        right_rgb_sensor.resolution = camera_resolution
        # TODO place right camera
        right_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0.025 * habitat_sim.geo.RIGHT + 1.1 * habitat_sim.geo.BACK

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
        right_depth_sensor.position = (0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT)
        right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # 2.5-depth camera
        surface_distance_sensor = habitat_sim.CameraSensorSpec()
        surface_distance_sensor.uuid = "surface_dist"
        surface_distance_sensor.resolution = [1, 1]
        surface_distance_sensor.orientation = [-90 / 180 * np.pi, 0, 0]
        surface_distance_sensor.position = (0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT)
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
        cube_template_handle = obj_templates_mgr.get_template_handles('cubeSolid')[0]

        # Editing cube properties
        cube_attributes: habitat_sim.attributes.ObjectAttributes
        cube_attributes = obj_templates_mgr.get_template_by_handle(handle=cube_template_handle)
        drone_dim = [drone_prms['width'], drone_prms['height'], drone_prms['width']]
        cube_attributes.scale = drone_dim
        cube_attributes.collision_asset_size = drone_dim
        obj_templates_mgr.register_template(cube_attributes, specified_handle='my_cube')

        cube_obj: habitat_sim.physics.ManagedRigidObject
        cube_obj = rigid_obj_mgr.add_object_by_template_handle('my_cube', sim.agents[0].scene_node, light_setup_key='0')

        # 4-Drone parameters
        cube_obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        if allow_gravity:
            cube_obj.mass = drone_prms['m']
        cube_obj.intertia_diagonal = [drone_prms['ix'], drone_prms['iy'], drone_prms['iz']]
        self.sim: habitat_sim.simulator.Simulator = sim
        self.agent: habitat_sim.physics.ManagedRigidObject = cube_obj
        self.drone_prms: Dict[str, float] = drone_prms
        self.ref_tf_global_agent = None
        self.ref_lin_vel_global = None

        # 5-simulation initialisation
        # 5.1-Drone model init
        assert self.drone_prms['ix'] == self.drone_prms['iy']
        theta = torch.FloatTensor([self.drone_prms['kt'] / self.drone_prms['m'],
                                   self.drone_prms['kt'] * self.drone_prms['l'] / self.drone_prms['ix'],
                                   self.drone_prms['km'] / self.drone_prms['iz'],
                                   self.drone_prms['iz'] / self.drone_prms['ix']])
        self.quadrotor = DynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=DTYPE)

    def generate_reference_procedurally(self, uniform_speed: float, plot_topdown: bool, dt: float):
        """

        :param uniform_speed: The number of step to cut the trajectory.

        :param plot_topdown: If true will plot topdown view.

        :param dt: dt to use for position derivation.
        """
        # Defining the drone's volume
        navmesh_settings: habitat_sim.nav.NavMeshSettings = habitat_sim.NavMeshSettings()
        navmesh_settings.agent_height = 0.2  # 1.5
        navmesh_settings.agent_max_climb = 2.0  # 0.2
        navmesh_settings.agent_max_slope = 88.0  # 45.0
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
                                                                               seed=0)

    def one_step_sim(self, step: float, state: np.ndarray, motor: np.ndarray) -> Tuple[np.ndarray, Any, bool]:
        assert state.shape == (13,)
        assert motor.shape == (4,)
        # 1. Initializing agent
        # quat: x, y, z, w
        trans, quat, linear_velocity, angular_velocity = state[:3], state[3:7], state[7:10], state[10:]
        tf_global__agent = Transform.from_quat_trans(quat=quat, trans=trans).as_matrix()
        self.agent.transformation = tf_global__agent
        self.agent.linear_velocity = linear_velocity
        self.agent.angular_velocity = angular_velocity

        rot_global__agent_flu = Transform.from_matrix(tf_global__agent @ TF_HABITAT__HABITAT_FLU).as_rot_mat()

        # 2. Sensing
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        z = obs["surface_dist"][0]

        # 3. Motor and Ground Effect (gravity is done by habitat)
        motor_torsor = self.quadrotor.motor_dynamic(motor=torch.tensor(motor, dtype=DTYPE)).numpy()
        motor_torques = np.array([self.drone_prms['kt'] * self.drone_prms['l'],
                                  self.drone_prms['kt'] * self.drone_prms['l'],
                                  self.drone_prms['km']]) * motor_torsor[:3]

        motor_forces = np.zeros_like(motor_torques)
        motor_forces[2] = self.drone_prms['kt'] * motor_torsor[3]

        ground_forces = np.zeros_like(motor_torques)
        ground_forces[1] = (np.sum(motor) ** 2 * self.drone_prms['kt'] * 1
                            * (self.drone_prms['prop_radius'] / (4 * z)) ** 2)

        # gravity = magnum.Vector3([0, -9.8 * self.drone_prms['m'], 0])

        self.agent.apply_force(rot_global__agent_flu @ motor_forces + ground_forces,
                               relative_position=np.zeros(3))
        self.agent.apply_torque(rot_global__agent_flu @ motor_torques)

        # 4-Simulating for step seconds
        self.sim.step_physics(step)
        quat, trans = Transform.from_matrix(np.array(self.agent.transformation)).as_quat_trans()
        next_state = np.concatenate([trans, quat, self.agent.linear_velocity, self.agent.angular_velocity], axis=-1)
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
        has_collided = self.agent.contact_test()
        return next_state, obs, has_collided

    def interactive_sim(self, controller, camera_rgb: bool, camera_depth: bool, step: float, tf_follow):
        # 1. Agent init
        rot_agent_flu = Transform.from_matrix(TF_HABITAT__HABITAT_FLU).as_rot_mat()
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
            # print(f'self.agent.transformation {self.agent.transformation}')
            rot_global_drone = Transform.from_matrix(tf_global_agent).as_rot_mat()
            rot_global_flu = rot_global_drone @ rot_agent_flu
            if controller is not None:
                motor = controller
            else:
                motor = np.zeros((1000, 4))

            # Plotting sensors
            print(simulation_iteration)
            print(f'Height: {z}')

            # 3-Motor and Ground Effect (gravity is done by habitat)
            motor_torsor = self.quadrotor.motor_dynamic(motor=torch.tensor(motor[simulation_iteration],
                                                                           dtype=DTYPE)).numpy()
            motor_torques = np.array([self.quadrotor.theta[1],
                                      self.quadrotor.theta[1],
                                      self.quadrotor.theta[2]]) * motor_torsor[:3]

            motor_forces = np.zeros_like(motor_torques)
            motor_forces[2] = self.quadrotor.theta[0] * motor_torsor[3]

            ground_forces = np.zeros_like(motor_torques)
            ground_forces[1] = (np.sum(motor[simulation_iteration]) ** 2 * self.drone_prms['kt'] * 1
                                * (self.drone_prms['prop_radius'] / (4 * z)) ** 2)

            self.agent.apply_force(rot_global_flu @ motor_forces + ground_forces,
                                   relative_position=magnum.Vector3([0, 0, 0]))
            self.agent.apply_torque(rot_global_flu @ motor_torques)
            # self.agent.apply_force(magnum.Vector3([0, self.drone_prms['m']*9.81, 0]), relative_position=magnum.Vector3([0, 0, 0]))
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


def gen_data(scene_id, flight_number, plot_ref: bool):
    # profiler()
    # Building the environment in habitat
    scene_path = f"{WD_PATH}data/habitat/scene_datasets/mp3d/{scene_id}/{scene_id}.glb"
    # scene_path = f"{WD_PATH}data/habitat/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    camera_resolution = np.array([720, 1280])
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
    horizon_cmd = 5
    horizon_pred = 40
    gamma = 0.95

    # Building the simulation
    simulation = Simulation(scene_path=scene_path, camera_resolution=camera_resolution, drone_prms=drone_prms,
                            allow_gravity=True)
    simulation.generate_reference_procedurally(uniform_speed=2, plot_topdown=False, dt=dt)

    # simulation.interactive_sim(controller=None, camera_rgb=True, camera_depth=False, step=step)

    def mpc_model(_state, _u):
        _next_state = simulation.quadrotor.forward(state=torch.tensor(_state),
                                                   motor=torch.tensor(_u),
                                                   dt=torch.tensor([dt])).numpy()
        return _next_state

    # Setting the drone at the first reference
    ref_tf_global_flu__agent_flu = TF_HABITAT_FLU__HABITAT @ simulation.ref_tf_global_agent @ TF_HABITAT__HABITAT_FLU
    rot_habitat_flu__habitat = Transform.from_matrix(TF_HABITAT_FLU__HABITAT).as_rot_mat()
    ref_lin_vel_flu = (rot_habitat_flu__habitat @ simulation.ref_lin_vel_global.T).T
    if plot_ref:
        quat, trans = Transform.from_matrix(ref_tf_global_flu__agent_flu).as_quat_trans()
        trace_trajectory3d(trans, quat, pause=1e-2, step=1)

    # Initializing flight in habitat
    quat0, trans0 = Transform.from_matrix(simulation.ref_tf_global_agent[0]).as_quat_trans()
    state = [np.concatenate([trans0, quat0, np.zeros(6)])]
    hover_cmd = np.sqrt(drone_prms['m'] * g / (4 * drone_prms['kt']))
    motor_horizon = np.repeat(hover_cmd, horizon_cmd * 4).reshape((horizon_cmd, 4))
    motor = []
    mpc_error = []
    time_end = len(simulation.ref_tf_global_agent) - 1
    has_collided = False
    obs_tab = {'camera': [], 'depth': [], 'surface_dist': [], }
    for i in tqdm(range(1, time_end)):
        motor_horizon, err = mpc_controller(model=mpc_model,
                                            ref_tf=ref_tf_global_flu__agent_flu[i:i + horizon_pred],
                                            ref_lin_vel=ref_lin_vel_flu[i:i + horizon_pred],
                                            state0=state_habitat_2_state_mpc(state[-1]),
                                            u0=motor_horizon,
                                            horizon_cmd=horizon_cmd,
                                            horizon_pred=horizon_pred,
                                            gamma=gamma)
        motor.append(motor_horizon[0])
        mpc_error.append(err)
        if err > 100:
            print()
        next_state, obs, has_collided = simulation.one_step_sim(step=dt, state=state[-1], motor=motor[-1])
        if has_collided:
            break
        obs_tab['camera'].append(obs['left_camera'])
        obs_tab['depth'].append(obs['left_depth_camera'])
        obs_tab['surface_dist'].append(obs['surface_dist'])

        state.append(next_state)
    state_tab, motor_tab, mpc_error_tab = (np.stack(e) for e in (state, motor, mpc_error))
    position_ref = Transform.from_matrix(simulation.ref_tf_global_agent).as_trans()
    position_real = np.stack(state_tab[:, :3])
    quat_real = np.stack(state_tab[:, 3:7])

    # DEBUG
    state_flu = np.zeros_like(state_tab)
    for i, e in enumerate(state_tab):
        state_flu[i] = state_habitat_2_state_mpc(e)
    pose_real_flu = np.stack(state_flu[:, :3])
    quat_real_flu = np.stack(state_flu[:, 3:7])
    position_ref = Transform.from_matrix(ref_tf_global_flu__agent_flu).as_trans()
    tf_real = Transform.from_quat_trans(quat_real, position_real).as_matrix()

    trace_trajectory3d(pose_real_flu, quat_real_flu, pause=1e-4, step=1, position_ref=position_ref[:time_end])
    simulation.interactive_sim(controller=None, camera_rgb=True, camera_depth=False, step=dt, tf_follow=tf_real)
    pickle.dump({'motor': motor_tab,
                 'state': state_tab,
                 'ref': ref_tf_global_flu__agent_flu[:len(state_tab)],
                 'camera': obs_tab['camera'],
                 'depth': obs_tab['depth'],
                 'surface_dist': obs_tab['surface_dist']}, open(f'{DATA_HABITAT_PATH}/{scene_id}_{flight_number}.pkl'))


def state_habitat_2_state_mpc(state_habitat):
    trans, quat, linear_velocity, angular_velocity = (state_habitat[:3], state_habitat[3:7],
                                                      state_habitat[7:10], state_habitat[10:])
    tf_global_agent = Transform.from_quat_trans(quat, trans).as_matrix()
    tf_global_flu__agent_flu = TF_HABITAT_FLU__HABITAT @ tf_global_agent @ TF_HABITAT__HABITAT_FLU
    quat_flu_flu, trans_flu_flu = Transform.from_matrix(tf_global_flu__agent_flu).as_quat_trans()

    rot_habitat_flu__habitat = Transform.from_matrix(TF_HABITAT_FLU__HABITAT).as_rot_mat()
    linear_velocity__flu = rot_habitat_flu__habitat @ linear_velocity  # linear velocity in global_flu

    rot_agent_flu__global = Transform.from_matrix(tf_global_agent @ TF_HABITAT__HABITAT_FLU).as_rot_mat().T
    angular_velocity__flu = rot_agent_flu__global @ angular_velocity  # angular velocity in agent_flu

    state_flu = np.concatenate([trans_flu_flu, quat_flu_flu, linear_velocity__flu,
                                angular_velocity__flu], axis=-1)
    return state_flu


def main():
    scene_id = ['17DRP5sb8fy']
    gen_data(scene_id=scene_id[0], flight_number=0, plot_ref=False)


if __name__ == "__main__":
    main()
