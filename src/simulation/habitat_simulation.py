import cv2
import habitat_sim
import magnum
import numpy as np
import torch
from typing import Dict

from src.drone_models.analytical_models.quadrotor_4parms import DynamicalModel
from src.utils.transform import Transform
from src.utils.paths import WD_PATH
from src.simulation.generate_reference import generate_reference
from src.simulation.mpc import mpc

cv2.namedWindow("stereo_pair")

# habitat agent and global reference frame are oriented with x-left, y-up and z-backward.
TF_AGENT_FLU = Transform.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
                                          trans=np.zeros(3)).as_matrix()
TF_FLU_AGENT = Transform.from_matrix(TF_AGENT_FLU).inv().as_matrix()


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
        right_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT + 1.1 * habitat_sim.geo.BACK

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
        surface_distance_sensor.orientation = np.array([-90 / 180 * np.pi, 0, 0], dtype=np.float32)
        surface_distance_sensor.position = (0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT)
        surface_distance_sensor.sensor_type = habitat_sim.SensorType.DEPTH

        # 3-Instantiating drone
        # 3.1-Connecting the drone to its sensors
        agent_config = habitat_sim.AgentConfiguration()
        agent_config.sensor_specifications = [surface_distance_sensor, left_rgb_sensor, right_rgb_sensor,
                                              left_depth_sensor,
                                              right_depth_sensor]
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
        cube_attributes.scale = magnum.Vector3([0.05, 0.025, 0.05])
        cube_attributes.collision_asset_size = magnum.Vector3([0.05, 0.025, 0.05])
        obj_templates_mgr.register_template(cube_attributes, specified_handle='my_cube')

        cube_obj: habitat_sim.physics.ManagedRigidObject
        cube_obj = rigid_obj_mgr.add_object_by_template_handle('my_cube', sim.agents[0].scene_node, light_setup_key='1')

        # 4-Drone parameters
        cube_obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        if allow_gravity:
            cube_obj.mass = drone_prms['m']
        cube_obj.intertia_diagonal = [drone_prms['ix'], drone_prms['iy'], drone_prms['iz']]
        self.sim: habitat_sim.simulator.Simulator = sim
        self.agent: habitat_sim.physics.ManagedRigidObject = cube_obj
        self.drone_prms: Dict[str, float] = drone_prms
        self.ref_tf_global_agent = None

        # 5-simulation initialisation
        # 5.1-Drone model init
        assert self.drone_prms['ix'] == self.drone_prms['iy']
        theta = torch.FloatTensor([self.drone_prms['kt'] / self.drone_prms['m'],
                                   self.drone_prms['kt'] * self.drone_prms['l'] / self.drone_prms['ix'],
                                   self.drone_prms['km'] / self.drone_prms['iz'],
                                   self.drone_prms['iz'] / self.drone_prms['ix']])
        self.quadrotor = DynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=torch.float32)

    def generate_reference_procedurally(self, uniform_step_number: int, plot_topdown: bool):
        """

        :param uniform_step_number: The number of step to cut the trajectory.

        :param plot_topdown: If true will plot topdown view.
        """
        # Defining the drone's volume
        navmesh_settings: habitat_sim.nav.NavMeshSettings = habitat_sim.NavMeshSettings()
        navmesh_settings.agent_height = 0.2  # 1.5
        navmesh_settings.agent_max_climb = 2.0  # 0.2
        navmesh_settings.agent_max_slope = 88.0  # 45.0
        navmesh_settings.agent_radius = 0.2  # 0.1
        navmesh_settings.cell_height = 0.05  # 0.2
        navmesh_settings.cell_size = 0.08  # 0.05
        navmesh_settings.detail_sample_dist = 6.0  # 6.0
        navmesh_settings.detail_sample_max_error = 0.01  # 1.0
        navmesh_settings.edge_max_error = 0.01  # 1.3
        navmesh_settings.edge_max_len = 12.0  # 12.0
        navmesh_settings.filter_ledge_spans = True  # True
        navmesh_settings.filter_low_hanging_obstacles = False  # True
        navmesh_settings.filter_walkable_low_height_spans = True  # True
        navmesh_settings.region_merge_size = 20.0  # 20.0
        navmesh_settings.region_min_size = 20.0  # 20.0
        navmesh_settings.verts_per_poly = 6.0  # 6.0
        self.ref_tf_global_agent = generate_reference(self.sim, uniform_step_number, navmesh_settings, plot_topdown)

    def one_step_sim(self, step: float, u: np.ndarray) -> np.ndarray:
        next_state = None
        return next_state

    def interactive_sim(self, controller, camera_rgb: bool, camera_depth: bool, step: float):
        """

        :param controller:

        :param camera_rgb:

        :param camera_depth:

        :param step:

        :return:
        """
        # 1. Agent init
        rot_agent_flu = Transform.from_matrix(TF_AGENT_FLU).as_rot_mat()
        simulation_iteration = 0
        self.agent.transformation = magnum.Matrix4(self.ref_tf_global_agent[0])
        obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone

        is_running = True
        while is_running:
            # 2. Sensors
            rgb_stereo_pair = np.concatenate([obs["left_camera"], obs["right_camera"]], axis=1)
            depth_stereo_pair = np.concatenate([obs["left_depth_camera"], obs["right_depth_camera"]], axis=1)
            z = obs["surface_dist"][0]
            tf_global_agent = np.array(self.agent.transformation)  # transform origin body
            rot_global_drone = Transform.from_matrix(tf_global_agent).as_rot_mat()
            rot_global_flu = rot_global_drone @ rot_agent_flu
            if controller:
                # TODO include controller
                pass
            else:
                motor = torch.FloatTensor([120, 120, 120, 120])

            # Plotting sensors
            print(simulation_iteration)
            print(f'Height: {z}')

            # 3-Motor and Ground Effect (gravity is done by habitat)
            motor_torsor = self.quadrotor.motor_dynamic(motor=motor).numpy()
            motor_torques = np.array(
                [self.drone_prms['kt'] * self.drone_prms['l'], self.drone_prms['kt'] * self.drone_prms['l'],
                 self.drone_prms['km']]) * motor_torsor[:3]

            motor_forces = np.zeros_like(motor_torques)
            motor_forces[2] = self.drone_prms['kt'] * motor_torsor[3]

            ground_forces = np.zeros_like(motor_torques)
            ground_forces[1] = (np.sum(motor.numpy()) ** 2 * self.drone_prms['kt'] * 1
                                * (self.drone_prms['prop_radius'] / (4 * z)) ** 2)

            self.agent.apply_force(magnum.Vector3(rot_global_flu @ motor_forces + ground_forces),
                                   relative_position=magnum.Vector3([0, 0, 0]))
            self.agent.apply_torque(magnum.Vector3(rot_global_flu @ motor_torques))

            # 4-Display and interactivity
            if camera_rgb:
                cv2.imshow("rgb_stereo_pair", rgb_stereo_pair)
            if camera_depth:
                depth_stereo_pair = np.clip(depth_stereo_pair, 0, 10)
                depth_stereo_pair /= 10.0
                cv2.imshow("depth_stereo_pair", depth_stereo_pair)
            key_pressed = cv2.waitKey()

            # Immobilizing the drone
            saved_velocity = (self.agent.linear_velocity, self.agent.angular_velocity)
            self.agent.linear_velocity = magnum.Vector3(rot_global_flu @ np.array([+0, 0, 0]))
            self.agent.angular_velocity = magnum.Vector3(rot_global_flu @ np.array([+0, 0, 0]))

            # Exploring the scene
            if key_pressed == ord("w"):
                self.agent.translate(magnum.Vector3(rot_global_flu @ np.array([+0.1, 0, 0])))
            if key_pressed == ord("s"):
                self.agent.translate(magnum.Vector3(rot_global_flu @ np.array([-0.1, 0, 0])))
            if key_pressed == ord("a"):
                # self.agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, +0.1, 0])))
                self.agent.rotate_y_local(magnum.Rad(+20 / 180 * np.pi))
            if key_pressed == ord("d"):
                # self.agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, -0.1, 0])))
                self.agent.rotate_y_local(magnum.Rad(-20 / 180 * np.pi))
            if key_pressed == ord("x"):
                self.agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, 0, +0.1])))
            if key_pressed == ord("z"):
                self.agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, 0, -0.1])))
            # Yaw
            if key_pressed == ord("q"):
                self.agent.rotate_z_local(magnum.Rad(+20 / 180 * np.pi))
            if key_pressed == ord("e"):
                self.agent.rotate_z_local(magnum.Rad(-20 / 180 * np.pi))
            # Pitch
            if key_pressed == ord("r"):
                self.agent.rotate_x_local(magnum.Rad(+20 / 180 * np.pi))
            if key_pressed == ord("f"):
                self.agent.rotate_x_local(magnum.Rad(-20 / 180 * np.pi))

            # Following the reference
            if key_pressed == ord("v"):
                simulation_iteration += 1
                self.agent.transformation = magnum.Matrix4(self.ref_tf_global_agent[simulation_iteration])
            if key_pressed == ord('c'):
                simulation_iteration -= 1
                self.agent.transformation = magnum.Matrix4(self.ref_tf_global_agent[simulation_iteration])

            # Resume the dynamic
            if key_pressed == ord(' '):
                self.agent.linear_velocity = magnum.Vector3(saved_velocity[0])
                self.agent.angular_velocity = magnum.Vector3(saved_velocity[1])

            if simulation_iteration == len(self.ref_tf_global_agent) - 1:
                break

            # 9-Simulating for step seconds
            self.sim.step_physics(step)
            s = self.sim.get_agent(0).state
            s.sensor_states['surface_dist'].rotation = np.quaternion(np.cos(0.25 * np.pi),
                                                                     -np.sin(0.25 * np.pi), 0, 0)
            self.sim.get_agent(0).set_state(s, False, False)
            obs = self.sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone


def main():
    drone_prms = {'m': 0.027,  # kg
                  'kt': 6e-6,  # 3.16e-10 / (2 * np.pi / 60) ** 2  #
                  'km': 1e-8,  # 7.94e-12 / (2 * np.pi / 60) ** 2,  #
                  'l': 0.0397,  # m
                  'ix': 1.4e-5,  # kg.m2
                  'iy': 1.4e-5,  # kg.m2
                  'iz': 2.17e-5,  # kg.m2
                  'prop_radius': 2.31348e-2}  # m
    scene_path = f"{WD_PATH}data/habitat/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    camera_resolution = np.array([720, 1280])

    simulation = Simulation(scene_path=scene_path, camera_resolution=camera_resolution, drone_prms=drone_prms,
                            allow_gravity=True)
    simulation.generate_reference_procedurally(uniform_step_number=1000, plot_topdown=False)

    simulation_mpc = Simulation(scene_path=scene_path, camera_resolution=camera_resolution, drone_prms=drone_prms,
                                allow_gravity=True)
    motor = mpc(model=lambda u: simulation_mpc.one_step_sim(step=1/60, u=u),
                reference=simulation.ref_tf_global_agent,
                horizon=10)

    # simulation.interactive_sim(controller=None, camera_rgb=True, camera_depth=False, step=1/60)


if __name__ == "__main__":
    main()
