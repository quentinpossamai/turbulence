import csv
import cv2
import habitat_sim
import magnum
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# from scipy.spatial.transform.rotation import Rotation
import torch
from typing import Tuple, Dict

from src.simulation.generate_reference import generate_reference
from src.drone_models.analytical_models.quadrotor_4parms import DynamicalModel
from src.utils.transform import Transform
from src.utils.paths import WD_PATH

cv2.namedWindow("stereo_pair")


def sim_agent_configuration(resolution=np.array([720, 1280])) -> Tuple[habitat_sim.simulator.Simulator,
                                                                       habitat_sim.physics.ManagedRigidObject,
                                                                       habitat_sim.CameraSensorSpec]:
    # 1-Initializing the simulation and loading the scene
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.enable_physics = True
    backend_cfg.scene_id = f"{WD_PATH}data/habitat/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"

    # 2-Adding a stereo RGB-D sensors to the drone
    # 2.1-left rgb camera
    left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    left_rgb_sensor.uuid = "left_camera"  # uuid is used to retrieve the observation
    left_rgb_sensor.resolution = resolution
    left_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.LEFT

    # 2.2-right rgb camera
    right_rgb_sensor = habitat_sim.CameraSensorSpec()
    right_rgb_sensor.uuid = "right_camera"
    right_rgb_sensor.resolution = resolution
    right_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT

    # 2.3-left depth camera
    left_depth_sensor = habitat_sim.CameraSensorSpec()
    left_depth_sensor.uuid = "left_depth_camera"
    left_depth_sensor.resolution = resolution
    left_depth_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.LEFT
    left_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    # 2.4-right depth camera
    right_depth_sensor = habitat_sim.CameraSensorSpec()
    right_depth_sensor.uuid = "right_depth_camera"
    right_depth_sensor.resolution = resolution
    right_depth_sensor.position = (0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT)
    right_depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    # 2.5-depth camera
    surface_distance_sensor = habitat_sim.CameraSensorSpec()
    surface_distance_sensor.uuid = "surface_distance_sensor"
    surface_distance_sensor.resolution = [1, 1]
    surface_distance_sensor.orientation = np.array([-90 / 180 * np.pi, 0, 0], dtype=np.float32)
    surface_distance_sensor.position = (0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.RIGHT)
    surface_distance_sensor.sensor_type = habitat_sim.SensorType.DEPTH

    # 3-Instantiating drone
    # 3.1-Connecting the drone to its sensors
    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [surface_distance_sensor, left_rgb_sensor, right_rgb_sensor, left_depth_sensor,
                                          right_depth_sensor]
    # 3.2-Adding the drone to the simulation
    sim: habitat_sim.physics.ManagedRigidObject = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg,
                                                                                                  [agent_config]))
    # 3.3-Adding a rigid cube as a body to the drone
    # prim_templates_mgr = sim.get_asset_template_manager()
    obj_templates_mgr: habitat_sim.attributes_managers.ObjectAttributesManager = sim.get_object_template_manager()
    rigid_obj_mgr: habitat_sim.physics.RigidObjectManager = sim.get_rigid_object_manager()
    cube_template_handle = obj_templates_mgr.get_template_handles('cubeSolid')[0]
    cube_obj: habitat_sim.physics.ManagedRigidObject
    cube_obj = rigid_obj_mgr.add_object_by_template_handle(cube_template_handle, sim.agents[0].scene_node)
    # agent: habitat_sim.agent.Agent = sim.agents[0]
    # # cube_obj.user_attributes = {'agent': agent}
    # agent_state = habitat_sim.AgentState()
    # agent_state.sensor_states['surface_distance_sensor'].rotation = rot_x
    return sim, cube_obj, surface_distance_sensor


# Helper function to render observations from the stereo agent
def simulate(sim: habitat_sim.simulator.Simulator,
             agent: habitat_sim.physics.ManagedRigidObject,
             drone_prms: Dict[str, float], display: bool, depth: bool = False,
             motor=torch.FloatTensor([120, 120, 120, 120])):
    ref_path = f'/Users/quentin/phd/turbulence/data/blackbird/raw/ampersand/yawForward/maxSpeed1p0/csv/' \
               f'blackbird_slash_pose_ref.csv'
    with open(ref_path, newline='') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader)
        data = [e for e in csv_reader]
        time = np.array([float(e[0]) * 1e-9 for e in data])
        trans = np.array([np.array([float(e[9]), float(e[10]), float(e[11])]) for e in data])
        quat = np.array([np.array([float(e[13]), float(e[14]), float(e[15]), float(e[16])]) for e in data])
        reference = {"t": time, "trans": trans, "quat": quat}
        tf_flu0_fludrone = Transform.from_quat_trans(quat=quat, trans=trans).as_matrix()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        start, stop = 2000, len(trans) - 2000
        # ax.plot(xs=position[start:stop, 0], ys=position[start:stop, 1], zs=position[start:stop, 2])
        # plt.show()
    pi_rotation = Transform.from_euler_trans("zyx", np.array([0, 0, np.pi]),
                                             degrees=False, trans=np.array([0, 0, 0])).as_matrix()
    tf_flu0_fludrone = pi_rotation @ tf_flu0_fludrone[start:stop] @ pi_rotation
    tf_global_drone0 = np.array(agent.transformation)  # transform origin body
    tf_drone_flu = Transform.from_euler_trans(seq='xyz', angles=np.array([-90, 90, 0]), degrees=True,
                                              trans=np.zeros(3)).as_matrix()
    rot_drone_flu = Transform.from_matrix(tf_drone_flu).as_rot_mat()
    reference_tf_global_fludrone = tf_global_drone0.reshape((1, 4, 4)) @ \
                                   tf_drone_flu.reshape((1, 4, 4)) @ \
                                   tf_flu0_fludrone
    is_running = True
    obs = sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
    simulation_iteration = 0
    while is_running:
        # Put the two stereo observations next to each other
        rgb_stereo_pair = np.concatenate([obs["left_camera"], obs["right_camera"]], axis=1)
        depth_stereo_pair = np.concatenate([obs["left_depth_camera"], obs["right_depth_camera"]], axis=1)
        z = obs["surface_distance_sensor"][0]

        # Drone state and transforms
        tf_global_drone = np.array(agent.transformation)  # transform origin body
        rot_global_drone = Transform.from_matrix(tf_global_drone).as_rot_mat()
        rot_global_flu = rot_global_drone @ rot_drone_flu

        # Dynamic
        assert drone_prms['ix'] == drone_prms['iy']
        theta = torch.FloatTensor([drone_prms['kt'] / drone_prms['m'],
                                   drone_prms['kt'] * drone_prms['l'] / drone_prms['ix'],
                                   drone_prms['km'] / drone_prms['iz'],
                                   drone_prms['iz'] / drone_prms['ix']])
        model = DynamicalModel(param={'permutation': [0, 1, 2, 3], 'theta': theta}, dtype=torch.float32)
        motor_torsor = model.motor_dynamic(motor=motor).numpy()
        motor_torques = np.array([drone_prms['kt'] * drone_prms['l'], drone_prms['kt'] * drone_prms['l'],
                                  drone_prms['km']]) * motor_torsor[:3]

        motor_forces = np.zeros_like(motor_torques)
        motor_forces[2] = drone_prms['kt'] * motor_torsor[3]

        ground_forces = np.zeros_like(motor_torques)
        ground_forces[1] = np.sum(motor.numpy()) ** 2 * drone_prms['kt'] * 1 * \
                           (drone_prms['prop_radius'] / (4 * z)) ** 2

        # Simulation
        # If it is a depth pair, manually normalize into [0, 1] so that images are always consistent
        if depth:
            depth_stereo_pair = np.clip(depth_stereo_pair, 0, 10)
            depth_stereo_pair /= 10.0

        # If in RGB/RGBA format, change first to RGB and change to BGR
        if len(depth_stereo_pair.shape) > 2:
            depth_stereo_pair = depth_stereo_pair[..., 0:3][..., ::-1]
        step = 1 / 60
        # display=False is used for the smoke test
        if display:
            cv2.imshow("rgb_stereo_pair", rgb_stereo_pair)
            cv2.imshow("depth_stereo_pair", depth_stereo_pair)
            k = cv2.waitKey()

            # agent.apply_force(magnum.Vector3(rot_global_flu @ motor_forces + ground_forces),
            #                   relative_position=magnum.Vector3([0, 0, 0]))
            # agent.apply_torque(magnum.Vector3(rot_global_flu @ motor_torques))
            # print(agent.transformation)

            if k == ord("g"):
                is_running = False
            if k == ord("w"):
                agent.translate(magnum.Vector3(rot_global_flu @ np.array([+0.1, 0, 0])))
            if k == ord("s"):
                agent.translate(magnum.Vector3(rot_global_flu @ np.array([-0.1, 0, 0])))
            if k == ord("a"):
                # agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, +0.1, 0])))
                agent.rotate_y_local(magnum.Rad(+20 / 180 * np.pi))
            if k == ord("d"):
                # agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, -0.1, 0])))
                agent.rotate_y_local(magnum.Rad(-20 / 180 * np.pi))
            if k == ord("x"):
                agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, 0, +0.1])))
            if k == ord("z"):
                agent.translate(magnum.Vector3(rot_global_flu @ np.array([0, 0, -0.1])))
            # Yaw
            if k == ord("q"):
                agent.rotate_z_local(magnum.Rad(+20 / 180 * np.pi))
            if k == ord("e"):
                agent.rotate_z_local(magnum.Rad(-20 / 180 * np.pi))
            # Pitch
            if k == ord("r"):
                agent.rotate_x_local(magnum.Rad(+20 / 180 * np.pi))
            if k == ord("f"):
                agent.rotate_x_local(magnum.Rad(-20 / 180 * np.pi))
            saved_speed = (agent.linear_velocity, agent.angular_velocity)
            agent.linear_velocity = magnum.Vector3(rot_global_flu @ np.array([+0, 0, 0]))
            agent.angular_velocity = magnum.Vector3(rot_global_flu @ np.array([+0, 0, 0]))
            if k == ord(" "):
                agent.linear_velocity = saved_speed[0]
                agent.angular_velocity = saved_speed[1]
                pass

            # Follow reference
            # Transform.from_matrix(reference_tf_global_fludrone[simulation_iteration]).as_quat_trans()
            # agent.angular_velocity = magnum.Vector3([0, 0, 0])
            # agent.linear_velocity = magnum.Vector3([0, 0, 0])
            # agent.transformation = magnum.Matrix4(reference_tf_global_fludrone[simulation_iteration])
            # print(f'simulation_iteration: {simulation_iteration}'
            #       f'reference_tf_global_fludrone[simulation_iteration]: {reference_tf_global_fludrone[simulation_iteration]}')

            sim.step_physics(step)
            s = sim.get_agent(0).state
            s.sensor_states['surface_distance_sensor'].rotation = np.quaternion(np.cos(0.25 * np.pi),
                                                                                -np.sin(0.25 * np.pi), 0, 0)
            sim.get_agent(0).set_state(s, False, False)
            obs = sim.get_sensor_observations(agent_ids=0)  # 0 is the agent_id of the drone
            simulation_iteration += 1


def main():
    sim, agent, surface_distance_sensor = sim_agent_configuration()

    # Drone parameters
    agent.motion_type = habitat_sim.physics.MotionType.DYNAMIC
    drone_prms = {'m': 0.027,  # kg
                  'kt': 6e-6,  # 3.16e-10 / (2 * np.pi / 60) ** 2  #
                  'km': 1e-8,  # 7.94e-12 / (2 * np.pi / 60) ** 2,  #
                  'l': 0.0397,  # m
                  'ix': 1.4e-5,  # kg.m2
                  'iy': 1.4e-5,  # kg.m2
                  'iz': 2.17e-5,  # kg.m2
                  'prop_radius': 2.31348e-2,  # m
                  }
    agent.mass = drone_prms['m']
    agent.intertia_diagonal = [drone_prms['ix'], drone_prms['iy'], drone_prms['iz']]
    simulate(sim, agent, drone_prms=drone_prms, display=True, depth=True)
    pass


if __name__ == "__main__":
    main()
