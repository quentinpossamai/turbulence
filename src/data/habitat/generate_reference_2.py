import matplotlib.text
from Christofides import christofides
import habitat_sim
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import magnum
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import splev, splrep
import sys
from tqdm import tqdm
from typing import Union, Dict

from src.data.habitat.hatbitat2flu import Habitat2FLU
from src.utils.transform import Transform, Rotation

matplotlib.use('Qt5Agg')


# matplotlib.use('MacOSX')
# matplotlib.use('TkAgg')
# matplotlib.use('Qt4Agg')

def display_topdown(sim, point, travel_order, entire_path):
    # Display
    resolution = 0.03
    h = sim.get_agent(0).state.position[1]
    topdown = sim.pathfinder.get_topdown_view(resolution, h)
    low, high = sim.pathfinder.get_bounds()
    img = np.zeros(list(topdown.shape) + [3], dtype=np.uint8)
    img[topdown] = np.array([255, 255, 255], dtype=np.uint8)

    # Place points on image
    point_img = ((point - low) / resolution).astype(int)
    img[point_img[:, 2], point_img[:, 0]] = np.array([255, 0, 0], dtype=np.uint8)

    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax_img = ax.imshow(img)
    for i, order in enumerate(travel_order[:-1]):
        ax.text(point_img[order, 0], point_img[order, 2], s=f'{i}', c='r')

    # Place trajectory on image
    entire_path_img = ((entire_path - low) / resolution).astype(int)
    for ind, _ in enumerate(entire_path_img[:-1]):
        plt.plot(entire_path_img[ind:ind + 2, 0], entire_path_img[ind:ind + 2, 2], c='g')

    axcolor = 'lightgoldenrodyellow'
    ax_wn = plt.axes([0.05, 0.1, 0.0225, 0.83], facecolor=axcolor)
    slider = Slider(ax=ax_wn,
                    label="Height",
                    valmin=0,
                    valmax=1.5,
                    valinit=0,
                    orientation="vertical")

    # The function to be called anytime a slider's value changes
    def _update(_):
        print(slider.val)
        _topdown = sim.pathfinder.get_topdown_view(resolution, slider.val)
        _low, _high = sim.pathfinder.get_bounds()
        _img = np.zeros(list(_topdown.shape) + [3], dtype=np.uint8)
        _img[_topdown] = np.array([255, 255, 255], dtype=np.uint8)
        jj, _, ii = ((point - _low) / resolution).astype(int).T
        _img[ii, jj] = np.array([255, 0, 0], dtype=np.uint8)
        ax_img.set_data(_img)
        fig.canvas.draw_idle()

    # register the update function with each slider
    slider.on_changed(_update)
    plt.show()


def get_orientation_from_position(x_time, x, point, trans_ref, flight_height):
    future_point = point[np.where(x_time < x)[0][0]]
    # In global
    x_vect = (future_point - trans_ref)
    x_vect = x_vect / np.linalg.norm(x_vect)
    y_vect = np.cross(np.array([0, 1, 0]), x_vect)
    y_vect = y_vect / np.linalg.norm(y_vect)
    z_vect = np.cross(x_vect, y_vect)

    trans_ref = flight_height * z_vect + trans_ref
    tf_global_flu = Transform.from_3_axis_rot_trans(x=x_vect, y=y_vect, z=z_vect, trans=trans_ref).as_matrix()
    return tf_global_flu @ Habitat2FLU.tf_flu__hab  # Todo check


def polynomial_interpolation(main_point, num_sample, degree, flight_height):
    """
    Sample points between main_point with polynomial interpolation.

    param main_point:

    param num_sample:

    param degree:

    param flight_height:

    return:
    """
    # Percentage of advancement of the trajectory
    traj_time = [0]
    for i in range(1, len(main_point)):
        traj_time.append(np.linalg.norm(main_point[i] - main_point[i - 1]))
    traj_time = np.cumsum(traj_time)
    traj_time = traj_time / traj_time[-1]

    trans = np.zeros((num_sample, 3))
    for i in range(3):
        rep = splrep(x=traj_time, y=main_point[:, i], k=degree)
        trans[:, i] = splev(np.linspace(0, 1, num_sample), rep)
    dist = []
    for point in main_point:
        dist.append(np.min(np.linalg.norm(point - trans, axis=1)))
    print(f'Max reference error {np.max(dist)}', flush=True)

    # Computing of the orientation from the position
    ref_tf_global_agent = []
    for x_time, interp_ref_time in zip(np.linspace(0, 1, num_sample), trans):
        if x_time == 1.0:
            rot = Transform.from_matrix(ref_tf_global_agent[-1]).as_rot_mat()
            ref_tf_global_agent.append(Transform.from_rot_mat_trans(rot=rot, trans=interp_ref_time).as_matrix())
        else:
            ref_tf_global_agent.append(get_orientation_from_position(x_time, traj_time, main_point, interp_ref_time,
                                                                     flight_height))
    ref_tf_global_agent = np.stack(ref_tf_global_agent, axis=0)
    return ref_tf_global_agent


def navmesh_optimization(sim, navset_dict):
    navmesh_settings: habitat_sim.nav.NavMeshSettings = habitat_sim.NavMeshSettings()
    # for key, val in navset_dict.items():
    #     setattr(navmesh_settings, key, val)
    # sim.recompute_navmesh(sim.pathfinder, navmesh_settings=navmesh_settings)

    # Display
    resolution = 0.05
    h = sim.get_agent(0).state.position[1]
    h = 0.134
    topdown = sim.pathfinder.get_topdown_view(resolution, h)
    # low, high = sim.pathfinder.get_bounds()
    # img = np.zeros(list(topdown.shape) + [3], dtype=np.uint8)
    # img[topdown] = np.array([255, 255, 255], dtype=np.uint8)
    print(topdown.shape)

    fig, ax = plt.subplots(1, 1, figsize=(13.2, 8), dpi=100)
    fig: plt.Figure
    ax: plt.Axes
    ax.set_title(f'resolution: {resolution} | cell_size: {navset_dict["cell_size"]} |'
                 f'')
    ax_img = ax.imshow(topdown, animated=True, cmap='gray')
    ax.axis('off')
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.8, top=0.99)
    ax_height = plt.axes([0.06, 1 - 0.1, 0.7, 0.02], facecolor='lightgoldenrodyellow')  # left, bottom, width, height
    h_slider = Slider(ax=ax_height,
                      label="Height",
                      valmin=0,
                      valmax=1.5,
                      valinit=h,
                      orientation="horizontal")

    # The function to be called anytime a slider's value changes
    def _update(_):
        print(f'h_slider.val: {h_slider.val}', flush=True)
        _topdown = sim.pathfinder.get_topdown_view(resolution, h_slider.val)
        # _low, _high = sim.pathfinder.get_bounds()
        # _img = np.zeros(list(_topdown.shape) + [3], dtype=np.uint8)
        # _img[_topdown] = np.array([255, 255, 255], dtype=np.uint8)
        # fig.canvas.restore_region(bg)
        ax_img.set_data(_topdown)
        ax.draw_artist(ax_img)
        fig.canvas.update()  # only usable with Qt5Agg backend
        # fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()
        print(f'Update done' + 2 * '\n', flush=True)

    def _update_text_box(expression, a):
        if expression in ['True', 'False']:
            _val = bool(expression)
        else:
            _val = float(expression)
        print(f'New value: {a} {expression}')
        setattr(navmesh_settings, a, _val)
        sim.recompute_navmesh(sim.pathfinder, navmesh_settings=navmesh_settings)
        _update(None)

    from matplotlib.widgets import TextBox
    box_axs = []
    text_boxs = []
    i = 11
    for j in list(range(i)) + list(range(i + 1, 16)):
        key = list(navset_dict.keys())[j]
        key = (key + '.')[:-1]
        val = navset_dict[key]
        # ax_text: plt.Axes = plt.axes([0.96, 1 - 0.05 - 0.05 * j, 0.03, 0.04])  # [left, bottom, width, height]
        # ax_text.text(0, 0, f'{key}={val}')
        plt.annotate(f'{key}={val}', [0.8, 1 - 0.05 - 0.05 * j], xycoords='figure fraction', fontsize=10)
    key = list(navset_dict.keys())[i]
    key = (key + '.')[:-1]
    val = navset_dict[key]
    ax_text = plt.axes([0.96, 1 - 0.05 - 0.05 * i, 0.03, 0.04])  # [left, bottom, width, height]
    text_boxs.append(TextBox(ax_text, key, val))
    text_boxs[-1].on_submit(lambda x: _update_text_box(x, key))

    # register the update function with each slider
    h_slider.on_changed(_update)
    plt.show()


def generate_reference(sim: habitat_sim.Simulator,
                       initial_point: np.ndarray,
                       rigid_object: habitat_sim.physics.ManagedRigidObject,
                       uniform_speed: float,
                       navmesh_settings_dict: Dict['str', Union[bool, float]],
                       flight_height: float,
                       dt: float,
                       db_use_og_path: bool = False,
                       db_plot_navmesh: bool = False,
                       db_plot_topdown: bool = False) -> Union[np.ndarray, bool]:
    if db_plot_navmesh:
        navmesh_optimization(sim, navmesh_settings_dict)
    if not db_use_og_path:
        # Find the geodesic shortest path between those points
        point = initial_point
        shortest_path = []
        distance_matrix = np.zeros((len(point), len(point)))
        for i, p1 in enumerate(point):
            for j, p2 in enumerate(point):
                if i == j:
                    continue
                short_path = habitat_sim.nav.ShortestPath()
                short_path.requested_start = p1
                short_path.requested_end = p2
                sim.pathfinder.find_path(short_path)
                shortest_path.append((i, j, short_path.geodesic_distance, short_path))
                if j > i:
                    distance_matrix[i, j] = short_path.geodesic_distance
        # Check if every point is reachable
        for t in shortest_path:
            if t[2] == np.inf:
                print('Generated points non reachable, change seed', file=sys.stderr, flush=True)
                return False

        # Compute the order to navigate between those points
        tsp_solution = christofides.compute(distance_matrix)
        travel_order = tsp_solution['Christofides_Solution']

        # Add points to avoid obstacles. At the end the entire_path can be done by navigating in straight lines to the
        # next point
        entire_path = []
        for i, current_point_idx in enumerate(travel_order[:-1]):
            next_point_idx = travel_order[i + 1]
            dist = [dist for dist in shortest_path if current_point_idx == dist[0] and next_point_idx == dist[1]][0]
            print(f'current_point_idx: {current_point_idx} | next_point_idx: {next_point_idx} | dist: {dist}', flush=True)
            entire_path = entire_path + dist[3].points
        entire_path.append(point[travel_order[-1]])
        entire_path = np.stack(entire_path)

        # Remove duplicates
        remove_duplicate = []
        for i, e in enumerate(entire_path[:-1]):
            if np.any(entire_path[i + 1] != e):
                remove_duplicate.append(e)
        remove_duplicate.append(entire_path[-1])
        entire_path = np.stack(remove_duplicate)
    else:
        entire_path = initial_point

    if db_plot_topdown:  # display trajectory
        display_topdown(sim, point, travel_order, entire_path)

    # Sample path to impose a specific speed
    perimeter = np.linalg.norm(entire_path[1:] - entire_path[:-1], axis=1).sum()
    num_sample = int(np.round((perimeter / uniform_speed) / dt))
    # Interpolation
    ref_tf_global_agent_collidable = polynomial_interpolation(main_point=entire_path, num_sample=num_sample, degree=1,
                                                              flight_height=flight_height)
    if np.isnan(ref_tf_global_agent_collidable).any():
        print('Reference interpolation not possible, change seed', file=sys.stderr, flush=True)
        return False

    # Check if trajectory collides
    ref_tf_global_agent = np.zeros_like(ref_tf_global_agent_collidable)
    for i, tf in enumerate(tqdm(ref_tf_global_agent_collidable, desc='Starting generating trajectory')):
        rigid_object.transformation = tf
        has_collided = rigid_object.contact_test()

        while has_collided:
            quat, trans = Transform.from_matrix(tf).as_quat_trans()
            # Debug
            looking = False
            while looking:
                obs = sim.get_sensor_observations(agent_ids=1)  # 1 is the agent_id of the error agent
                dist = np.array([obs["forward_dist"][0][0],
                                 obs["backward_dist"][0][0],
                                 obs["left_dist"][0][0],
                                 obs["right_dist"][0][0],
                                 obs["up_dist"][0][0],
                                 obs["down_dist"][0][0]])
                print(f'Has collided: {has_collided} | i: {i} | dist: {dist}')
                import cv2
                obs = sim.get_sensor_observations(agent_ids=1)  # 1 is the agent_id of the error agent
                cv2.imshow("rgb", obs["error_camera"])
                key_pressed = cv2.waitKey()
                tf_global_agent = np.array(rigid_object.transformation)  # transform origin body
                rot_global_flu = Transform.from_matrix(tf_global_agent).as_rot_mat() @ Habitat2FLU.rot_hab__flu
                # 6-Exploring the scene
                # 6.1. Forward, Backward
                if key_pressed == ord("w"):  # Front
                    rigid_object.translate(rot_global_flu @ np.array([+0.1, 0, 0]))
                if key_pressed == ord("s"):  # Back
                    rigid_object.translate(rot_global_flu @ np.array([-0.1, 0, 0]))
                # 6.2 Thrust
                if key_pressed == ord("x"):  # Up
                    rigid_object.translate(rot_global_flu @ np.array([0, 0, +0.1]))
                if key_pressed == ord("z"):  # Down
                    rigid_object.translate(rot_global_flu @ np.array([0, 0, -0.1]))
                # 6.3. Roll
                if key_pressed == ord("q"):  # Left
                    rigid_object.rotate_z_local(magnum.Rad(+20 / 180 * np.pi))
                if key_pressed == ord("e"):  # Right
                    rigid_object.rotate_z_local(magnum.Rad(-20 / 180 * np.pi))
                # 6.4. Pitch
                if key_pressed == ord("r"):  # Up
                    rigid_object.rotate_x_local(magnum.Rad(+20 / 180 * np.pi))
                if key_pressed == ord("f"):  # Down
                    rigid_object.rotate_x_local(magnum.Rad(-20 / 180 * np.pi))
                # 6.5. Yaw
                if key_pressed == ord("a"):  # Left
                    rigid_object.rotate_y_local(magnum.Rad(+20 / 180 * np.pi))
                if key_pressed == ord("d"):  # Right
                    rigid_object.rotate_y_local(magnum.Rad(-20 / 180 * np.pi))
                if key_pressed == ord("k"):  # Right
                    cv2.destroyAllWindows()
                    looking = False
                if trans[1] > 3:  # 3 meters
                    print('Collision non avoidable in reference, change seed')
                    return False
                if has_collided:
                    dtrans = Rotation.from_quat(quat).as_matrix() @ np.array([0, 0.1, 0])
                    trans = trans + dtrans
                    tf = Transform.from_quat_trans(quat, trans).as_matrix()
                    rigid_object.transformation = tf
                    has_collided = rigid_object.contact_test()
            if has_collided and not looking:
                dtrans = Rotation.from_quat(quat).as_matrix() @ np.array([0, 0.1, 0])
                trans = trans + dtrans
                tf = Transform.from_quat_trans(quat, trans).as_matrix()
                rigid_object.transformation = tf
                has_collided = rigid_object.contact_test()
        ref_tf_global_agent[i] = tf

    # Compute velocity
    quat, trans = Transform.from_matrix(ref_tf_global_agent).as_quat_trans()
    lin_vel_global = (trans[1:] - trans[:-1]) / dt
    lin_vel_global = np.concatenate([np.zeros((1, 3)), lin_vel_global], axis=0)
    return np.concatenate([trans, quat, lin_vel_global, np.zeros((len(lin_vel_global), 3))], axis=1)
