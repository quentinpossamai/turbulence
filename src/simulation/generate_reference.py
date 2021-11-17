from Christofides import christofides
import habitat_sim
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

from src.utils.paths import WD_PATH


def aff(sim, point, travel_order, entire_path):
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
    color_map = cm.viridis(range(len(entire_path)))
    for ind, _ in enumerate(entire_path_img[:-1]):
        plt.plot(entire_path_img[ind:ind + 2, 0], entire_path_img[ind:ind + 2, 2], c=color_map[ind])

    axcolor = 'lightgoldenrodyellow'
    ax_wn = plt.axes([0.05, 0.1, 0.0225, 0.83], facecolor=axcolor)
    slider = Slider(ax=ax_wn,
                    label="Height",
                    valmin=0,
                    valmax=1.5,
                    valinit=0,
                    orientation="vertical")

    # The function to be called anytime a slider's value changes
    def update(_):
        resolution = 0.03
        h = sim.get_agent(0).state.position[1]
        print(slider.val)
        topdown = sim.pathfinder.get_topdown_view(resolution, slider.val)
        low, high = sim.pathfinder.get_bounds()
        img = np.zeros(list(topdown.shape) + [3], dtype=np.uint8)
        img[topdown] = np.array([255, 255, 255], dtype=np.uint8)
        j, _, i = ((point - low) / resolution).astype(int).T
        img[i, j] = np.array([255, 0, 0], dtype=np.uint8)
        # ax.clear()
        ax_img.set_data(img)
        fig.canvas.draw_idle()

    # register the update function with each slider
    slider.on_changed(update)
    plt.show()


def generate_reference(scene_id: str, display: bool = False):
    # Defining the drone's volume
    navmesh_settings: habitat_sim.nav.NavMeshSettings = habitat_sim.NavMeshSettings()
    navmesh_settings.agent_height = 0.1  # 1.5
    navmesh_settings.agent_max_climb = 99.0  # 0.2
    navmesh_settings.agent_max_slope = 90.0  # 45.0
    navmesh_settings.agent_radius = 0.1  # 0.1
    navmesh_settings.cell_height = 0.05  # 0.2
    navmesh_settings.cell_size = 0.05  # 0.05
    navmesh_settings.detail_sample_dist = 1.0  # 6.0
    navmesh_settings.detail_sample_max_error = 0.1  # 1.0
    navmesh_settings.edge_max_error = 1.3  # 1.3
    navmesh_settings.edge_max_len = 0.1  # 12.0
    navmesh_settings.filter_ledge_spans = False  # True
    navmesh_settings.filter_low_hanging_obstacles = False  # True
    navmesh_settings.filter_walkable_low_height_spans = False  # True
    navmesh_settings.region_merge_size = 20.0  # 20.0
    navmesh_settings.region_min_size = 20.0  # 20.0
    navmesh_settings.verts_per_poly = 6.0  # 6.0

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.enable_physics = True
    backend_cfg.scene_id = scene_id

    # 2-Adding a stereo RGB-D sensors to the drone
    # 2.1-left rgb camera
    left_rgb_sensor = habitat_sim.bindings.CameraSensorSpec()
    left_rgb_sensor.uuid = "left_camera"  # uuid is used to retrieve the observation
    left_rgb_sensor.resolution = [1280, 720]
    left_rgb_sensor.position = 0 * habitat_sim.geo.UP + 0 * habitat_sim.geo.LEFT

    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [left_rgb_sensor]

    sim = habitat_sim.simulator.Simulator(habitat_sim.Configuration(backend_cfg, [agent_config]))
    seed = 0
    sim.seed(seed)
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings=navmesh_settings)
    point = np.stack([sim.pathfinder.get_random_navigable_point() for _ in range(10)])

    is_navigable = False
    while not is_navigable:
        sim.seed(seed)
        point = np.stack([sim.pathfinder.get_random_navigable_point() for _ in range(10)])
        shortest_path = []
        distance_matrix = np.zeros((len(point), len(point)))
        for i, p1 in enumerate(point):
            for j, p2 in enumerate(point):
                if i == j:
                    continue
                short_path = habitat_sim.nav.ShortestPath()  # Todo problem ?
                short_path.requested_start = p1
                short_path.requested_end = p2
                sim.pathfinder.find_path(short_path)
                shortest_path.append((i, j, short_path.geodesic_distance, short_path))
                if j > i:
                    distance_matrix[i, j] = short_path.geodesic_distance
        is_navigable = True
        for t in shortest_path:
            if t[2] == np.inf:
                is_navigable = False
        if not is_navigable:
            seed += 1

    tsp_solution = christofides.compute(distance_matrix)
    travel_order = tsp_solution['Christofides_Solution']

    entire_path = []
    for i, current_point_idx in enumerate(travel_order[:-1]):
        next_point_idx = travel_order[i + 1]
        dist = [dist for dist in shortest_path if current_point_idx == dist[0] and next_point_idx == dist[1]][0]
        print(f'current_point_idx: {current_point_idx} | next_point_idx: {next_point_idx} | dist: {dist}')
        entire_path = entire_path + dist[3].points
    entire_path.append(point[travel_order[-1]])
    entire_path = np.stack(entire_path)

    if display:
        aff(sim, point, travel_order, entire_path)

    return entire_path


def main():
    generate_reference(f"{WD_PATH}data/habitat/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb", display=True)


if __name__ == '__main__':
    main()
