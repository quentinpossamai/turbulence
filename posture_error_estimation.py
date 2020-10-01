from typing import Union, Iterable, Tuple, List, Dict, Any
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import numpy as np
import pandas as pd
from cv2 import VideoWriter, VideoWriter_fourcc
from scipy.spatial.transform import Rotation
from util import Progress, Transform, angle_arccos, DataFolder, merge_two_arrays


def main():
    f = DataFolder('data_drone2')
    vol_number = 1

    # mc preparation
    print('Loading data')
    mc_measure = f.pickle_load_file('.pkl', f.folders['raw_python'][vol_number], 'mc_measure')
    mc_measure = mc_measure[['time', 'pose']]
    # Set flu0
    flu0_tf_origin = mc_measure['pose'][0]
    for i, drone_tf_origin in enumerate(mc_measure['pose']):
        mc_measure.loc[i, 'pose'] = drone_tf_origin @ flu0_tf_origin.inv()

    # MC - ANALYSIS 1 : Finding first synchro
    # mc_scissors(mc_measure, ind_inf=0, ind_sup=len(mc_measure)-1, saving_path=f.folders['intermediate'][vol_number])

    # MC - ANALYSIS 2 : Finding second synchro
    # mc_drone_flu0(mc_measure, ind_inf=9000, ind_sup=9150, saving_path=f.folders['intermediate'][vol_number])
    # mc_drone_flu0(mc_measure, ind_inf=0, ind_sup=len(mc_measure) - 1,
    #               saving_path=f.folders['intermediate'][vol_number])

    # tara preparation
    tara_raw = f.pickle_load_file('.pkl', f.folders['raw_python'][vol_number], 'tara_left', True)
    topic = 'tara/left/image_raw'
    images = pd.DataFrame({'time': [tara_raw[topic][idx]['t'] for idx in tara_raw[topic]],
                           'image': [tara_raw[topic][idx]['image'] for idx in tara_raw[topic]]})
    # Time begin at 0
    images['time'] -= images['time'][0]

    # # TARA - ANALYSIS 1 : Finding first synchro
    # tara_slider_plot(images, topic, ind_inf=0, ind_sup=len(images))

    # TARA - ANALYSIS 2 : Finding first and second synchro
    # saving_path = f.folders['intermediate'][vol_number] + 'tara.mkv'
    # build_video(saving_path=saving_path, image_df=images, fps=30)

    # Index of mc data and drone data
    print('Synchronizing data')
    synchro = {0: {'mc': 4.125,
                   'images': np.nan},  # Scissors closing not shown on data
               1: {'mc': [470, 9077],
                   'images': [2902, 5430]}}

    # Tara start sync TODO not cut here
    assert synchro[vol_number]['images'][0] < synchro[vol_number]['images'][1], "Synchro failed, tara indices error."
    # images_sync = images[synchro[vol_number]['images'][0]:].reset_index(drop=True)
    images_sync = images.copy()
    images_sync['time'] -= images_sync['time'][synchro[vol_number]['images'][0]]

    # MC start sync
    assert synchro[vol_number]['mc'][0] < synchro[vol_number]['mc'][1], "Synchro failed, mc indices error."
    # mc_sync = mc_measure[synchro[vol_number]['mc'][0]:].reset_index(drop=True)
    mc_sync = mc_measure.copy()
    mc_sync['time'] -= mc_sync['time'][synchro[vol_number]['mc'][0]]
    mc_sync['time'] *= (images_sync['time'][synchro[vol_number]['images'][1]] /
                        mc_sync['time'][synchro[vol_number]['mc'][1]])

    # Measure sync
    mc_ids, image_ids = merge_two_arrays(mc_sync['time'], images_sync['time'])
    input_data = pd.concat([mc_sync.loc[mc_ids].reset_index(drop=True).rename({'time': 'pose_time'}),
                            images_sync.loc[image_ids].reset_index(drop=True).rename({'time': 'image_time'})], axis=1)

    estimator = ErrorEstimation(input_data, tara_raw['tara/left/camera_info'][0])
    estimator.p3_generator()
    saving_path = f.folders['results'][vol_number] + 'pose_camera_correspondence_video.mkv'
    estimator.p32video(fps=30, saving_path=saving_path)


class ErrorEstimation(object):
    def __init__(self, input_data: pd.DataFrame,
                 camera_parameters: Dict[str, np.ndarray]):
        """
        Create a video to verify with the eye if the position estimation and video flux
        :param input_data: pandas.DataFrame with columns :
            'pose_time' : float
            'pose' : Each pose correspond to drone_tf_mc_origin, a util.Transform object.
            'image_time' : float
            'image' : numpy.ndarray of shape (width, height, 1)
        :param camera_parameters: The intrinsic parameters of the camera.
         {'P': np.ndarray of shape (3, 4)
          'K': np.ndarray of shape (3, 3)
          'R': np.ndarray of shape (3, 3)
          'D': np.ndarray of shape (5,)}
        """

        self.df = input_data
        self.camera_parameters = camera_parameters
        self.p = camera_parameters['P']
        self.p3 = []

        self.frames_width = self.df['image'][0].shape[1]  # width along the second axis
        self.frames_height = self.df['image'][0].shape[0]

        # Set flu0
        flu0_tf_origin = self.df['pose'][0]
        for i, drone_tf_origin in enumerate(self.df['pose']):
            self.df.loc[i, 'pose'] = drone_tf_origin @ flu0_tf_origin.inv()

        # Vincent input : (0.105, 0, 0, -1.57, 0.0, -2.0943) (convention x, y, z, roll, pitch, yaw)
        drone_d_camera = np.array([0.105, 0, 0])
        drone_rot_camera = Rotation.from_euler('xyz', np.array([-120, 0, -90]), degrees=True).as_quat()
        self.camera_tf_drone = Transform().from_pose(drone_d_camera, drone_rot_camera)

    def get_camera_angle_to_horizontal_plane(self, iteration):
        z_camera_in_camera = np.array([0, 0, 1])  # Vector only rotation are important and NOT translations

        drone_tf_camera = self.camera_tf_drone.inv()
        drone_rot_camera = drone_tf_camera.get_rot()
        z_camera_in_drone = drone_rot_camera @ z_camera_in_camera

        drone_tf_flu0 = self.df['pose'][iteration]
        flu0_tf_drone = drone_tf_flu0.inv()
        flu0_rot_drone = flu0_tf_drone.get_rot()
        z_camera_in_flu0 = flu0_rot_drone @ z_camera_in_drone

        camera_angle_to_horizontal_plane = np.arcsin(np.abs(z_camera_in_flu0[2]) / np.linalg.norm(z_camera_in_flu0))
        return camera_angle_to_horizontal_plane

    def image23d(self, idx, image_point):
        # self.xyz and self.quaternions are from FLU0 to Drone

        p_left = self.p
        fx = p_left[0, 0]
        fy = p_left[1, 1]
        cx = p_left[0, 2]
        cy = p_left[1, 2]

        image_x = image_point[0]
        image_y = image_point[1]
        temp_z = 10
        x = (temp_z / fx) * (image_x - cx)
        y = (temp_z / fy) * (image_y - cy)
        vect1 = np.array([x, y, temp_z])

        middle_x = cx
        middle_y = cy
        x = (temp_z / fx) * (middle_x - cx)
        y = (temp_z / fy) * (middle_y - cy)
        vect2 = np.array([x, y, temp_z])
        pixel_angle = angle_arccos(vect1, vect2) * 180 / np.pi

        # pixel_angle = 8.34  # ° between optical center (cx, cy) and (cx, cy + cy / 2)

        camera_angle_to_horizontal_plane = self.get_camera_angle_to_horizontal_plane(idx) * 180 / np.pi
        theta = (camera_angle_to_horizontal_plane + pixel_angle) * np.pi / 180

        h = np.round(self.df['pose'][idx].inv().get_trans()[2], 3)
        h = (h if h > 0 else 0) + 0.1  # Todo to complete

        true_z = h / np.cos(np.pi / 2 - theta)

        x = (true_z / fx) * (image_x - cx)
        y = (true_z / fy) * (image_y - cy)

        return np.array([x, y, true_z])

    def _is_out_of_image(self, image_point):
        assert len(image_point) == 2
        return not ((0 <= image_point[0] < self.frames_width) and (0 <= image_point[1] < self.frames_height))

    def _space2image(self, point3d):
        u, v, w = (self.p @ np.append(point3d, 1)).flatten()
        return u / w, v / w

    def p3_generator(self):
        """
        p3 are points expressed in an image with a coordinate system expressed by ROS.
        cf. http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Image.html
        and http://wiki.ros.org/image_pipeline/CameraInfo

        :return: p3 points list.
        """
        drone_0_tf_flu0 = self.df['pose'][0]  # in Reference Frame flu0 to drone
        drone_1_tf_flu0 = drone_0_tf_flu0
        cx, cy = self.p[0, 2], self.p[1, 2]
        p = (cx, cy + cy / 2)
        p1 = self.image23d(0, p)
        p3_list = []
        progress = Progress(len(self.df), 'Computing p3 points')
        for i in range(len(self.df)):
            drone_i_tf_flu0 = self.df['pose'][i]
            r1cam_tf_flu0 = self.camera_tf_drone @ drone_1_tf_flu0
            ricam_tf_flu0 = self.camera_tf_drone @ drone_i_tf_flu0
            ricam_tf_r1cam = ricam_tf_flu0 @ r1cam_tf_flu0.inv()
            p2 = ricam_tf_r1cam @ p1
            p3 = self._space2image(p2)
            if self._is_out_of_image(p3):
                drone_1_tf_flu0 = self.df['pose'][i]
                p1 = self.image23d(i, p)  # RF camera
                p3 = (40, 40)
            p3_list.append(p3)
            progress.update_pgr()
        self.p3 = p3_list
        # p3df = pd.DataFrame({'x': [e[0] for e in p3_list],
        #                      'y': [e[1] for e in p3_list]})
        # pickle.dump(self.data, open(ABSOLUTE_PATH + self.flight_name + 'clean_data/data.pkl', 'wb'))

    def p32video(self, fps: int, saving_path: str):
        codec = VideoWriter_fourcc(*'H264')
        # name = ABSOLUTE_PATH + 'error_estimation_videos/' + part1 + '_' + part2 + '_error_estimation.mkv'
        video = VideoWriter(saving_path, codec, float(fps), (self.frames_width, self.frames_height), True)
        print(f'Video will be saved at : {saving_path}')

        prg = Progress(len(self.df), 'Creating video')
        shape = (self.frames_height, self.frames_width, 3)
        for i, measure in enumerate(self.df['image']):
            p = np.int32(np.round(self.p3[i]))
            color_frame = np.uint8(np.zeros(shape))
            for j in range(3):
                color_frame[:, :, j] = measure
            fig = plt.figure()
            ax = plt.axes([0, 0, 1, 1])
            ax.imshow(color_frame)
            ax.scatter(p[0], p[1], color='r')
            xyz = self.df['pose'][i].inv().get_trans()
            angles = Rotation.from_matrix(self.df['pose'][i].inv().get_rot()).as_euler('xyz', degrees=True)
            legend = (f'Drone pose - FLU0\n'
                      f'      x : {xyz[0]:8.3f}m\n'
                      f'      y : {xyz[1]:8.3f}m\n'
                      f'      z : {xyz[2]:8.3f}m\n'
                      f'Rot x : {angles[0]:8.3f}°\n'
                      f'Rot y : {angles[1]:8.3f}°\n'
                      f'Rot z : {angles[2]:8.3f}°')
            ax.annotate(legend, (600, 40), xycoords='data', size=12, ha='right', va='top',
                        bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
            ax.axis('off')
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
            plt.close('all')

            video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
            prg.update_pgr()

            # plt.imshow(data)
            # plt.show()

        video.release()
        cv2.destroyAllWindows()
        # del video


def tara_slider_plot(images: pd.DataFrame, topic: str, ind_inf: int, ind_sup: int):
    """
    Function to be used in main with main's variables as argument after tara import.
    Plot's all tara images.

    :param images: DataFrame with 'time' and 'image' columns.
    :param topic: topic containing images.
    :param ind_inf: Inf of the interval where the data will be plotted.
    :param ind_sup: Sup of the interval where the data will be plotted.
    """
    # tara synchronizing plot
    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(bottom=0.25)
    i = 0
    ax.imshow(images['image'][i])
    # ax.axis('off')
    legend = f'Index : {i}\nTime : {images["image"][i]}'
    ann = ax.annotate(legend, (600, 50), xycoords='data', size=12, ha='right', va='top',
                      bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
    axcolor = 'lightgoldenrodyellow'
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    index_slider = Slider(ax_slider, 'Frame Id', ind_inf, ind_sup, valinit=0, valstep=1)

    def update(val):
        j = index_slider.val
        ax.imshow(images['image'][j])
        legend1 = f'Index : {j}\nTime : {images["image"][j]:0.3f}'
        global ann
        ann.remove()
        ann = ax.annotate(legend1, (600, 50), xycoords='data', size=12, ha='right', va='top',
                          bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
        fig.canvas.draw()

    index_slider.on_changed(update)
    plt.show()


def mc_scissors(mc_measure: pd.DataFrame, ind_inf: int, ind_sup: int, saving_path: str = None):
    """
    Function to be used in main with main's variables as argument after motion capture measures import.
    Plot the distance between two points of the scissors over time.

    :param mc_measure: data imported in main().
    :param ind_inf: Inf of the interval where the data will be plotted.
    :param ind_sup: Sup of the interval where the data will be plotted.
    :param saving_path: The path to save the plot.
    """
    # mc synchronizing plot
    # Finding first synchro - scissors
    fig, ax = plt.subplots(1, 1)
    # _, ind_inf = merge_two_arrays(ind_inf, mc_measure['time'])
    # _, ind_sup = merge_two_arrays(ind_sup, mc_measure['time'])
    mc_time_zoomed = mc_measure['time'][ind_inf:ind_sup]
    diff_x = mc_measure['clapet_sup_2_x'][ind_inf:ind_sup] - mc_measure['clapet_inf2_x'][ind_inf:ind_sup]
    diff_y = mc_measure['clapet_sup_2_y'][ind_inf:ind_sup] - mc_measure['clapet_inf2_y'][ind_inf:ind_sup]
    diff_z = mc_measure['clapet_sup_2_z'][ind_inf:ind_sup] - mc_measure['clapet_inf2_z'][ind_inf:ind_sup]
    scissors = np.hstack([diff.to_numpy().reshape((-1, 1)) for diff in [diff_x, diff_y, diff_z]])
    # ax.plot(mc_time_zoomed, np.linalg.norm(scissors, axis=1), label='Scissors length')
    ax.plot(np.linalg.norm(scissors, axis=1), label='Scissors length')
    ax.set_xlabel('(s)')
    ax.set_ylabel('(m)')
    ax.legend()
    if saving_path is not None:
        plt.savefig(saving_path + 'mc_scissors.png')
    plt.show()


def mc_drone_flu0(mc_measure: pd.DataFrame, ind_inf: int, ind_sup: int, saving_path: str = None):
    """
    Function to be used in main with main's variables as argument after motion capture measures import.
    Plot the position of the drone relative to flu0 over time.

    :param mc_measure: data imported in main().
    :param ind_inf: Inf of the interval where the data will be plotted.
    :param ind_sup: Sup of the interval where the data will be plotted.
    :param saving_path: The path to save the plot.

    """
    # Finding second synchro
    fig, axs = plt.subplots(3, 1)
    # _, ind_inf = merge_two_arrays(ind_inf, mc_measure['time'])
    # _, ind_sup = merge_two_arrays(ind_sup, mc_measure['time'])
    mc_time_zoomed = mc_measure['time'][ind_inf:ind_sup]
    position = np.vstack([mc_measure['pose'][i].inv().get_trans() for i in mc_measure.index])
    for i, (ax, name) in enumerate(zip(axs, ['x', 'y', 'z'])):
        # ax.plot(mc_time_zoomed, position[ind_inf:ind_sup, i], label=name)
        ax.plot(range(ind_inf, ind_sup), position[ind_inf:ind_sup, i], label=name)
        ax.set_xlabel('(s)')
        ax.set_ylabel('(m)')
        ax.legend()
    fig.suptitle('Drone position over time relative to FLU a time 0')
    if saving_path is not None:
        plt.savefig(saving_path + 'mc_drone_flu0.png')
    plt.show()


def build_video(saving_path: str, image_df: pd.DataFrame, fps: int):
    """
    Create a video. Printing the time (indicated in image_df) and index of each the frame in the video.
    :param saving_path: Path where to save the video.
    :param image_df: DataFrame with 'image' an 'time' column.
    :param fps: The number of fps.
    :return: A video saved in saving_path.
    """
    codec = VideoWriter_fourcc(*'H264')
    # name = ABSOLUTE_PATH + 'error_estimation_videos/' + part1 + '_' + part2 + '_error_estimation.mkv'
    frames_height, frames_width = image_df['image'][0].shape
    video = VideoWriter(saving_path, codec, float(fps), (frames_width, frames_height), True)
    print(f'Video will be saved at : {saving_path}')

    prg = Progress(len(image_df), 'Creating video')
    shape = (frames_height, frames_width, 3)
    for i, measure in enumerate(image_df['image']):
        color_frame = np.uint8(np.zeros(shape))
        for j in range(3):
            color_frame[:, :, j] = measure
        fig = plt.figure()
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(color_frame)
        legend = (f'Camera\n'
                  f'  index : {image_df.index[i]:8}\n'
                  f'   time : {image_df["time"][i]:8.3f}s')
        ax.annotate(legend, (600, 40), xycoords='data', size=12, ha='right', va='top',
                    bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
        ax.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
        plt.close('all')

        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
        prg.update_pgr()
    video.release()
    cv2.destroyAllWindows()


def aff3d(xyz_array, quat_array, video_path):
    assert len(xyz_array) == len(quat_array)
    print(video_path)
    x_pos = np.array([1, 0, 0])
    y_pos = np.array([0, 1, 0])
    z_pos = np.array([0, 0, 1])
    # minx, miny, minz = np.min(xyz_array, axis=0) - 1
    # maxx, maxy, maxz = np.max(xyz_array, axis=0) + 1

    width = 640
    height = 480

    codec = VideoWriter_fourcc(*'H264')
    fps = 30
    video = VideoWriter(video_path, codec, float(fps), (width, height), True)
    prg = Progress(len(xyz_array))

    for xyz, quat in zip(xyz_array, quat_array):
        ref_tf_pos = Transform().from_pose(xyz, quat).inv()

        x_ref = ref_tf_pos.get_rot() @ x_pos
        y_ref = ref_tf_pos.get_rot() @ y_pos
        z_ref = ref_tf_pos.get_rot() @ z_pos

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.quiver(xyz[0], xyz[1], xyz[2], x_ref[0], x_ref[1], x_ref[2], length=1, normalize=False, color='r')
        ax.quiver(xyz[0], xyz[1], xyz[2], y_ref[0], y_ref[1], y_ref[2], length=1, normalize=False, color='g')
        ax.quiver(xyz[0], xyz[1], xyz[2], z_ref[0], z_ref[1], z_ref[2], length=1, normalize=False, color='b')

        ax.scatter(xs=xyz[0] + 1, ys=xyz[1] + 1, zs=xyz[2] + 1, alpha=0)
        ax.scatter(xs=xyz[0] - 1, ys=xyz[1] - 1, zs=xyz[2] - 1, alpha=0)

        # ax.scatter(xs=minx, ys=miny, zs=minz, alpha=0)
        # ax.scatter(xs=maxx, ys=maxy, zs=maxz, alpha=0)

        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((height, width, 3))

        plt.close('all')
        # plt.show()

        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
        prg.update_pgr()

    video.release()
    cv2.destroyAllWindows()
    print()


if __name__ == '__main__':
    main()
