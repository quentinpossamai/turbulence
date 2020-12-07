from typing import Union, Iterable, Tuple, List, Dict, Any
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc

import util
import posture_error_estimation


def data_processing():
    f = util.DataFolder('data_drone2')
    flight_number = 1

    # 1 - CAMERA PARAMETERS
    tara_raw = f.pickle_load_file('.pkl', f.folders['raw_python'][flight_number], 'tara_left', True)
    topic = 'tara/left/camera_info'
    camera_parameters = {'K': tara_raw[topic][0]['K'],
                         'D': tara_raw[topic][0]['D'],
                         'fps': 30}

    # 2 - MC PREPARATION
    print('Loading data')
    # Load drone_tf_origin
    mc_raw = f.pickle_load_file('.pkl', f.folders['raw_python'][flight_number], 'mc_measure')
    mc = mc_raw[['time', 'pose']].rename({'time': 'pose_time'}, axis=1).reset_index(drop=True)
    # TODO : tf accuracy ?
    # Vincent input : (0.105, 0, 0, -1.57, 0.0, -2.0943) (convention x, y, z, roll, pitch, yaw)
    drone_d_camera = np.array([0.105, 0, 0])
    drone_rot_camera = Rotation.from_euler('xyz', np.array([-120, 0, -90]), degrees=True).as_quat()
    camera_tf_drone = util.Transformation().from_pose(drone_d_camera, drone_rot_camera).inv()
    mc['pose'] = mc['pose'].apply(lambda x: camera_tf_drone @ x)
    mc['pose_time'] = (mc['pose_time'] - mc['pose_time'][0])

    # 3 - TARA PREPARATION
    topic = 'tara/left/image_raw'
    image = pd.DataFrame({'image_time': [tara_raw[topic][idx]['t'] for idx in tara_raw[topic]],
                          'image': [tara_raw[topic][idx]['image'] for idx in tara_raw[topic]]})
    image['image'] = image['image'].apply(lambda x: np.repeat(x.reshape((x.shape[0], x.shape[1], 1)), 3, axis=2))
    # Time begin at 0
    image['image_time'] = (image['image_time'] - image['image_time'][0])

    # # 4 - ANALYSIS TODO : Synchro
    # # mc 1 : Finding first synchro the scissors
    # mc_scissors(mc_raw, ind_inf=400, ind_sup=600, saving_path=f.folders['intermediate'][flight_number])

    # mc 2 : Finding second synchro with the position
    # mc_drone_origin(mc, ind_inf=7690, ind_sup=7730)
    # mc_drone_origin(mc, ind_inf=0, ind_sup=len(mc))

    # tara 1 : Finding synchro frame by frame
    # tara_slider_plot(image, ind_inf=0, ind_sup=len(image))

    # # tara 2 : Finding synchro video
    # saving_path = f.folders['intermediate'][flight_number] + 'tara.mkv'
    # build_video(saving_path=saving_path, image_df=images, fps=30)

    # Index of mc data and drone data
    print('Synchronizing data')
    synchro = {0: {'mc': [np.nan, np.nan],
                   'image': [np.nan, np.nan]},  # Scissors closing not shown on data
               1: {'mc': [470, 7700],
                   'image': [2902, 5117]}}

    # Tara start sync
    i1, i2 = synchro[flight_number]['image']
    j1, j2 = synchro[flight_number]['mc']
    assert i1 < i2, "Synchro failed, tara indices error."
    assert j1 < j2, "Synchro failed, mc indices error."
    # img_time = time_coef * mc_time + time_bias
    time_coef = (image['image_time'][i1] - image['image_time'][i2]) / (mc['pose_time'][j1] - mc['pose_time'][j2])
    time_bias = image['image_time'][i1] - time_coef * mc['pose_time'][j1]
    mc['pose_time'] = time_coef * mc['pose_time'] + time_bias

    mc_ids, image_ids = util.merge_two_arrays(mc['pose_time'], image['image_time'])
    input_data = pd.concat([mc.loc[mc_ids],
                            image.loc[image_ids]],
                           axis=1)
    pickle.dump(input_data, open(f.folders['intermediate'][flight_number] + 'posture_error_input_data.pkl', 'wb'))
    pickle.dump(camera_parameters, open(f.folders['intermediate'][flight_number] + 'camera_parameters.pkl', 'wb'))


def error_estimation():
    f = util.DataFolder('data_drone2')
    flight_number = 1

    input_data = pickle.load(open(f.folders['intermediate'][flight_number] + 'posture_error_input_data.pkl', 'rb'))
    camera_parameters = pickle.load(open(f.folders['intermediate'][flight_number] + 'camera_parameters.pkl', 'rb'))

    estimator = posture_error_estimation.ErrorEstimation(input_data=input_data,
                                                         calibration_matrix=camera_parameters['K'],
                                                         distortion_coefficient=camera_parameters['D'],
                                                         max_distance=5)
    estimator.p3_generator()
    saving_path = f.folders['results'][flight_number] + 'pose_camera_correspondence_video.mkv'
    estimator.p32video(fps=camera_parameters['fps'], saving_path=saving_path)


class tara_slider_plot:
    def __init__(self, image: pd.DataFrame, ind_inf: int, ind_sup: int):
        """
        Function to be used in main_data_drones with main_data_drones's variables as argument after tara import.
        Plot's all tara images.

        :param image: DataFrame with 'time' and 'image' columns.
        :param ind_inf: Inf of the interval where the data will be plotted.
        :param ind_sup: Sup of the interval where the data will be plotted.

        Inspired by https://stackoverflow.com/questions/40126176/fast-live-plotting-in-matplotlib-pyplot
        """
        self.image = image
        self.ind_inf = ind_inf
        self.ind_sup = ind_sup
        self.i = 0
        height, width, _ = self.image['image'][self.i].shape

        # tara synchronizing plot
        self.fig = plt.figure(figsize=(width / 100., height / 100.))
        self.ax = self.fig.add_subplot(1, 1, 1)
        plt.subplots_adjust(bottom=0.25)
        self.img = self.ax.imshow(self.image['image'][self.i])
        legend = f'Index : {self.i}\nTime : {self.image["image_time"][self.i]:0.3f}'
        self.ann = self.ax.annotate(legend, (width - 40, 40), xycoords='data', size=12, ha='right', va='top',
                                    bbox=dict(boxstyle='round', alpha=0.5, fc='w'))

        axcolor = 'lightgoldenrodyellow'
        slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        self.slider = Slider(slider_ax, 'Frame Id', ind_inf, ind_sup, valinit=0, valstep=1)
        self.fig.canvas.mpl_connect('key_press_event', self._on_press)
        self.slider.on_changed(self._update)

        self.fig.canvas.draw()
        self.ax_background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        plt.show(block=False)

        # plt.figure(figsize=(width / 100., height / 100.))
        # plt.imshow(self.image['image'][1358])
        # plt.show()

        self._aff(self.i)

    def _aff(self, df_idx):
        # Update data
        self.img.set_data(self.image['image'][df_idx])
        self.ann.set_text(f'Index : {df_idx}\nTime : {self.image["image_time"][df_idx]:0.3f}')

        # Restore background
        self.fig.canvas.restore_region(self.ax_background)

        # redraw just what changed
        self.ax.draw_artist(self.img)
        self.ax.draw_artist(self.ann)

        # fill in the axes rectangle
        self.fig.canvas.blit(self.ax.bbox)
        self.fig.canvas.flush_events()

    def _update(self, val):
        self.i = int(self.slider.val)
        self._aff(self.i)
        self.fig.canvas.draw()

    def _on_press(self, event):
        if (event.key == 'e') and (self.i + 1 < self.ind_sup):
            self.i += 1
            self._aff(self.i)
        elif (event.key == 'a') and (self.i - 1 >= self.ind_inf):
            self.i -= 1
            self._aff(self.i)
        elif event.key == 'q':
            plt.close(self.fig)
        self.slider.set_val(self.i)
        self.fig.canvas.draw()


def mc_scissors(mc_measure: pd.DataFrame, ind_inf: int, ind_sup: int, saving_path: str = None):
    """
    Function to be used in main_data_drones with main_data_drones's variables as argument after motion capture measures
     import.
    Plot the distance between two points of the scissors over time.

    :param mc_measure: data imported in main_data_drones().
    :param ind_inf: Inf of the interval where the data will be plotted.
    :param ind_sup: Sup of the interval where the data will be plotted.
    :param saving_path: The path to save the plot.
    """
    # mc synchronizing plot
    # Finding first synchro - scissors
    fig, ax = plt.subplots(1, 1)
    xx = list(range(ind_inf, ind_sup))
    diff_x = mc_measure['clapet_sup_2_x'][ind_inf:ind_sup] - mc_measure['clapet_inf2_x'][ind_inf:ind_sup]
    diff_y = mc_measure['clapet_sup_2_y'][ind_inf:ind_sup] - mc_measure['clapet_inf2_y'][ind_inf:ind_sup]
    diff_z = mc_measure['clapet_sup_2_z'][ind_inf:ind_sup] - mc_measure['clapet_inf2_z'][ind_inf:ind_sup]
    scissors = np.hstack([diff.to_numpy().reshape((-1, 1)) for diff in [diff_x, diff_y, diff_z]])
    ax.plot(xx, np.linalg.norm(scissors, axis=1), label='Scissors length')
    ax.set_xlabel('(s)')
    if len(xx) < 100:
        ax.set_xticks(xx)
        ax.set_xticklabels(xx, rotation=45)
        ax.grid(True, axis='x')
    ax.set_ylabel('(m)')
    ax.legend()
    if saving_path is not None:
        plt.savefig(saving_path + 'mc_scissors.png')
    plt.show()


def mc_drone_origin(mc_measure: pd.DataFrame, ind_inf: int, ind_sup: int, saving_path: str = None):
    """
    Function to be used in main_data_drones with main_data_drones's variables as argument after motion capture measures
    import.
    Plot the position of the drone relative to flu0 over time.

    :param mc_measure: data imported in main_data_drones().
    :param ind_inf: Inf of the interval where the data will be plotted.
    :param ind_sup: Sup of the interval where the data will be plotted.
    :param saving_path: The path to save the plot.

    """
    # Finding second synchro
    fig, axs = plt.subplots(3, 2)
    position = np.vstack([mc_measure['pose'][i].inv().get_trans() for i in mc_measure.index])
    angle = np.vstack([mc_measure['pose'][i].inv().get_rot_euler('xyz', True) for i in mc_measure.index])
    xx = list(range(ind_inf, ind_sup))
    for i in range(2):
        if i == 0:
            to_loop = enumerate(zip([ax[0] for ax in axs], ['x', 'y', 'z']))
            y_unit = '(m)'
            to_plot = position
        else:
            to_loop = enumerate(zip([ax[1] for ax in axs], ['rot x', 'rot y', 'rot z']))
            y_unit = '(°)'
            to_plot = angle
        for j, (ax, name) in to_loop:
            # ax.plot(mc_time_zoomed, position[ind_inf:ind_sup, i], label=name)
            ax.plot(xx, to_plot[ind_inf:ind_sup, j], label=name)
            ax.set_xlabel('(s)')
            ax.set_ylabel(y_unit)
            ax.legend()
            if len(xx) < 100:
                ax.set_xticks(xx)
                ax.set_xticklabels(xx, rotation=45)
                ax.grid(True, axis='x')
    fig.suptitle('Drone position over time relative motion capture origin.')
    if saving_path is not None:
        plt.savefig(saving_path + 'mc_drone_origin.png')
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

    prg = util.Progress(len(image_df), 'Creating video')
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
    prg = util.Progress(len(xyz_array))

    for xyz, quat in zip(xyz_array, quat_array):
        ref_tf_pos = util.Transformation().from_pose(xyz, quat)

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
    data_processing()
    error_estimation()
