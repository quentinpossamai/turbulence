import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import imageio
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal
from scipy.spatial.transform import Rotation
import torch
from tqdm import tqdm

# import src.data.euroc_mav.loading as euroc
# from src.drone_models.analytical_models.hexarotor_4params import DynamicalModel
# from src.drone_models.residual.gru_cnn import GRUCNNResidual
from src.utils.paths import mkdir_if_not_exist, ANALYSIS_DATA_VIS_EUROC_MAV, ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING

matplotlib.use('TKAgg')
mkdir_if_not_exist([ANALYSIS_DATA_VIS_EUROC_MAV])


def plot_butterworth(quat, den_quat, fc_quat, period):
    # Butterworth quaternion
    wn_list = range(1, 15)
    n_list = range(1, 15)
    den_quat_butter = np.zeros([len(n_list)] + [len(wn_list)] + list(quat.shape))
    for n_ind, n in enumerate(n_list):
        for wn_ind, wn in enumerate(wn_list):
            sos = signal.butter(N=n, Wn=wn, btype='lowpass', fs=1 / period, analog=False, output='sos')
            for i in range(4):
                den_quat_butter[n_ind, wn_ind, :, i] = signal.sosfilt(sos, quat[:, i])
    den_quat_butter = den_quat_butter / np.linalg.norm(den_quat_butter, axis=-1)[..., np.newaxis]  # Normalize

    i = 0
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    plt.plot(quat[:, i], label='gt')
    plt.plot(den_quat[:, i], label=f'ideal fc_quat={fc_quat}')
    line = plt.plot(den_quat_butter[0, 0, :, i], label=f'butter')
    plt.grid()
    plt.legend()
    axcolor = 'lightgoldenrodyellow'

    # Make a vertically oriented slider to control the amplitude
    ax_n = plt.axes([0.05, 0.1, 0.0225, 0.83], facecolor=axcolor)
    n_slider = Slider(ax=ax_n,
                      label="butter order",
                      valmin=n_list[0],
                      valmax=n_list[-1],
                      valinit=n_list[0],
                      orientation="vertical")
    ax_wn = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    wn_slider = Slider(ax=ax_wn,
                       label="butter frequency",
                       valmin=wn_list[0],
                       valmax=wn_list[-1],
                       valinit=wn_list[0],
                       orientation="horizontal")

    # The function to be called anytime a slider's value changes
    def update(_):
        line[0].set_ydata(den_quat_butter[int(np.round(n_slider.val)),
                          int(np.round(wn_slider.val)), :, i])
        fig.canvas.draw_idle()

    # register the update function with each slider
    n_slider.on_changed(update)
    wn_slider.on_changed(update)
    print()
    plt.show()


def main():
    pass


if __name__ == '__main__':
    main()
