from matplotlib.widgets import Slider
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy import signal

import src.data.euroc_mav.loading as euroc


def plot_y_x():
    flight = euroc.FLIGHT_LIST[0]
    # list[dict_keys(['timestamp', 'embedding', 'MSE', 'GradMSE'])]
    depth_emb = pickle.load(open(f"{euroc.INTERMEDIATE_DATASET_PATH}{flight}depth_embedding.pkl", "rb"))
    depth_emb = sorted(depth_emb, key=lambda x: x["timestamp"])

    # list[dict_keys(['timestamp', 'embedding', 'MSE', 'CrossProduct'])]
    norm_emb = pickle.load(open(f"{euroc.INTERMEDIATE_DATASET_PATH}{flight}normal_embedding.pkl", "rb"))
    norm_emb = sorted(norm_emb, key=lambda x: x["timestamp"])

    depth_grad_mse = np.array([e["GradMSE"] for e in depth_emb])
    depth_mse = np.array([e["MSE"] for e in depth_emb])
    norm_cross_prod = np.array([e["CrossProduct"] for e in norm_emb])
    norm_psnr = np.array([e["MSE"] for e in norm_emb])
    depth_psnr = 10 * np.log10(4 / depth_mse)
    norm_psnr = 10 * np.log10(4 / norm_psnr)

    # 1-DEPTH EMBEDDINGS
    # 1.1-Depth grad mse histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(depth_grad_mse, bins=100)
    ax.grid()
    ax.set_xlabel("Grad MSE")
    ax.set_ylabel("Number of samples")
    ax.set_title("1.1-Depth grad mse histogram")

    # 1.2-Depth PSNR histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(depth_psnr, bins=100)
    ax.grid()
    ax.set_xlabel("PSNR")
    ax.set_ylabel("Number of samples")
    ax.set_title("1.2-Depth PSNR histogram")

    # 1.3-Depth scatter grad mse, mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, depth_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Grad MSE")
    ax.set_ylabel("PSNR")
    ax.set_title("1.3-Depth scatter grad mse, PSNR")

    # 2-NORM EMBEDDINGS
    # 2.1-Norm cross product histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(norm_cross_prod, bins=100)
    ax.grid()
    ax.set_xlabel("Cross Product")
    ax.set_ylabel("Number of samples")
    ax.set_title("2.1-Norm cross product histogram")

    # 2.2-Depth PSNR histogram
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.hist(norm_psnr, bins=100)
    ax.grid()
    ax.set_xlabel("PSNR")
    ax.set_ylabel("Number of samples")
    ax.set_title("2.2-Norm PSNR histogram")

    # 2.3-Norm scatter cross product, mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(norm_cross_prod, norm_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Cross Product")
    ax.set_ylabel("PSNR")
    ax.set_title("2.3-Norm scatter cross product, PSNR")

    # 3-DEPTH-NORM COMPARISON
    # 3.1-Depth grad mse, norm mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, norm_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's grad MSE")
    ax.set_ylabel("Normal's PSNR")
    ax.set_title("3.1-Depth grad mse, norm PSNR")

    # 3.2-Depth grad mse, cross product
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_grad_mse, norm_cross_prod, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's grad MSE")
    ax.set_ylabel("Norm's cross product")
    ax.set_title("3.2-Depth grad mse, norm cross product")

    # 3.3-Depth PSNR, norm mse
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_psnr, norm_psnr, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's PSNR")
    ax.set_ylabel("Normal's PSNR")
    ax.set_title("3.3-Depth PSNR, norm PSNR")

    # 3.4-Depth PSNR, norm's cross product
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    ax.scatter(depth_psnr, norm_cross_prod, s=0.5)
    ax.grid()
    ax.set_xlabel("Depth's PSNR")
    ax.set_ylabel("Norm's cross product")
    ax.set_title("3.4-Depth PSNR, norm cross product")
    plt.show()


def plot_butterworth(quat, den_quat, fc_quat, period):
    # Butterworth quaternion
    wn_list = range(1, 15)
    n_list = range(1, 15)
    den_quat_butter = np.zeros([len(n_list)] + [len(wn_list)] + list(quat.shape))
    for n_ind, n in enumerate(n_list):
        for wn_ind, wn in enumerate(wn_list):
            sos = signal.butter(N=n, Wn=wn, btype='lowpass', fs=1 / period, analog=False, output='sos')
            # sos = signal.butter(N=n, Wn=wn, btype='lowpass', analog=True, output='sos')
            for i in range(4):
                den_quat_butter[n_ind, wn_ind, :, i] = signal.sosfilt(sos, quat[:, i])
    den_quat_butter = den_quat_butter / np.linalg.norm(den_quat_butter, axis=-1)[..., np.newaxis]  # Normalize

    i = 0
    fig, ax = plt.subplots(1, 1, figsize=(19.2, 10.8), dpi=100)
    plt.plot(quat[:, i], label='gt')
    plt.plot(den_quat[:, i], label=f'ideal {fc_quat=}')
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
    def update(val):
        line[0].set_ydata(den_quat_butter[int(np.round(n_slider.val)),
                                          int(np.round(wn_slider.val)), :, i])
        fig.canvas.draw_idle()

    # register the update function with each slider
    n_slider.on_changed(update)
    wn_slider.on_changed(update)
    print()


def main():
    plot_y_x()


if __name__ == '__main__':
    main()
