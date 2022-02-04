import matplotlib.pyplot as plt
import numpy as np
from Dataloaders.EurocMAVLoader import EurocMavLoader
from Dataloaders.IRSLoader import IRSLoader
from Models.model_CoPhy_like import Autoencoder
from Models.model_IRS_like import NormNetS
import torch
from torch.utils.data import DataLoader

plt.style.use("seaborn")
depth_irs_irs = {32: [0.3248, 17.04],
                 64: [0.2901, 17.78],
                 128: [0.2692, 18.36],
                 256: [0.2486, 19.16],
                 512: [0.2236, 19.78]}

depth_irs_eurocmav = {32: [0.3080, 11.86],
                      64: [0.2699, 12.05],
                      128: [0.2761, 12.02],
                      256: [0.2639, 12.10],
                      512: [0.2858, 11.97]}

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def draw_pointcloud(fig):
    ax = [fig.add_subplot(4, 2, 1), fig.add_subplot(4, 2, 2)]

    ax[0].set_title("Depth2Normal IRS->IRS", fontsize=10, pad=0)
    ax[1].set_title("Depth2Normal IRS->EurocMav", fontsize=10, pad=0)
    for k in depth_irs_irs.keys():
        x_irs, y_irs = depth_irs_irs[k]
        x_euroc, y_euroc = depth_irs_eurocmav[k]
        ax[0].scatter(x_irs, y_irs, c="black")
        ax[1].scatter(x_euroc, y_euroc, c="black")

        ax[0].text(x_irs * 1.005, y_irs * 1.005, str(k))
        ax[1].text(x_euroc * 1.005, y_euroc * 1.005, str(k))

        ax[0].set_ylim(15, 22)
        ax[0].set_xlim(0.2, 0.35)
        ax[1].set_ylim(11, 13)
        ax[1].set_xlim(0.25, 0.32)

    ax[0].set_xlabel("Cross Product", fontsize=8)
    ax[0].set_ylabel("PSNR", fontsize=8)
    ax[1].set_xlabel("Cross Product", fontsize=8)
    ax[1].set_ylabel("PSNR", fontsize=8)


def get_img(z, dataset):
    dataloader = dataset(split='test')

    model = Autoencoder(feature_size=z, in_channel=1, out_channel=3).to(device)
    try:
        state_dict = torch.load(
            f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/autoencoder_normal_irs_{z}_irs.nn",
            map_location=device)
    except FileNotFoundError:
        state_dict = torch.load(
            f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/autoencoder_normal_irs_{z}_irs.nn",
            map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        depthmap = dataloader[0]['depth'].to(device).unsqueeze(0)
        out = model(depthmap)
    return (out["output"][0] + 1) / 2


def draw_irs(fig):
    img_irs = [(IRSLoader(split="test")[0]['normal'].numpy().transpose(1, 2, 0) + 1) / 2]
    for z in depth_irs_irs.keys():
        img_irs.append(get_img(z, IRSLoader).numpy().transpose(1, 2, 0))

    axs = [fig.add_subplot(3, 6, 6 + i) for i in range(1, 7)]

    titles = ("Ground Truth", "Z=32", "Z=64", "Z=128", "Z=256", "Z=512")
    for i, img in enumerate(img_irs):
        axs[i].imshow(img, cmap=plt.get_cmap("viridis"))
        axs[i].axis("off")
        axs[i].set_title(titles[i], fontsize=8, pad=0)


def draw_euroc(fig):
    img_irs = [(EurocMavLoader(split="test")[0]['normal'].numpy().transpose(1, 2, 0) + 1) / 2]
    for z in depth_irs_irs.keys():
        img_irs.append(get_img(z, EurocMavLoader).numpy().transpose(1, 2, 0))

    axs = [fig.add_subplot(3, 6, 12 + i) for i in range(1, 7)]

    titles = ("Ground Truth", "Z=32", "Z=64", "Z=128", "Z=256", "Z=512")
    for i, img in enumerate(img_irs):
        axs[i].imshow(img, cmap=plt.get_cmap("viridis"))
        axs[i].axis("off")
        axs[i].set_title(titles[i], fontsize=8, pad=0)


if __name__ == '__main__':
    fig = plt.figure()
    draw_pointcloud(fig)
    draw_irs(fig)
    draw_euroc(fig)
    plt.show()
