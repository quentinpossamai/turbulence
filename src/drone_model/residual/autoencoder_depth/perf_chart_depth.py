import matplotlib.pyplot as plt
import numpy as np
from Dataloaders.EurocMAVLoader import EurocMavLoader
from Dataloaders.IRSDataset import IRSLoader
from Models.model_CoPhy_like import Autoencoder
from Models.model_IRS_like import NormNetS
import torch
from torch.utils.data import DataLoader

plt.style.use("seaborn")
depth_irs_irs = {32: [3.4532, 27.34],
                 64: [3.0938, 28.77],
                 128: [2.1211, 30.71],
                 256: [1.9558, 31.40],
                 512: [1.4701, 33.30]}

depth_irs_eurocmav = {32: [0.0382, 27.02],
                      64: [0.0286, 29.51],
                      128: [0.0330, 31.73],
                      256: [0.0282, 33.21],
                      512: [0.0307, 32.52], }

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")


def draw_pointcloud(fig):
    ax = [fig.add_subplot(4, 2, 1), fig.add_subplot(4, 2, 2)]

    ax[0].set_title("Depth2Depth IRS->IRS", fontsize=10, pad=0)
    ax[1].set_title("Depth2Depth IRS->EurocMav", fontsize=10, pad=0)
    for k in depth_irs_irs.keys():
        x_irs, y_irs = depth_irs_irs[k]
        x_euroc, y_euroc = depth_irs_eurocmav[k]
        ax[0].scatter(x_irs, y_irs, c="black")
        ax[1].scatter(x_euroc, y_euroc, c="black")

        ax[0].text(x_irs * 1.005, y_irs * 1.005, str(k))
        ax[1].text(x_euroc * 1.005, y_euroc * 1.005, str(k))

        ax[0].set_ylim(25, 35)
        ax[0].set_xlim(0, 6)
        ax[1].set_ylim(25, 35)
        ax[1].set_xlim(0, 0.06)

    ax[0].set_xlabel("MSE on Gradient", fontsize=8)
    ax[0].set_ylabel("PSNR", fontsize=8)
    ax[1].set_xlabel("MSE on Gradient", fontsize=8)
    ax[1].set_ylabel("PSNR", fontsize=8)


def get_img(z, dataset):
    dataloader = dataset(split='test')

    model = Autoencoder(feature_size=z, in_channel=1, out_channel=1).to(device)
    try:
        state_dict = torch.load(
            f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/autoencoder_depth_irs_{z}.nn",
            map_location=device)
    except FileNotFoundError:
        state_dict = torch.load(
            f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/autoencoder_depth_irs_{z}.nn",
            map_location=device)

    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        depthmap = dataloader[0]['depth'].to(device).unsqueeze(0)
        out = model(depthmap)
    return out["output"][0]


def draw_irs(fig):
    img_irs = [IRSLoader(split="test")[0]['depth'].numpy().transpose(1, 2, 0)]
    for z in depth_irs_irs.keys():
        img_irs.append(get_img(z, IRSLoader).numpy().transpose(1, 2, 0))

    axs = [fig.add_subplot(3, 6, 6 + i) for i in range(1, 7)]

    titles = ("Ground Truth", "Z=32", "Z=64", "Z=128", "Z=256", "Z=512")
    for i, img in enumerate(img_irs):
        axs[i].imshow(img, cmap=plt.get_cmap("viridis"))
        axs[i].axis("off")
        axs[i].set_title(titles[i], fontsize=12, pad=0)


def draw_euroc(fig):
    img_irs = [EurocMavLoader(split="test")[0]['depth'].numpy().transpose(1, 2, 0)]
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
