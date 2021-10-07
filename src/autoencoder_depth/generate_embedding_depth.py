import argparse
import cv2
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.autoencoder_depth.Models.model_CoPhy_like import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--flight', default="V1_01_easy", type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--feature_size', default=128, type=int)
parser.add_argument('--name', default='autoencoder_depth_irs_128', type=str)
parser.add_argument('--weight_grad', default=1, type=int)


args = parser.parse_args()

use_cuda = True
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

MSE = nn.MSELoss()

ZMAX = 7.48


class EurocMavLoader(Dataset):
    def __init__(self, split="V1_01"):
        super(EurocMavLoader, self).__init__()
        self.dataloc = f"/Volumes/Samsung_T5/BlackBird/euroc_mav/intermediate/{split}/"
        if not os.path.isdir(self.dataloc):
            self.dataloc = f"/storage/Datasets/BlackBird/euroc_mav/intermediate/{split}/"

        self.index = os.listdir(self.dataloc + "depth_estimation")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        dir = self.dataloc + "depth_estimation/" + self.index[item]
        try:
            depth_map = torch.FloatTensor(cv2.imread(dir)[:, :, 0]).unsqueeze(0)
        except TypeError as e:
            print(self.index[item])
            raise TypeError(".DS_Store probably in data") from e
        depth_map = depth_map / 255 * 2 - 1
        depth_map = depth_map  # * ZMAX

        return {"depth": depth_map, "ex": self.index[item]}


def compute_grad(rgb):
    a = torch.Tensor([[1, 0, -1],
                      [2, 0, -2],
                      [1, 0, -1]]).view(1, 1, 3, 3).to(device)
    b = torch.Tensor([[1, 2, 1],
                      [0, 0, 0],
                      [-1, -2, -1]]).view(1, 1, 3, 3).to(device)
    sobel = (F.conv2d(rgb, a, padding=1) ** 2 + F.conv2d(rgb, b, padding=1) ** 2).squeeze(1)
    return sobel


def get_loss(gt, prediction):
    pred = prediction["output"][:, :, :gt.shape[-2], :gt.shape[-1]]
    mse_normal = MSE(gt, pred)
    mse_grad = MSE(compute_grad(gt), compute_grad(pred))
    return 1000 * mse_normal + args.weight_grad * mse_grad, mse_normal, mse_grad


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataloader = DataLoader(EurocMavLoader(split=args.flight), batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

    model = Autoencoder(feature_size=args.feature_size, in_channel=1, out_channel=1).to(device)
    try:
        state_dict = torch.load(
            # f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/{args.name}.nn",
            f"/home/qpossamai/phd/turbulence/models/autoencoder_cophylike/{args.name}.nn",
            map_location=device)
    except FileNotFoundError:
        state_dict = torch.load(
            f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/{args.name}.nn",
            map_location=device)

    model.load_state_dict(state_dict)
    model.eval()

    data = []
    for i, x in enumerate(tqdm(dataloader)):
        depthmap = x['depth'].to(device)
        ex = float(x["ex"][0].split(".")[0].replace("_", '.'))
        out = model(depthmap)
        costs = get_loss(depthmap, out)
        data.append(
            {'timestamp': ex, 'embedding': out["feature"].cpu(),
             "MSE": costs[1].cpu().detach().numpy(), "GradMSE": costs[-1].cpu()})
    pickle.dump(data, open(f"/storage/Datasets/BlackBird/{args.flight}_depth.pkl"))


if __name__ == '__main__':
    main()
