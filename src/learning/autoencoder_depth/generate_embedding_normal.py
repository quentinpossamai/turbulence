import argparse
import cv2
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.learning.autoencoder_depth.Models.model_CoPhy_like import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--flight', default="V1_01_easy", type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--feature_size', default=128, type=int)
parser.add_argument('--name', default='autoencoder_normal_irs_128_irs', type=str)

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

        dir = dir.replace("depth_estimation", "normal")
        normal_map = torch.FloatTensor(cv2.imread(dir).transpose(2, 0, 1))
        normal_map = normal_map / 255
        normal_map = normal_map * 2 - 1
        return {"depth": depth_map, "normal": normal_map, "ex": self.index[item]}



def get_loss(gt, prediction):
    mse_normal = MSE(gt, prediction["output"])
    gt_transpose = gt.permute(0, 2, 3, 1).reshape(-1, 3)
    prediction_transpose = prediction["output"].permute(0, 2, 3, 1).reshape(-1, 3)
    cross_product_loss = torch.sqrt((torch.cross(gt_transpose, prediction_transpose, dim=-1) ** 2).sum(-1)).mean()
    return 0, mse_normal, 0, cross_product_loss


def main():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataloader = DataLoader(EurocMavLoader(split=args.flight), batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=True)

    model = Autoencoder(feature_size=args.feature_size, in_channel=1, out_channel=3).to(device)
    try:
        state_dict = torch.load(
            # f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/{args.name}.nn",
            # f"/home/qpossamai/phd/turbulence/models/depth_normal_autoencoder/{args.name}.nn",
            f"/storage/Datasets/BlackBird/euroc_mav/depth_normal_autoencoder/{args.name}.nn",
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
        normal = x['normal'].to(device)
        ex = float(x["ex"][0].split(".")[0].replace("_", '.'))
        out = model(depthmap)
        costs = get_loss(normal, out)
        data.append(
            {'timestamp': ex, 'embedding': out["feature"][0].cpu().detach().numpy(),
             "MSE": costs[1].cpu().detach().numpy(), "CrossProduct": costs[-1].cpu().detach().numpy()})
    pickle.dump(data, open(f"/storage/Datasets/BlackBird/euroc_mav/"
                           f"intermediate/{args.flight}/normal_embedding.pkl", "wb"))


if __name__ == '__main__':
    main()
