import argparse
import cv2
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import src.utils.paths as paths
from src.learning.autoencoder_depth.Models.model_CoPhy_like import Autoencoder
import src.data.habitat.loading as habitat

parser = argparse.ArgumentParser()
parser.add_argument('--flight', default="V1_01_easy", type=str)
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


def main_euroc_mav():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataloader = DataLoader(EurocMavLoader(split=args.flight), batch_size=1,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=False)

    model = Autoencoder(feature_size=args.feature_size, in_channel=1, out_channel=1).to(device)
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
        ex = float(x["ex"][0].split(".")[0].replace("_", '.'))
        out = model(depthmap)
        costs = get_loss(depthmap, out)
        data.append(
            {'timestamp': ex, 'embedding': out["feature"].cpu().detach().numpy(),
             "MSE": costs[1].cpu().detach().numpy(), "GradMSE": costs[-1].cpu().detach().numpy()})
    # pickle.dump(data, open(f"/storage/Datasets/BlackBird/euroc_mav/"
    #                        f"intermediate/{args.flight}/depth_embedding.pkl", "wb"))


def main_habitat():
    print(args)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataloader = DataLoader(habitat.HabitatEmbeddingDataset(), batch_size=1,
                            shuffle=False,
                            num_workers=0 if torch.cuda.is_available() else 0,
                            pin_memory=False)

    with torch.no_grad():
        model = Autoencoder(feature_size=args.feature_size, in_channel=1, out_channel=1).to(device)
        try:
            state_dict = torch.load(
                # f"/home/sjanny/codes/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/{args.name}.nn",
                # f"/home/qpossamai/phd/turbulence/models/depth_normal_autoencoder/{args.name}.nn",
                f"/storage/Datasets/BlackBird/euroc_mav/depth_normal_autoencoder/{args.name}.nn",
                map_location=device)
        except FileNotFoundError:
            state_dict = torch.load(
                # f"/Users/steevenjanny/Desktop/turbulence/src/autoencoder_depth/trained_models/CoPhy_Like/{args.name}.nn",
                f"/Users/quentin/phd/turbulence/models/depth_normal_autoencoder/{args.name}.nn",
                map_location=device)

        model.load_state_dict(state_dict)
        model.eval()

        for i, trajectory in enumerate(tqdm(dataloader)):
            depthmap = trajectory['depth'].to(device)[0].unsqueeze(1)

            depth_emb = np.empty((len(depthmap), 128), dtype=np.float32) * np.nan
            depth_mse = np.empty((len(depthmap)), dtype=np.float32) * np.nan
            depth_gradmse = np.empty((len(depthmap)), dtype=np.float32) * np.nan

            for j, depthmap_gt in enumerate(depthmap):
                depthmap_gt = depthmap_gt.unsqueeze(0)
                out = model(depthmap_gt)  # feature, output
                embedding, depthmap_hat = out['feature'][0], out['output']
                sce_num, flight_number = trajectory['name']
                sce_num = sce_num.item()
                flight_number = flight_number.item()
                folder = f'{habitat.MP3D_DATA}{habitat.MP3D_SCENE_LIST[sce_num]}/flight/{flight_number}/depth_emb/'
                paths.mkdir_if_not_exist([folder])

                mse_normal = MSE(depthmap_gt, depthmap_hat)
                mse_grad = MSE(compute_grad(depthmap_gt), compute_grad(depthmap_hat))

                # store
                depth_emb[j] = embedding.cpu().detach().numpy().astype(np.float32)
                depth_mse[j] = mse_normal.cpu().detach().numpy().astype(np.float32)
                depth_gradmse[j] = mse_grad.cpu().detach().numpy().astype(np.float32)

            assert not np.isnan(depth_emb).any()
            assert not np.isnan(depth_mse).any()
            assert not np.isnan(depth_gradmse).any()
            np.save(f'{folder}depth_emb.npy', depth_emb)
            np.save(f'{folder}depth_mse.npy', depth_mse)
            np.save(f'{folder}depth_gradmse.npy', depth_gradmse)


if __name__ == '__main__':
    main_habitat()
