from torch.utils.data import Dataset
import src.data.habitat.loading as Habitat
import src.data.euroc_mav.loading as EurocMAV
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

import torchvision.transforms.functional as f


INTERMEDIATE_DATASET_PATH = EurocMAV.INTERMEDIATE_DATASET_PATH


# INTERMEDIATE_DATASET_PATH = "/Volumes/Samsung_T5/EurocMav/"


class EurocMAVDatasetHorizon2(Dataset):
    fa_mean = Habitat.HabitatResidualDataset.fa_mean
    fa_std = Habitat.HabitatResidualDataset.fa_std

    z_mean = Habitat.HabitatResidualDataset.z_mean
    z_std = Habitat.HabitatResidualDataset.z_std

    def __init__(self, train_valid_test: str,
                 past: int,
                 load_depth: bool,
                 load_depth_emb: bool,
                 load_norm_emb: bool,
                 normed_depth: bool):
        self.train_valid_test = train_valid_test

        self.splits = [("V1_01_easy/", i) for i in range(25800 + past * 10, 27169, 10)] + [
            ("V1_02_medium/", i) for i in range(14000 + past * 10, 15537, 10)] + \
                      [("V1_03_difficult/", i) for i in range(18600 + past * 10, 19333, 10)]

        self.data = {key: pickle.load(open(INTERMEDIATE_DATASET_PATH + key + "/data.pkl", 'rb')) for key in
                     ["V1_01_easy/",
                      "V1_02_medium/",
                      "V1_03_difficult/"]}
        try:
            self.heights = pickle.load(open(INTERMEDIATE_DATASET_PATH + "depth_iros_test/height.pkl", 'rb'))
        except FileNotFoundError:
            raise FileNotFoundError("Tu as pas copiere-collere depth_iros_test depuis /storage vers ton data...")
        self.past = past
        self.load_depth = load_depth
        self.load_depth_emb = load_depth_emb
        self.load_norm_emb = load_norm_emb
        self.normed_depth = normed_depth

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, item):
        flight, idx = self.splits[item]
        data = self.data[flight]

        data_isolated = {key: data[key][idx] for key in data.keys()}
        data_isolated["flight"] = flight

        depth = []
        for i in range(idx - (self.past - 1) * 10, idx + 10, 10):
            t = data['t'][i]
            depth.append(np.load(
                INTERMEDIATE_DATASET_PATH + 'depth_iros_test/' + flight + str(t).replace('.', '_') + '.npy'))
        t = data['t'][idx]
        idx = list(self.heights[flight][:, 0]).index(t)
        heights = self.heights[flight][idx, 1]

        depth = torch.from_numpy(np.stack(depth, axis=0).astype(np.float32))
        if self.normed_depth:
            depth = f.resized_crop(depth, (480 - 408) // 2, 0, 408, 640, [240, 376])
            depth = np.clip(depth / 20, 0, 1) * 2 - 1

        heights = (heights - EurocMAVDatasetHorizon2.z_mean) / EurocMAVDatasetHorizon2.z_std
        data_isolated['residual'] = ((data_isolated["residual"] - EurocMAVDatasetHorizon2.fa_mean)
                                     / EurocMAVDatasetHorizon2.fa_std)

        to_return = {'state': torch.from_numpy(data_isolated["state"].astype(np.float32)),
                     'motor': torch.from_numpy(data_isolated["motor"].astype(np.float32)),
                     'ground_force': torch.from_numpy(data_isolated["residual"].astype(np.float32)),
                     'surface_dist': torch.from_numpy(np.array(heights).astype(np.float32)).view(1),
                     'depth': depth}

        return to_return


def estimate_cheeseman():
    data = {key: pickle.load(open(INTERMEDIATE_DATASET_PATH + key + "/data.pkl", 'rb')) for key in
            ["V1_01_easy/",
             "V1_02_medium/",
             "V1_03_difficult/"]}
    heights = pickle.load(open(INTERMEDIATE_DATASET_PATH + "depth_iros_test/height.pkl", 'rb'))

    from src.utils.paths import WD_PATH
    z_hat = pickle.load(open(f'{WD_PATH}z_hat_fine_euroc_mav.pkl', 'rb'))
    fa_hat = pickle.load(open(f'{WD_PATH}fa_hat_fine_euroc_mav.pkl', 'rb'))

    fa, n = [], []
    for key in heights.keys():
        for i in range(heights[key].shape[0]):
            idx = list(data[key]['t']).index(heights[key][i, 0])
            fa.append(data[key]['residual'][idx])
            n.append(np.sum(data[key]["motor"][idx] ** 2))
        break

    n = np.stack(n, axis=0).reshape((-1, 1))
    fa = np.stack(fa, axis=0)

    height = np.stack(heights["V1_01_easy/"][:, 1], axis=0).reshape((-1, 1))

    # height = np.clip(height, 0.5, 100000)
    faz = fa[:, 2:3]

    def error(x):
        fa = np.sum((faz - x[0] * n / (height ** 2 - x[1])) ** 2)
        return fa

    import scipy.optimize
    sol = scipy.optimize.fmin(error, np.ones((2)) * 1e-2)

    print(sol)
    import matplotlib.pyplot as plt

    fa, n = [], []
    for key in heights.keys():
        for i in range(heights[key].shape[0]):
            idx = list(data[key]['t']).index(heights[key][i, 0])
            fa.append(data[key]['residual'][idx])
            n.append(np.sum(data[key]["motor"][idx] ** 2))
    n = np.stack(n, axis=0).reshape((-1, 1))
    fa = np.stack(fa, axis=0)[:, 2]
    height = np.concatenate([heights[key][:, 1] for key in heights.keys()], axis=0).reshape((-1, 1))
    fapred = sol[0] * n / (height ** 2 - sol[1])
    plt.plot(fa)
    plt.plot(fapred)
    plt.plot(fa_hat)
    plt.show()

    plt.plot(z_hat)
    plt.plot(height)
    print(sol)


if __name__ == '__main__':
    estimate_cheeseman()
