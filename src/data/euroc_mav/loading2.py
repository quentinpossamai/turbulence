from torch.utils.data import Dataset
import src.data.habitat.loading as Habitat
import src.data.euroc_mav.loading as EurocMAV
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

INTERMEDIATE_DATASET_PATH = EurocMAV.INTERMEDIATE_DATASET_PATH


class EurocMAVDatasetHorizon2(Dataset):
    fa_mean = Habitat.HabitatResidualDataset.fa_mean
    fa_std = Habitat.HabitatResidualDataset.fa_std

    z_mean = Habitat.HabitatResidualDataset.z_mean
    z_std = Habitat.HabitatResidualDataset.z_std

    def __init__(self, train_valid_test: str,
                 past: int,
                 load_depth: bool,
                 load_depth_emb: bool,
                 load_norm_emb: bool):
        self.train_valid_test = train_valid_test

        self.splits = [("V1_01_easy/", i) for i in range(25800 + past * 10, 27179, 10)] + [
            ("V1_02_medium/", i) for i in range(14000 + past * 10, 15547, 10)] + [("V1_03_difficult/", i)
                                                                                  for i
                                                                                  in
                                                                                  range(18600 + past * 10,
                                                                                        19343, 10)]
        try:
            self.data = {key: pickle.load(open(INTERMEDIATE_DATASET_PATH + key + "/data.pkl", 'rb')) for key in
                         ["V1_01_easy/",
                          "V1_02_medium/",
                          "V1_03_difficult/"]}
        except FileNotFoundError:
            raise FileNotFoundError("Tu as pas copiere-collere depth_iros_test depuis /storage vers ton data...")
        self.past = past
        self.heights = pickle.load(open(INTERMEDIATE_DATASET_PATH + "depth_iros_test/height.pkl", 'rb'))

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

        depth = np.stack(depth, axis=0)

        heights = (heights - EurocMAVDatasetHorizon2.z_mean) / EurocMAVDatasetHorizon2.z_std
        data_isolated['residual'] = (data_isolated["residual"] - EurocMAVDatasetHorizon2.fa_mean) / EurocMAVDatasetHorizon2.fa_std

        to_return = {'state': torch.from_numpy(data_isolated["state"]),
                     'motor': torch.from_numpy(data_isolated["motor"]),
                     'ground_force': torch.from_numpy(data_isolated["residual"]),
                     'surface_dist': torch.from_numpy(np.array(heights)).view(1),
                     'depth': torch.from_numpy(depth)}
        # TODO : Deal with motors

        return to_return


if __name__ == '__main__':
    d = EurocMAVDatasetHorizon2("train", past=40, load_depth=False, load_depth_emb=False, load_norm_emb=False)
    timestamps = {'V1_01_easy/': [], 'V1_02_medium/': [], 'V1_03_difficult/': [], }
    for x in d:
        timestamps[x["flight"]].append(x["timestamp"])
    for key in timestamps.keys():
        timestamps[key] = np.concatenate(timestamps[key])
        timestamps[key] = np.unique(timestamps[key])
