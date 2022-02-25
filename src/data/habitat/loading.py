import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as f

from typing import Dict, Tuple, Any, Union, List

from src.utils.paths import WD_PATH

DATA_HABITAT_PATH = f'{WD_PATH}data/habitat/'
MP3D_SCENE_LIST = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_scene_list.pkl', 'rb'))
GEN_DATA_KEY = ['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force']
MP3D_DATA = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/'

SPLIT = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_split.pkl', 'rb'))

TRAIN = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_train.pkl', 'rb'))
VALID = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_valid.pkl', 'rb'))
TEST = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_test.pkl', 'rb'))

FLIGHT = 'flight2'


def load_file(name: str, scene_number: int, flight_number: int):
    """
    param name: {'stats', 'state', 'motor', 'mpc_error', 'rgb', 'depth', 'surface_dist', 'ground_force', 'depth_emb',
                 'norm_emb'}
    param scene_number: In [0, 90]

    param flight_number: in Z
    """
    scene_id = MP3D_SCENE_LIST[scene_number]
    folder = f'{MP3D_DATA}{scene_id}/{FLIGHT}/{flight_number}/'
    assert name in ['stats', 'state', 'motor', 'mpc_error', 'rgb', 'depth', 'surface_dist', 'ground_force',
                    'depth_emb', 'norm_emb']
    if name == 'stats':
        return pickle.load(open(f'{folder}stats.pkl', 'rb'))
    elif name in ['depth_emb', 'norm_emb']:
        return np.load(f'{folder}{name}/{name}.npy', mmap_mode='r', allow_pickle=False)
    else:
        return np.load(f'{folder}{name}.npy', mmap_mode='r', allow_pickle=False)


def build_index_valid_flight(check_all_file: bool = True):
    index = {i: {} for i in range(len(MP3D_SCENE_LIST))}
    to_test = ['depth.npy', 'depth_emb/depth_emb.npy', 'ground_force.npy', 'motor.npy', 'mpc_error.npy',
               'norm_emb/norm_emb.npy', 'state.npy', 'stats.pkl', 'surface_dist.npy']
    for scene_number, scene_name in enumerate(MP3D_SCENE_LIST):
        try:  # Try finding flight folder
            flights = sorted([int(e) for e in next(os.walk(f'{MP3D_DATA}{scene_name}/{FLIGHT}/'))[1]])
            for number in flights:
                if check_all_file:
                    is_valid = all([os.path.isfile(f'{MP3D_DATA}{scene_name}/{FLIGHT}/'
                                                   f'{number}/{test}') for test in to_test])
                else:
                    is_valid = True
                if is_valid:  # if every needed files are present
                    stats = load_file('stats', scene_number, number)
                    if stats['done'] is not None:  # If the drone has flight at least one step
                        mpc_error = load_file('mpc_error', scene_number, number)
                        if (mpc_error < 5).all() and len(mpc_error) >= 200:  # If controller is not wrong and traj long
                            # enough
                            index[scene_number][number] = len(mpc_error)
        except StopIteration:
            continue
    pickle.dump(index, open(f'{DATA_HABITAT_PATH}mp3d_index.pkl', 'wb'))
    return index


class HabitatResidualDataset(Dataset):
    fa_mean = 0.0025
    fa_std = 0.0091

    z_mean = 0.27047664
    z_std = 0.45976904

    def __init__(self, train_valid_test: str,
                 past: int,
                 load_depth: bool,
                 load_depth_emb: bool,
                 load_norm_emb: bool,
                 normed_depth: bool):
        build_index_valid_flight()

        assert train_valid_test in ['train', 'valid', 'test']
        self.train_valid_test = train_valid_test

        assert isinstance(past, int)
        self.past = past

        self.valid_index: Dict[int, Dict[int, int]] = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_index.pkl', 'rb'))
        if self.train_valid_test == 'train':
            include = TRAIN
        elif self.train_valid_test == 'valid':
            include = VALID
        else:
            assert self.train_valid_test == 'test'
            include = TEST

        self.index = {}
        for i in include:
            if self.valid_index[i] != {}:
                self.index[i] = self.valid_index[i]
        self.index: Dict[int, Dict[int, int]]

        self.len = np.sum([e - self.past for val in self.index.values() for e in val.values()])

        self.path = []
        i = 0
        for scene_number, scene_trajectories in self.index.items():
            for flight_number, traj_len in scene_trajectories.items():
                for j in range(past, traj_len):
                    self.path.append([scene_number, flight_number, j - self.past, j])
                    i += 1

        self.load_depth = load_depth
        self.load_depth_emb = load_depth_emb
        self.load_norm_emb = load_norm_emb
        self.normed_depth = normed_depth
        self.depth_list = []
        if self.load_depth:
            self.depth_list.append('depth')
        if self.load_depth_emb:
            self.depth_list.append('depth_emb')
        if self.load_norm_emb:
            self.depth_list.append('norm_emb')

    def __len__(self):
        return self.len

    def __getitem__(self, i) -> Dict[str, torch.FloatTensor]:
        scene_number, flight_number, start, stop = self.path[i]
        to_return = {'state': torch.from_numpy(load_file('state', scene_number, flight_number)[stop - 1].copy()),
                     'motor': torch.from_numpy(load_file('motor', scene_number, flight_number)[stop - 1].copy()),
                     'ground_force': torch.from_numpy(load_file('ground_force',
                                                                scene_number,
                                                                flight_number)[stop - 1].copy()),
                     'surface_dist': torch.from_numpy(load_file('surface_dist',
                                                                scene_number,
                                                                flight_number)[stop - 1].copy())
                     }
        to_return['ground_force'] = (to_return['ground_force'] - self.fa_mean) / self.fa_std
        to_return['surface_dist'] = (to_return['surface_dist'] - self.z_mean) / self.z_std
        for key in self.depth_list:
            to_return[key] = torch.from_numpy(load_file(key, scene_number, flight_number)[start:stop].copy())

        if self.normed_depth:
            to_return['depth'] = f.resized_crop(to_return['depth'], (480 - 408) // 2, 0, 408, 640, [240, 376])

            # Divide by 20 because 20 is the ZMAX of CophyLike autoencoder trained on IRS
            to_return['depth'] = np.clip(to_return['depth'] / 20, 0, 1) * 2 - 1
        return to_return


class HabitatNormedDepthDataset(Dataset):
    def __init__(self):
        """
        Len is all the trajectories for all scenes.

        Getitem returns: {
             'depth': (n, 240, 376), all the depth for the entire trajectory

             'name': scene_number, flight_number}
        """
        build_index_valid_flight(False)
        self.valid_index: Dict[int, Dict[int, int]] = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_index.pkl', 'rb'))

        self.len = np.sum([1 for scene_traj in self.valid_index.values() for _ in scene_traj.values()])
        self.path = []
        i = 0
        for scene_number, scene_trajectories in self.valid_index.items():
            for flight_number, traj_len in scene_trajectories.items():
                self.path.append([scene_number, flight_number])
                i += 1

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        scene_number, flight_number = self.path[i]
        depth = torch.from_numpy(load_file('depth', scene_number, flight_number))
        depth_dim = f.resized_crop(depth, (480 - 408) // 2, 0, 408, 640, [240, 376])

        # Divide by 20 because 20 is the ZMAX of CophyLike autoencoder trained on IRS
        depth_norm = np.clip(depth_dim / 20, 0, 1) * 2 - 1
        return {'depth': depth_norm, 'name': self.path[i]}


def main():
    dataset = HabitatResidualDataset('train', 40, True, False, False, False)
    # dataloader = DataLoader(dataset, batch_size=int(dataset.len), shuffle=False, num_workers=1, pin_memory=True,
    #                         drop_last=False)
    a = dataset[0]
    print()

    index = build_index_valid_flight()
num_traj = 0
num_not_empty = 0
num_len = 0
for scene_number, flight in index.items():
    if flight != {}:
        num_not_empty += 1
    for flight_number, len in flight.items():
        num_traj += 1
        num_len += len


    # from tqdm import tqdm
    # print('Instanciating dataset', flush=True)
    # dataset = HabitatNormedDepthDataset()
    # print('Computing max depth', flush=True)
    # for i in tqdm(range(len(dataset))):
    #     a = dataset[i]


if __name__ == '__main__':
    main()
