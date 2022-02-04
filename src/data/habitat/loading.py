import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as f

from typing import Dict, Tuple, Any, Union, List

from src.utils.paths import WD_PATH

DATA_HABITAT_PATH = f'{WD_PATH}data/habitat/'
MP3D_SCENE_LIST = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_scene_list.pkl', 'rb'))
GEN_DATA_KEY = ['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force']
SCENE_PATH = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/'

SPLIT = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_split.pkl', 'rb'))

TRAIN = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_train.pkl', 'rb'))
VALID = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_valid.pkl', 'rb'))
TEST = pickle.load(open(f'{DATA_HABITAT_PATH}mp3d_test.pkl', 'rb'))


def load_file(name: str, scene_number: int, flight_number: int):
    scene_id = MP3D_SCENE_LIST[scene_number]
    folder = f'{SCENE_PATH}{scene_id}/flight/{flight_number}/'
    assert name in ['stats', 'state', 'motor', 'mpc_error', 'rgb', 'depth', 'surface_dist', 'ground_force']
    if name == 'stats':
        return pickle.load(open(f'{folder}stats.pkl', 'rb'))
    else:
        return np.load(f'{folder}{name}.npy', mmap_mode='r', allow_pickle=False)


def build_index_valid_flight():
    index = {i: None for i in range(len(MP3D_SCENE_LIST))}
    for scene_number, scene_name in enumerate(MP3D_SCENE_LIST):
        try:
            folders = next(os.walk(f'{SCENE_PATH}{scene_name}/flight/'))[1]
        except StopIteration:
            index[scene_number] = {}
            continue
        index[scene_number] = {}
        flight_list = np.sort(np.array(folders).astype(int))
        for flight_number in flight_list:
            stats = load_file('stats', scene_number, flight_number)
            if stats['done'] is not None:
                mpc_error = load_file('mpc_error', scene_number, flight_number)
                if (mpc_error < 5).all() and len(mpc_error) >= 200:
                    index[scene_number][flight_number] = len(mpc_error)
    # print(f'MP3D_SCENE_LIST: {MP3D_SCENE_LIST}')
    # print(f'SCENE_PATH: {SCENE_PATH}')
    # print(f'index: {index}')
    pickle.dump(index, open(f'{DATA_HABITAT_PATH}mp3d_index.pkl', 'wb'))


class ResidualDataset(Dataset):
    def __init__(self, train_valid_test: str, past: int):
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

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        scene_number, flight_number, start, stop = self.path[i]
        return {'state': torch.from_numpy(load_file('state', scene_number, flight_number)[stop - 1]),
                'motor': torch.from_numpy(load_file('motor', scene_number, flight_number)[stop - 1]),
                'depth': torch.from_numpy(load_file('depth', scene_number, flight_number)[start:stop]),
                'ground_force': torch.from_numpy(load_file('ground_force', scene_number, flight_number)[stop - 1])}


class EmbeddingDataset(Dataset):
    def __init__(self):
        build_index_valid_flight()
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
        print(f'scene_number: {scene_number} - flight_number: {flight_number} - max_depth: {torch.max(depth)}',
              flush=True)
        return {'depth': depth_dim}


def main():
    # build_index_valid_flight()
    # dataset = HabitatDataset('train', 40)
    # import time
    # tic = time.time()
    # for i in range(len(dataset)-1):
    #     a = dataset[i]['state'] * dataset[i+1]['state']
    # print(f'train {time.time() - tic}')
    from tqdm import tqdm
    print('Instanciating dataset', flush=True)
    dataset = EmbeddingDataset()
    print('Computing max depth', flush=True)
    for i in tqdm(range(len(dataset))):
        a = dataset[i]


if __name__ == '__main__':
    main()
