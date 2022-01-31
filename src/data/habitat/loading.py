import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Tuple, Any, Union

from src.data.habitat.habitat_simulation import GEN_DATA_KEY, DATA_HABITAT_PATH, MP3_SCENE_LIST


def load_trajectory(name: str, scene_number: int, flight_number: int):
    scene_id = MP3_SCENE_LIST[scene_number]
    folder = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight/{flight_number}/'
    assert name in ['state', 'motor', 'mpc_error', 'camera', 'depth', 'surface_dist', 'ground_force']
    return np.load(f'{folder}{name}.npy')


def build_index_valid_flight():
    index = {scene: {} for scene in MP3_SCENE_LIST}
    for scene_number in MP3_SCENE_LIST:
        scene_id = MP3_SCENE_LIST[scene_number]
        os.listdir(f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight/')
    flight_number = 0
    mpc_error = load_trajectory('mpc_error', 0, 0)
    if len(mpc_error) > 200:
        mpc_error




class HabitatDataset(Dataset):
    def __init__(self, train_valid_test: str, horizon: float):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def main():
    data = load_trajectory(0, 0)
    import matplotlib.pyplot as plt

    plt.plot(data['mpc_error'][:])
    print()
    pass


if __name__ == '__main__':
    main()
