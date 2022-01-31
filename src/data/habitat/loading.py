import argparse
import cv2
import habitat_sim
import magnum
import numpy as np
import pickle
import sys
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Tuple, Any, Union

from src.data.habitat.habitat_simulation import GEN_DATA_KEY, DATA_HABITAT_PATH, MP3_SCENE_LIST
from src.data.habitat.mpc import mpc_controller
from src.data.habitat.hatbitat2flu import Habitat2FLU
from src.utils.paths import WD_PATH, mkdir_if_not_exist
from src.utils.trajectory import trace_trajectory3d
from src.utils.transform import Transform


def load_trajectory(scene_number, traj_number):
    scene_id = MP3_SCENE_LIST[scene_number]
    folder = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/{scene_id}/flight/{traj_number}/'
    return {'state': np.load(f'{folder}state.npy'),
            'motor': np.load(f'{folder}motor.npy'),
            'mpc_error': np.load(f'{folder}mpc_error.npy'),
            'camera': np.load(f'{folder}camera.npy'),
            'depth': np.load(f'{folder}depth.npy'),
            'surface_dist': np.load(f'{folder}surface_dist.npy'),
            'ground_force': np.load(f'{folder}ground_force.npy')}


class HabitatDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass


def main():
    load_trajectory(0, 0)
    pass


if __name__ == '__main__':
    main()
