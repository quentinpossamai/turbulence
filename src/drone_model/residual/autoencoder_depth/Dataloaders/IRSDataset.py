import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as f
from tqdm import tqdm
from typing import Tuple, List

import src.utils.path as path


def default_transform(scale_size=None):
    rgb_list = [
        transforms.ToTensor(),
    ]
    if scale_size is not None:
        rgb_list.append(transforms.Resize(scale_size))
    return transforms.Compose(rgb_list)


class IRSDataset(Dataset):

    def __init__(self, split: str, list_to_load: List[str]):
        """
        Args:
            split: 'train' or 'test'.
            list_to_load: A list containing any of the following elements: [left, right, depth, normal] to load.
        """
        assert split in ['train', 'test']
        self.split = split
        self.irs_path = f'{path.WD_PATH}data/IRS/'
        self.data_path = f"{self.irs_path}IRSDataset/"

        assert len(np.unique(list_to_load)) == len(list_to_load)
        assert [e in ['left', 'right', 'depth', 'normal'] for e in list_to_load]
        self.list_to_load = list_to_load

        if split == 'train':
            with open(f'{self.data_path}list/IRSDataset_TRAIN.list', "r") as file:  # 87678
                self.index = file.read().split('\n')
        else:
            with open(f'{self.data_path}list/IRSDataset_TEST.list', "r") as file:  # 15638
                self.index = file.read().split('\n')

        self.focal = 480.0  # confirmed by author
        self.sx = 540
        self.sy = 960

        # define the IRS dataset baseline distance
        self.baseline = 0.1  # confirmed by author

        def resized_crop(x):
            x = x.permute(2, 0, 1)  # [C, H, W]
            x = f.resized_crop(x, 0, (960 - 720) // 2, 540, 720, [480, 640])
            x = x.permute(1, 2, 0)  # [H, W, C]
            return x

        def load_rgb(x):
            rgb = cv2.imread(f'{x}')[..., ::-1]
            assert rgb.shape == (540, 960, 3)
            rgb = torch.from_numpy(rgb.copy())
            return resized_crop(rgb)

        def load_depth(x):
            disparity = cv2.imread(f'{x}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            assert disparity.shape == (540, 960, 3)
            assert np.linalg.norm(disparity[..., 0]) == 0
            assert np.linalg.norm(disparity[..., 2]) == 0
            disparity = disparity[..., 1]
            depth = self.baseline * self.focal / disparity
            depth = torch.from_numpy(depth.copy()).unsqueeze(-1)
            return resized_crop(depth).squeeze()

        def load_normal(x):
            normal = cv2.imread(f'{x}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            assert normal.shape == (540, 960, 3)
            normal = torch.from_numpy(normal.copy())
            normal = resized_crop(normal)
            normal = normal / np.linalg.norm(normal, axis=2, keepdims=True)
            return normal

        self.loader = {'left': lambda x: load_rgb(x),
                       'right': lambda x: load_rgb(x),
                       'depth': lambda x: load_depth(x),
                       'normal': lambda x: load_normal(x)}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
        left and right [0., 255.]
        depth R+
        normal [0., 1.]
        """
        img_names = self.index[idx].rstrip().split()  # from txt to list of 4 images: left, right, disparity, normal
        # self.irs_path = f'{path.WD_PATH}data/irs/'
        # img_names = ['l_9.png', 'r_9.png', 'd_9.exr', 'n_9.exr']
        map_path = {'left': os.path.join(self.irs_path, img_names[0]),
                    'right': os.path.join(self.irs_path, img_names[1]),
                    'depth': os.path.join(self.irs_path, img_names[2]),
                    'normal': os.path.join(self.irs_path, img_names[3])}

        to_return = {}
        for key in self.list_to_load:
            to_return[key] = self.loader[key](map_path[key])

        return to_return


def _test():
    dataset = IRSDataset('train', ['left', 'right', 'depth', 'normal'])
    import matplotlib.pyplot as plt
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        for key in sample:
            assert sample[key].shape in [(480, 640, 3), (480, 640)]
            if i < 10:
                fig, axs = plt.subplots(2, 2)
                axs[0, 0].imshow(sample['left'])
                axs[0, 1].imshow(sample['right'])
                axs[1, 0].imshow(torch.clip(sample['depth'], 0, 10))
                axs[1, 1].imshow(sample['normal'])
                plt.savefig(f'{dataset.data_path}fig{i}.png')


def main():
    _test()
    # d = IRSLoader(scale_size=(240, 376), with_stereo=True)
    # x = d[0]['normal']
    #
    # print((x ** 2).sum(0))
    # print(x.min(), x.max())
    # print(x.shape)


if __name__ == '__main__':
    main()
