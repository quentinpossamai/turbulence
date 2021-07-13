from __future__ import print_function, division
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from Dataloaders.utils import *

ZMAX = 20


class IRSLoader(Dataset):

    def __init__(self, split='test', scale_size=(240, 376), with_stereo=False):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
        """

        self.dataloc = "/Volumes/Samsung_T5/IRS/"
        if not os.path.isdir(self.dataloc):
            self.dataloc = "/home/sjanny/IRS/"

        try:
            file = f"splits/irs_{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')
        except FileNotFoundError:
            file = f"/home/sjanny/codes/turbulence/src/autoencoder_depth/splits/irs_{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')

        if 'Samsung' in self.dataloc:
            self.index = [k for k in self.index if 'Home' not in k]
        self.scale_size = scale_size
        self.split = split
        self.with_stereo = with_stereo

        self.F = 480.0
        self.sx = 540
        self.sy = 960

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        img_names = self.index[idx].rstrip().split()

        img_left_name = os.path.join(self.dataloc, img_names[0])
        img_right_name = os.path.join(self.dataloc, img_names[1])
        gt_disp_name = os.path.join(self.dataloc, img_names[2])
        gt_norm_name = os.path.join(self.dataloc, img_names[3])

        def load_rgb(filename):
            img = cv2.imread(filename)
            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.pad(img, ((0, 0), (0, 0), (0, 2)), 'constant')
                img[:, :, 1] = img[:, :, 0]
                img[:, :, 2] = img[:, :, 0]
            h, w, c = img.shape
            if c == 4:
                img = img[:, :, :3]
            return img

        def load_depth(filename):
            return cv2.imread(filename)[..., 0] / 255 * 2.0 - 1.0

        def load_norm(filename):
            normal = cv2.imread(filename) / 255 * 2.0 - 1.0
            # fix opposite normal
            # m = normal >= 0
            # m[:, :, 0:2] = False
            # normal[m] = - normal[m]
            return normal

        resize_transform = default_transform(self.scale_size)

        depthmap = resize_transform(load_depth(gt_disp_name))  # * ZMAX
        normalmap = resize_transform(load_norm(gt_norm_name))

        sample = {'img_names': img_names, 'depth': depthmap.float(),
                  'normal': normalmap.float()}

        if self.with_stereo:
            img_left = load_rgb(img_left_name)
            img_right = load_rgb(img_right_name)
            img_left = resize_transform(img_left)
            img_right = resize_transform(img_right)
            sample['img_left'] = img_left
            sample['img_right'] = img_right

        return sample


if __name__ == '__main__':
    d = IRSLoader(scale_size=(240, 376), with_stereo=True)
    x = d[0]
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # plt.imshow(x['depth'].numpy().transpose(1, 2, 0))
    # plt.figure()
    # plt.imshow(x['normal'].numpy().transpose(1, 2, 0))
    # plt.show()
