from __future__ import print_function, division
import os
import OpenEXR
import Imath
import cv2
import numpy as np
from torch.utils.data import Dataset
from autoencoder_depth.Dataloaders.utils import *


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
            file = f"../splits/irs_{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')

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

        def load_disp(filename):
            return load_exr(filename)

        def load_norm(filename):
            normal = load_exr(filename) * 2.0 - 1.0
            ncv = cv2.imread(filename, cv2.IMREAD_UNCHANGED) * 2 - 1
            # fix opposite normal
            m = normal >= 0
            m[:, :, 0:2] = False
            normal[m] = - normal[m]
            return normal

        resize_transform = default_transform(self.scale_size)

        gt_disp = resize_transform(load_disp(gt_disp_name))
        normalmap = resize_transform(load_norm(gt_norm_name))

        gt_disp[gt_disp == 0] = 0.001
        depthmap = self.F * 0.1 / gt_disp

        sample = {'img_names': img_names, 'depth': depthmap,
                  'normal': normalmap}

        if self.with_stereo:
            img_left = load_rgb(img_left_name)
            img_right = load_rgb(img_right_name)
            img_left = resize_transform(img_left)
            img_right = resize_transform(img_right)
            sample['img_left'] = img_left
            sample['img_right'] = img_right

        return sample


def load_exr(filename):
    File = OpenEXR.InputFile(filename)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if CNum > 1:
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
    if CNum == 1:
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
    else:
        hdr[:, :, 0] = np.reshape(Pixels[0], (Size[1], Size[0]))
        hdr[:, :, 1] = np.reshape(Pixels[1], (Size[1], Size[0]))
        hdr[:, :, 2] = np.reshape(Pixels[2], (Size[1], Size[0]))
    h, w, c = hdr.shape
    if c == 1:
        hdr = np.squeeze(hdr)
    return hdr


if __name__ == '__main__':
    d = IRSLoader(scale_size=(240, 376), with_stereo=True)
    x = d[0]
