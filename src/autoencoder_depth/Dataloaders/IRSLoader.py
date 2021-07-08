from __future__ import print_function, division
import os
import OpenEXR
import Imath
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from torchvision import transforms
from IRS.Dataloaders.utils import *


class IRSDataset(Dataset):

    def __init__(self, split='test', scale_size=(576, 960), with_stereo=False):
        """
        Args:
            txt_file [string]: Path to the image list
            transform (callable, optional): Optional transform to be applied                on a sample
        """

        self.dataloc = "/Volumes/Samsung_T5/IRS/"
        if not os.path.isdir(self.dataloc):
            self.dataloc = "/home/sjanny/IRS/"

        try:
            file = f"splits/{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')
        except FileNotFoundError:
            file = f"../splits/{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')

        self.scale_size = scale_size
        self.split = split
        self.with_stereo = with_stereo

        self.fx = 480.0
        self.fy = 480.0
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
            img = io.imread(filename)
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
            gt_norm = None
            if filename.endswith('exr'):
                gt_norm = load_exr(filename)
                gt_norm = gt_norm * 2.0 - 1.0

                # fix opposite normal
                m = gt_norm >= 0
                m[:, :, 0] = False
                m[:, :, 1] = False
                gt_norm[m] = - gt_norm[m]

            return gt_norm

        if self.split == 'test':
            rgb_transform = default_transform(self.scale_size)
        else:
            rgb_transform = inception_color_preproccess(self.scale_size)

        gt_disp = load_disp(gt_disp_name)
        gt_norm = load_norm(gt_norm_name)

        gt_disp = gt_disp[np.newaxis, :]
        gt_disp = torch.from_numpy(gt_disp.copy()).float()

        gt_norm = gt_norm.transpose([2, 0, 1])
        gt_norm = torch.from_numpy(gt_norm.copy()).float()

        sample = {'img_names': img_names, 'gt_disp': gt_disp,
                  'gt_norm': gt_norm}

        if self.with_stereo:
            img_left = load_rgb(img_left_name)
            img_right = load_rgb(img_right_name)
            img_left = rgb_transform(img_left)
            img_right = rgb_transform(img_right)
            sample['img_left'] = img_left
            sample['img_right'] = img_right

        return sample


def load_exr(filename):
    File = OpenEXR.InputFile(filename)
    PixType = Imath.PixelType(Imath.PixelType.FLOAT)
    DW = File.header()['dataWindow']
    CNum = len(File.header()['channels'].keys())
    if (CNum > 1):
        Channels = ['R', 'G', 'B']
        CNum = 3
    else:
        Channels = ['G']
    Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)
    Pixels = [np.fromstring(File.channel(c, PixType), dtype=np.float32) for c in Channels]
    hdr = np.zeros((Size[1], Size[0], CNum), dtype=np.float32)
    if (CNum == 1):
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
    d = IRSDataset(scale_size=(240, 376), with_stereo=False)
    x = d[0]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2)

    l = x["img_left"].numpy().transpose(1, 2, 0)
    r = x["img_right"].numpy().transpose(1, 2, 0)
    d = x["gt_disp"].numpy().transpose(1, 2, 0)
    n = x["gt_norm"].numpy().transpose(1, 2, 0)

    ax[0][0].imshow(l)
    ax[0][1].imshow(r)
    ax[1][0].imshow(d)
    ax[1][1].imshow(n)

    ax[0][0].axis("off")
    ax[0][1].axis("off")
    ax[1][0].axis("off")
    ax[1][1].axis("off")

    plt.show()
    print(x["img_left"].shape)
    print(x["img_right"].shape)
    print(x["gt_disp"].shape)
    print(x["gt_norm"].shape)
