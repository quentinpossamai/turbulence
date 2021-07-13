from __future__ import print_function, division
import os
import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

ZMAX = 20


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
    DIR = "/home/sjanny/IRS/"
    maxz, minz = 0, 100000
    disparity_ex, normal_ex = [], []
    for scene in os.listdir(DIR):
        for detail in os.listdir(DIR + "/" + scene):
            dataloc = DIR + "/" + scene + "/" + detail
            for file in os.listdir(dataloc):
                if "exr" in file:
                    if "d" in file:
                        disparity_ex.append(dataloc + "/" + file)
                    elif "n" in file:
                        normal_ex.append(dataloc + "/" + file)

    for ex in tqdm(disparity_ex):
        disp = load_exr(ex)
        disp[disp == 0] = 0.001
        z = 48 / disp
        assert z.min() >= 0
        z = np.clip(z, 0, ZMAX)
        z = z / ZMAX
        plt.imsave(ex.replace('exr', 'png'), np.stack([z, z, z], axis=-1))

    for ex in tqdm(normal_ex):
        n = load_exr(ex)
        assert n.min() >= 0 and n.max() <= 1
        plt.imsave(ex.replace('exr', 'png'), n)