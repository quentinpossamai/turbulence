import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.habitat.dataviz as dataviz

image_num = 20 # [152]
rgb1, z1, fa1, z1_hat = dataviz.get_image_from_generated_traj(6, 18, np.linspace(550, 600, image_num, dtype=int))
# np.linspace(0, 200, image_num, dtype=int)
rgb2, z2, fa2, z2_hat = dataviz.get_image_from_generated_traj(2, 14, np.linspace(110, 180, image_num, dtype=int))
# np.linspace(340, 390, image_num, dtype=int) 16
rgb3, z3, fa3, z3_hat = dataviz.get_image_from_generated_traj(24, 16, np.linspace(40, 183, image_num, dtype=int))
pickle.dump([[rgb1, rgb2, rgb3],
             [z1, z2, z3],
             [fa1, fa2, fa3],
             [z1_hat, z2_hat, z3_hat]], open(f'{dataviz.habitat.WD_PATH}data/habitat/3x7.pkl', 'wb'))

[[rgb1, rgb2, rgb3],
 [z1, z2, z3],
 [fa1, fa2, fa3],
 [z1_hat, z2_hat, z3_hat]] = pickle.load(open(f'{dataviz.habitat.WD_PATH}data/habitat/3x7.pkl', 'rb'))

shape = [480, 640]
fig, axs = plt.subplots(3, image_num, figsize=(image_num * shape[1] / 250, 3 * shape[0] / 250))
for rgb, i, z, fa, z_hat in zip([rgb1, rgb2, rgb3],  #
                                [0, 1, 2],  #
                                [z1, z2, z3],  #
                                [fa1, fa2, fa3],  #
                                [z1_hat, z2_hat, z3_hat]):  #
    for j, img in enumerate(rgb):
        ax = axs[i, j]
        ax.imshow(img)
        ax.axis('off')
        ax.annotate(f'd: {z[j]:0.2f}\nd_hat: {z_hat[j]:0.2f}', (shape[1] - 10, 10), xycoords='data', size=8,
                    ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.02, hspace=0.02)

plt.savefig(f'{dataviz.habitat.WD_PATH}reports/IROS_01_03_2022/graphics/3x72.pdf')
