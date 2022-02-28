import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.habitat.dataviz as dataviz

image_num = 5
# rgb1 = dataviz.get_image_from_generated_traj(0, 6, np.linspace(874, 898, image_num, dtype=int))
# rgb2 = dataviz.get_image_from_generated_traj(3, 13, np.linspace(0, 200, image_num, dtype=int))
rgb3 = dataviz.get_image_from_generated_traj(24, 12, np.linspace(413, 880, image_num, dtype=int))
pickle.dump([rgb1, rgb2, rgb3], open(f'{dataviz.habitat.WD_PATH}data/habitat/3x7.pkl', 'wb'))

rgb1, rgb2, rgb3 = pickle.load(open(f'{dataviz.habitat.WD_PATH}data/habitat/3x7.pkl', 'rb'))

shape = [480, 640]
fig, axs = plt.subplots(3, image_num, figsize=(image_num * shape[1] / 250, 3 * shape[0] / 250))
for rgb, i in zip([rgb1, rgb2, rgb3],
                  [0, 1, 2]):
    for j, img in enumerate(rgb):
        ax = axs[i, j]
        ax.imshow(img)
        ax.axis('off')
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.02, hspace=0.02)

plt.savefig(f'{dataviz.habitat.WD_PATH}reports/IROS_01_03_2022/graphics/3x7.pdf')
