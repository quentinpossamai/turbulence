import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import src.data.habitat.dataviz as dataviz

plt.style.use('seaborn-deep')
plt.rcParams.update({"text.usetex": False,
                     "font.family": "serif",
                     "font.sans-serif": ["Computer Modern Roman"]})

xx = np.arange(200, 275)
# rgb1, z1, fa1, z1_hat, z_hat_nogru = dataviz.get_image_from_generated_traj(64, 7, xx)

# pickle.dump([rgb1, z1, fa1, z1_hat, z_hat_nogru], open(f'{dataviz.habitat.WD_PATH}figure6.pkl', 'wb'))
[rgb1, z1, fa1, z1_hat, z_hat_nogru] = pickle.load(open(f'{dataviz.habitat.WD_PATH}figure6.pkl', 'rb'))

z_gt = np.array(pickle.load(open(f'{dataviz.habitat.WD_PATH}figure6/z_gt.pkl', 'rb')))
z_hat = np.array(pickle.load(open(f'{dataviz.habitat.WD_PATH}figure6/z_hat.pkl', 'rb')))
z_hat2 = np.array(pickle.load(open(f'{dataviz.habitat.WD_PATH}figure6/z_hat2.pkl', 'rb')))
fa_gt = np.array(pickle.load(open(f'{dataviz.habitat.WD_PATH}figure6/fa_gt.pkl', 'rb')))
fa_hat = np.array(pickle.load(open(f'{dataviz.habitat.WD_PATH}figure6/fa_hat.pkl', 'rb')))
fa_hat2 = np.array(pickle.load(open(f'{dataviz.habitat.WD_PATH}figure6/fa_hat2.pkl', 'rb')))

xx = np.arange(215, 275)
xx2 = np.arange(39, len(fa_gt) + 39)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(xx2[xx - 39] * 1/20, fa_gt[xx - 39], label='GT')
ax.plot(xx2[xx - 39] * 1/20, fa_hat[xx - 39], label='Ours')
ax.plot(xx2[xx - 39] * 1/20, fa_hat2[xx - 39], label='No GRU')
ax.grid()
ax.set_xlabel('Time (s)')
ax.set_ylabel('fa (N)')
# ax.set_ylim([0.1, 1.2])
ax.axvline(240 * 1/20, c='black')
ax.axvline(246 * 1/20, c='black')
ax.legend()
plt.subplots_adjust(left=0.1, bottom=0.15, right=0.99, top=0.99, wspace=0., hspace=0.)
plt.savefig(f'{dataviz.habitat.WD_PATH}figure6/figure6.pdf')


fig, ax = plt.subplots()
ax.imshow(rgb1[170 - 200])

drone_width, drone_height = 0.1, 0.025
volumetric_mass = 200
g = 9.81
m = volumetric_mass * drone_width ** 2 * drone_height
drone_prms = {'width': drone_width,  # [m]
              'height': drone_height,  # [m]
              'm': m,  # [kg]
              'kt': m * g / (4 * 150 ** 2),  #
              'km': m * g / (4 * 150 ** 2) * 2 ** 0.5 * drone_width / 2,  #
              'l': 2 ** 0.5 * drone_width / 2,  # [m]
              'ix': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
              'iy': 1 / 3 * (drone_width ** 2 + drone_height ** 2) * m,  # [kg.m2]
              'iz': 1 / 3 * (drone_width ** 2 + drone_width ** 2) * m,  # [kg.m2]
              'prop_radius': drone_width * 0.1}  # [m]
theta = np.array([drone_prms['kt'] / drone_prms['m'],
                  drone_prms['kt'] * drone_prms['l'] / drone_prms['ix'],
                  drone_prms['km'] / drone_prms['iz'],
                  drone_prms['iz'] / drone_prms['ix']])


def _apply(_z, _motor):
    mu = 1
    cap = 0.1
    ground_force = np.zeros(3)
    if _z < np.sqrt(mu) * drone_prms['prop_radius'] / 4:
        ground_force[2] = cap * drone_prms['m'] * g
    else:
        mu_r_z = mu * (drone_prms['prop_radius'] / (4 * _z)) ** 2
        ground_force[2] = np.clip(np.sum(_motor ** 2) * drone_prms['kt'] * mu_r_z / (1 - mu_r_z),
                                  a_min=0, a_max=cap * drone_prms['m'] * g)
    return ground_force


fig, ax = plt.subplots()
ax.plot(xx, z1, label='GT')
ax.plot(xx, z1_hat, label='Ours')
ax.plot(xx, z_hat_nogru, label='No GRU')
ax.grid()
ax.set_xlabel('Sample #')
ax.set_ylabel('d (m)')
# ax.set_ylim([0.1, 1.2])
ax.axvline(311, c='black')
ax.legend()
plt.sav
print()


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
        symbole = '$\hat{d}$'
        ax.annotate(f'd: {z[j]:0.2f}\n{symbole}: {z_hat[j]:0.2f}', (shape[1] - 10, 10), xycoords='data', size=8,
                    ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.02, hspace=0.02)
plt.savefig(f'{dataviz.habitat.WD_PATH}reports/IROS_01_03_2022/graphics/3x72.pdf')
plt.show()

