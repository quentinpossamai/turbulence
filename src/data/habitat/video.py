import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

import src.data.habitat.dataviz as dataviz

plt.rcParams.update({"text.usetex": True,
                     "font.family": "serif",
                     "font.sans-serif": ["Computer Modern Roman"]})
matplotlib.use('TKAgg')

rgb1, z1, z1_hat, z1_nogru = dataviz.get_image_from_generated_traj(9, 0, np.arange(400, 450, 1))
rgb2, z2, fa2, z2_hat = dataviz.get_image_from_generated_traj(2, 14, np.arange(110, 180, 1, dtype=int))
rgb3, z3, fa3, z3_hat = dataviz.get_image_from_generated_traj(6, 0, np.arange(400, 450))
pickle.dump([[rgb1, rgb2, rgb3],
             [z1, z2, z3],
             [fa1, fa2, fa3],
             [z1_hat, z2_hat, z3_hat]], open(f'{dataviz.habitat.WD_PATH}data/habitat/video.pkl', 'wb'))

# [[rgb1, rgb2, rgb3],
#  [z1, z2, z3],
#  [fa1, fa2, fa3],
#  [z1_hat, z2_hat, z3_hat]] = pickle.load(open(f'{dataviz.habitat.WD_PATH}data/habitat/video.pkl', 'rb'))

shape = [480, 640, 3]
for rgb, i, z, fa, z_hat in zip([rgb1, rgb2, rgb3],  #
                                [0, 1, 2],  #
                                [z1, z2, z3],  #
                                [fa1, fa2, fa3],  #
                                [z1_hat, z2_hat, z3_hat]):  #
    saving_path = f'{dataviz.habitat.WD_PATH}reports/IROS_01_03_2022/graphics/video/video{i}.mkv'
    fps = 10
    codec = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(saving_path, codec, float(fps), (shape[1], shape[0]), True)
    print(f'Video destination : {saving_path}')

    for j, img in enumerate(rgb):
        fig = plt.figure(figsize=(shape[1] / 100., shape[0] / 100.))
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.axis('off')
        symbol = '$\hat{d}$'
        ax.annotate(f'd: {z[j]:0.2f}\n{symbol}: {z_hat[j]:0.2f}', (shape[1] - 10, 10), xycoords='data', size=8,
                    ha='right', va='top', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
        plt.close('all')
        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
    video.release()
    cv2.destroyAllWindows()
