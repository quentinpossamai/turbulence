import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.restoration import inpaint

DATA_DIR = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy/depth/"
OUT_DIR = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy/depth_postprocessed/"

KERNEL_SIZE = 2

if __name__ == '__main__':
    for f in tqdm(os.listdir(DATA_DIR)):
        data = np.load(DATA_DIR + f).astype(np.float32)
        mask = np.isnan(data).astype(np.float32)
        filled_data = inpaint.inpaint_biharmonic(data, mask)

        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(data)
        ax[1].imshow(filled_data)
        ax[0].axis('off')
        ax[1].axis("off")
        fig.tight_layout()
        plt.show()
