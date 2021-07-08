import numpy as np
import os
from tqdm import tqdm

mini, maxi = [], []

PATH = "/storage/Datasets/BlackBird/euroc_mav/raw/V1_03_difficult/depth_estimation/"
for i in tqdm(os.listdir(PATH)):
    d = np.load(PATH + i)
    mini.append(d.min())
    maxi.append(d.max())

PATH = "/storage/Datasets/BlackBird/euroc_mav/raw/V1_01_easy/depth_estimation/"
for i in tqdm(os.listdir(PATH)):
    d = np.load(PATH + i)
    mini.append(d.min())
    maxi.append(d.max())

PATH = "/storage/Datasets/BlackBird/euroc_mav/raw/V1_02_medium/depth_estimation/"
for i in tqdm(os.listdir(PATH)):
    d = np.load(PATH + i)
    mini.append(d.min())
    maxi.append(d.max())

print(min(mini), np.mean(mini), np.std(mini))
print(min(maxi), np.mean(maxi), np.std(maxi))
