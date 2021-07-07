import numpy as np
import os

PATH = "/storage/Datasets/BlackBird/euroc_mav/raw/V1_03_difficult/depth/"

mini, maxi = [], []

for i in os.listdir(PATH):
    d = np.load(PATH+i)
    mini.append(d.min())
    maxi.append(d.max())

print(min(mini), np.mean(mini), np.std(mini))
print(min(maxi), np.mean(maxi), np.std(maxi))

