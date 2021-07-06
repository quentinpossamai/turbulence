import pptk
import numpy as np
from tqdm import tqdm
data = []
color = []
file = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_02_medium/pointcloud0/data.ply"

def read_ply():
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in tqdm(range(11, len(lines))):
            parsed = lines[i].split(" ")
            x = [float(parsed[k]) for k in range(3)]
            data.append(x)
    return data

data = read_ply()
data = np.stack(data, axis=0)
np.save("pointcloud_V1_02.npy", data)
print(data.shape)
v = pptk.viewer(data)