import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image

if __name__ == '__main__':
    files = os.listdir("/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy/depth/")

    depth_files = [f for f in files if "depth" in f]
    depth_files = sorted(depth_files, key=lambda x: int(x.split("depth")[0]))
    for d in depth_files:
        plt.clf()
        data = np.load("/Volumes/Samsung_T5/BlackBird/eurocmav/raw/V1_01_easy/depth/" + d)
        plt.imshow(data)
        plt.title(d)
        plt.pause(0.1)
    plt.show()
