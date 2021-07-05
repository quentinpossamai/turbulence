import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.image

if __name__ == '__main__':
    files = os.listdir("demo")

    depth_files = [f for f in files if "depth" in f]
    depth_files = sorted(depth_files, key = lambda x : int(x.split("depth")[0]))
    for d in depth_files:
        plt.clf()
        data = np.load("demo/" + d)
        plt.imshow(data)
        plt.title(d)
        plt.pause(0.5)
    plt.show()