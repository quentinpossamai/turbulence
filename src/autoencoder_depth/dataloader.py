from torch.utils.data import Dataset
import numpy as np
import os
import PIL.PngImagePlugin


class Depthmap_Loader(Dataset):
    def __init__(self, split="train"):
        super(Depthmap_Loader, self).__init__()
        self.dataloc = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/"
        if not os.path.isdir(self.dataloc):
            self.dataloc = "/storage/Datasets/BlackBird/euroc_mav/raw/"

        try:
            file = f"autoencoder_depth/splits/{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')
        except FileNotFoundError:
            file = f"../autoencoder_depth/splits/{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        depth_map = PIL.PngImagePlugin.PngImageFile(self.dataloc + self.index[item])
        depth_map = np.divide(depth_map, 2 ** 8 - 1, dtype=np.float32)[:, :, 0]
        depth_map = depth_map * 2 - 1
        return depth_map


if __name__ == '__main__':
    d = Depthmap_Loader()
    print(d[0].shape)
