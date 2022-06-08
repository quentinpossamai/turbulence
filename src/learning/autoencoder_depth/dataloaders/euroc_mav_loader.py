from torch.utils.data import Dataset
import torch
import os
import cv2

ZMAX = 7.48


class EurocMavLoader(Dataset):
    def __init__(self, split="train"):
        super(EurocMavLoader, self).__init__()
        self.dataloc = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/"
        if not os.path.isdir(self.dataloc):
            self.dataloc = "/storage/Datasets/BlackBird/euroc_mav/raw/"

        try:
            file = f"splits/eurocmav_{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')
        except FileNotFoundError:
            file = f"/home/sjanny/codes/turbulence/src/autoencoder_depth/splits/eurocmav_{split}.txt"
            with open(file, "r") as file:
                self.index = file.read().split('\n')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        dir = self.dataloc + self.index[item]
        depth_map = torch.FloatTensor(cv2.imread(dir)[:, :, 0]).unsqueeze(0)
        depth_map = depth_map / 255 * 2 - 1
        depth_map = depth_map  # * ZMAX

        dir = dir.replace("depth", "normal")
        normal_map = torch.FloatTensor(cv2.imread(dir).transpose(2, 0, 1))
        normal_map = normal_map / 255
        normal_map = normal_map * 2 - 1
        return {"depth": depth_map, "normal": normal_map}


if __name__ == '__main__':
    d = EurocMavLoader(split="train")
    print(d[1]["normal"].shape)
    import matplotlib.pyplot as plt

    plt.imshow((d[0]['normal'].numpy().transpose(1, 2, 0) + 1) / 2)
    plt.show()
