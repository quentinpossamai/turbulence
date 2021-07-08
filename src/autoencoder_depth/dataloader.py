from torch.utils.data import Dataset
import numpy as np
import os
import torch


class Depthmap_Loader(Dataset):
    def __init__(self):
        super(Depthmap_Loader, self).__init__()
        self.dataloc = "/Volumes/Samsung_T5/BlackBird/euroc_mav/"
        if not os.path.isdir(self.dataloc):
            self.dataloc = "/storage/Datasets/BlackBird/euroc_mav/"

