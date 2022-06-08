from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import re
import socket
import sys
import time
from typing import Union, Tuple, List, Iterable

WD_PATH = os.getcwd() + "/"
print(f'WD_PATH: {WD_PATH}', flush=True)
MODELS_PATH = WD_PATH + "models/"
ANALYSIS_PATH = WD_PATH + "analysis/"
ANALYSIS_DATA_VIS_EUROC_MAV = ANALYSIS_PATH + 'data_vis_euroc_mav/'  # euroc_mav.datavis.py

# HABITAT
DATA_HABITAT_PATH = f'{WD_PATH}data/habitat/'
SCENE_PATH = f'{DATA_HABITAT_PATH}scene_datasets/mp3d/'


def mkdir(paths: list):
    if "steeven" in os.getcwd():
        pass
    else:
        assert isinstance(paths, list), "Path given must be a list."
        assert WD_PATH in ["/Users/quentin/phd/turbulence/",
                           "/home/qpossamai/phd/turbulence/",
                           "/storage/home/qpossamai/phd/turbulence/"]
        for path in paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)


mkdir([MODELS_PATH, ANALYSIS_PATH])


def is_cluster():
    return socket.gethostname() in ["joey", "deedee", "marky", "bob", "dgx"]
