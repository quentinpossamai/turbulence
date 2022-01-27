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
print(f'WD_PATH: {WD_PATH}')
MODELS_PATH = WD_PATH + "models/"
ANALYSIS_PATH = WD_PATH + "analysis/"
ANALYSIS_EUROC_MAV_RESIDUAL_TRAINING = ANALYSIS_PATH + "euroc_mav_residual_training/"  # train.py
ANALYSIS_DATA_VIS_EUROC_MAV = ANALYSIS_PATH + 'data_vis_euroc_mav/'  # euroc_mav.data_vis.py
DATA_HABITAT_PATH = f'{WD_PATH}data/habitat/'


def mkdir_if_not_exist(paths: list):
    if "steeven" in os.getcwd():
        pass
    else:
        assert isinstance(paths, list), "Path given must be a list."
        assert WD_PATH in ["/Users/quentin/phd/turbulence/",
                           "/home/qpossamai/phd/turbulence/",
                           "/storage/home/qpossamai/phd/turbulence/"]
        for path in paths:
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)


mkdir_if_not_exist([MODELS_PATH, ANALYSIS_PATH])


def is_cluster():
    return socket.gethostname() in ["joey", "deedee", "marky", "bob", "dgx"]
