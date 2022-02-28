import pickle
import numpy as np

import src.utils.paths as paths

z_gt = pickle.load(open(f'{paths.WD_PATH}z_gt_nogru_habitat.pkl', 'rb'))
z_hat = pickle.load(open(f'{paths.WD_PATH}z_hat_nogru_habitat.pkl', 'rb'))
fa_gt = pickle.load(open(f'{paths.WD_PATH}fa_gttest_nogru_habitat.pkl', 'rb'))
fa_hat = pickle.load(open(f'{paths.WD_PATH}fa_hat_nogru_habitat.pkl', 'rb'))

np.sqrt(((fa_hat - fa_gt) ** 2).mean())
np.min(fa_gt)

print()