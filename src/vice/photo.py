import cv2
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

import src.utils.paths as paths
import src.data.euroc_mav.loading as euroc
import src.data.locobot.loading as locobot
import src.data.blackbird.loading as blackbird


def make_mpl_photo():
    vice_path_euroc = f'{euroc.INTERMEDIATE_DATASET_PATH}{euroc.FLIGHT_LIST[0]}vice/'
    vice_path_locobot = f'{locobot.LOCOBOT_DATA_PATH}{locobot.TRAJ_LIST[1]}'
    vice_path_blackbird = f'{blackbird.VICE_PATH}'

    euroc_data = {}
    data = pickle.load(open(f'{vice_path_euroc}vice_data.pkl', 'rb'))
    euroc_data['width'] = 752
    euroc_data['height'] = 480
    euroc_data['img'] = np.uint8(np.zeros((len(data['img_path']), euroc_data['height'], euroc_data['width'], 3)))
    for i, image_path in enumerate(data['img_path']):
        tmp_img = np.array(imageio.imread(image_path))
        color_frame = np.uint8(np.zeros((euroc_data['height'], euroc_data['width'], 3)))
        for j in range(3):
            color_frame[:, :, j] = tmp_img
        euroc_data['img'][i] = color_frame
    suffix1, suffix2 = '', 'imu'
    prefix = 'ground'
    euroc_data['p_gt'] = pickle.load(open(f'{vice_path_euroc}labels.pkl', 'rb'))
    euroc_data['p_mocap'] = {}
    euroc_data['p_imu'] = {}
    for lab_num in range(4):
        euroc_data['p_mocap'][lab_num] = pickle.load(open(f'{vice_path_euroc}{prefix}{suffix1}{lab_num}.pkl', 'rb'))
        euroc_data['p_imu'][lab_num] = pickle.load(open(f'{vice_path_euroc}{prefix}{suffix2}{lab_num}.pkl', 'rb'))

    locobot_data = {}
    locobot_data['img'] = pickle.load(open(f'{vice_path_locobot}vice_data.pkl', 'rb'))['img']
    _, locobot_data['height'], locobot_data['width'], _ = locobot_data['img'].shape
    suffix1, suffix2 = '_odom', '_slam'
    locobot_data['p_gt'] = pickle.load(open(f'{vice_path_locobot}labels.pkl', 'rb'))
    locobot_data['p_mocap'] = {}
    locobot_data['p_imu'] = {}
    for lab_num in range(4):
        locobot_data['p_mocap'][lab_num] = pickle.load(open(f'{vice_path_locobot}{prefix}{suffix1}{lab_num}.pkl', 'rb'))
        locobot_data['p_imu'][lab_num] = pickle.load(open(f'{vice_path_locobot}{prefix}{suffix2}{lab_num}.pkl', 'rb'))

    blackbird_data = {}
    blackbird_data['width'] = 1024
    blackbird_data['height'] = 768
    img_path = pickle.load(open(f'{vice_path_blackbird}vice_data.pkl', 'rb'))['img_path']
    img = np.zeros((len(img_path), blackbird_data['height'], blackbird_data['width'], 3), dtype=int)
    blackbird_data['img'] = img
    for i, image_path in enumerate(img_path):
        img[i] = np.array(imageio.imread(image_path))
    suffix1, suffix2 = '', 'imu'
    blackbird_data['p_gt'] = pickle.load(open(f'{vice_path_blackbird}labels.pkl', 'rb'))
    blackbird_data['p_mocap'] = {}
    blackbird_data['p_imu'] = {}
    for lab_num in range(4):
        blackbird_data['p_mocap'][lab_num] = pickle.load(open(f'{vice_path_blackbird}{prefix}{suffix1}{lab_num}.pkl', 'rb'))
        blackbird_data['p_imu'][lab_num] = pickle.load(open(f'{vice_path_blackbird}{prefix}{suffix2}{lab_num}.pkl', 'rb'))

    # pickle.dump(euroc_data, open(f'/Volumes/NO NAME/vice_figure/euroc_data.pkl', 'wb'))
    # pickle.dump(locobot_data, open(f'/Volumes/NO NAME/vice_figure/locobot_data.pkl', 'wb'))
    pickle.dump(blackbird_data, open(f'/Users/quentin/phd/turbulence/data/blackbird/blackbird_data.pkl', 'wb'), protocol=4)

    param = {'edgecolors': 'w', 'linewidths': 2}

    def scatter_point(dataset, ax_to_plot, ind):
        if dataset == 'euroc_mav':
            p_gt = euroc_data['p_gt']
            p_mocap = euroc_data['p_mocap']
            p_imu = euroc_data['p_imu']
        elif dataset == 'locobot':
            p_gt = locobot_data['p_gt']
            p_mocap = locobot_data['p_mocap']
            p_imu = locobot_data['p_imu']
        else:
            assert dataset == 'blackbird'
            p_gt = blackbird_data['p_gt']
            p_mocap = blackbird_data['p_mocap']
            p_imu = blackbird_data['p_imu']
        for _lab_num in range(3):
            point = p_mocap[_lab_num][ind][0]
            if isinstance(point, np.ndarray):
                ax_to_plot.scatter(point[0], point[1], s=50, c='r', alpha=1, marker='o')
            else:
                return False
            point = p_imu[_lab_num][ind][0]
            if isinstance(point, np.ndarray):
                ax_to_plot.scatter(point[0], point[1], s=100, c='lime', alpha=1, marker='x')
            else:
                return False
            try:
                _point, _is_keyframe = p_gt[_lab_num][ind]
                ax_to_plot.scatter(_point[0], _point[1], s=100, c='c', alpha=1, marker='.')
            except KeyError:
                return False
        return True

    euroc_ind = np.array([0, 10, 20, 30, 40, 50, 60]) + 185
    locobot_ind = np.array([0, 10, 20, 30, 40, 50, 60]) + 245
    blackbird_ind = np.array([0, 10, 20, 30, 40, 50, 60]) + 156

    shape = (np.min([euroc_data['height'], locobot_data['height'], blackbird_data['height']]),
             np.min([euroc_data['width'], locobot_data['width'], blackbird_data['width']]))

    # fig = plt.figure(figsize=(shape[1] * 7 / 100, shape[0] * 3 / 100))
    fig, axs = plt.subplots(3, 7, figsize=(7 * shape[1] / 500, 3 * shape[0] / 500))
    for j, ind, data, name in zip([1, 2, 3],
                                  [euroc_ind, locobot_ind, blackbird_ind],
                                  [euroc_data, locobot_data, blackbird_data],
                                  ['euroc_mav', 'locobot', 'blackbird']):
        for i, e in enumerate(ind):
            ax = axs[j-1, i]
            # dim = [i / 7, 1 - j * 0.33, 1 / 7, 0.33]
            # ax = plt.axes(dim)
            ax.axis('off')
            ax.imshow(data['img'][e][:shape[0], :shape[1]])
            has_worked = scatter_point(name, ax, e)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.02, hspace=0.05)
    plt.savefig(f'{paths.WD_PATH}data/fig.pdf')


def main():
    make_mpl_photo()


if __name__ == '__main__':
    main()
