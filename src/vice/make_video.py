import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import imageio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
# from tqdm import tqdm

import src.data.euroc_mav.loading as euroc
import src.data.locobot.loading as locobot
import src.data.blackbird.loading as blackbird

matplotlib.use('TKAgg')


def make_video(number: int, include_label: bool, dataset: str):
    assert dataset in ['blackbird', 'euroc_mav', 'locobot']
    if dataset == 'euroc_mav':
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{euroc.FLIGHT_LIST[number]}vice/'
        data = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))
        width = 752
        height = 480
        img = np.zeros((len(data['img_path']), height, width, 3), dtype=int)
        for i, image_path in enumerate(data['img_path']):
            tmp_img = np.array(imageio.imread(image_path))
            color_frame = np.uint8(np.zeros((height, width, 3)))
            for j in range(3):
                color_frame[:, :, j] = tmp_img
            img[i] = color_frame
        fps = 10
    elif dataset == 'locobot':
        vice_path = f'{locobot.LOCOBOT_DATA_PATH}{locobot.TRAJ_LIST[number]}'
        img = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))['img']
        fps = 10
        print(img.shape)
        _, height, width, _ = img.shape
        # width = 640
        # height = 480
    else:
        assert dataset == 'blackbird'
        vice_path = blackbird.VICE_PATH
        fps = 10
        width = 1024
        height = 768
        img_path = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))['img_path']
        img = np.zeros((len(img_path), height, width, 3), dtype=int)
        for i, image_path in enumerate(tqdm(img_path, desc='Loading img')):
            img[i] = np.array(imageio.imread(image_path))

    if include_label:
        label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
        saving_path = f'{vice_path}labelled.mkv'
    else:
        saving_path = f'{vice_path}raw.mkv'
    codec = VideoWriter_fourcc(*'H264')

    video = VideoWriter(saving_path, codec, float(fps), (width, height), True)
    print(f'Video destination : {saving_path}')

    shape = (height, width, 3)
    for i, single_img in enumerate(img):
        fig = plt.figure(figsize=(width / 100., height / 100.))
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(single_img)
        if include_label:
            try:
                point, is_keyframe = label[i]
                ax.scatter(point[0], point[1], c='g' if is_keyframe else 'r')
            except KeyError:
                ax.annotate('No label placed.', (width / 2, height / 2), xycoords='data', size=12,
                            ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
        ax.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
        plt.close('all')
        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
    video.release()
    cv2.destroyAllWindows()


def make_video_all(number: int, depth: bool, dataset: str):
    assert dataset in ['blackbird', 'euroc_mav', 'locobot']
    if dataset == 'euroc_mav':
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{euroc.FLIGHT_LIST[number]}vice/'
        data = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))
        fps = 20
        width = 752
        height = 480
        img = np.uint8(np.zeros((len(data['img_path']), height, width, 3)))
        for i, image_path in enumerate(data['img_path']):
            tmp_img = np.array(imageio.imread(image_path))
            color_frame = np.uint8(np.zeros((height, width, 3)))
            for j in range(3):
                color_frame[:, :, j] = tmp_img
            img[i] = color_frame
        title = f'EuRoC MAV - {euroc.FLIGHT_LIST[number][:-1]}'
        suffix1, suffix2 = '', 'imu'
    elif dataset == 'locobot':
        vice_path = f'{locobot.LOCOBOT_DATA_PATH}{locobot.TRAJ_LIST[number]}'
        img = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))['img']
        fps = 30
        print(img.shape)
        _, height, width, _ = img.shape
        title = f'CITI Locobot'
        suffix1, suffix2 = '_odom', '_slam'
    else:
        assert dataset == 'blackbird'
        vice_path = blackbird.VICE_PATH
        fps = 30
        width = 1024
        height = 768
        img_path = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))['img_path']
        img = np.zeros((len(img_path), height, width, 3), dtype=int)
        for i, image_path in enumerate(img_path):
            img[i] = np.array(imageio.imread(image_path))
        title = f'Blackbird - Patrick yawConstant maxSpeed3p0'
        suffix1, suffix2 = '', 'imu'
    prefix = 'depth' if depth else 'ground'
    p_gt = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
    p_mocap = pickle.load(open(f'{vice_path}{prefix}{suffix1}.pkl', 'rb'))
    p_imu = pickle.load(open(f'{vice_path}{prefix}{suffix2}.pkl', 'rb'))

    name = 'all_depth' if depth else 'all_ground'
    saving_path = f'{vice_path}{name}.mkv'
    codec = VideoWriter_fourcc(*'H264')
    video = VideoWriter(saving_path, codec, float(fps), (width, height), True)
    print(f'Video destination : {saving_path}')

    # Keep lasting legend
    fig = plt.figure(figsize=(width / 100., height / 100.))
    ax = plt.axes([0, 0, 1, 1])
    param = {'s': 100, 'edgecolors': 'w', 'linewidths': 2}
    if dataset == 'locobot':
        ax.scatter(0, 0, c='r', alpha=1, label='Odom', **param)
        ax.scatter(0, 0, c='g', alpha=1, label='SLAM', **param)
        ax.scatter(0, 0, c='c', alpha=1, label='Label', **param)
    else:
        ax.scatter(0, 0, c='r', alpha=1, label='MoCap', **param)
        ax.scatter(0, 0, c='g', alpha=1, label='IMU', **param)
        ax.scatter(0, 0, c='c', alpha=1, label='Label', **param)
    ax.legend(loc=2, labelcolor='black', fontsize='medium', frameon=True)
    handles, labels = ax.get_legend_handles_labels()

    shape = (height, width, 3)
    for i, single_img in enumerate(img):
        fig = plt.figure(figsize=(width / 100., height / 100.))
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(single_img)
        if isinstance(p_mocap[i][0], np.ndarray):
            ax.scatter(p_mocap[i][0][0], p_mocap[i][0][1], c='r', alpha=1, **param)
        if isinstance(p_imu[i][0], np.ndarray):
            ax.scatter(p_imu[i][0][0], p_imu[i][0][1], c='g', alpha=1, **param)
        try:
            point, is_keyframe = p_gt[i]
            ax.scatter(point[0], point[1], c='c', alpha=1, **param)
        except KeyError:
            ax.annotate('No label placed.', (width / 2, height / 2), xycoords='data', size=12,
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
        ax.legend(handles, labels, loc=2, fontsize='medium', frameon=True)
        ax.annotate(title, (width * 0.02, height * 0.98), xycoords='data', size=12, c='w',
                    ha='left', va='bottom')
        ax.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
        plt.close('all')
        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
    video.release()
    cv2.destroyAllWindows()


def main():
    make_video(0, include_label=False, dataset='euroc_mav')
    # make_video(1, include_label=False, dataset='euroc_mav')
    # make_video(2, include_label=False, dataset='euroc_mav')
    #
    # make_video(1, include_label=True, dataset='locobot')
    #
    # make_video(0, include_label=True, dataset='blackbird')

    # for i in range(3):
    #     for depth in [False, True]:
    #         make_video_all(euroc.FLIGHT_LIST[i], depth)

    # make_video_all(0, False, dataset='blackbird')
    # make_video_all(0, True, dataset='blackbird')
    # make_video_all(1, False, dataset='locobot')
    # make_video_all(1, True, dataset='locobot')

    # make_video_all(0, False, dataset='euroc_mav')
    # make_video_all(1, False, dataset='euroc_mav')
    # make_video_all(2, False, dataset='euroc_mav')
    # make_video_all(0, True, dataset='euroc_mav')
    # make_video_all(1, True, dataset='euroc_mav')
    # make_video_all(2, True, dataset='euroc_mav')

    # make_video(0, include_label=True, dataset='blackbird')


if __name__ == '__main__':
    main()
