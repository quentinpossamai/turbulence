import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import imageio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from tqdm import tqdm

import src.data.euroc_mav.loading as euroc
import src.data.locobot.loading as locobot
import src.data.blackbird.loading as blackbird

matplotlib.use('TKAgg')


def make_video(number: int, include_label: bool, dataset: str):
    assert dataset in ['blackbird', 'euroc_mav', 'locobot']
    if dataset == 'euroc_mav':
        vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{euroc.FLIGHT_LIST[number]}vice/'
        data = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))
        height, width = data['img'][0].shape
        img = np.zeros_like((len(data), height, width, 3))
        for i, image_path in enumerate(tqdm(data['img'], desc='')):
            tmp_img = np.array(imageio.imread(image_path))
            color_frame = np.uint8(np.zeros((height, width, 3)))
            for j in range(3):
                color_frame[:, :, j] = tmp_img
            img[i] = color_frame
        fps = 20
        width = 752
        height = 480
    elif dataset == 'locobot':
        vice_path = f'{locobot.LOCOBOT_DATA_PATH}{locobot.TRAJ_LIST[number]}'
        img = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))['img']
        fps = 30
        print(img.shape)
        _, height, width, _ = img.shape
        # width = 640
        # height = 480
    else:
        assert dataset == 'blackbird'
        blackbird.VICE_PATH

    if include_label:
        label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
        saving_path = f'{vice_path}labelled.mkv'
    else:
        saving_path = f'{vice_path}raw.mkv'
    codec = VideoWriter_fourcc(*'H264')

    video = VideoWriter(saving_path, codec, float(fps), (width, height), True)
    print(f'Video destination : {saving_path}')

    shape = (height, width, 3)
    for i, single_img in enumerate(tqdm(img, desc="Creating a video")):
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


def make_video_all(flight: str, depth: bool):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    data = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))
    prefix = 'depth' if depth else 'ground'
    p_gt = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
    p_mocap = pickle.load(open(f'{vice_path}{prefix}.pkl', 'rb'))
    p_imu = pickle.load(open(f'{vice_path}{prefix}imu.pkl', 'rb'))

    name = 'alldepth' if depth else 'all'
    saving_path = f'{vice_path}{name}.mkv'
    codec = VideoWriter_fourcc(*'H264')
    fps = 20
    width = 752
    height = 480
    video = VideoWriter(saving_path, codec, float(fps), (width, height), True)
    print(f'Video destination : {saving_path}')

    shape = (height, width, 3)
    for i, image_path in enumerate(tqdm(data['img'], desc="Creating a video")):
        img = np.array(imageio.imread(image_path))
        color_frame = np.uint8(np.zeros((shape[0], shape[1], 3)))
        for j in range(3):
            color_frame[:, :, j] = img
        img = color_frame

        fig = plt.figure(figsize=(width / 100., height / 100.))
        ax = plt.axes([0, 0, 1, 1])
        ax.imshow(img)
        try:
            point, is_keyframe = p_gt[i]
            ax.scatter(point[0], point[1], c='c')
        except KeyError:
            ax.annotate('No label placed.', (width / 2, height / 2), xycoords='data', size=12,
                        ha='center', va='center', bbox=dict(boxstyle='round', alpha=0.5, fc='w'))
        if isinstance(p_mocap[i][0], np.ndarray):
            ax.scatter(p_mocap[i][0][0], p_mocap[i][0][1], c='r')
        if isinstance(p_imu[i][0], np.ndarray):
            ax.scatter(p_imu[i][0][0], p_imu[i][0][1], c='g')

        ax.axis('off')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(shape)
        plt.close('all')
        video.write(data[:, :, ::-1])  # Because VideoWriter.write take BGR images
    video.release()
    cv2.destroyAllWindows()


def main():
    # make_video(euroc.FLIGHT_LIST[2], include_label=True)
    #
    # for i in range(3):
    #     for depth in [False, True]:
    #         make_video_all(euroc.FLIGHT_LIST[i], depth)

    make_video(1, include_label=True, for_locobot=True)


if __name__ == '__main__':
    main()
