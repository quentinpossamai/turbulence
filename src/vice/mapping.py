import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import imageio
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import datetime
from tqdm import tqdm

import src.data.euroc_mav.loading as euroc

matplotlib.use('TKAgg')


FRAME_NUM = 0
INVERT = False


def make_video(flight: str, include_label: bool):
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    data = pickle.load(open(f'{vice_path}vice_data.pkl', 'rb'))
    if include_label:
        label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
        saving_path = f'{vice_path}labelled.mkv'
    else:
        saving_path = f'{vice_path}raw.mkv'

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


def labelize(flight):
    global FRAME_NUM
    vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    video = cv2.VideoCapture(f'{vice_path}/raw.mkv')
    img = []
    is_reading, frame = video.read()
    while is_reading:
        img.append(frame[:, :, ::-1])
        is_reading, frame = video.read()
    img = np.stack(img[:-1])
    _, height, width, _ = img.shape

    try:
        msg = f'Previous label found at: {vice_path}label.pkl'
        label = pickle.load(open(f'{vice_path}label.pkl', 'rb'))
    except FileNotFoundError:
        msg = f'No previous label found.'
        label = {}
    print(msg)

    FRAME_NUM = 0
    # fig, ax = plt.subplots(1, 1, figsize=(2.5 * 6.4, 2 * 4.8))
    fig, ax = plt.subplots(1, 1, figsize=(2 * 6.4, 1.5 * 4.8))
    ax.set_title(f'Frame: {FRAME_NUM}')
    img_obj = ax.imshow(img[FRAME_NUM])
    scatter_obj = ax.scatter(width / 2, height / 2, c='r', alpha=0)
    if 0 in label:
        scatter_obj.set_alpha(1)
        value = label[0]
        scatter_obj.set_offsets(value[0])
        scatter_obj.set_color('g' if value[1] else 'r')

    def display_other_frame(increment):
        global FRAME_NUM
        global INVERT
        FRAME_NUM = FRAME_NUM + increment
        if INVERT:
            img_obj.set_data(255 - img[FRAME_NUM])
        else:
            img_obj.set_data(img[FRAME_NUM])
        if FRAME_NUM in label:
            val = label[FRAME_NUM]
            scatter_obj.set_alpha(1)
            if val[1]:  # keyframe
                ax.set_title(f'Frame: {FRAME_NUM} | Keyframe')
                scatter_obj.set_offsets(label[FRAME_NUM][0])
                scatter_obj.set_color('g')
            else:
                ax.set_title(f'Frame: {FRAME_NUM} | Normal frame ')
                scatter_obj.set_color('r')
                scatter_obj.set_offsets(label[FRAME_NUM][0])
        else:
            ax.set_title(f'Frame: {FRAME_NUM}')
            scatter_obj.set_alpha(0)
        fig.canvas.draw()

    # def on_click(event):
    #     if event.button.name == 'LEFT':
    #         label[FRAME_NUM] = np.array([np.round(event.xdata), np.round(event.ydata)])
    #         display_other_frame(1)

    def on_key(event):
        global INVERT
        if event.key == 'e':
            label[FRAME_NUM] = (np.array([np.round(event.xdata), np.round(event.ydata)]), False)  # not a key frame
            display_other_frame(1)
        if event.key == ' ':
            label[FRAME_NUM] = (np.array([np.round(event.xdata), np.round(event.ydata)]), True)  # a key frame
            display_other_frame(1)
        if event.key == '1':
            display_other_frame(-1)
        if event.key == '2':
            display_other_frame(1)
        if event.key == '3':
            display_other_frame(-10)
        if event.key == '4':
            display_other_frame(10)
        if event.key == '5':
            display_other_frame(-100)
        if event.key == '6':
            display_other_frame(100)
        if event.key == 'r':
            INVERT = not INVERT
            display_other_frame(0)
        if event.key == 'd':
            del label[FRAME_NUM]
            display_other_frame(0)
        if event.key == 'w':
            h = datetime.datetime.now()
            print(f'{h.ctime()}: Saving at: {vice_path}label.pkl')
            pickle.dump(label, open(f'{vice_path}label.pkl', 'wb'))
    # fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def main():
    make_video(euroc.FLIGHT_LIST[2], include_label=True)
    # labelize(euroc.FLIGHT_LIST[2])


if __name__ == '__main__':
    main()
