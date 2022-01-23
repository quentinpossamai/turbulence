import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import datetime

import src.data.euroc_mav.loading as euroc
import src.data.locobot.loading as locobot
import src.data.blackbird.loading as blackbird

matplotlib.use('TKAgg')

FRAME_NUM = 0
INVERT = False


def label_data(video_path: str, label_path: str):
    """
    Label the frame of a video file with using matplotlib and cursor position.

    param video_path: The path to load the video from.
    param label_path: The path to save the label from.
    :return:
    """
    global FRAME_NUM
    video = cv2.VideoCapture(video_path)
    img = []
    is_reading, frame = video.read()
    height, width, _, = frame.shape
    while is_reading:
        # read video is in bgr.
        img.append(np.array(frame[:, :, ::-1]))
        is_reading, frame = video.read()
    img = np.stack(img)
    _, height, width, _ = img.shape

    try:
        msg = f'Previous label found at: {label_path}'
        label = pickle.load(open(f'{label_path}', 'rb'))
    except FileNotFoundError:
        msg = f'No previous label found.'
        label = {}
    print(msg)

    FRAME_NUM = 0
    # fig, ax = plt.subplots(1, 1, figsize=(2.5 * 6.4, 2 * 4.8))
    fig, ax = plt.subplots(1, 1, figsize=(1.2 * width / 100, 1.2 * height / 100))
    ax.set_title(f'Frame: {FRAME_NUM}')
    img_obj = ax.imshow(img[FRAME_NUM])
    ax.grid(None)
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

    # If the labelling is done with the click.
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
            print(f'{h.ctime()}: Saving at: {label_path}')
            pickle.dump(label, open(f'{label_path}', 'wb'))

    # fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def main():
    # flight = euroc.FLIGHT_LIST[0]
    # vice_path = f'{euroc.INTERMEDIATE_DATASET_PATH}{flight}vice/'
    # label_data(f'{vice_path}raw.mkv', label_path=f'{vice_path}label.pkl')

    # traj = locobot.TRAJ_LIST[1]
    # vice_path = f'{locobot.LOCOBOT_DATA_PATH}{traj}'
    # label_data(video_path=f'{vice_path}raw.mkv', label_path=f'{vice_path}label.pkl')

    label_data(video_path=f'{blackbird.VICE_PATH}raw.mkv', label_path=f'{blackbird.VICE_PATH}label.pkl')


if __name__ == '__main__':
    main()
