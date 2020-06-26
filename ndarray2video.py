import numpy as np
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import os
import glob
import re
from util2 import Progress
# Extract arrays
# Listing data
# noinspection DuplicatedCode
ABSOLUTE_PATH = '/Users/quentin/work/turbulence/data_drone/'


def extract_data_path():
    bags_n_flights = []
    for day in sorted(next(os.walk(ABSOLUTE_PATH))[1]):
        temp_path = ABSOLUTE_PATH + day + '/'
        for flight_name in sorted(next(os.walk(temp_path))[1]):
            flight_path = temp_path + flight_name + '/'
            for bag in sorted(glob.glob(flight_path + '*.npy')):
                bags_n_flights.append((bag, flight_path))
    return bags_n_flights


# Frame size for the Tara camera
WIDTH = 640
HEIGHT = 480

BAGS_N_FLIGHTS = extract_data_path()  # List of tuple (path to .bag file, path to .npy folder)

TARA_ARRAYS_PATHS = [e for e in BAGS_N_FLIGHTS if re.search('.*(/tara.*)', e[0]) is not None]  # List of tuple (path to
# tara.npy)

# Creating a video to identify data
for e in TARA_ARRAYS_PATHS:
    array_file_path = e[0]
    array_folder_path = e[1]

    array = np.load(array_file_path)
    FPS = 28

    codec = VideoWriter_fourcc(*'H264')
    video = VideoWriter(array_file_path[:-4] + '.mkv', codec, float(FPS), (WIDTH, HEIGHT), False)

    print(array_file_path)

    prg = Progress(len(array), '\n')
    for i, frame in enumerate(array):
        video.write(frame)
        prg.update_pgr()

    video.release()
    cv2.destroyAllWindows()
    del video

    print('   ---   ')
