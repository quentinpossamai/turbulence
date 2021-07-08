import random
import os


DIR = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/"
FLIGHTS = ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]

if __name__ == '__main__':
    random.seed(0)
    files = []
    for flight in FLIGHTS:
        for f in os.listdir(DIR + flight + "/depth_estimation"):
            files.append(flight + "/depth_estimation/" + f)

    with open("splits/train.txt", 'w') as f:
        f.write('\n'.join(files))
