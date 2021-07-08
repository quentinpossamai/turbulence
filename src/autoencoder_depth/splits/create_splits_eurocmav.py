import random
import os

DIR = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/"
FLIGHTS = ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]

random.seed(0)
files = []
for flight in FLIGHTS:
    for f in os.listdir(DIR + flight + "/depth"):
        files.append(flight + "/depth/" + f)

with open("eurocmav_train.txt", 'w') as f:
    f.write('\n'.join(files))
