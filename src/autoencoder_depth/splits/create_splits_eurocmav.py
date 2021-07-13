import random
import os

DIR = "/Volumes/Samsung_T5/BlackBird/eurocmav/raw/"
FLIGHTS = ["V1_01_easy", "V1_02_medium", "V1_03_difficult"]

random.seed(0)
files = []
for flight in FLIGHTS:
    for f in os.listdir(DIR + flight + "/depth"):
        files.append(flight + "/depth/" + f)

for _ in range(50):
    random.shuffle(files)

val_size = int(len(files) * 0.2)

val = files[:val_size]
test = files[val_size:2 * val_size]
train = files[val_size * 2:]

with open("eurocmav_test.txt", 'w') as f:
    f.write('\n'.join(test))

with open("eurocmav_train.txt", 'w') as f:
    f.write('\n'.join(train))

with open("eurocmav_val.txt", 'w') as f:
    f.write('\n'.join(val))
