import os
import re
import random

random.seed(0)

DIR = "/home/sjanny/IRS/"
ex = []
for scene in os.listdir(DIR):
    for detail in os.listdir(DIR + "/" + scene):
        all_files = os.listdir(DIR + "/" + scene + "/" + detail)
        dict = {}

        for f in all_files:
            res = re.findall(r'([a-z])_([0-9]*).*', f)[0]
            if int(res[1]) not in dict.keys():
                dict[int(res[1])] = {}
            dict[int(res[1])][res[0]] = f
        for key in dict.keys():
            str = f"{scene}/{detail}/{dict[key]['l']} {scene}/{detail}/{dict[key]['r']} {scene}/{detail}/{dict[key]['d']} {scene}/{detail}/{dict[key]['n']}"
            ex.append(str)

for _ in range(50):
    random.shuffle(ex)

val_size = int(len(ex) * 0.2)

val = ex[:val_size]
test = ex[val_size:2 * val_size]
train = ex[val_size * 2:]

with open("irs_test.txt", 'w') as f:
    f.write('\n'.join(test))

with open("irs_train.txt", 'w') as f:
    f.write('\n'.join(train))

with open("irs_val.txt", 'w') as f:
    f.write('\n'.join(val))
