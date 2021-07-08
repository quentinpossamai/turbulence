import os
import re
import random

random.seed(0)

DIR = "/Volumes/Samsung_T5/IRS"
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

with open("irs_test.txt", 'w') as f:
    f.write('\n'.join(ex))
