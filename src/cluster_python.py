import os
import sys
import shutil

path = f'/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/'
os.chdir(path)
scenes = os.listdir(path)

for e in scenes:
    tmp = f'{path}{e}/flight3/'
    try:
        os.listdir(tmp)
        shutil.rmtree(tmp)
    except FileNotFoundError:
        print(f'{e} has no flight directory')

for e in scenes:
    tmp = f'{path}{e}/'
    print(os.listdir(tmp))

import subprocess
# Get the total size of the workspace
os.chdir(f'/home/qpossamai/phd/turbulence/')
cmd = f'du -sh .'
# os.system(cmd)
terra_size = float(subprocess.check_output(cmd, shell=True).decode('utf-8')[:3])

# for e in scenes:
#     tmp = f'{path}{e}/flight2/'
#     for i in range(20):
#         cmd += f' --exclude=./data/habitat/scene_datasets/mp3d/{e}/flight2/{i}/rgb.pkl'
# cmd += ' .'
# Count file size without rgb.pkl
total = 0
cmd = 'du -sh --exclude=rgb.npy .'
cmd = f'du -chd 1 --exclude=rgb.npy ./data/'
os.system(cmd)