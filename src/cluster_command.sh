#!/bin/bash
find . -regextype posix-egrep -regex ".*surface_dist\.npy" > to_dl.txt
rsync -rtv --exclude rgb.npy --exclude depth.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/ ./