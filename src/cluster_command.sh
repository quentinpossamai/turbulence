#!/bin/bash
#rsync -rtv --exclude rgb.npy --exclude depth.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' --delete remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/ ./

cd ~/phd/turbulence/data/habitat/scene_datasets/mp3d/17DRP5sb8fy/flight2/3/
rsync -rtv --exclude rgb.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' --delete remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/17DRP5sb8fy/flight2/3/ ./

cd ~/phd/turbulence/data/habitat/scene_datasets/mp3d/2azQ1b91cZZ/flight2/0/
rsync -rtv --exclude rgb.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' --delete remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/2azQ1b91cZZ/flight2/0/ ./

cd ~/phd/turbulence/data/habitat/scene_datasets/mp3d/2t7WUuJeko7/flight2/0/
rsync -rtv --exclude rgb.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' --delete remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/2t7WUuJeko7/flight2/0/ ./

rsync -atv  --include='*.png' --exclude='*' remsto:~/phd/turbulence/ ./
rsync -atv  --include='*.png' --exclude='*' ./ remsto:~/phd/turbulence/