#!/bin/bash
# Mount cluster
sshfs remsto:/home/qpossamai/phd/turbulence /Users/quentin/phd/cluster_mount
umount /Users/quentin/phd/cluster_mount/

#rsync -rtv --exclude rgb.npy --exclude depth.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' --delete remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/ ./

cd ~/phd/turbulence/data/habitat/scene_datasets/mp3d/17DRP5sb8fy/flight2/3/
rsync -rtv --exclude rgb.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' --delete remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/17DRP5sb8fy/flight2/3/ ./

cd ~/phd/turbulence/data/habitat/scene_datasets/mp3d/2azQ1b91cZZ/flight2/0/
rsync -rtv --exclude rgb.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' --delete remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/2azQ1b91cZZ/flight2/0/ ./

cd ~/phd/turbulence/data/habitat/scene_datasets/mp3d/
rsync -rtvP --exclude depth.npy --exclude rgb.npy --exclude '*.glb' --exclude '*.navmesh' --exclude '*.house' --exclude '*.ply' remsto:/home/qpossamai/phd/turbulence/data/habitat/scene_datasets/mp3d/ ./

rsync -rtvP ./labels.pkl remsto:/home/qpossamai/phd/turbulence/data/habitat/labels.pkl


rsync -atv --progress --include='*.pkl' --exclude='*' remsto:~/phd/turbulence/ ~/phd/turbulence/
rsync -atv  --include='*.png' --exclude='*' ./ remsto:~/phd/turbulence/

rsync -avP remsto:~/phd/turbulence/analysis/ ~/phd/turbulence/analysis
rsync -avP ~/phd/turbulence/analysis remsto:~/phd/turbulence/analysis/

rsync -avP remsto:~/phd/turbulence/models/ ~/phd/turbulence/models
rsync -avP ~/phd/turbulence/models/ remsto:~/phd/turbulence/models

rsync -avP ~/phd/turbulence/reports/ remsto:~/phd/turbulence/reports
rsync -avP remsto:~/phd/turbulence/reports/ ~/phd/turbulence/reports

rsync -avP ~/phd/turbulence/data/ remsto:~/phd/turbulence/data
rsync -avP remsto:~/phd/turbulence/data/ ~/phd/turbulence/data

list=$(find . -regextype posix-egrep -regex ".*.pz")
for f in $list; do echo $f ; done;
for f in $list; do echo ${f:0:23}${f:24::} ; done;
for f in $list; do mv $f ${f:0:22}; done;