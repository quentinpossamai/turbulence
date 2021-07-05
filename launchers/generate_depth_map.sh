#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/turbulence.sif \
python3 -u ../src/depth/depth_from_pointcloud.py \
--room_path="/storage/Datasets/Blackbird/eurocmav/raw/V1_01_easy" \
--output_path="/storage/Datasets/Blackbird/eurocmav/raw/V1_01_easy/depth."