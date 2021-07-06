#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/turbulence.sif \
python3 -u ../src/depth/depth_from_pointcloud.py \
--room_path="/storage/Datasets/BlackBird/euroc_mav/raw/V1_03_difficult" \
--output_path="/storage/Datasets/BlackBird/euroc_mav/raw/V1_03_difficult/depth"
