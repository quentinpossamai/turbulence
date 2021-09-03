#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.depth_estimation.depth_from_pointcloud\
  --room_path 2 --output_path nan --integrated True