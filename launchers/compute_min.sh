#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/turbulence.sif \
python3 -u ../src/depth/compute_depth_range.py \
