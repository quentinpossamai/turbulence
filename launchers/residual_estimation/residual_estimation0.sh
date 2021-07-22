#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.residual_estimation\
  --batch_size 1 --epochs 200 --lr 1e-4 --past 20 --horizon 1 --unit_size 128
