#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.residual_estimation\
  --batch_size 128 --epochs 10000 --lr 1e-3 --past 50 --horizon 1 --unit_size 50
