#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.residual_estimation\
 --batch_size 128\
 --epochs 10000\
 --lr 1e-6\
 --horizon 1\
 --past 60\
 --unit_size 50\
 --activation tanh\
 --layer 3\
 --embedding norm