#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.system_identification  --batch_size 32 --epochs 1 --lr 1e-4 --past 10 --horizon 1
