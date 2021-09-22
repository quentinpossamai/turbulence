#!/bin/bash
cd ~/phd/turbulence/
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.residual_lightning\
 --batch_size 128\
 --activation tanh\
 --embedding both\
 --horizon 1\
 --layer 3\
 --lr 1e-6\
 --past 60\
 --unit_size 50\
 --max_epochs -1