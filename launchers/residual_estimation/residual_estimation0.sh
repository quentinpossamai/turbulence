#!/bin/bash
cd ~/phd/turbulence/
~/anaconda3/envs/turbulence/bin/python -m src.learning.train_residual\
 --activation tanh\
 --batch_size 4\
 --lr 1e-6\
 --max_epochs 10000\
 --past 40\
 --unit_size 50
# --load_weights $file