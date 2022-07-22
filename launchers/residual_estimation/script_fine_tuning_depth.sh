#!/bin/bash
cd ~/phd/turbulence/
~/anaconda3/envs/turbulence/bin/python -m src.learning.residual.fine_tuning\
 --activation relu\
 --embedding norm\
 --batch_size 4\
 --l2 0\
 --lr 1e-6\
 --max_epochs 10000\
 --past 40