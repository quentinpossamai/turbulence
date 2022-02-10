#!/bin/bash
cd ~/phd/turbulence/
file="/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
file+="VariantA/20220204_104115/default/20220204_104115/checkpoints/epoch\=0-step\=120728.ckpt"
~/anaconda3/envs/turbulence/bin/python -m src.learning.train_residual\
 --activation tanh\
 --batch_size 4\
 --lr 1e-6\
 --max_epochs 10000\
 --past 40\
 --unit_size 50\
 --load_weights $file