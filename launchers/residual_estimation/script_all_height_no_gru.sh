#!/bin/bash
cd ~/phd/turbulence/
#file="/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
#file+="VariantA/20220204_104115/default/20220204_104115/checkpoints/epoch\=0-step\=120728.ckpt"
~/anaconda3/envs/turbulence/bin/python -m src.learning.residual.all_height_no_gru\
 --activation relu\
 --batch_size 320\
 --l2 0.0001\
 --lr 1e-6\
 --max_epochs 10000\
 --past 1
# --resume_from_checkpoint $file