#!/bin/bash
cd ~/phd/turbulence/
#file="/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
#file+="VariantA/20220204_104115/default/20220204_104115/checkpoints/epoch\=0-step\=120728.ckpt"
~/anaconda3/envs/turbulence/bin/python -m src.drone_models.residual.fine_tuning\
 --activation relu\
 --batch_size 4\
 --l2 1e-4\
 --lr 1e-6\
 --max_epochs 10000\
 --past 40
# --resume_from_checkpoint $file