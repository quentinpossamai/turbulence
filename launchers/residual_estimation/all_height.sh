#!/bin/bash
cd ~/phd/turbulence/
#file="/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
#file+="VariantA/20220204_104115/default/20220204_104115/checkpoints/epoch\=0-step\=120728.ckpt"
~/anaconda3/envs/turbulence/bin/python -m src.drone_models.residual.all_height\
 --activation relu\
 --batch_size 20\
 --dropout 0.\
 --l2 0\
 --lr 1e-5\
 --max_epochs 10000\
 --past 20
# --resume_from_checkpoint $file