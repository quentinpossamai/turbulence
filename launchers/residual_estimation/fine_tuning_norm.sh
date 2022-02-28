#!/bin/bash
cd ~/phd/turbulence/
#file="/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
#file+="FineTunning/20220225_185029/epoch\=1-step\=98011-val_loss\=0.53.ckpt"
~/anaconda3/envs/turbulence/bin/python -m src.drone_models.residual.fine_tuning\
 --activation relu\
 --embedding norm\
 --batch_size 4\
 --l2 0\
 --lr 1e-7\
 --max_epochs 10000\
 --past 40
# --resume_from_checkpoint $file