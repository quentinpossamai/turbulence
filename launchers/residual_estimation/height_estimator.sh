#!/bin/bash
cd ~/phd/turbulence/
#file="/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
#file+="VariantA/20220204_104115/default/20220204_104115/checkpoints/epoch\=0-step\=120728.ckpt"
~/anaconda3/envs/turbulence/bin/python -m src.drone_models.residual.height_estimator\
 --activation relu\
 --batch_size 10\
 --dropout 0.5\
 --epoch 200\
 --l2 0\
 --lr 1e-5\
 --past 20\
 --stage irs_depth_head