#!/bin/bash
cd ~/phd/turbulence/
#file="/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
#file+="VariantA/20220204_104115/default/20220204_104115/checkpoints/epoch\=0-step\=120728.ckpt"
~/anaconda3/envs/turbulence/bin/python -m src.drone_models.residual.gru_cnn_var_d\
 --activation tanh\
 --batch_size 200\
 --lr 1e-3\
 --max_epochs 10000\
 --past 40
# --load_weights $file