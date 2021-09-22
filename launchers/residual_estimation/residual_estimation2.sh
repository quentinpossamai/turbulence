#!/bin/bash
cd ~/phd/turbulence/
file="/home/qpossamai/phd/turbulence/analysis/euroc_mav_residual_training/"
file+="GRUCNNResidual_norm/default/20210920_151801_GRUCNNResidual_bs128_epch10000_lr1e-06_h1_p60_us50_tanh_l3_norm/"
file+="checkpoints/epoch=9999-step=2929999.ckpt"
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.residual_lightning\
 --batch_size 64\
 --activation tanh\
 --embedding norm\
 --horizon 1\
 --layer 3\
 --lr 1e-6\
 --past 60\
 --unit_size 50\
 --max_epochs 20000\
 --resume_from_checkpoint $file
