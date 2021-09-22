#!/bin/bash
cd ~/phd/turbulence/
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.residual_lightning\
 --batch_size 128\
 --activation tanh\
 --embedding none\
 --horizon 1\
 --layer 3\
 --lr 1e-4\
 --past 60\
 --unit_size 50\
 --max_epochs -1\
 --resume_from_checkpoint "/home/qpossamai/phd/turbulence/analysis/euroc_mav_residual_training/GRUCNNResidual_depth/default/20210920_151437_GRUCNNResidual_bs128_epch10000_lr1e-06_h1_p60_us50_tanh_l3_depth/checkpoints/epoch=7269-step=2130109.ckpt"