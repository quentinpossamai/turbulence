#!/bin/bash
cd ~/phd/turbulence/
file="/home/qpossamai/phd/turbulence/analysis/euroc_mav_residual_training/GRUCNNResidual_None/default/"
file+="20210920_151437_GRUCNNResidual_bs128_epch10000_lr0.0001_h1_p60_us50_tanh_l3_None/"
file+="checkpoints/epoch=9999-step=2929999.ckpt"
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.learning.residual_lightning\
 --batch_size 64\
 --activation tanh\
 --embedding None\
 --horizon 1\
 --layer 3\
# --load_weights \
 --lr 1e-4\
 --past 60\
 --unit_size 50\
 --max_epochs 20000
# --resume_from_checkpoint