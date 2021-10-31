#!/bin/bash
cd ~/phd/turbulence/
#file="/home/qpossamai/phd/turbulence/analysis/euroc_mav_residual_training/GRUCNNResidual_None/default/"
#file+="20210927_091013_GRUCNNResidual_bs64_epch10000_lr0.0001_h1_p60_us50_tanh_l3_None/"
#file+="checkpoints/epoch=358-step=210373.ckpt"
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence_conda.sif python3 -m src.learning.train_residual\
 --activation tanh\
 --batch_size 256\
 --embedding None\
 --layer 3\
 --lr 1e-6\
 --max_epochs 10000\
 --past 60\
 --unit_size 50
# --load_weights $file

# Train classifier
##file="/home/qpossamai/phd/turbulence/analysis/euroc_mav_residual_training/GRUCNNResidual_None/default/"
##file+="20210927_091013_GRUCNNResidual_bs64_epch10000_lr0.0001_h1_p60_us50_tanh_l3_None/"
##file+="checkpoints/epoch=358-step=210373.ckpt"
#singularity_3.7.1 exec --nv --bind /storage launchers/turbulence_conda.sif python3 -m src.learning.train\
# --batch_size 256\
# --activation relu\
# --embedding depth\
# --l1_coef 0\
# --l2_coef 0\
# --layer 3\
# --lr 1e-6\
# --unit_size 50\
# --max_epochs 10000
## --load_weights $file