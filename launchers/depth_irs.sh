#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/turbulence.sif \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 150 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 1024 \
--name "autoencoder_depth"
