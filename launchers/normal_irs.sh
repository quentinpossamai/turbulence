#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 150 \
--lr 0.00001 \
--dataset 'irs' \
--feature_size 1024 \
--name "autoencoder_normal"
