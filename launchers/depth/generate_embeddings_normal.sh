#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/generate_embedding_normal.py \
--flight "V1_03_difficult" \
--feature_size 128 \
--name "autoencoder_normal_irs_128_irs"
