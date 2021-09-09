#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/generate_embedding_depth.py \
--flight "V1_03_difficult"
