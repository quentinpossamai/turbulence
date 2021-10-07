#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/singularity_steeven.img python3 -m src.autoencoder_depth.generate_embedding_normal\
  --flight "V1_01_easy"