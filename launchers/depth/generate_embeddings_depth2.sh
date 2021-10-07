#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif python3 -m src.autoencoder_depth.generate_embedding_depth\
  --flight "V1_02_medium"