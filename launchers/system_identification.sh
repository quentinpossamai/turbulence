#!/bin/bash
cd ~/phd/turbulence/
singularity_3.7.1 exec --nv --bind /storage turbulence.sif python3 src/system_identification.py