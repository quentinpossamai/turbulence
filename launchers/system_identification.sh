#!/bin/bash
workspace=~/phd/turbulence/
cd $workspace
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence.sif "PYTHONPATH=~/phd/turbulence/ python3 src/system_identification.py"
