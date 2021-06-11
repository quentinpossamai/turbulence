#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage turbulence.sif python3 ~/phd/turbulence/src/system_identification.py