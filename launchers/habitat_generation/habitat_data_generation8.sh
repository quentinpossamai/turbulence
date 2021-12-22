#!/bin/bash
cd ~/phd/turbulence/
singularity_3.7.1 exec --nv --bind /storage launchers/turbulence_conda.sif python3 -m src.simulation.habitat_simulation\
 --scene_number 0\
 --flight_number 8\
 --cluster True