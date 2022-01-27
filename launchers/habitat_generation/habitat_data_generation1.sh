#!/bin/bash
cd ~/phd/turbulence/
eval "$(conda shell.bash hook)"
conda activate turbulence
python -m src.simulation.habitat_simulation --scene_number 0 --flight_number 1 --cluster True