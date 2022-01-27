#!/bin/bash
cd ~/phd/turbulence/
conda activate turbulence
python -m src.simulation.habitat_simulation --scene_number 0 --flight_number 4 --cluster True