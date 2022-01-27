#!/bin/bash
cd /home/qpossamai/phd/turbulence/
echo "$PWD"
echo "$HOME"
/home/qpossamai/anaconda3/envs/turbulence/bin/python -m src.simulation.habitat_simulation --scene_number 0 --flight_number 0 --cluster True