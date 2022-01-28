#!/bin/bash
cd /home/qpossamai/phd/turbulence/
/home/qpossamai/anaconda3/envs/turbulence/bin/python -m src.simulation.habitat_simulation\
--scene_start 6\
--scene_stop 20\
--fn_start 0\
--fn_stop 10\
--cluster True