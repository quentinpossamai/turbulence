#!/bin/bash
cd ~/phd/turbulence/
~/anaconda3/envs/turbulence/bin/python -m src.simulation.data_gen --scene_start 0 --scene_stop 23 --fn_start 0 --fn_stop 10 --cluster True