#!/bin/bash
cd ~/phd/turbulence/
~/anaconda3/envs/turbulence/bin/python -um src.data.habitat.control_test\
  --test_number 5\
  --estimate_fa True\
  --hold_fa False
