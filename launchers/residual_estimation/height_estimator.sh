#!/bin/bash
cd ~/phd/turbulence/
~/anaconda3/envs/turbulence/bin/python -m src.learning.residual.height_estimator\
 --activation relu\
 --batch_size 5\
 --dropout 0.5\
 --epoch 200\
 --l2 0.\
 --lr 7e-6\
 --cnn_layer_num 9\
 --cnn_channel_num 64\
 --regression_size 50\
 --past 20\
 --stage irs_normal_finetune\
 --resume 20220701_222109
