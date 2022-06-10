#!/bin/bash
cd ~/phd/turbulence/
~/anaconda3/envs/turbulence/bin/python -m src.learning.autoencoder_depth.train_autoencoder_depth\
 --activation relu\
 --batch_size 160\
 --dataset irs\
 --encoding depth\
 --embedding_size 128\
 --epoch 200\
 --l2 0\
 --lr 1e-5\
 --model cnn\
 --weight_grad 1