#!/bin/bash
cd ~/phd/turbulence/
~/anaconda3/envs/turbulence/bin/python -m src.learning.autoencoder_depth.train_autoencoder_depth\
 --activation relu\
 --batch_size 2\
 --dataset irs\
 --embedding_size 128\
 --encoder_channel 100\
 --encoder_layer 6\
 --encoding depth\
 --epoch 200\
 --l2 0\
 --lr 1e-5\
 --model cnn\
 --resume none\
 --weight_grad 1