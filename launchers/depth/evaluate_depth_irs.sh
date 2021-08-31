#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 32 \
--name "autoencoder_depth_irs_32_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 64 \
--name "autoencoder_depth_irs_64"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 128 \
--name "autoencoder_depth_irs_128"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 256 \
--name "autoencoder_depth_irs_256"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 512 \
--name "autoencoder_depth_irs_512_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 32 \
--name "autoencoder_depth_irs_32_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 64 \
--name "autoencoder_depth_irs_64"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 128 \
--name "autoencoder_depth_irs_128"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 256 \
--name "autoencoder_depth_irs_256"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../src/autoencoder_depth/train_autoencoder_depth.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 512 \
--name "autoencoder_depth_irs_512_irs"

