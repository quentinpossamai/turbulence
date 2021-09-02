#!/bin/bash
singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 32 \
--name "autoencoder_normal_irs_32_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 64 \
--name "autoencoder_normal_irs_64_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 128 \
--name "autoencoder_normal_irs_128_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 256 \
--name "autoencoder_normal_irs_256_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'irs' \
--feature_size 512 \
--name "autoencoder_normal_irs_512_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 32 \
--name "autoencoder_normal_irs_32_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 64 \
--name "autoencoder_normal_irs_64_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 128 \
--name "autoencoder_normal_irs_128_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 256 \
--name "autoencoder_normal_irs_256_irs"

singularity_3.7.1 exec --nv --bind /storage $HOME/singularity_dgx.img \
python3 -u ../../src/autoencoder_depth/train_autoencoder_normal.py \
--epoch 0 \
--lr 0.0001 \
--dataset 'eurocmav' \
--feature_size 512 \
--name "autoencoder_normal_irs_512_irs"