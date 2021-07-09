import os
print(f"CWD: {os.getcwd()}")
# Locally import tensorboard logs and model weights from the cluster and launch it.
tb_cluster_path = "/home/qpossamai/phd/turbulence/analysis/"
tb_local_path = "analysis/"
# os.system(f"scp -r remsto:{tb_cluster_path} {tb_local_path}")
os.system(f"rsync -av remsto:{tb_cluster_path} {tb_local_path}")

# Launch tensorboard
tensorboard_cmd = f"tensorboard --logdir /Users/quentin/phd/turbulence/analysis/euroc_mav_learned_hexarotor_training/" \
                  "training/ --load_fast=false --samples_per_plugin scalars=999999999"
print(tensorboard_cmd)
# os.system(tensorboard_cmd)
