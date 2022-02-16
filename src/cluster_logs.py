import os
print(f"CWD: {os.getcwd()}")
# Locally import tensorboard logs and model weights from the cluster and launch it.
# tb_cluster_path = "/home/qpossamai/phd/turbulence/analysis/euroc_mav_residual_training/"
# tb_local_path = "/Users/quentin/phd/turbulence/analysis/euroc_mav_residual_training/"
tb_cluster_path = "/home/qpossamai/phd/turbulence/analysis/habitat_residual_training/"
tb_local_path = "/Users/quentin/phd/turbulence/analysis/habitat_residual_training/"
os.system(f"rsync -rtv --exclude=.DS_Store remsto:{tb_cluster_path} {tb_local_path}")
os.system(f"rsync -rtv --exclude=.DS_Store {tb_local_path} remsto:{tb_cluster_path}")
# os.system(f'ssh remsto /bin/bash << EOF\n'
#           # f'cd /home/qpossamai &&\n'
#           # f'chmod -R 755 {tb_cluster_path} &&\n'
#           # f'echo "changing permission in analysis done"  &&\n'
#           f'find . -name ".DS_Store" -delete &&\n'
#           f'echo "Removing .DS_Store in homedir"\n'
#           f'EOF')
# os.system(f"rsync -av --delete remsto:{tb_cluster_path} {tb_local_path}")  # exception

# Launch tensorboard
tensorboard_cmd = f"tensorboard --logdir {tb_local_path} --load_fast=false --samples_per_plugin scalars=999999999"
print(tensorboard_cmd)
# os.system(tensorboard_cmd)

# Synchronise data from mac to cluster
# rsync -av ~/phd/turbulence/data/euroc_mav/intermediate/ remsto:~/phd/turbulence/data/euroc_mav/intermediate/
