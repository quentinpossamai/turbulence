import os

print(f"CWD: {os.getcwd()}")
# Locally import tensorboard logs and model weights from the cluster and launch it.
# tb_cluster_path = "/home/qpossamai/phd/turbulence/analysis/irs_encoder_decoder/"
# tb_local_path = "/Users/quentin/phd/turbulence/analysis/irs_encoder_decoder/"
# os.system(f"rsync -avP --exclude=.DS_Store --delete remsto:{tb_cluster_path} {tb_local_path}")

# os.system(f'ssh remsto /bin/bash << EOF\n'
#           # f'cd /home/qpossamai &&\n'
#           # f'chmod -R 755 {tb_cluster_path} &&\n'
#           # f'echo "changing permission in analysis done"  &&\n'
#           f'find . -name ".DS_Store" -delete &&\n'
#           f'echo "Removing .DS_Store in homedir"\n'
#           f'EOF')
# os.system(f"rsync -av --delete remsto:{tb_cluster_path} {tb_local_path}")  # exception

tb_cluster_path = "/home/qpossamai/phd/turbulence/"
tb_local_path = "/Users/quentin/phd/turbulence/"
cmd = (f"rsync -avP --exclude .DS_Store --exclude *pyc --exclude __pycache__ --exclude .git --exclude .idea"  
       f" --exclude /analysis --exclude /data" 
       f" {tb_local_path} remsto:{tb_cluster_path}")
print(f'{cmd}')
os.system(cmd)

tb_cluster_path = "/home/qpossamai/phd/turbulence/src/"
tb_local_path = "/Users/quentin/phd/turbulence/src/"
cmd = (f"rsync -avP --exclude .DS_Store --exclude *pyc --exclude __pycache__ --delete --delete-excluded"  
       f" {tb_local_path} remsto:{tb_cluster_path}")
print(f'{cmd}')
os.system(cmd)
