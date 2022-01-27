# Get and update conda
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda update -n base -c defaults conda

# One turbulence.yaml is ready send it and create env
# sync with git ! # rsync -rtv ~/phd/turbulence/turbulence.yaml remsto:~/phd/turbulence/launchers/
conda env create -f turbulence.yaml  # on cluster or mac. Don't forget to edit it accordingly.
conda env create -f rosbag.yaml  # on the mac
conda activate turbulence

# Christofides:
# correct one from remsto:~/christofides.py
rsync -rtv ~/christofides.py ~/anaconda3/envs/turbulence/lib/python3.7/site-packages/Christofides/christofides.py  # run on cluster

# If installation failed to reset installation:
conda remove -n turbulence --all
conda clean --all