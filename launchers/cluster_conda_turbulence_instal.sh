# Get and update conda
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
conda update -n base -c defaults conda

# One cluster_turbulence.yaml is ready send it and create env
# sync with git ! # rsync -rtv ~/phd/turbulence/cluster_turbulence.yaml remsto:~/phd/turbulence/launchers/
conda env create -f cluster_turbulence.yaml  # on cluster
conda env create -f mac_turbulence.yaml  # on the mac
conda activate turbulence

# Christofides:
# /opt/anaconda3/envs/turbulence/lib/python3.7/site-packages/Christofides/christofides.py ou bien remsto:~/christofides.py
rsync -rtv ~/christofides.py ~/anaconda3/envs/turbulence/lib/python3.7/site-packages/Christofides/christofides.py  # run on cluster

# If installation failed to reset installation:
conda remove -n turbulence --all
conda clean --all

# Tmp

conda create -n turbulence python==3.7
conda install -y numpy
conda install -y pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -y pytorch-lightning==1.4.9 -c conda-forge
conda install -y habitat-sim withbullet headless -c conda-forge -c aihabitat
conda install -y matplotlib==3.4.2 -c conda-forge
conda install -y tensorboard==2.6.0 -c conda-forge
conda install -y torchdiffeq==0.2.2 -c conda-forge


conda install -y pickleshare==0.7.5 -c anaconda
conda install -y scipy==1.6.3 -c conda-forge
conda install -y imageio==2.9.0 -c anaconda
conda install -y pandas==1.1.3 -c anaconda
conda install -y ffmpeg==4.2.2 -c anaconda
conda install -y fire==0.4.0 -c conda-forge
conda install -y open3d==0.11.2 -c open3d-admin
conda install -y pillow==8.3.1
conda install -y scikit-learn==0.24.2 -c conda-forge
conda install -y tqdm==4.62.3 -c conda-forge
conda install -y wget==1.20.1 -c anaconda
conda install -y yaml==0.2.5 -c anaconda
conda install -y gitpython==3.1.24 -c conda-forge
conda install numpy-quaternion==2021.11.4.15.26.3
conda install imageio-ffmpeg==0.4.5



pip install christofides==1.0.1
pip install opencv-python==4.5.3.56