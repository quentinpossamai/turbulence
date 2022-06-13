cluster_logs &&
ssh remsto /bin/bash << EOF
#  find . -name ".DS_Store" -delete &&
#  echo "Removing .DS_Store in homedir" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
#  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o irs_depth_finetune_%A.out --time=8-00:00:00 height_estimator.sh
#  sbatch --mem 10G -p part1 -J predict -o predict_all_%A.out --time=8-00:00:00 predict.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J general -o ae_lr_curve_%A.out --time=8-00:00:00 general_script.sh
  sbatch --mem 10G --gres=gpu:1 -p part1 -J ae_training -o ae_training_%A.out --time=8-00:00:00 ae_irs.sh
#  conda activate turbulence
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J ae_training -o ae_training_sweep_%A.out --time=8-00:00:00 ./launchers/residual_estimation/launch_sweep.sh
EOF