cluster_logs &&
ssh remsto /bin/bash << EOF
#  find . -name ".DS_Store" -delete &&
#  echo "Removing .DS_Store in homedir" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o cnn_20_%A.out --time=8-00:00:00 height_estimator.sh
#  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o lr_curve%A_past.out --time=8-00:00:00 general_script.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J predict -o predict_%A.out --time=8-00:00:00 predict.sh
EOF