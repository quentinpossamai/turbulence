cluster_logs &&
ssh remsto /bin/bash << EOF
#  find . -name ".DS_Store" -delete &&
#  echo "Removing .DS_Store in homedir" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o euroc_mav_validation_%A.out --time=8-00:00:00 euroc_mav_validation.sh
#  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o all_height_%A.out --time=8-00:00:00 all_height.sh
#  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o irs_normal_finetune_%A.out --time=8-00:00:00 height_estimator.sh

EOF