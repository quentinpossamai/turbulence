cluster_logs &&
ssh remsto /bin/bash << EOF
#  find . -name ".DS_Store" -delete &&
#  echo "Removing .DS_Store in homedir" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
#  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o euroc_mav_validation_%A.out --time=8-00:00:00 script_euroc_mav_validation.sh
#  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o all_height_%A.out --time=8-00:00:00 all_height.sh
#  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o irs_normal_finetune_%A.out --time=8-00:00:00 height_estimator.sh

#  sbatch --mem 20G --gres=gpu:1 -p part1 -J exec -o all_height_no_gru_%A.out --time=8-00:00:00 script_all_height_no_gru.sh
  sbatch --mem 20G --gres=gpu:1 -p part3 -J exec -o all_height_%A.out --time=8-00:00:00 script_all_height.sh
#  sbatch --mem 20G --gres=gpu:1 -p part1 -J exec -o emb_height_%A.out --time=8-00:00:00 script_emb_height.sh
#  sbatch --mem 20G --gres=gpu:1 -p part1 -J exec -o emb_height2_%A.out --time=8-00:00:00 script_emb_height2.sh
#  sbatch --mem 20G --gres=gpu:1 -p part1 -J exec -o fine_tuning_depth_%A.out --time=8-00:00:00 script_fine_tuning_depth.sh
#  sbatch --mem 20G --gres=gpu:1 -p part1 -J exec -o fine_tuning_norm2_%A.out --time=8-00:00:00 script_fine_tuning_norm2.sh

EOF