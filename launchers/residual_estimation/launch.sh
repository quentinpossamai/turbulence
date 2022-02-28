git push github main &&
echo "Git: push from homedir to github done"  &&
ssh remsto /bin/bash << EOF
#  find . -name ".DS_Store" -delete &&
#  echo "Removing .DS_Store in homedir" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
  git pull github main &&
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J all_height_d -o log-all_height_d-%A.out --time=8-00:00:00 all_height.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J emb_height -o log-emb_height_1-%A.out --time=8-00:00:00 emb_height.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J emb_height2 -o log-emb_height_2-%A.out --time=8-00:00:00 emb_height2.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J fine_tuning_depth -o log-depth_ft-%A.out --time=8-00:00:00 fine_tuning_depth.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J fine_tuning_norm -o log-norm_ft-%A.out --time=8-00:00:00 fine_tuning_norm.sh
#  sbatch --mem 10G --gres=gpu:8 -p part3 -J fine_tuning_norm2 -o log-norm_ft2-%A.out --time=8-00:00:00 fine_tuning_norm2.sh
  sbatch --mem 10G --gres=gpu:1 -p part2   -J dataviz -o log-dataviz-%A.out --time=8-00:00:00 dataviz.sh
#
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J tr_a -o log-tr_a-%A.out --time=8-00:00:00 variant_a.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J tr_d -o log-tr_d-%A.out --time=8-00:00:00 variant_d.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J residual_depth -o log-residual_depth-%A.out residual_estimation1.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J residual_norm -o log-residual_norm-%A.out residual_estimation2.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J residual_both -o log-residual_both-%A.out residual_estimation3.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua4 -o log-residua4-%A.out residual_estimation4.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua5 -o log-residua5-%A.out residual_estimation5.sh
EOF