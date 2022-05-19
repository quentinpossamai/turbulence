git push github main &&
echo "Git: push from homedir to github done"  &&
ssh remsto /bin/bash << EOF
#  find . -name ".DS_Store" -delete &&
#  echo "Removing .DS_Store in homedir" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
  git pull github main &&
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J all_height -o log_all_height_%A.out --time=8-00:00:00 all_height.sh
  sbatch --mem 10G -J predict -o log_predict_%A.out --time=8-00:00:00 predict.sh
EOF