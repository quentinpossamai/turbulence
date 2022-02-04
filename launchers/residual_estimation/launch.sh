git push github main &&
echo "Git: push from homedir to github done"  &&
ssh remsto /bin/bash << EOF
  find . -name ".DS_Store" -delete &&
  echo "Removing .DS_Store in homedir" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
  git pull github main &&
  sbatch --mem 10G --gres=gpu:1 -p part3 -J residual_none -o log-residual_none-%A.out residual_estimation0.sh
  sbatch --mem 10G --gres=gpu:1 -p part3 -J residual_depth -o log-residual_depth-%A.out residual_estimation1.sh
  sbatch --mem 10G --gres=gpu:1 -p part3 -J residual_norm -o log-residual_norm-%A.out residual_estimation2.sh
  sbatch --mem 10G --gres=gpu:1 -p part3 -J residual_both -o log-residual_both-%A.out residual_estimation3.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua4 -o log-residua4-%A.out residual_estimation4.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua5 -o log-residua5-%A.out residual_estimation5.sh
EOF