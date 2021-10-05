rsync -rtv --delete ~/phd/turbulence/data/euroc_mav/intermediate/ remsto:~/phd/turbulence/data/euroc_mav/intermediate/ &&
echo "rsync from macbook to homedir done"  &&
git push github main &&
echo "Git: push from homedir to github done"  &&
ssh remsto /bin/bash << EOF
  chmod -R 755 ~/phd/turbulence/data/euroc_mav/intermediate/ &&
  echo "changing permission in homedir done"  &&
  rsync -rv ~/phd/turbulence/data/euroc_mav/intermediate/ /storage/Datasets/BlackBird/euroc_mav/intermediate/ &&
  echo "rsync from homedir to storage done"  &&
  chmod -fR 775 /storage/Datasets/BlackBird/euroc_mav/intermediate/
  echo "chmod done"  &&
  chgrp -fR robotics /storage/Datasets/BlackBird/
  echo "chgrp done"  &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
  git pull github main &&
  sbatch --mem 10G --gres=gpu:1 -p part2 -J debug_include_fa -o log-debug_include_fa-%A.out debug_include_fa.sh
  sbatch --mem 20G --gres=gpu:1 -p part2 -J classifier_norm -o log-classifier_norm-%A.out residual_estimation1.sh
  sbatch --mem 10G --gres=gpu:1 -p part2 -J classifier_both -o log-classifier_both-%A.out residual_estimation2.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J both -o log-both-%A.out residual_estimation3.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua4 -o log-residua4-%A.out residual_estimation4.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua5 -o log-residua5-%A.out residual_estimation5.sh
EOF