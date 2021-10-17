rsync -rtv --exclude=.DS_Store ~/phd/turbulence/data/euroc_mav/intermediate/ remsto:~/phd/turbulence/data/euroc_mav/intermediate/ &&
# rsync -rtv --exclude=.DS_Store remsto:~/phd/turbulence/data/euroc_mav/intermediate/ ~/phd/turbulence/data/euroc_mav/intermediate/  # revert
echo "rsync from macbook to homedir done"  &&
git push github main &&
echo "Git: push from homedir to github done"  &&
ssh remsto /bin/bash << EOF
  chmod -R 755 ~/phd/turbulence/data/euroc_mav/intermediate/ &&
  echo "changing permission in homedir done"  &&
  find . -name ".DS_Store" -delete &&
  echo "Removing .DS_Store in homedir" &&
  rsync -rtv --exclude=.DS_Store --delete ~/phd/turbulence/data/euroc_mav/intermediate/ /storage/Datasets/BlackBird/euroc_mav/intermediate/ &&
#  rsync -rtv --exclude=.DS_Store /storage/Datasets/BlackBird/euroc_mav/intermediate/ ~/phd/turbulence/data/euroc_mav/intermediate/ && # revert
  echo "rsync from homedir to storage done"  &&
  chmod -fR 775 /storage/Datasets/BlackBird/euroc_mav/intermediate/
  echo "chmod done"  &&
  chgrp -fR robotics /storage/Datasets/BlackBird/
  echo "chgrp done"  &&
  cd /storage/Datasets/BlackBird &&
  find . -name ".DS_Store" -delete &&
  echo "Removing .DS_Store in /storage/Datasets/BlackBird" &&
# rsync -rtv --exclude=.DS_Store --delete /storage/Datasets/BlackBird/ oggy:/storage/Datasets/BlackBird/ &&
# rsync -rtv --exclude=.DS_Store oggy:/storage/Datasets/BlackBird/ /storage/Datasets/BlackBird/ &&  # revert
#  echo "Synchronise Oggy with remsto" &&
  cd ~/phd/turbulence/launchers/residual_estimation/ &&
  git pull github main &&
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J classifier_depth -o log-classifier_depth-%A.out residual_estimation0.sh
  sbatch --mem 10G --gres=gpu:1 -p part3 -J classifier_norm -o log-classifier_norm-%A.out residual_estimation1.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J classifier_both -o log-classifier_both-%A.out residual_estimation2.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J both -o log-both-%A.out residual_estimation3.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua4 -o log-residua4-%A.out residual_estimation4.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua5 -o log-residua5-%A.out residual_estimation5.sh
EOF