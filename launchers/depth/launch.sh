rsync -rtv ~/phd/turbulence/data/euroc_mav/intermediate/ remsto:~/phd/turbulence/data/euroc_mav/intermediate/ &&
echo "rsync from macbook to homedir done"  &&
git push github main &&
echo "Git: push from homedir to github done"  &&
ssh remsto /bin/bash << EOF
  chmod -R 755 ~/phd/turbulence/data/euroc_mav/intermediate/ &&
  echo "changing permission in homedir done"  &&
  find . -name ".DS_Store" -delete &&
  echo "Removing .DS_Store in homedir" &&
  rsync -rtv --delete ~/phd/turbulence/data/euroc_mav/intermediate/ /storage/Datasets/BlackBird/euroc_mav/intermediate/ &&
  echo "rsync from homedir to storage done"  &&
  chmod -fR 775 /storage/Datasets/BlackBird/euroc_mav/intermediate/
  echo "chmod done"  &&
  chgrp -fR robotics /storage/Datasets/BlackBird/
  echo "chgrp done"  &&
  cd /storage/Datasets/BlackBird &&
  find . -name ".DS_Store" -delete &&
  echo "Removing .DS_Store in /storage/Datasets/BlackBird" &&
# rsync -rtv --delete /storage/Datasets/BlackBird/ oggy:/storage/Datasets/BlackBird/ &&
#  echo "Synchronise Oggy with remsto" &&
  cd ~/phd/turbulence/launchers/depth/ &&
  git pull github main &&
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J gen0 -o log-gen0-%A.out generate0.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J gen1 -o log-gen1-%A.out generate1.sh
#  sbatch --mem 10G --gres=gpu:1 -p part3 -J gen2 -o log-gen2-%A.out generate2.sh
  sbatch --mem 10G -p part1 -J gen_depth1 -o log-gen_depth1-%A.out generate_embeddings_depth1.sh
  sbatch --mem 10G -p part1 -J gen_depth2 -o log-gen_depth2-%A.out generate_embeddings_depth2.sh
  sbatch --mem 10G -p part1 -J gen_depth3 -o log-gen_depth3-%A.out generate_embeddings_depth3.sh
  sbatch --mem 10G -p part1 -J gen_norm1 -o log-gen_norm1-%A.out generate_embeddings_normal1.sh
  sbatch --mem 10G -p part1 -J gen_norm2 -o log-gen_norm2-%A.out generate_embeddings_normal2.sh
  sbatch --mem 10G -p part1 -J gen_norm3 -o log-gen_norm3-%A.out generate_embeddings_normal3.sh
EOF
