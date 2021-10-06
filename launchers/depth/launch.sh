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
  sbatch --mem 10G -p part3 -J gen0 -o log-depth0-%A.out generate_depth0.sh
  sbatch --mem 10G -p part3 -J gen1 -o log-depth1-%A.out generate_depth1.sh
  sbatch --mem 10G -p part3 -J gen2 -o log-depth2-%A.out generate_depth2.sh
EOF
