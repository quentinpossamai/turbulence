#rsync -rtv --exclude=.DS_Store remsto:~/phd/turbulence/data/euroc_mav/intermediate/ ~/phd/turbulence/data/euroc_mav/intermediate/  # revert
echo "rsync from macbook to homedir done"  &&
git push github main &&
echo "Git: push from homedir to github done"  &&
rsync -rtv --exclude=.DS_Store --delete ~/phd/turbulence/data/habitat remsto:~/phd/turbulence/data/ &&
ssh remsto /bin/bash << EOF
  cd ~/phd/turbulence/launchers/habitat_generation/ &&
  git pull github main &&
  sbatch --mem 10G -p part1 -J hab_gen0 -o log-hab_gen0-%A.out habitat_data_generation0.sh
#  sbatch --mem 10G -p part1 -J hab_gen1 -o log-hab_gen1-%A.out habitat_data_generation1.sh
#  sbatch --mem 10G -p part1 -J hab_gen2 -o log-hab_gen2-%A.out habitat_data_generation2.sh
#  sbatch --mem 10G -p part1 -J hab_gen3 -o log-hab_gen3-%A.out habitat_data_generation3.sh
#  sbatch --mem 10G -p part1 -J hab_gen4 -o log-hab_gen4-%A.out habitat_data_generation4.sh
#  sbatch --mem 10G -p part1 -J hab_gen5 -o log-hab_gen5-%A.out habitat_data_generation5.sh
#  sbatch --mem 10G -p part1 -J hab_gen6 -o log-hab_gen6-%A.out habitat_data_generation6.sh
#  sbatch --mem 10G -p part1 -J hab_gen7 -o log-hab_gen7-%A.out habitat_data_generation7.sh
#  sbatch --mem 10G -p part1 -J hab_gen8 -o log-hab_gen8-%A.out habitat_data_generation8.sh
#  sbatch --mem 10G -p part1 -J hab_gen9 -o log-hab_gen9-%A.out habitat_data_generation9.sh
EOF