#rsync -rtv --exclude=.DS_Store remsto:~/phd/turbulence/data/euroc_mav/intermediate/ ~/phd/turbulence/data/euroc_mav/intermediate/  # revert
echo "rsync from macbook to homedir done"  &&
git push github main &&
echo "Git: push from homedir to github done"  &&
#rsync -rtv --exclude=.DS_Store --delete ~/phd/turbulence/data/habitat remsto:~/phd/turbulence/data/ &&
ssh remsto /bin/bash << EOF
  cd ~/phd/turbulence/launchers/habitat_generation/ &&
  git pull github main &&
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J hab_gen0 -o log-hab_gen0-%A.out --time=8-00:00:00 habitat_data_generation0.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J hab_gen1 -o log-hab_gen1-%A.out --time=8-00:00:00 habitat_data_generation1.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J hab_gen2 -o log-hab_gen2-%A.out --time=8-00:00:00 habitat_data_generation2.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J hab_gen3 -o log-hab_gen3-%A.out --time=8-00:00:00 habitat_data_generation3.sh
  sbatch --mem 10G --gres=gpu:1 -p part3 -J emb_normal -o log-emb_normal-%A.out --time=8-00:00:00 gen_emb.sh
EOF