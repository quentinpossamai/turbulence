rsync -av --delete ~/phd/turbulence/data/euroc_mav/intermediate/ remsto:~/phd/turbulence/data/euroc_mav/intermediate/
ssh remsto /bin/bash << EOF
  rsync -av ~/phd/turbulence/data/euroc_mav/intermediate/ /storage/Datasets/BlackBird/euroc_mav/intermediate/
  chmod -R 774 /storage/Datasets/BlackBird/euroc_mav/intermediate/
  cd ~/phd/turbulence/launchers/residual_estimation/
  git pull github main
  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua0 -o log-residua0-%A.out residual_estimation0.sh
  sbatch --mem 10G --gres=gpu:1 -p part1 -J res_emb1 -o log-res_emb1-%A.out residual_estimation1.sh
  sbatch --mem 10G --gres=gpu:1 -p part1 -J res_emb2 -o log-res_emb2-%A.out residual_estimation2.sh
  sbatch --mem 10G --gres=gpu:1 -p part1 -J res_emb3 -o log-res_emb3-%A.out residual_estimation3.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua4 -o log-residua4-%A.out residual_estimation4.sh
#  sbatch --mem 10G --gres=gpu:1 -p part1 -J residua5 -o log-residua5-%A.out residual_estimation5.sh
EOF