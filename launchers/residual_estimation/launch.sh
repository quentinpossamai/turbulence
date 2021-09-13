sbatch --mem 10G --gres=gpu:1 -p part1 -J residua0 -o log-residua0-%A.out residual_estimation0.sh
#sbatch --mem 10G --gres=gpu:1 -p part3 -J res_emb1 -o log-res_emb1-%A.out residual_estimation1.sh
#sbatch --mem 10G --gres=gpu:1 -p part3 -J res_emb2 -o log-res_emb2-%A.out residual_estimation2.sh
#sbatch --mem 10G --gres=gpu:1 -p part1 -J residua3 -o log-residua3-%A.out residual_estimation3.sh
#sbatch --mem 10G --gres=gpu:1 -p part1 -J residua4 -o log-residua4-%A.out residual_estimation4.sh
#sbatch --mem 10G --gres=gpu:1 -p part1 -J residua5 -o log-residua5-%A.out residual_estimation5.sh