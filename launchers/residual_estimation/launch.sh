sbatch --mem 10G --gres=gpu:1 -p part1 -J residua0 -o log-residua0-%A.out residual_estimation0.sh
sbatch --mem 10G --gres=gpu:1 -p part1 -J residua1 -o log-residua1-%A.out residual_estimation1.sh
sbatch --mem 10G --gres=gpu:1 -p part1 -J residua2 -o log-residua2-%A.out residual_estimation2.sh
sbatch --mem 10G --gres=gpu:1 -p part1 -J residua3 -o log-residua3-%A.out residual_estimation3.sh
sbatch --mem 10G --gres=gpu:1 -p part1 -J residua4 -o log-residua4-%A.out residual_estimation4.sh
sbatch --mem 10G --gres=gpu:1 -p part1 -J residua5 -o log-residua5-%A.out residual_estimation5.sh