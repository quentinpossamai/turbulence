sbatch --mem 10G -J residua0 -o log-residua0-%A_%a.out residual_estimation0.sh
sbatch --mem 10G -J residua1 -o log-residua1-%A_%a.out residual_estimation1.sh
sbatch --mem 10G -J residua2 -o log-residua2-%A_%a.out residual_estimation2.sh
sbatch --mem 10G -J residua3 -o log-residua3-%A_%a.out residual_estimation3.sh
sbatch --mem 10G -J residua4 -o log-residua4-%A_%a.out residual_estimation4.sh
sbatch --mem 10G -J residua5 -o log-residua5-%A_%a.out residual_estimation5.sh