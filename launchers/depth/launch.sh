sbatch --mem 10G -p part3 -J depth0 -o log-depth0-%A.out generate_depth0.sh
sbatch --mem 10G -p part3 -J depth1 -o log-depth1-%A.out generate_depth1.sh
sbatch --mem 10G -p part3 -J depth2 -o log-depth2-%A.out generate_depth2.sh
#sbatch --mem 10G --gres=gpu:1 -p part1 -J residua1 -o log-residua1-%A.out residual_estimation1.sh