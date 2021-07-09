sbatch --mem 10G -J residua4 -o log-%A.out residual_estimation/residual_estimation0.sh
sbatch --mem 10G -J residua1 -o log-%A.out residual_estimation/residual_estimation1.sh
sbatch --mem 10G -J residua2 -o log-%A.out residual_estimation/residual_estimation2.sh
sbatch --mem 10G -J residua3 -o log-%A.out residual_estimation/residual_estimation3.sh