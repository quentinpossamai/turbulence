cluster_logs &&
ssh remsto /bin/bash << EOF
  cd ~/phd/turbulence/launchers/ &&
  sbatch --mem 10G --gres=gpu:1 -p part1 -J exec -o control_%A.out --time=8-00:00:00 script_control.sh
EOF