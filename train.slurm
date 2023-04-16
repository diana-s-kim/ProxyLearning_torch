#!/bin/bash -l

#SBATCH --job-name=train_proxy
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=[v100]    
#SBATCH --time=01:00:00
#SBATCH --output=./outslurm/%j-slurm.out

module load machine_learning
srun ./script/train.sh
