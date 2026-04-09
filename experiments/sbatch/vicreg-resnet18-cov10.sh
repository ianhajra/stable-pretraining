#!/bin/bash
#SBATCH --job-name=vicreg-resnet18-cov10
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

mkdir -p /oscar/scratch/"$USER"/rerankme/logs
mkdir -p /oscar/scratch/"$USER"/rerankme/checkpoints/vicreg-resnet18-cov10

python experiments/cifar10/vicreg-resnet18-cov10.py

cp ~/scratch/rerankme/checkpoints/vicreg-resnet18-cov10/last.ckpt ~/data/rerankme/checkpoints/vicreg-resnet18-cov10/ 2>/dev/null || true
