#!/bin/bash
#SBATCH --job-name=simclr-resnet18-tau0.07
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

mkdir -p /oscar/scratch/"$USER"/rerankme/logs
mkdir -p /oscar/scratch/"$USER"/rerankme/checkpoints/simclr-resnet18-tau0.07

python experiments/cifar10/simclr-resnet18-tau0.07.py

cp ~/scratch/rerankme/checkpoints/simclr-resnet18-tau0.07/last.ckpt ~/data/rerankme/checkpoints/simclr-resnet18-tau0.07/ 2>/dev/null || true
