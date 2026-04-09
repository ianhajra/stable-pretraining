#!/bin/bash
#SBATCH --job-name=simclr-resnet18-tau0.40
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load anaconda3/2023.09-0-aqbc
# shellcheck source=/dev/null
source activate spt

mkdir -p /oscar/scratch/"$USER"/rerankme/logs
mkdir -p /oscar/scratch/"$USER"/rerankme/checkpoints/simclr-resnet18-tau0.40

python experiments/cifar10/simclr-resnet18-tau0.40.py

cp ~/scratch/rerankme/checkpoints/simclr-resnet18-tau0.40/last.ckpt ~/data/rerankme/checkpoints/simclr-resnet18-tau0.40/ 2>/dev/null || true
