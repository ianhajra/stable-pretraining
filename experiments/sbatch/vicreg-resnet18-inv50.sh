#!/bin/bash
#SBATCH --job-name=vicreg-resnet18-inv50
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
mkdir -p /oscar/scratch/"$USER"/rerankme/checkpoints/vicreg-resnet18-inv50

python experiments/cifar10/vicreg-resnet18-inv50.py

cp ~/scratch/rerankme/checkpoints/vicreg-resnet18-inv50/last.ckpt ~/data/rerankme/checkpoints/vicreg-resnet18-inv50/ 2>/dev/null || true
