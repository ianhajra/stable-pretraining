#!/bin/bash
# Usage: bash experiments/launch_all.sh <username>

if [ -z "$1" ]; then
    echo "Error: username required."
    echo "Usage: bash experiments/launch_all.sh <username>"
    exit 1
fi

USERNAME=$1

mkdir -p /oscar/scratch/"$USERNAME"/rerankme/logs
mkdir -p /oscar/scratch/"$USERNAME"/rerankme/checkpoints

submitted=0
for f in experiments/sbatch/*.sh; do
    RUN_NAME=$(basename "$f" .sh)
    sbatch --output=/oscar/scratch/"$USERNAME"/rerankme/logs/"$RUN_NAME"-%j.out \
           --error=/oscar/scratch/"$USERNAME"/rerankme/logs/"$RUN_NAME"-%j.err \
           "$f"
    submitted=$((submitted + 1))
    echo "Submitted $RUN_NAME"
done

echo "Submitted $submitted jobs total."
echo "Monitor with: squeue -u $USERNAME"
echo "Check storage with: checkquota"
