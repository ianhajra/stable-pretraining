#!/bin/bash
# Usage: bash experiments/archive_results.sh <username>

if [ -z "$1" ]; then
    echo "Error: username required."
    echo "Usage: bash experiments/archive_results.sh <username>"
    exit 1
fi

USERNAME=$1
SCRATCH=/oscar/scratch/$USERNAME/rerankme
DATA=/oscar/home/$USERNAME/data/rerankme

mkdir -p "$DATA"/checkpoints
for run_dir in "$SCRATCH"/checkpoints/*/; do
    run_name=$(basename "$run_dir")
    mkdir -p "$DATA"/checkpoints/"$run_name"
    cp "$run_dir"/epoch=299.ckpt "$DATA"/checkpoints/"$run_name"/ 2>/dev/null && \
        echo "Archived: $run_name" || \
        echo "WARNING: missing final checkpoint for $run_name"
done

mkdir -p "$DATA"/logs
cp -r "$SCRATCH"/logs/. "$DATA"/logs/

echo "Done. Run checkquota to verify storage."
