#!/bin/bash
# run_transform_ablations.sh
#
# Submits SimCLR transform ablation jobs (one per transform, fixed window size).
# Run with:
#
#   bash scripts/train/run_transform_ablations.sh [--script PATH]
#
# Jobs
# ----
#   For each transform in the ablation set, submits a job with a single transform.

set -e

mkdir -p logs

# ─── Activate conda environment ───────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -z "$CONDA_BASE" ]]; then
    for _p in "$HOME/miniconda3" "$HOME/anaconda3" "/oscar/runtime/software/miniconda/23.11.0"; do
        if [[ -f "$_p/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="$_p"
            break
        fi
    done
fi
if [[ -z "$CONDA_BASE" ]]; then
    echo "ERROR: could not locate conda installation" >&2
    exit 1
fi
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spt
PYTHON=python

# ─── Parse arguments ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABLATION_SCRIPT="$SCRIPT_DIR/ablation_transformations.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --script|-s)
            ABLATION_SCRIPT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ─── Parameters ──────────────────────────────────────────────────────────────
DATA_DIR="/oscar/scratch/ihajra/finance/sp500_encoded/gaf_mtf/w063"
WINDOW_SIZE=63
ENCODING="gaf_mtf"
DATASET="sp500"
NUM_CLASSES=11
SEED=42
BATCH_SIZE=256
NUM_WORKERS=4

# ─── Transform names (must match those in ablation_transformations.py) ────────
TRANSFORM_NAMES=(
    RandomResizedCrop
    RandomHorizontalFlip
    ColorJitter
    RandomGrayscale
    GaussianBlur
    MagnitudeScaling
    GaussianNoiseInjection
    RandomTemporalMasking
)

# ─── Tracking ─────────────────────────────────────────────────────────────────
N_SUBMITTED=0

# ─── Helper ───────────────────────────────────────────────────────────────────
header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

submit_job() {
    local job_name="$1"
    local transform_name="$2"

    sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err

CONDA_BASE=\$(conda info --base 2>/dev/null)
if [[ -z "\$CONDA_BASE" ]]; then
    for _p in "\$HOME/miniconda3" "\$HOME/anaconda3" "/oscar/runtime/software/miniconda/23.11.0"; do
        if [[ -f "\$_p/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="\$_p"
            break
        fi
    done
fi
if [[ -z "\$CONDA_BASE" ]]; then
    echo "ERROR: could not locate conda installation" >&2
    exit 1
fi
source "\$CONDA_BASE/etc/profile.d/conda.sh"
conda activate spt
PYTHON=python

$PYTHON $ABLATION_SCRIPT \
    --data_dir $DATA_DIR \
    --num_classes $NUM_CLASSES \
    --window_size $WINDOW_SIZE \
    --encoding $ENCODING \
    --dataset $DATASET \
    --seed $SEED \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --transform_name $transform_name
EOF

    N_SUBMITTED=$((N_SUBMITTED + 1))
}

# ─── Submit jobs ──────────────────────────────────────────────────────────────
header "SimCLR Transform Ablation (1 per transform)"

for TNAME in "${TRANSFORM_NAMES[@]}"; do
    JOB_NAME="abl_trans_${TNAME,,}"
    echo "  Submitting: $JOB_NAME"
    submit_job "$JOB_NAME" "$TNAME"
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUBMISSION COMPLETE"
echo "================================================================"
echo "  Jobs submitted : $N_SUBMITTED / ${#TRANSFORM_NAMES[@]}"
echo "  Ablation script: $ABLATION_SCRIPT"
echo ""
echo "  Data: $DATA_DIR"
echo "  Window size: $WINDOW_SIZE"
echo "  Encoding: $ENCODING"
echo "  Dataset: $DATASET"
echo "  Logs: logs/abl_trans_*_%j.out / .err"
echo "================================================================"
