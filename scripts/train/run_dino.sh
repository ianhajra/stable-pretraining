#!/usr/bin/env bash
# run_dino.sh — Submit DINO training jobs for all dataset/encoding combos.
#
# Usage:
#   bash scripts/train/run_dino.sh [--script PATH]
#
# Each encoding gets its own job.  Dataset paths, window sizes, and
# augmentation lists all come from config.sh — do not hardcode them here.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p logs

# ── Conda ─────────────────────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -z "$CONDA_BASE" ]]; then
    for _p in "$HOME/miniconda3" "$HOME/anaconda3" "/oscar/runtime/software/miniconda/23.11.0"; do
        if [[ -f "$_p/etc/profile.d/conda.sh" ]]; then
            CONDA_BASE="$_p"
            break
        fi
    done
fi
[[ -z "$CONDA_BASE" ]] && { echo "ERROR: could not locate conda installation" >&2; exit 1; }

# ── Training script path ───────────────────────────────────────────────────────
TRAIN_SCRIPT="$SCRIPT_DIR/dino.py"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --script|-s) TRAIN_SCRIPT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ── Tracking ──────────────────────────────────────────────────────────────────
N_SUBMITTED=0

# ── Helpers ───────────────────────────────────────────────────────────────────
header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

join_array() { local IFS=" "; echo "$*"; }

submit_job() {
    local job_name="$1"
    local data_dir="$2"
    local num_classes="$3"
    local window_size="$4"
    local encoding="$5"
    local dataset="$6"
    local augmentations="$7"

    sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=${job_name}
#SBATCH --partition=${SBATCH_PARTITION}
#SBATCH --nodes=${SBATCH_NODES}
#SBATCH --ntasks=${SBATCH_NTASKS}
#SBATCH --gres=${SBATCH_GPUS}
#SBATCH --cpus-per-task=${SBATCH_CPUS_PER_TASK}
#SBATCH --mem=${SBATCH_MEM}
#SBATCH --time=${SBATCH_TIME}
#SBATCH --output=logs/${job_name}_%j.out
#SBATCH --error=logs/${job_name}_%j.err

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate spt
PYTHON=python

\$PYTHON ${TRAIN_SCRIPT} \\
    --data_dir    ${data_dir} \\
    --num_classes ${num_classes} \\
    --window_size ${window_size} \\
    --encoding    ${encoding} \\
    --dataset     ${dataset} \\
    --augmentations ${augmentations}
EOF

    N_SUBMITTED=$((N_SUBMITTED + 1))
    echo "  Submitted: ${job_name}"
}

# ── SP500 × gaf_mtf ───────────────────────────────────────────────────────────
header "SP500 × gaf_mtf"
submit_job \
    "dino_gaf_sp500_w$(printf '%03d' $WINDOW_SIZE_GAF)" \
    "${SP500_BASE}/${ENCODING_SP500_GAF}/w$(printf '%03d' $WINDOW_SIZE_GAF)" \
    11 \
    "${WINDOW_SIZE_GAF}" \
    "${ENCODING_SP500_GAF}" \
    "sp500" \
    "$(join_array "${AUGMENTATIONS_GAF[@]}")"

# ── SP500 × candlestick ───────────────────────────────────────────────────────
header "SP500 × candlestick"
submit_job \
    "dino_candle_sp500_w$(printf '%03d' $WINDOW_SIZE_CANDLE)" \
    "${SP500_BASE}/${ENCODING_SP500_CANDLE}/w$(printf '%03d' $WINDOW_SIZE_CANDLE)" \
    11 \
    "${WINDOW_SIZE_CANDLE}" \
    "${ENCODING_SP500_CANDLE}" \
    "sp500" \
    "$(join_array "${AUGMENTATIONS_CANDLE[@]}")"

# ── FF × heatmap ──────────────────────────────────────────────────────────────
header "FF × heatmap"
submit_job \
    "dino_heatmap_ff_w$(printf '%03d' $WINDOW_SIZE_FF)" \
    "${FF_BASE}/${ENCODING_FF_HEATMAP}/w$(printf '%03d' $WINDOW_SIZE_FF)" \
    5 \
    "${WINDOW_SIZE_FF}" \
    "${ENCODING_FF_HEATMAP}" \
    "ff" \
    "$(join_array "${AUGMENTATIONS_HEATMAP[@]}")"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUBMISSION COMPLETE"
echo "================================================================"
echo "  Jobs submitted : ${N_SUBMITTED} / 3"
echo "  Training script: ${TRAIN_SCRIPT}"
echo "  Logs           : logs/dino_*_%j.out / .err"
echo "================================================================"
