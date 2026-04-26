#!/usr/bin/env bash
# run_retry.sh — Resubmit the 3 jobs that failed with CUDA illegal memory access.
#
# Failed jobs (node-level CUDA error during teardown, not a code bug):
#   lejepa_heatmap_ff_w020      (LeJEPA × FF × heatmap)
#   sup_gaf_sp500_w126          (Supervised × SP500 × gaf_mtf)
#   sup_candle_sp500_w020       (Supervised × SP500 × candlestick)
#
# Usage:
#   bash scripts/train/run_retry.sh

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
    local train_script="$2"
    local data_dir="$3"
    local num_classes="$4"
    local window_size="$5"
    local encoding="$6"
    local dataset="$7"
    local augmentations="$8"

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

\$PYTHON ${train_script} \\
    --data_dir    ${data_dir} \\
    --num_classes ${num_classes} \\
    --window_size ${window_size} \\
    --encoding    ${encoding} \\
    --dataset     ${dataset} \\
    --augmentations ${augmentations}
EOF

    echo "  Submitted: ${job_name}"
}

N_SUBMITTED=0

# ── 1. LeJEPA × FF × heatmap ─────────────────────────────────────────────────
header "LeJEPA × FF × heatmap"
submit_job \
    "lejepa_heatmap_ff_w$(printf '%03d' $WINDOW_SIZE_FF)" \
    "$SCRIPT_DIR/lejepa.py" \
    "${FF_BASE}/${ENCODING_FF_HEATMAP}/w$(printf '%03d' $WINDOW_SIZE_FF)" \
    5 \
    "${WINDOW_SIZE_FF}" \
    "${ENCODING_FF_HEATMAP}" \
    "ff" \
    "$(join_array "${AUGMENTATIONS_HEATMAP[@]}")"
N_SUBMITTED=$((N_SUBMITTED + 1))

# ── 2. Supervised × SP500 × gaf_mtf ──────────────────────────────────────────
header "Supervised × SP500 × gaf_mtf"
submit_job \
    "sup_gaf_sp500_w$(printf '%03d' $WINDOW_SIZE_GAF)" \
    "$SCRIPT_DIR/supervised.py" \
    "${SP500_BASE}/${ENCODING_SP500_GAF}/w$(printf '%03d' $WINDOW_SIZE_GAF)" \
    11 \
    "${WINDOW_SIZE_GAF}" \
    "${ENCODING_SP500_GAF}" \
    "sp500" \
    "$(join_array "${AUGMENTATIONS_GAF[@]}")"
N_SUBMITTED=$((N_SUBMITTED + 1))

# ── 3. Supervised × SP500 × candlestick ──────────────────────────────────────
header "Supervised × SP500 × candlestick"
submit_job \
    "sup_candle_sp500_w$(printf '%03d' $WINDOW_SIZE_CANDLE)" \
    "$SCRIPT_DIR/supervised.py" \
    "${SP500_BASE}/${ENCODING_SP500_CANDLE}/w$(printf '%03d' $WINDOW_SIZE_CANDLE)" \
    11 \
    "${WINDOW_SIZE_CANDLE}" \
    "${ENCODING_SP500_CANDLE}" \
    "sp500" \
    "$(join_array "${AUGMENTATIONS_CANDLE[@]}")"
N_SUBMITTED=$((N_SUBMITTED + 1))

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUBMISSION COMPLETE"
echo "================================================================"
echo "  Jobs submitted : ${N_SUBMITTED} / 3"
echo "  Logs           : logs/*_%j.out / .err"
echo "================================================================"
