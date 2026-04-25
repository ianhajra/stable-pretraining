#!/usr/bin/env bash
# run_all.sh — Submit ALL method training jobs for every dataset/encoding combo.
#
# Usage:
#   bash scripts/train/run_all.sh
#
# This script is the single source of truth for sbatch headers, dataset paths,
# window sizes, and augmentation lists.  Changing a value here propagates to
# every method automatically.
#
# Note: individual run_*.sh scripts source config.sh independently and remain
# fully usable standalone.  This master inlines its own config and calls sbatch
# directly — sourcing config.sh inside a child script would overwrite any
# variables exported from here, so the individual launchers are not called.
#
# To skip a method, comment out its line in the METHODS array below.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED CONFIG — edit here, changes propagate to every method.
# ══════════════════════════════════════════════════════════════════════════════

# ── SLURM directives ──────────────────────────────────────────────────────────
SBATCH_PARTITION="gpu"
SBATCH_NODES=1
SBATCH_NTASKS=1
SBATCH_GPUS="gpu:1"
SBATCH_CPUS_PER_TASK=6
SBATCH_MEM="32G"
SBATCH_TIME="12:00:00"

# ── Dataset base paths ────────────────────────────────────────────────────────
SP500_BASE="/oscar/scratch/ihajra/finance/sp500_encoded"
FF_BASE="/oscar/scratch/ihajra/finance/ff_encoded"

# ── Encoding names ────────────────────────────────────────────────────────────
ENCODING_SP500_GAF="gaf_mtf"
ENCODING_SP500_CANDLE="candlestick"
ENCODING_FF_HEATMAP="heatmap"

# ── Window sizes ──────────────────────────────────────────────────────────────
WINDOW_SIZE_GAF=126       # SP500 × gaf_mtf
WINDOW_SIZE_CANDLE=20     # SP500 × candlestick
WINDOW_SIZE_FF=20         # FF × heatmap

# ── Augmentations (one array per encoding) ────────────────────────────────────
# To add a new augmentation for a given encoding, append its name to the
# relevant array below.  Valid names:
#   random_resized_crop  horizontal_flip  color_jitter  random_grayscale
#   gaussian_blur        magnitude_scaling gaussian_noise  temporal_masking
#
# TODO: add augmentation names here to include them in the gaf_mtf pipeline,
#       e.g. AUGMENTATIONS_GAF+=("my_new_transform")
AUGMENTATIONS_GAF=(
    "random_resized_crop"
    "horizontal_flip"
    "color_jitter"
    "random_grayscale"
    "gaussian_blur"
    "magnitude_scaling"
    "gaussian_noise"
    "temporal_masking"
)

# TODO: add augmentation names here to include them in the candlestick pipeline,
#       e.g. AUGMENTATIONS_CANDLE+=("my_new_transform")
AUGMENTATIONS_CANDLE=(
    "random_resized_crop"
    "horizontal_flip"
    "color_jitter"
    "random_grayscale"
    "gaussian_blur"
    "magnitude_scaling"
    "gaussian_noise"
    "temporal_masking"
)

# TODO: add augmentation names here to include them in the heatmap pipeline,
#       e.g. AUGMENTATIONS_HEATMAP+=("my_new_transform")
AUGMENTATIONS_HEATMAP=(
    "random_resized_crop"
    "horizontal_flip"
    "color_jitter"
    "random_grayscale"
    "gaussian_blur"
    "magnitude_scaling"
    "gaussian_noise"
    "temporal_masking"
)

# ── Methods ───────────────────────────────────────────────────────────────────
# Each entry: "job_prefix|training_script_filename"
# To skip a method, comment out its line.
METHODS=(
    "simclr|simclr.py"
    "barlow|barlow.py"
    "vicreg|vicreg.py"
    "mae|mae.py"
    "ijepa|ijepa.py"
    "lejepa|lejepa.py"
    "sup|supervised.py"
)

# ══════════════════════════════════════════════════════════════════════════════
#  END OF CONFIG — nothing below this line needs to be edited.
# ══════════════════════════════════════════════════════════════════════════════

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

# ── Tracking ──────────────────────────────────────────────────────────────────
N_SUBMITTED=0
N_METHODS=${#METHODS[@]}
N_TOTAL=$((N_METHODS * 3))

# ── Helpers ───────────────────────────────────────────────────────────────────
header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

join_array() { local IFS=" "; echo "$*"; }

# submit_job ARGS:
#   $1  train_script  — absolute path to the Python training script
#   $2  job_name
#   $3  data_dir      — full path to the encoded dataset split on disk
#   $4  num_classes   — number of probe classes (11 SP500, 5 FF)
#   $5  window_size   — integer window size
#   $6  encoding      — encoding name string
#   $7  dataset       — "sp500" or "ff"
#   $8  augmentations — space-separated augmentation names
submit_job() {
    local train_script="$1"
    local job_name="$2"
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

    N_SUBMITTED=$((N_SUBMITTED + 1))
    echo "  Submitted: ${job_name}"
}

# ── Submissions ───────────────────────────────────────────────────────────────
for method_entry in "${METHODS[@]}"; do
    prefix="${method_entry%%|*}"
    train_script="$SCRIPT_DIR/${method_entry##*|}"

    header "${prefix} × SP500 × gaf_mtf"
    submit_job \
        "$train_script" \
        "${prefix}_gaf_sp500_w$(printf '%03d' $WINDOW_SIZE_GAF)" \
        "${SP500_BASE}/${ENCODING_SP500_GAF}/w$(printf '%03d' $WINDOW_SIZE_GAF)" \
        11 \
        "${WINDOW_SIZE_GAF}" \
        "${ENCODING_SP500_GAF}" \
        "sp500" \
        "$(join_array "${AUGMENTATIONS_GAF[@]}")"

    header "${prefix} × SP500 × candlestick"
    submit_job \
        "$train_script" \
        "${prefix}_candle_sp500_w$(printf '%03d' $WINDOW_SIZE_CANDLE)" \
        "${SP500_BASE}/${ENCODING_SP500_CANDLE}/w$(printf '%03d' $WINDOW_SIZE_CANDLE)" \
        11 \
        "${WINDOW_SIZE_CANDLE}" \
        "${ENCODING_SP500_CANDLE}" \
        "sp500" \
        "$(join_array "${AUGMENTATIONS_CANDLE[@]}")"

    header "${prefix} × FF × heatmap"
    submit_job \
        "$train_script" \
        "${prefix}_heatmap_ff_w$(printf '%03d' $WINDOW_SIZE_FF)" \
        "${FF_BASE}/${ENCODING_FF_HEATMAP}/w$(printf '%03d' $WINDOW_SIZE_FF)" \
        5 \
        "${WINDOW_SIZE_FF}" \
        "${ENCODING_FF_HEATMAP}" \
        "ff" \
        "$(join_array "${AUGMENTATIONS_HEATMAP[@]}")"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUBMISSION COMPLETE"
echo "================================================================"
echo "  Jobs submitted : ${N_SUBMITTED} / ${N_TOTAL}"
echo "  Logs           : logs/*_%j.out / .err"
echo "================================================================"
