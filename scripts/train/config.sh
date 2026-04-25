#!/usr/bin/env bash
# config.sh — shared configuration for all training launcher scripts.
# Source this file at the top of any launcher:
#
#   source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# ── SLURM directives ────────────────────────────────────────────────────────
# Taken verbatim from the existing launcher scripts.
# Override individual values in the launcher after sourcing if needed.

SBATCH_PARTITION="gpu"
SBATCH_NODES=1
SBATCH_NTASKS=1
SBATCH_GPUS="gpu:1"
SBATCH_CPUS_PER_TASK=6
SBATCH_MEM="32G"
SBATCH_TIME="12:00:00"
# Log paths use the job name and SLURM job ID; launchers expand these at
# submission time: logs/${JOB_NAME}_%j.out / .err


# ── Dataset base paths ───────────────────────────────────────────────────────

SP500_BASE="/oscar/scratch/ihajra/finance/sp500_encoded"
FF_BASE="/oscar/scratch/ihajra/finance/ff_encoded"

# Encoding subdirectory names (appended to the base path as {BASE}/{ENCODING}/w{NNN}/)
ENCODING_SP500_GAF="gaf_mtf"
ENCODING_SP500_CANDLE="candlestick"
ENCODING_FF_HEATMAP="heatmap"


# ── Window sizes ─────────────────────────────────────────────────────────────
# From run_transform_ablations.sh: one fixed window size per encoding.

WINDOW_SIZE_GAF=126       # SP500 × gaf_mtf
WINDOW_SIZE_CANDLE=20     # SP500 × candlestick
WINDOW_SIZE_FF=20         # FF × heatmap  (stored as w020 on disk)


# ── Augmentation pipelines ───────────────────────────────────────────────────
# Each array lists the augmentation names that will be included in the
# training pipeline for that encoding.  Names must match the --transform
# choices accepted by ablation_transforms.py.
#
# To add a new augmentation for a specific encoding:
#   1. Implement it in ablation_transforms.py (add a branch to _make_view).
#   2. Add its name as a new element in the array for the encoding(s) you
#      want it applied to — it will automatically be included in every job
#      submitted for that encoding.

# SP500 × gaf_mtf augmentations
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

# SP500 × candlestick augmentations
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

# FF × heatmap augmentations
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
