#!/bin/bash
# submit_transforms_ablations.sh
#
# Submits all 24 SimCLR single-transform ablation jobs (8 transforms × 3 dataset/encoding combos).
# Window size is fixed at 63 for all runs.
# Run with:
#
#   bash scripts/experiments/submit_transforms_ablations.sh [--script PATH]
#
# Jobs
# ----
#   SP500 × gaf_mtf     × 8 transforms
#   SP500 × candlestick × 8 transforms
#   FF    × heatmap     × 8 transforms

set -e

mkdir -p logs

# ─── Activate conda environment ───────────────────────────────────────────────
# SLURM runs a non-interactive shell, so conda must be initialised explicitly.
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
ABLATION_SCRIPT="$SCRIPT_DIR/ablation_transforms.py"

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


SP500_BASE="/oscar/scratch/ihajra/finance/sp500_encoded"
FF_BASE="/oscar/scratch/ihajra/finance/ff_encoded"
WINDOW_SIZE_GAF=126
WINDOW_SIZE_CANDLE=20
WINDOW_SIZE_FF=126

TRANSFORMS=(
    "random_resized_crop"
    "horizontal_flip"
    "color_jitter"
    "random_grayscale"
    "gaussian_blur"
    "magnitude_scaling"
    "gaussian_noise"
    "temporal_masking"
)

# Short codes for job names
declare -A ABBREV=(
    ["random_resized_crop"]="rrc"
    ["horizontal_flip"]="hflip"
    ["color_jitter"]="cj"
    ["random_grayscale"]="gray"
    ["gaussian_blur"]="gblur"
    ["magnitude_scaling"]="mag"
    ["gaussian_noise"]="gnoise"
    ["temporal_masking"]="tmask"
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
    local data_dir="$2"
    local num_classes="$3"
    local window_size="$4"
    local encoding="$5"
    local dataset="$6"
    local transform="$7"

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

\$PYTHON $ABLATION_SCRIPT \\
    --data_dir $data_dir \\
    --num_classes $num_classes \\
    --window_size $window_size \\
    --encoding $encoding \\
    --dataset $dataset \\
    --transform $transform
EOF

    N_SUBMITTED=$((N_SUBMITTED + 1))
}


# ─── SP500: gaf_mtf ───────────────────────────────────────────────────────────
# header "SP500 × gaf_mtf (8 transforms)"

# for T in "${TRANSFORMS[@]}"; do
#     JOB_NAME="abl_tfm_${ABBREV[$T]}_gaf_sp500_w${WINDOW_SIZE_GAF}"
#     echo "  Submitting: $JOB_NAME  (transform=$T)"
#     submit_job "$JOB_NAME" "$SP500_BASE/gaf_mtf/w${WINDOW_SIZE_GAF}" 11 $WINDOW_SIZE_GAF gaf_mtf sp500 $T
# done


# # ─── SP500: candlestick ───────────────────────────────────────────────────────
# header "SP500 × candlestick (8 transforms)"

# for T in "${TRANSFORMS[@]}"; do
#     PADDED_CANDLE=$(printf "%03d" $WINDOW_SIZE_CANDLE)
#     JOB_NAME="abl_tfm_${ABBREV[$T]}_candle_sp500_w${PADDED_CANDLE}"
#     echo "  Submitting: $JOB_NAME  (transform=$T)"
#     submit_job "$JOB_NAME" "$SP500_BASE/candlestick/w${PADDED_CANDLE}" 11 $WINDOW_SIZE_CANDLE candlestick sp500 $T
# done

# ─── FF: heatmap ──────────────────────────────────────────────────────────────
header "FF × heatmap (8 transforms)"

for T in "${TRANSFORMS[@]}"; do
    JOB_NAME="abl_tfm_${ABBREV[$T]}_heat_ff_w${WINDOW_SIZE_FF}"
    echo "  Submitting: $JOB_NAME  (transform=$T)"
    submit_job "$JOB_NAME" "$FF_BASE/heatmap/w${WINDOW_SIZE_FF}" 5 $WINDOW_SIZE_FF heatmap ff $T
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUBMISSION COMPLETE"
echo "================================================================"
echo "  Jobs submitted : $N_SUBMITTED / 24"
echo "  Ablation script: $ABLATION_SCRIPT"
echo ""
echo "  Transforms (8):"
for T in "${TRANSFORMS[@]}"; do
    echo "    $T"
done
echo ""
echo "  Encodings:"
echo "    $SP500_BASE/gaf_mtf/w063/"
echo "    $SP500_BASE/candlestick/w063/"
echo "    $FF_BASE/heatmap/w063/"
echo ""
echo "  Logs: logs/abl_tfm_*_%j.out / .err"
echo "================================================================"