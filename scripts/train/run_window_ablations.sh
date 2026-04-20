#!/bin/bash
# submit_window_ablations.sh
#
# Submits all 12 SimCLR window-size ablation jobs (3 encodings × 4 window sizes).
# Run with:
#
#   bash scripts/experiments/submit_window_ablations.sh [--script PATH]
#
# Jobs
# ----
#   SP500 × gaf_mtf      × windows 20, 63, 126, 252
#   SP500 × candlestick  × windows 20, 63, 126, 252
#   FF    × heatmap      × windows 20, 63, 126, 252

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
ABLATION_SCRIPT="$SCRIPT_DIR/ablation_window_size.py"

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
WINDOW_SIZES=(20 63 126 252)

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

    sbatch <<EOF
#!/bin/bash

#SBATCH --job-name=$job_name
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
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
    --dataset $dataset
EOF

    N_SUBMITTED=$((N_SUBMITTED + 1))
}

# ─── SP500: gaf_mtf ───────────────────────────────────────────────────────────
header "SP500 × gaf_mtf (4 window sizes)"

for W in "${WINDOW_SIZES[@]}"; do
    JOB_NAME="abl_gaf_mtf_sp500_w$(printf '%03d' $W)"
    echo "  Submitting: $JOB_NAME"
    submit_job "$JOB_NAME" "$SP500_BASE/gaf_mtf/w$W" 11 $W gaf_mtf sp500
done

# ─── SP500: candlestick ───────────────────────────────────────────────────────
header "SP500 × candlestick (4 window sizes)"

for W in "${WINDOW_SIZES[@]}"; do
    JOB_NAME="abl_candle_sp500_w$(printf '%03d' $W)"
    echo "  Submitting: $JOB_NAME"
    submit_job "$JOB_NAME" "$SP500_BASE/candlestick/w$W" 11 $W candlestick sp500
done

# ─── FF: heatmap ──────────────────────────────────────────────────────────────
header "FF × heatmap (4 window sizes)"

for W in "${WINDOW_SIZES[@]}"; do
    JOB_NAME="abl_heatmap_ff_w$(printf '%03d' $W)"
    echo "  Submitting: $JOB_NAME"
    submit_job "$JOB_NAME" "$FF_BASE/heatmap/w$W" 5 $W heatmap ff
done

# ─── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  SUBMISSION COMPLETE"
echo "================================================================"
echo "  Jobs submitted : $N_SUBMITTED / 12"
echo "  Ablation script: $ABLATION_SCRIPT"
echo ""
echo "  Encodings:"
echo "    $SP500_BASE/gaf_mtf/w{020,063,126,252}/"
echo "    $SP500_BASE/candlestick/w{020,063,126,252}/"
echo "    $FF_BASE/heatmap/w{020,063,126,252}/"
echo ""
echo "  Logs: logs/abl_*_%j.out / .err"
echo "================================================================"