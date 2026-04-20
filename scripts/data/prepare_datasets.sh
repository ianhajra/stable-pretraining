#!/bin/bash
# prepare_datasets.sh
#
# Orchestrates the full dataset preparation pipeline for stable-pretraining.
# Submit with:
#
#   sbatch scripts/data/prepare_datasets.sh [--force] [--output-dir PATH]
#
# Steps
# -----
#   1. Download S&P 500 OHLCV parquet files           (scripts/data/download_sp500.py)
#   2. Encode S&P 500 images for all four window sizes  (scripts/data/encode_sp500.py)
#   3. Download Fama-French 30-industry data            (scripts/data/download_ff_30_industry.py)
#   4. Encode FF heatmap images                         (scripts/data/encode_ff.py)
#
# Each step is skipped if its sentinel output already exists unless --force is given.

#SBATCH --job-name=prepare_datasets
#SBATCH --output=logs/prepare_datasets_%j.out
#SBATCH --error=logs/prepare_datasets_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00

set -e

mkdir -p logs

# ─── Activate conda environment ───────────────────────────────────────────────
# SLURM runs a non-interactive shell, so conda must be initialised explicitly.
CONDA_BASE=$(conda info --base 2>/dev/null)
if [[ -z "$CONDA_BASE" ]]; then
    # Fallback: check common install locations
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
FORCE=0
OUTPUT_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=1
            shift
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# Set default output dir to /oscar/scratch/ihajra/finance if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="/oscar/scratch/ihajra/finance"
else
    # Remove trailing slash if present
    OUTPUT_DIR="${OUTPUT_DIR%/}"
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# ─── Sentinel paths ───────────────────────────────────────────────────────────
SP500_SENTINEL="$OUTPUT_DIR/sp500/sp500_train.parquet"
SP500_ENC_SENTINEL="$OUTPUT_DIR/sp500_encoded/gaf_mtf/w063/metadata.csv"
FF_SENTINEL="$OUTPUT_DIR/ff/30_industry_daily_vw.parquet"
FF_ENC_SENTINEL="$OUTPUT_DIR/ff_encoded/heatmap/w063/metadata.csv"

# ─── Tracking ─────────────────────────────────────────────────────────────────
STEP1_STATUS="skipped"
STEP2_STATUS="skipped"
STEP3_STATUS="skipped"
STEP4_STATUS="skipped"

# ─── Helper ───────────────────────────────────────────────────────────────────
header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

# ─── Step 1: Download S&P 500 parquets ───────────────────────────────────────
header "Step 1 / 4 — Download S&P 500 OHLCV data"

if [[ $FORCE -eq 1 ]] || [[ ! -f "$SP500_SENTINEL" ]]; then
    echo "  Running: python scripts/data/download_sp500.py"
    $PYTHON scripts/data/download_sp500.py --output_dir "$OUTPUT_DIR/sp500/"
    STEP1_STATUS="run"
else
    echo "  SKIPPED — $SP500_SENTINEL already exists (use --force to re-run)"
fi

# ─── Step 2: Encode S&P 500 images ───────────────────────────────────────────
header "Step 2 / 4 — Encode S&P 500 images (all four window sizes)"

if [[ $FORCE -eq 1 ]] || [[ ! -f "$SP500_ENC_SENTINEL" ]]; then
    echo "  Running: python scripts/data/encode_sp500.py"
    $PYTHON scripts/data/encode_sp500.py --input_dir "$OUTPUT_DIR/sp500" --output_dir "$OUTPUT_DIR/sp500_encoded/"
    STEP2_STATUS="run"
else
    echo "  SKIPPED — $SP500_ENC_SENTINEL already exists (use --force to re-run)"
fi

# ─── Step 3: Download FF 30-industry parquet ─────────────────────────────────
header "Step 3 / 4 — Download Fama-French 30-industry data"

if [[ $FORCE -eq 1 ]] || [[ ! -f "$FF_SENTINEL" ]]; then
    echo "  Running: python scripts/data/download_ff_30_industry.py"
    $PYTHON scripts/data/download_ff_30_industry.py --output_dir "$OUTPUT_DIR/ff/"
    STEP3_STATUS="run"
else
    echo "  SKIPPED — $FF_SENTINEL already exists (use --force to re-run)"
fi

# ─── Step 4: Download FF factors + encode heatmap images ─────────────────────
header "Step 4 / 4 — Encode FF 30-industry heatmap images (all four window sizes)"

if [[ $FORCE -eq 1 ]] || [[ ! -f "$FF_ENC_SENTINEL" ]]; then
    echo "  Running: python scripts/data/encode_ff.py"
    $PYTHON scripts/data/encode_ff.py --input_parquet "$OUTPUT_DIR/ff/30_industry_daily_vw.parquet" --output_dir "$OUTPUT_DIR/ff_encoded/"
    STEP4_STATUS="run"
else
    echo "  SKIPPED — $FF_ENC_SENTINEL already exists (use --force to re-run)"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  PIPELINE COMPLETE"
echo "================================================================"
echo "  Step 1 (S&P 500 download)        : $STEP1_STATUS"
echo "  Step 2 (S&P 500 encoding)        : $STEP2_STATUS"
echo "  Step 3 (FF 30-industry download) : $STEP3_STATUS"
echo "  Step 4 (FF heatmap encoding)     : $STEP4_STATUS"
echo ""
echo "  Outputs:"
echo "    $OUTPUT_DIR/sp500/                          (parquet files)"
echo "    $OUTPUT_DIR/sp500_encoded/gaf_mtf/w{020,063,126,252}/"
echo "    $OUTPUT_DIR/sp500_encoded/candlestick/w{020,063,126,252}/"
echo "    $OUTPUT_DIR/ff_encoded/heatmap/w{020,063,126,252}/"
echo "================================================================"
