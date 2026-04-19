#!/usr/bin/env bash
# prepare_datasets.sh
#
# Orchestrates the full dataset preparation pipeline for stable-pretraining.
# Must be run from the repository root:
#
#   bash scripts/data/prepare_datasets.sh [--force]
#
# Steps
# -----
#   1. Download S&P 500 OHLCV parquet files          (scripts/data/download_sp500.py)
#   2. Encode S&P 500 images for all four window sizes (scripts/data/encode_sp500.py)
#   3. Download and encode FF heatmap images           (scripts/data/encode_ff.py)
#
# Each step is skipped if its sentinel output already exists unless --force is given.

set -e

# ─── Resolve Python executable ────────────────────────────────────────────────
# Prefer the conda env's Python if it exists, otherwise fall back to python3.
CONDA_PYTHON="$HOME/miniconda3/envs/stable-pretraining/bin/python"
if [[ ! -x "$CONDA_PYTHON" ]]; then
    CONDA_PYTHON="$HOME/anaconda3/envs/stable-pretraining/bin/python"
fi
if [[ -x "$CONDA_PYTHON" ]]; then
    PYTHON="$CONDA_PYTHON"
    echo "Using conda env Python: $PYTHON"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    echo "Using system python3: $(which python3)"
else
    echo "ERROR: no Python interpreter found" >&2
    exit 1
fi

# ─── Parse arguments ──────────────────────────────────────────────────────────
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ─── Sentinel paths ───────────────────────────────────────────────────────────
SP500_SENTINEL="data/sp500/sp500_train.parquet"
SP500_ENC_SENTINEL="data/sp500_encoded/gaf_mtf/w063/metadata.csv"
FF_SENTINEL="data/ff/30_industry_daily_vw.parquet"
FF_ENC_SENTINEL="data/ff_encoded/heatmap/w063/metadata.csv"

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
    $PYTHON scripts/data/download_sp500.py
    STEP1_STATUS="run"
else
    echo "  SKIPPED — $SP500_SENTINEL already exists (use --force to re-run)"
fi

# ─── Step 2: Encode S&P 500 images ───────────────────────────────────────────
header "Step 2 / 4 — Encode S&P 500 images (all four window sizes)"

if [[ $FORCE -eq 1 ]] || [[ ! -f "$SP500_ENC_SENTINEL" ]]; then
    echo "  Running: python scripts/data/encode_sp500.py"
    $PYTHON scripts/data/encode_sp500.py
    STEP2_STATUS="run"
else
    echo "  SKIPPED — $SP500_ENC_SENTINEL already exists (use --force to re-run)"
fi

# ─── Step 3: Download FF 30-industry parquet ─────────────────────────────────
header "Step 3 / 4 — Download Fama-French 30-industry data"

if [[ $FORCE -eq 1 ]] || [[ ! -f "$FF_SENTINEL" ]]; then
    echo "  Running: python scripts/data/download_ff_30_industry.py"
    $PYTHON scripts/data/download_ff_30_industry.py
    STEP3_STATUS="run"
else
    echo "  SKIPPED — $FF_SENTINEL already exists (use --force to re-run)"
fi

# ─── Step 4: Download FF factors + encode heatmap images ─────────────────────
header "Step 4 / 4 — Encode FF 30-industry heatmap images (all four window sizes)"

if [[ $FORCE -eq 1 ]] || [[ ! -f "$FF_ENC_SENTINEL" ]]; then
    echo "  Running: python scripts/data/encode_ff.py"
    $PYTHON scripts/data/encode_ff.py
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
echo "    data/sp500/                          (parquet files)"
echo "    data/sp500_encoded/gaf_mtf/w{020,063,126,252}/"
echo "    data/sp500_encoded/candlestick/w{020,063,126,252}/"
echo "    data/ff_encoded/heatmap/w{020,063,126,252}/"
echo "================================================================"
