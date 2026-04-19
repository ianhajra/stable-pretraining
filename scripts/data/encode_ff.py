"""Encode Fama-French 30-Industry daily returns as heatmap images for SSL pre-training.

Reads industry returns from the pre-downloaded parquet at data/ff/30_industry_daily_vw.parquet.
Downloads the FF 5-factor daily data directly from Kenneth French's website
(same approach as download_ff_30_industry.py, no pandas_datareader needed).

Temporal split (strict calendar boundaries):
  Train : start  — 2019-12-31
  Val   : 2020-01-01 — 2021-12-31
  Test  : 2022-01-01 — 2023-12-31

Each image is (224, 224, 3) uint8 RGB, derived from a (30, window_size) log-return
matrix that is cross-sectionally z-scored, clipped to [-3, 3], and rescaled to
[0, 255].

Factor labels (1–5 quintile bins per window per factor) are computed from
cumulative factor returns within the window. Quintile boundaries are fitted on
the train split only and clamped for val/test.

Output
------
  data/ff_encoded/
    heatmap/
      dataset_dict.json     ← detected by spt.data.HFDataset
      train/ val/ test/     ← HF arrow files
      images/train/…        ← source PNGs (path-referenced from arrow)
      metadata.csv

  Load in training scripts:
    spt.data.HFDataset("data/ff_encoded/heatmap", split="train", transform=...)
    # sample keys: "image" (PIL), "label" (Mkt-RF quintile 1–5),
    #              "label_mktrf", "label_smb", "label_hml", "label_rmw", "label_cma",
    #              "start_date", "end_date"

Usage
-----
  python scripts/data/encode_ff.py [--output_dir PATH] [--input_parquet PATH]
                                   [--window_size N] [--stride N] [--seed N]
"""

import argparse
import io
import re
import random
import zipfile
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pandas as pd
import requests
from PIL import Image
from scipy.ndimage import zoom

# ─── Constants ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "ff_encoded"
DEFAULT_PARQUET = _REPO_ROOT / "data" / "ff" / "30_industry_daily_vw.parquet"
DEFAULT_STRIDE = 2
DEFAULT_SEED = 42

WINDOW_SIZES = [20, 63, 126, 252]  # all four window sizes rendered in one run

FACTORS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

TRAIN_END = "2019-12-31"
VAL_START = "2020-01-01"
VAL_END = "2021-12-31"
TEST_START = "2022-01-01"
TEST_END = "2023-12-31"

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
N_QUINTILES = 5
IMG_SIZE = 224
SPLITS = ["train", "val", "test"]


# ─── Data download ────────────────────────────────────────────────────────────


def load_industry_data(parquet_path: Path) -> pd.DataFrame:
    """Load pre-downloaded 30-industry daily returns from parquet."""
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    df = df / 100.0  # parquet stores percentage returns
    return df


def download_factors() -> pd.DataFrame:
    """Download FF 5-factor daily data directly from Kenneth French's website.
    Returns a DataFrame with columns [Mkt-RF, SMB, HML, RMW, CMA] in decimal.
    """
    print(f"Downloading 5-factor data from {FACTORS_URL} ...")
    resp = requests.get(FACTORS_URL, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        raw_text = zf.read(csv_name).decode("latin-1")

    # Find the data block: skip header lines until we hit the column header row
    lines = raw_text.splitlines()
    data_lines = []
    header_found = False
    for line in lines:
        stripped = line.strip()
        if not header_found:
            # Column header row starts with a comma or has recognisable factor names
            if re.match(r"^\s*,?\s*Mkt", stripped, re.IGNORECASE):
                data_lines.append(line)
                header_found = True
        else:
            # Data rows start with 8-digit dates; stop at blank lines after data
            if re.match(r"^\s*\d{8}", stripped):
                data_lines.append(line)
            elif data_lines and not stripped:
                break  # end of first data block

    raw = "\n".join(data_lines)
    fac = pd.read_csv(io.StringIO(raw), index_col=0)
    fac.index = pd.to_datetime(fac.index.astype(str), format="%Y%m%d")
    fac.index.name = "date"
    fac.columns = fac.columns.str.strip()
    # Keep only the 5 model factors (drop RF)
    fac = fac[[c for c in FACTORS if c in fac.columns]]
    fac = fac / 100.0  # convert from percentage
    return fac


def load_data(parquet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load industry returns from parquet and download 5-factor data."""
    ind = load_industry_data(parquet_path)
    fac = download_factors()

    common = ind.index.intersection(fac.index)
    ind = ind.loc[common]
    fac = fac.loc[common]

    print(
        f"  Industry data  : {ind.shape}  {ind.index[0].date()} — {ind.index[-1].date()}"
    )
    print(
        f"  Factor data    : {fac.shape}  {fac.index[0].date()} — {fac.index[-1].date()}"
    )
    return ind, fac


# ─── Image encoding ───────────────────────────────────────────────────────────


def encode_heatmap(window_returns: np.ndarray) -> np.ndarray:
    """Encode a (30, T) float64 log-return matrix into a (224, 224, 3) uint8 image.

    Steps:
      1. Rank-transform each column: replace the 30 values with their ordinal
         ranks (1–30), then rescale to [0, 255].  Robust to outliers by
         construction — a single extreme return cannot dominate — and naturally
         exposes cross-sectional banding because industries with similar factor
         loadings receive similar ranks consistently across time.
      2. Bilinear resize from (30, T) to (224, 224) via scipy.ndimage.zoom
      3. Replicate single channel → 3-channel RGB
    """
    n_assets, T = window_returns.shape  # (30, T)

    # 1. Column-wise rank transform: argsort twice gives ranks (0-indexed)
    ranks = np.argsort(np.argsort(window_returns, axis=0), axis=0).astype(np.float32)
    # Rescale ranks [0, n_assets-1] → [0, 255]
    rescaled = ranks / (n_assets - 1) * 255.0  # (30, T)

    # 2. Bilinear resize (30, T) → (224, 224)
    zoom_r = IMG_SIZE / rescaled.shape[0]
    zoom_c = IMG_SIZE / rescaled.shape[1]
    resized = zoom(rescaled, (zoom_r, zoom_c), order=1)  # bilinear
    resized = np.clip(resized, 0.0, 255.0).astype(np.uint8)

    # 3. Replicate to RGB
    rgb = np.stack([resized, resized, resized], axis=2)  # (224, 224, 3)
    return rgb


# ─── Window generation ────────────────────────────────────────────────────────


def generate_windows(
    ind: pd.DataFrame,
    fac: pd.DataFrame,
    window_size: int,
    stride: int,
) -> list[dict]:
    """Slide a window over ind and fac and return a list of window dicts containing:
    start_idx, end_idx (integer positions into ind/fac),
    start_date, end_date (YYYYMMDD strings),
    log_returns (30, window_size) float64,
    cum_factors dict {factor_name: float}
    """
    n = len(ind)
    ind_vals = ind.values  # (N, 30)
    fac_vals = fac.values  # (N, 5)
    dates = ind.index

    windows = []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size  # exclusive
        r = ind_vals[start:end]  # (T, 30)
        log_r = np.log1p(r).T  # (30, T)

        f = fac_vals[start:end]  # (T, 5)
        cum_factors = {FACTORS[i]: float(f[:, i].sum()) for i in range(len(FACTORS))}

        windows.append(
            {
                "start_idx": start,
                "end_idx": end,
                "start_date": dates[start].strftime("%Y%m%d"),
                "end_date": dates[end - 1].strftime("%Y%m%d"),
                "log_returns": log_r,
                "cum_factors": cum_factors,
            }
        )
    return windows


# ─── Quintile labelling ───────────────────────────────────────────────────────


def fit_quintile_boundaries(windows: list[dict]) -> dict[str, np.ndarray]:
    """Compute quintile boundaries for each factor from training windows."""
    boundaries: dict[str, np.ndarray] = {}
    for factor in FACTORS:
        vals = np.array([w["cum_factors"][factor] for w in windows])
        # 5 quintiles → 4 inner boundaries (0%, 20%, 40%, 60%, 80%, 100%)
        boundaries[factor] = np.percentile(vals, [20, 40, 60, 80])
    return boundaries


def assign_quintile(value: float, boundaries: np.ndarray) -> int:
    """Return quintile bin in {1, ..., 5}, clamped to the boundary range."""
    return int(np.searchsorted(boundaries, value, side="right")) + 1


def label_windows(windows: list[dict], boundaries: dict[str, np.ndarray]) -> list[dict]:
    """Attach quintile label columns to each window dict (mutates in place)."""
    for w in windows:
        for factor in FACTORS:
            key = f"label_{factor.lower().replace('-', '')}"
            w[key] = assign_quintile(w["cum_factors"][factor], boundaries[factor])
    return windows


# ─── Split + render ───────────────────────────────────────────────────────────


def render_split(
    split: str,
    windows: list[dict],
    img_dir: Path,
    window_size: int,
) -> list[dict]:
    """Render PNGs for all windows in a split to img_dir and return metadata rows.
    Each row includes the absolute image path under key "image_path".
    """
    img_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []

    for w in windows:
        fname = f"ff30_{w['end_date']}_w{window_size:03d}.png"
        arr = encode_heatmap(w["log_returns"])
        img_path = img_dir / fname
        Image.fromarray(arr, mode="RGB").save(img_path)

        row = {
            "filename": fname,
            "image_path": str(img_path),
            "split": split,
            "start_date": w["start_date"],
            "end_date": w["end_date"],
        }
        for factor in FACTORS:
            key = f"label_{factor.lower().replace('-', '')}"
            row[key] = w[key]
        meta_rows.append(row)

    return meta_rows


def build_hf_dataset_dict(
    all_meta: list[dict],
    heatmap_root: Path,
) -> None:
    """Build a HuggingFace DatasetDict from rendered metadata and save to
    heatmap_root using save_to_disk(), creating dataset_dict.json that
    spt.data.HFDataset detects for local loading.

    Primary label is the Mkt-RF quintile (1–5).  All five factor quintile
    labels are also stored as integer columns.
    """
    # Quintile class names 1–5
    quintile_names = [str(i) for i in range(1, N_QUINTILES + 1)]

    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "label": hf_datasets.ClassLabel(names=quintile_names),  # Mkt-RF quintile
            "label_mktrf": hf_datasets.Value("int32"),
            "label_smb": hf_datasets.Value("int32"),
            "label_hml": hf_datasets.Value("int32"),
            "label_rmw": hf_datasets.Value("int32"),
            "label_cma": hf_datasets.Value("int32"),
            "start_date": hf_datasets.Value("string"),
            "end_date": hf_datasets.Value("string"),
        }
    )

    split_datasets: dict[str, hf_datasets.Dataset] = {}
    for split in SPLITS:
        rows = [r for r in all_meta if r["split"] == split]
        if not rows:
            continue
        split_datasets[split] = hf_datasets.Dataset.from_dict(
            {
                "image": [r["image_path"] for r in rows],
                "label": [
                    r["label_mktrf"] - 1 for r in rows
                ],  # 0-indexed for ClassLabel
                "label_mktrf": [r["label_mktrf"] for r in rows],
                "label_smb": [r["label_smb"] for r in rows],
                "label_hml": [r["label_hml"] for r in rows],
                "label_rmw": [r["label_rmw"] for r in rows],
                "label_cma": [r["label_cma"] for r in rows],
                "start_date": [r["start_date"] for r in rows],
                "end_date": [r["end_date"] for r in rows],
            },
            features=features,
        )

    dd = hf_datasets.DatasetDict(split_datasets)
    dd.save_to_disk(str(heatmap_root))
    print(
        f"  HF DatasetDict -> {heatmap_root}  "
        f"(splits: {', '.join(f'{k}={len(v)}' for k, v in dd.items())})"
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode FF 30-industry returns as heatmap images "
        "for all four window sizes (20, 63, 126, 252) in a single run."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root output directory (default: data/ff_encoded/)",
    )
    parser.add_argument(
        "--input_parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to 30_industry_daily_vw.parquet",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Window stride in trading days (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Load / download data once — shared across all window sizes
    ind, fac = load_data(args.input_parquet)

    # 2. Calendar-date splits (same boundaries for all window sizes)
    train_ind = ind.loc[:TRAIN_END]
    train_fac = fac.loc[:TRAIN_END]
    val_ind = ind.loc[VAL_START:VAL_END]
    val_fac = fac.loc[VAL_START:VAL_END]
    test_ind = ind.loc[TEST_START:TEST_END]
    test_fac = fac.loc[TEST_START:TEST_END]

    print("\nSplit sizes:")
    print(
        f"  train : {len(train_ind):,} days  ({train_ind.index[0].date()} — {train_ind.index[-1].date()})"
    )
    print(
        f"  val   : {len(val_ind):,} days  ({val_ind.index[0].date()} — {val_ind.index[-1].date()})"
    )
    print(
        f"  test  : {len(test_ind):,} days  ({test_ind.index[0].date()} — {test_ind.index[-1].date()})"
    )

    meta_cols = [
        "filename",
        "split",
        "start_date",
        "end_date",
        "label_mktrf",
        "label_smb",
        "label_hml",
        "label_rmw",
        "label_cma",
    ]

    # ── Loop over all window sizes ────────────────────────────────────────────
    for window_size in WINDOW_SIZES:
        ws_tag = f"w{window_size:03d}"
        print(f"\n{'=' * 60}")
        print(f"WINDOW SIZE: {window_size} days  ({ws_tag})")
        print("=" * 60)

        # 3. Generate windows per split
        print(f"  Generating windows (size={window_size}, stride={args.stride}) ...")
        train_windows = generate_windows(train_ind, train_fac, window_size, args.stride)
        val_windows = generate_windows(val_ind, val_fac, window_size, args.stride)
        test_windows = generate_windows(test_ind, test_fac, window_size, args.stride)
        print(f"  train: {len(train_windows):,} windows")
        print(f"  val  : {len(val_windows):,} windows")
        print(f"  test : {len(test_windows):,} windows")

        # 4. Fit quintile boundaries on train only; apply to all splits
        print("  Fitting quintile boundaries on train split ...")
        boundaries = fit_quintile_boundaries(train_windows)
        for factor in FACTORS:
            b = boundaries[factor]
            print(
                f"    {factor:<8s}: {b[0]:+.5f}  {b[1]:+.5f}  {b[2]:+.5f}  {b[3]:+.5f}"
            )

        label_windows(train_windows, boundaries)
        label_windows(val_windows, boundaries)
        label_windows(test_windows, boundaries)

        # 5. Render images per split  (images go to heatmap/{ws_tag}/{split}/)
        heatmap_ws = args.output_dir / "heatmap" / ws_tag
        all_meta: list[dict] = []

        for split, windows in [
            ("train", train_windows),
            ("val", val_windows),
            ("test", test_windows),
        ]:
            print(f"  Rendering {split} ({len(windows):,} images) ...")
            img_dir = heatmap_ws / split
            meta = render_split(split, windows, img_dir, window_size)
            all_meta.extend(meta)
            print(f"    Done — {len(meta):,} images written to {img_dir}")

        # 6. Metadata CSV
        heatmap_ws.mkdir(parents=True, exist_ok=True)
        csv_path = heatmap_ws / "metadata.csv"
        pd.DataFrame(all_meta)[meta_cols].to_csv(csv_path, index=False)
        print(f"  Metadata CSV -> {csv_path}")

        # 7. HuggingFace DatasetDict
        print("  Building HF DatasetDict ...")
        build_hf_dataset_dict(all_meta, heatmap_ws)

        print(f"  [{ws_tag}] Quintile bin boundaries (train-fitted):")
        for factor in FACTORS:
            b = boundaries[factor]
            print(
                f"    {factor:<8s}: {b[0]:+.5f}  {b[1]:+.5f}  {b[2]:+.5f}  {b[3]:+.5f}"
            )

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL WINDOW SIZES COMPLETE")
    print("=" * 60)
    print(f"  Seed        : {args.seed}")
    print(f"  Stride      : {args.stride} days")
    print(f"  Window sizes: {WINDOW_SIZES}")
    print("\n  Load in training scripts (example for w063):")
    print(
        '    spt.data.HFDataset("data/ff_encoded/heatmap/w063", split="train", transform=...)'
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
