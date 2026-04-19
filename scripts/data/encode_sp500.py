"""Encode S&P 500 OHLCV windows as GAF+MTF and Candlestick images for SSL pre-training.

Reads pre-downloaded parquet files from data/sp500/ and writes two HuggingFace
DatasetDicts — one per encoding — directly loadable by spt.data.HFDataset.

Output layout
-------------
  data/sp500_encoded/
    gaf_mtf/
      dataset_dict.json     ← detected by spt.data.HFDataset
      train/ val/ test/     ← arrow files (HF splits)
      images/train/…        ← source PNGs (path-referenced from arrow)
      metadata.csv
    candlestick/
      dataset_dict.json
      train/ val/ test/
      images/train/…
      metadata.csv

Usage in training scripts
-------------------------
  train_ds = spt.data.HFDataset("data/sp500_encoded/gaf_mtf", split="train",
                                 transform=transform)
  # sample keys: "image" (PIL), "label" (int sector idx), "ticker", "sector",
  #              "start_date", "end_date"

Encodings
---------
GAF+MTF (3-channel RGB):
  Ch1 GASF  = cos(φᵢ + φⱼ)      normalised close prices → arccos
  Ch2 GADF  = sin(φᵢ - φⱼ)
  Ch3 MTF   = empirical transition probabilities, Q=8 equal-frequency bins

Candlestick (RGB, black background):
  Top 80 %   — OHLC bars (green/red body + wick)
  Bottom 20 % — grey volume bars

Temporal split caps (by calendar-date boundaries already in the parquets):
  Train → 45,000   Val / Test → proportional to their share of train rows,
  minimum 1,000 per split.

CLI
---
  python scripts/data/encode_sp500.py [--output_dir PATH] [--window_size N]
                                      [--stride N] [--num_workers N] [--seed N]
"""

import argparse
import multiprocessing as mp
import random
from pathlib import Path

import datasets as hf_datasets
import numpy as np
import pandas as pd
from PIL import Image

# ─── Defaults ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = _REPO_ROOT / "data" / "sp500"
DEFAULT_OUTPUT = _REPO_ROOT / "data" / "sp500_encoded"
DEFAULT_STRIDE = 5
DEFAULT_WORKERS = 4
DEFAULT_SEED = 42

WINDOW_SIZES = [20, 63, 126, 252]  # all four window sizes rendered in one run

TRAIN_CAP = 45_000
MIN_CAP = 1_000
Q_BINS = 8
IMG_SIZE = 224
SPLITS = ["train", "val", "test"]
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


# ─── Image encoding ───────────────────────────────────────────────────────────


def _rescale_uint8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    span = hi - lo
    if span == 0:
        span = 1.0
    scaled = (arr - lo) / span
    return (np.clip(scaled, 0.0, 1.0) * 255).astype(np.uint8)


def compute_gaf_mtf(close: np.ndarray) -> np.ndarray:
    """Encode a 1-D close-price series as a (IMG_SIZE, IMG_SIZE, 3) uint8 image.
    Ch1=GASF, Ch2=GADF, Ch3=MTF.
    """
    mn, mx = close.min(), close.max()
    norm = 2.0 * (close - mn) / (mx - mn) - 1.0 if mx > mn else np.zeros_like(close)

    clipped = np.clip(norm, -1.0 + 1e-6, 1.0 - 1e-6)
    phi = np.arccos(clipped)

    phi_i = phi[:, None]
    phi_j = phi[None, :]
    gasf = np.cos(phi_i + phi_j)  # (T, T), range ≈ [-1, 1]
    gadf = np.sin(phi_i - phi_j)  # (T, T), range ≈ [-1, 1]

    # Equal-frequency binning for MTF
    T = len(norm)
    edges = np.percentile(norm, np.linspace(0, 100, Q_BINS + 1))
    bins = np.clip(np.digitize(norm, edges[1:-1]), 0, Q_BINS - 1)

    trans = np.zeros((Q_BINS, Q_BINS), dtype=np.float64)
    for t in range(T - 1):
        trans[bins[t], bins[t + 1]] += 1.0
    row_sums = trans.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    trans /= row_sums

    # MTF_ij = W[bin(xᵢ), bin(xⱼ)]
    mtf = trans[np.ix_(bins, bins)]  # (T, T), range [0, 1]

    ch1 = _rescale_uint8(gasf, -1.0, 1.0)
    ch2 = _rescale_uint8(gadf, -1.0, 1.0)
    ch3 = _rescale_uint8(mtf, 0.0, 1.0)

    raw = np.stack([ch1, ch2, ch3], axis=2)  # (T, T, 3)
    img = Image.fromarray(raw, mode="RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img)


def render_candlestick(ohlcv: np.ndarray) -> np.ndarray:
    """Rasterise OHLCV bars into a (IMG_SIZE, IMG_SIZE, 3) uint8 image.
    ohlcv: (T, 5) — open, high, low, close, volume
    """
    T = ohlcv.shape[0]
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    price_rows = int(IMG_SIZE * 0.8)  # rows 0 .. price_rows-1
    vol_rows = IMG_SIZE - price_rows  # rows price_rows .. IMG_SIZE-1

    opens, highs, lows, closes, volumes = (ohlcv[:, i] for i in range(5))

    p_min = lows.min()
    p_max = highs.max()
    p_range = p_max - p_min if p_max > p_min else 1.0
    v_max = volumes.max() if volumes.max() > 0 else 1.0

    def p2y(price: float) -> int:
        frac = (price - p_min) / p_range
        return int(np.clip((1.0 - frac) * (price_rows - 1), 0, price_rows - 1))

    bar_w = IMG_SIZE / T

    for i in range(T):
        x0 = int(i * bar_w)
        x1 = max(x0 + 1, int((i + 1) * bar_w))
        x_mid = (x0 + x1) // 2

        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        color: tuple[int, int, int] = (0, 180, 0) if c >= o else (180, 0, 0)

        y_open = p2y(o)
        y_close = p2y(c)
        y_high = p2y(h)
        y_low = p2y(l)

        y_body_top = min(y_open, y_close)
        y_body_bot = max(y_open, y_close)

        # Body (filled rectangle)
        img[y_body_top : y_body_bot + 1, x0:x1] = color
        # Upper wick (high → top of body), 1 px wide
        img[y_high : y_body_top + 1, x_mid : x_mid + 1] = color
        # Lower wick (bottom of body → low), 1 px wide
        img[y_body_bot : y_low + 1, x_mid : x_mid + 1] = color

        # Volume bar (grey)
        vh = max(1, int((volumes[i] / v_max) * vol_rows))
        img[IMG_SIZE - vh : IMG_SIZE, x0:x1] = (128, 128, 128)

    return img


# ─── Worker (module-level for multiprocessing pickling) ───────────────────────


def _render_ticker(args: tuple) -> list[dict]:
    """Render all sampled windows for a single ticker and return metadata records.

    args = (ticker, sector, dates, ohlcv, window_indices, gaf_dir, cs_dir, window_size)
      dates         : list[str] of YYYYMMDD strings (full ticker timeline)
      ohlcv         : (N, 5) float64 array (full ticker timeline)
      window_indices: list[(start_int, end_int)]
    """
    ticker, sector, dates, ohlcv, window_indices, gaf_dir, cs_dir, window_size = args
    results = []

    for start, end in window_indices:
        start_date = dates[start]
        end_date = dates[end - 1]
        fname = f"{ticker}_{end_date}_w{window_size:03d}.png"

        window_ohlcv = ohlcv[start:end]

        # GAF+MTF — uses only close prices
        gaf_arr = compute_gaf_mtf(window_ohlcv[:, 3])
        Image.fromarray(gaf_arr, mode="RGB").save(gaf_dir / fname)

        # Candlestick — uses all 5 columns
        cs_arr = render_candlestick(window_ohlcv)
        Image.fromarray(cs_arr, mode="RGB").save(cs_dir / fname)

        results.append(
            {
                "filename": fname,
                "gaf_path": str(gaf_dir / fname),
                "cs_path": str(cs_dir / fname),
                "ticker": ticker,
                "sector": sector,
                "start_date": start_date,
                "end_date": end_date,
            }
        )

    return results


# ─── HuggingFace DatasetDict builder ─────────────────────────────────────────


def _build_hf_dataset_dict(
    meta_list: list[dict],
    img_key: str,
    enc_dir: Path,
    sector_to_int: dict[str, int],
) -> None:
    """Build a HuggingFace DatasetDict with train/val/test splits and save to
    enc_dir using DatasetDict.save_to_disk().  This creates dataset_dict.json
    which spt.data.HFDataset detects for local loading.

    Each sample exposes:
      image      PIL Image (224x224 RGB) — loaded from PNG path on access
      label      int  (sector index, consistent across all splits)
      ticker     str
      sector     str
      start_date str  YYYYMMDD
      end_date   str  YYYYMMDD

    Load in training scripts:
      spt.data.HFDataset("data/sp500_encoded/gaf_mtf", split="train", transform=...)
    """
    sector_names = sorted(sector_to_int, key=sector_to_int.get)

    features = hf_datasets.Features(
        {
            "image": hf_datasets.Image(),
            "label": hf_datasets.ClassLabel(names=sector_names),
            "ticker": hf_datasets.Value("string"),
            "sector": hf_datasets.Value("string"),
            "start_date": hf_datasets.Value("string"),
            "end_date": hf_datasets.Value("string"),
        }
    )

    split_datasets: dict[str, hf_datasets.Dataset] = {}
    for split in SPLITS:
        rows = [r for r in meta_list if r["split"] == split]
        if not rows:
            continue
        split_datasets[split] = hf_datasets.Dataset.from_dict(
            {
                "image": [r[img_key] for r in rows],  # file paths; PIL decoded lazily
                "label": [sector_to_int[r["sector"]] for r in rows],
                "ticker": [r["ticker"] for r in rows],
                "sector": [r["sector"] for r in rows],
                "start_date": [r["start_date"] for r in rows],
                "end_date": [r["end_date"] for r in rows],
            },
            features=features,
        )

    dd = hf_datasets.DatasetDict(split_datasets)
    dd.save_to_disk(str(enc_dir))
    print(
        f"  HF DatasetDict -> {enc_dir}  "
        f"(splits: {', '.join(f'{k}={len(v)}' for k, v in dd.items())})"
    )


# ─── Pipeline helpers ─────────────────────────────────────────────────────────


def _collect_windows(df: pd.DataFrame, window_size: int, stride: int) -> list[dict]:
    """Slide a window over each ticker's sorted time series and return a list of
    window descriptors.  Windows containing any NaN are skipped.
    Tickers are processed in deterministic sorted order.
    """
    windows: list[dict] = []
    for ticker, tdf in df.groupby("ticker", sort=True):
        tdf = tdf.sort_values("date").reset_index(drop=True)
        sector = str(tdf["sector"].iloc[0])
        ohlcv = tdf[OHLCV_COLS].to_numpy(dtype=np.float64)
        dates = [str(d).replace("-", "") for d in tdf["date"]]
        n = len(tdf)
        for start in range(0, n - window_size + 1, stride):
            end = start + window_size
            if np.any(np.isnan(ohlcv[start:end])):
                continue
            windows.append(
                {
                    "ticker": ticker,
                    "sector": sector,
                    "start": start,
                    "end": end,
                    "dates": dates,  # shared ref — not duplicated in memory
                    "ohlcv": ohlcv,  # shared ref
                }
            )
    return windows


def _subsample(windows: list[dict], cap: int, rng: random.Random) -> list[dict]:
    if len(windows) <= cap:
        return list(windows)
    return rng.sample(windows, cap)


def _process_split(
    split: str,
    df: pd.DataFrame,
    cap: int,
    output_dir: Path,
    window_size: int,
    ws_tag: str,
    stride: int,
    num_workers: int,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """Generate, subsample, and render all windows for one split.
    PNGs are written to {enc}/{ws_tag}/{split}/.
    Returns (gaf_metadata_rows, cs_metadata_rows).
    """
    gaf_dir = output_dir / "gaf_mtf" / ws_tag / split
    cs_dir = output_dir / "candlestick" / ws_tag / split
    gaf_dir.mkdir(parents=True, exist_ok=True)
    cs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[{split}] Collecting windows ...")
    all_windows = _collect_windows(df, window_size, stride)
    print(f"[{split}] {len(all_windows):,} windows before subsampling")

    sampled = _subsample(all_windows, cap, rng)
    print(f"[{split}] {len(sampled):,} windows after subsampling (cap={cap:,})")

    # Group by ticker so each multiprocessing task carries one ticker's data once
    by_ticker: dict[str, list[dict]] = {}
    for w in sampled:
        by_ticker.setdefault(w["ticker"], []).append(w)

    tasks = [
        (
            ticker,
            wlist[0]["sector"],
            wlist[0]["dates"],
            wlist[0]["ohlcv"],
            [(w["start"], w["end"]) for w in wlist],
            gaf_dir,
            cs_dir,
            window_size,
        )
        for ticker, wlist in sorted(by_ticker.items())
    ]

    gaf_meta: list[dict] = []
    cs_meta: list[dict] = []
    tickers_done = 0

    with mp.Pool(processes=num_workers) as pool:
        for ticker_results in pool.imap_unordered(_render_ticker, tasks):
            for rec in ticker_results:
                gaf_meta.append({**rec, "split": split})
                cs_meta.append({**rec, "split": split})
            tickers_done += 1
            if tickers_done % 20 == 0 or tickers_done == len(tasks):
                print(
                    f"[{split}]   {tickers_done}/{len(tasks)} tickers, "
                    f"{len(gaf_meta):,} images written"
                )

    return gaf_meta, cs_meta


# ─── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode S&P 500 windows as GAF+MTF and candlestick images "
        "for all four window sizes (20, 63, 126, 252) in a single run."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Directory containing sp500_{train,val,test}.parquet",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Root directory for encoded images and metadata",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Window stride in trading days (default: 5)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel worker processes (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Global random seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load splits once — shared across all window sizes
    dfs: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        path = args.input_dir / f"sp500_{split}.parquet"
        dfs[split] = pd.read_parquet(path)
        print(
            f"Loaded {split}: {len(dfs[split]):,} rows, "
            f"{dfs[split]['ticker'].nunique()} tickers"
        )

    # Subsampling caps proportional to split row counts — same for all window sizes
    train_rows = len(dfs["train"])
    caps = {
        "train": TRAIN_CAP,
        "val": max(MIN_CAP, int(TRAIN_CAP * len(dfs["val"]) / train_rows)),
        "test": max(MIN_CAP, int(TRAIN_CAP * len(dfs["test"]) / train_rows)),
    }
    print(f"\nSubsampling caps: { {k: f'{v:,}' for k, v in caps.items()} }")

    meta_cols = ["filename", "split", "ticker", "start_date", "end_date", "sector"]

    # ── Loop over all window sizes ────────────────────────────────────────────
    for window_size in WINDOW_SIZES:
        ws_tag = f"w{window_size:03d}"
        print(f"\n{'=' * 60}")
        print(f"WINDOW SIZE: {window_size} days  ({ws_tag})")
        print("=" * 60)

        # Fresh RNG per window size so subsampling is reproducible independently
        rng = random.Random(args.seed)

        all_gaf_meta: list[dict] = []
        all_cs_meta: list[dict] = []

        for split in SPLITS:
            gaf_meta, cs_meta = _process_split(
                split=split,
                df=dfs[split],
                cap=caps[split],
                output_dir=args.output_dir,
                window_size=window_size,
                ws_tag=ws_tag,
                stride=args.stride,
                num_workers=args.num_workers,
                rng=rng,
            )
            all_gaf_meta.extend(gaf_meta)
            all_cs_meta.extend(cs_meta)

        # Build consistent sector → int mapping across all splits
        all_sectors = sorted({r["sector"] for r in all_gaf_meta})
        sector_to_int = {s: i for i, s in enumerate(all_sectors)}

        # Write metadata CSVs and HuggingFace DatasetDicts
        for enc_name, meta_list, img_key in [
            ("gaf_mtf", all_gaf_meta, "gaf_path"),
            ("candlestick", all_cs_meta, "cs_path"),
        ]:
            enc_dir = args.output_dir / enc_name / ws_tag
            enc_dir.mkdir(parents=True, exist_ok=True)

            csv_path = enc_dir / "metadata.csv"
            pd.DataFrame(meta_list)[meta_cols].to_csv(csv_path, index=False)
            print(f"\nMetadata CSV -> {csv_path}")

            _build_hf_dataset_dict(
                meta_list=meta_list,
                img_key=img_key,
                enc_dir=enc_dir,
                sector_to_int=sector_to_int,
            )

        print(f"\n[{ws_tag}] Sectors: {len(sector_to_int)}  ({', '.join(all_sectors)})")
        for enc_name, meta in [("GAF+MTF", all_gaf_meta), ("Candlestick", all_cs_meta)]:
            print(f"  {enc_name}:")
            for split in SPLITS:
                count = sum(1 for r in meta if r["split"] == split)
                print(f"    {split:5s}: {count:>7,} images")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ALL WINDOW SIZES COMPLETE")
    print("=" * 60)
    print(f"  Seed        : {args.seed}")
    print(f"  Stride      : {args.stride} days")
    print(f"  Window sizes: {WINDOW_SIZES}")
    print("\n  Load in training scripts (example for w063):")
    print(
        '    spt.data.HFDataset("data/sp500_encoded/gaf_mtf/w063", split="train", transform=...)'
    )
    print(
        '    spt.data.HFDataset("data/sp500_encoded/candlestick/w063", split="train", transform=...)'
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
