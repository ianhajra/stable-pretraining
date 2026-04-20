"""Download and preprocess Fama-French 30 Industry Portfolios (Daily).

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
File:   30_Industry_Portfolios_daily_CSV.zip

The raw file contains two tables:
  - Average Value Weighted Returns -- Daily
  - Average Equal Weighted Returns -- Daily

Each table has dates in YYYYMMDD format and returns in percentage points
(e.g. 0.05 means 0.05%).  Missing values are coded as -99.99 or -999.

Modelling universe
------------------
  Start : 1963-07-01  (start of broad CRSP coverage)
  End   : 2023-12-31

  Chronological 70 / 10 / 20 split by row count:
    Train : 1963-07-01 — ~mid-2013   (~10,570 rows)
    Val   : ~mid-2013  — ~late-2016  (~1,510 rows)
    Test  : ~late-2016 — 2023-12-31  (~3,020 rows)

Output (data/ff/):
  Full (filtered to modelling universe):
    30_industry_daily_vw.parquet
    30_industry_daily_ew.parquet

  Pre-split files (suffix _train / _val / _test):
    30_industry_daily_vw_train.parquet
    30_industry_daily_vw_val.parquet
    30_industry_daily_vw_test.parquet
    30_industry_daily_ew_train.parquet
    30_industry_daily_ew_val.parquet
    30_industry_daily_ew_test.parquet

Dates are stored as a DatetimeIndex named "date".
Missing values are converted to NaN.
"""

import argparse
import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_daily_CSV.zip"
MISSING = {-99.99, -999.0}
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "ff"

# Modelling universe: broad CRSP coverage start through end of 2023
UNIVERSE_START = "1963-07-01"
UNIVERSE_END = "2023-12-31"

# Chronological 70 / 10 / 20 split (fractions of row count)
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)


def _split_df(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) via a chronological 70/10/20 row-count split."""
    n = len(df)
    n_train = int(n * SPLIT_FRACTIONS[0])
    n_val = int(n * SPLIT_FRACTIONS[1])
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def _parse_blocks(text: str) -> dict[str, pd.DataFrame]:
    """Parse the fixed-width / comma-separated blocks inside the FF CSV file.

    Returns a dict mapping block title -> DataFrame with a DatetimeIndex.
    """
    # Split on blank lines to find blocks
    # Each block looks like:
    #   Average Value Weighted Returns -- Daily
    #   (blank)
    #   ,Food,Beer,...
    #   19260701,0.05,-1.39,...
    #   ...
    blocks: dict[str, pd.DataFrame] = {}
    current_title: str | None = None
    current_rows: list[str] = []

    def _flush(title: str, rows: list[str]) -> None:
        if not rows:
            return
        raw = "\n".join(rows)
        df = pd.read_csv(io.StringIO(raw), index_col=0)
        df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
        df.index.name = "date"
        df.columns = df.columns.str.strip()
        # Replace missing-value codes with NaN
        df = df.replace(-99.99, float("nan")).replace(-999.0, float("nan"))
        blocks[title] = df

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Detect a header line: non-empty, does not start with a digit or comma
        if line and not re.match(r"^[\d,]", line) and not line.startswith(" "):
            # Flush previous block if any
            if current_title is not None:
                _flush(current_title, current_rows)
            current_title = line
            current_rows = []
        elif line and current_title is not None:
            # Keep data lines (header row starts with comma, data rows start with digit)
            current_rows.append(lines[i])

        i += 1

    # Flush the final block
    if current_title is not None:
        _flush(current_title, current_rows)

    return blocks


def download_and_save(url: str = URL, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # There is exactly one CSV inside the zip
        csv_name = next(
            n for n in zf.namelist() if n.endswith(".CSV") or n.endswith(".csv")
        )
        print(f"Extracting {csv_name} ...")
        raw_text = zf.read(csv_name).decode("latin-1")

    blocks = _parse_blocks(raw_text)

    if not blocks:
        raise ValueError("No data blocks found — check the file format.")

    for title, df in blocks.items():
        # Map title to a clean filename stem
        if "value" in title.lower():
            stem = "30_industry_daily_vw"
        elif "equal" in title.lower():
            stem = "30_industry_daily_ew"
        else:
            stem = re.sub(r"[^\w]+", "_", title).strip("_").lower()

        # Restrict to modelling universe
        df = df.loc[UNIVERSE_START:UNIVERSE_END]

        # Save full (filtered) file
        out_path = out_dir / f"{stem}.parquet"
        df.to_parquet(out_path)
        print(f"  Saved {len(df):,} rows × {len(df.columns)} cols -> {out_path}")

        # Save pre-split files
        train, val, test = _split_df(df)
        for split_name, split_df in (("train", train), ("val", val), ("test", test)):
            split_path = out_dir / f"{stem}_{split_name}.parquet"
            split_df.to_parquet(split_path)
            print(
                f"    {split_name:5s}: {split_df.index[0].date()} — "
                f"{split_df.index[-1].date()}  ({len(split_df):,} rows)"
            )

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fama-French 30-industry data.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save output parquet files (default: data/ff/ relative to repo root).",
    )
    args = parser.parse_args()
    out_dir = args.output_dir if args.output_dir is not None else OUT_DIR
    download_and_save(out_dir=out_dir)
