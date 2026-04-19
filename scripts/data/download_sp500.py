"""Download and preprocess S&P 500 daily OHLCV data for SSL pre-training.

Downloads all current S&P 500 constituents from yfinance for the period
2013-01-01 through 2023-12-31, attaches GICS sector labels, filters low-
quality tickers, and saves chronologically split parquet files.

Temporal split (strict calendar boundaries):
  Train : 2013-01-01 — 2019-12-31  (~70 %)
  Val   : 2020-01-01 — 2021-12-31  (~10 %)
  Test  : 2022-01-01 — 2023-12-31  (~20 %)

Output columns per split parquet:
  ticker (str), date (date), open, high, low, close, volume (float),
  sector (str)

Usage:
  python scripts/data/download_sp500.py [--output_dir PATH] [--seed INT]
"""

import argparse
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


_TORCH_AVAILABLE = False
try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FULL_START = "2013-01-01"
FULL_END = "2023-12-31"

TRAIN_START, TRAIN_END = "2013-01-01", "2019-12-31"
VAL_START, VAL_END = "2020-01-01", "2021-12-31"
TEST_START, TEST_END = "2022-01-01", "2023-12-31"

MIN_TRADING_DAYS = 100
BATCH_SIZE = 50

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "sp500"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        import torch

        torch.manual_seed(seed)


def fetch_sp500_tickers() -> list[str]:
    """Return a hardcoded list of S&P 500 constituents (as of early 2025)."""
    return sorted(
        [
            "A",
            "AAL",
            "AAPL",
            "ABBV",
            "ABNB",
            "ABT",
            "ACGL",
            "ACI",
            "ACN",
            "ADBE",
            "ADI",
            "ADM",
            "ADP",
            "ADSK",
            "AEE",
            "AEP",
            "AES",
            "AFL",
            "AIG",
            "AIZ",
            "AJG",
            "AKAM",
            "ALB",
            "ALGN",
            "ALL",
            "ALLE",
            "AMAT",
            "AMCR",
            "AMD",
            "AME",
            "AMGN",
            "AMP",
            "AMT",
            "AMZN",
            "ANET",
            "AON",
            "AOS",
            "APA",
            "APD",
            "APH",
            "APTV",
            "ARE",
            "ATO",
            "AVB",
            "AVGO",
            "AVY",
            "AWK",
            "AXON",
            "AXP",
            "AZO",
            "BA",
            "BAC",
            "BALL",
            "BAX",
            "BBWI",
            "BBY",
            "BDX",
            "BEN",
            "BF-B",
            "BG",
            "BIIB",
            "BIO",
            "BK",
            "BKNG",
            "BKR",
            "BLDR",
            "BLK",
            "BMY",
            "BR",
            "BRK-B",
            "BRO",
            "BSX",
            "BWA",
            "BX",
            "BXP",
            "C",
            "CAG",
            "CAH",
            "CARR",
            "CAT",
            "CB",
            "CBOE",
            "CBRE",
            "CCI",
            "CCL",
            "CDNS",
            "CDW",
            "CE",
            "CEG",
            "CF",
            "CFG",
            "CHD",
            "CHRW",
            "CHTR",
            "CI",
            "CINF",
            "CL",
            "CLX",
            "CMA",
            "CMCSA",
            "CME",
            "CMG",
            "CMI",
            "CMS",
            "CNC",
            "CNP",
            "COF",
            "COO",
            "COP",
            "COST",
            "CPB",
            "CPAY",
            "CPT",
            "CRL",
            "CRM",
            "CRWD",
            "CSCO",
            "CSGP",
            "CSX",
            "CTAS",
            "CTLT",
            "CTRA",
            "CTSH",
            "CTVA",
            "CVS",
            "CVX",
            "CZR",
            "D",
            "DAL",
            "DAY",
            "DD",
            "DECK",
            "DEI",
            "DG",
            "DGX",
            "DHI",
            "DHR",
            "DIS",
            "DLR",
            "DLTR",
            "DOC",
            "DOV",
            "DOW",
            "DPZ",
            "DRI",
            "DTE",
            "DUK",
            "DVA",
            "DVN",
            "DXCM",
            "EA",
            "EBAY",
            "ECL",
            "ED",
            "EFX",
            "EG",
            "EIX",
            "EL",
            "ELV",
            "EMN",
            "EMR",
            "ENPH",
            "EOG",
            "EPAM",
            "EQIX",
            "EQR",
            "EQT",
            "ES",
            "ESS",
            "ETN",
            "ETR",
            "EVRG",
            "EW",
            "EXC",
            "EXPD",
            "EXPE",
            "EXR",
            "F",
            "FANG",
            "FAST",
            "FCX",
            "FDS",
            "FDX",
            "FE",
            "FFIV",
            "FI",
            "FICO",
            "FIS",
            "FITB",
            "FMC",
            "FOX",
            "FOXA",
            "FRT",
            "FSLR",
            "FTNT",
            "FTV",
            "GD",
            "GDDY",
            "GE",
            "GEHC",
            "GEN",
            "GEV",
            "GILD",
            "GIS",
            "GL",
            "GLW",
            "GM",
            "GNRC",
            "GOOGL",
            "GPC",
            "GPN",
            "GRMN",
            "GS",
            "GWW",
            "HAL",
            "HAS",
            "HBAN",
            "HCA",
            "HD",
            "HES",
            "HIG",
            "HII",
            "HLT",
            "HOLX",
            "HON",
            "HPE",
            "HPQ",
            "HRL",
            "HSIC",
            "HST",
            "HSY",
            "HUBB",
            "HUM",
            "HWM",
            "IBM",
            "ICE",
            "IDXX",
            "IEX",
            "IFF",
            "INCY",
            "INTC",
            "INTU",
            "INVH",
            "IP",
            "IPG",
            "IQV",
            "IR",
            "IRM",
            "ISRG",
            "IT",
            "ITW",
            "IVZ",
            "J",
            "JBHT",
            "JCI",
            "JKHY",
            "JNJ",
            "JNPR",
            "JPM",
            "K",
            "KDP",
            "KEY",
            "KEYS",
            "KHC",
            "KIM",
            "KKR",
            "KLAC",
            "KMB",
            "KMI",
            "KMX",
            "KO",
            "KR",
            "KVUE",
            "L",
            "LDOS",
            "LEN",
            "LH",
            "LHX",
            "LIN",
            "LKQ",
            "LLY",
            "LMT",
            "LNT",
            "LOW",
            "LRCX",
            "LULU",
            "LUV",
            "LVS",
            "LW",
            "LYB",
            "LYV",
            "MA",
            "MAA",
            "MAR",
            "MAS",
            "MCD",
            "MCHP",
            "MCK",
            "MCO",
            "MDLZ",
            "MDT",
            "MET",
            "META",
            "MGM",
            "MHK",
            "MKC",
            "MKTX",
            "MLM",
            "MMC",
            "MMM",
            "MNST",
            "MO",
            "MOH",
            "MOS",
            "MPC",
            "MPWR",
            "MRK",
            "MRNA",
            "MRO",
            "MS",
            "MSCI",
            "MSFT",
            "MSI",
            "MTB",
            "MTCH",
            "MTD",
            "MU",
            "NCLH",
            "NDAQ",
            "NEE",
            "NEM",
            "NFLX",
            "NI",
            "NKE",
            "NOC",
            "NOW",
            "NRG",
            "NSC",
            "NTAP",
            "NTRS",
            "NUE",
            "NVDA",
            "NVR",
            "NWS",
            "NWSA",
            "NXPI",
            "O",
            "ODFL",
            "OKE",
            "OMC",
            "ON",
            "ORCL",
            "ORLY",
            "OTIS",
            "OXY",
            "PANW",
            "PARA",
            "PAYC",
            "PAYX",
            "PCAR",
            "PCG",
            "PEG",
            "PEP",
            "PFE",
            "PFG",
            "PG",
            "PGR",
            "PH",
            "PHM",
            "PKG",
            "PLD",
            "PM",
            "PNC",
            "PNR",
            "PNW",
            "PODD",
            "POOL",
            "PPG",
            "PPL",
            "PRU",
            "PSA",
            "PSX",
            "PTC",
            "PWR",
            "PXD",
            "PYPL",
            "QCOM",
            "QRVO",
            "RCL",
            "REG",
            "REGN",
            "RF",
            "RJF",
            "RL",
            "RMD",
            "ROK",
            "ROL",
            "ROP",
            "ROST",
            "RSG",
            "RTX",
            "RVTY",
            "SBAC",
            "SBUX",
            "SCHW",
            "SHW",
            "SJM",
            "SLB",
            "SMCI",
            "SNA",
            "SNPS",
            "SO",
            "SOLV",
            "SPG",
            "SPGI",
            "SRE",
            "STE",
            "STLD",
            "STT",
            "STX",
            "STZ",
            "SWK",
            "SWKS",
            "SYF",
            "SYK",
            "SYY",
            "T",
            "TAP",
            "TDG",
            "TDY",
            "TECH",
            "TEL",
            "TER",
            "TFC",
            "TGT",
            "TJX",
            "TMO",
            "TMUS",
            "TPR",
            "TRGP",
            "TRMB",
            "TROW",
            "TRV",
            "TSCO",
            "TSLA",
            "TSN",
            "TT",
            "TTWO",
            "TXN",
            "TXT",
            "TYL",
            "UAL",
            "UBER",
            "UDR",
            "UHS",
            "ULTA",
            "UNH",
            "UNP",
            "UPS",
            "URI",
            "USB",
            "V",
            "VICI",
            "VLO",
            "VLTO",
            "VMC",
            "VRSK",
            "VRSN",
            "VRTX",
            "VST",
            "VTR",
            "VTRS",
            "VZ",
            "WAB",
            "WAT",
            "WBA",
            "WBD",
            "WDC",
            "WEC",
            "WELL",
            "WFC",
            "WM",
            "WMB",
            "WMT",
            "WRB",
            "WST",
            "WTW",
            "WY",
            "WYNN",
            "XEL",
            "XOM",
            "XYL",
            "YUM",
            "ZBH",
            "ZBRA",
            "ZTS",
        ]
    )


def fetch_sector(ticker: str) -> str | None:
    """Return the GICS sector string from yfinance, or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            info = yf.Ticker(ticker).info
        return info.get("sector") or None
    except Exception:
        return None


def download_batch(tickers: list[str]) -> pd.DataFrame:
    """Download adjusted OHLCV for a batch of tickers.

    Returns a long-format DataFrame with columns:
      ticker, date, open, high, low, close, volume
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            tickers,
            start=FULL_START,
            end=FULL_END,
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    if raw.empty:
        return pd.DataFrame()

    # yfinance returns MultiIndex columns (ticker, field) when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        frames = []
        for tkr in tickers:
            if tkr not in raw.columns.get_level_values(0):
                continue
            sub = raw[tkr].copy()
            sub.index.name = "date"
            sub = sub.reset_index()
            sub.insert(0, "ticker", tkr)
            frames.append(sub)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
    else:
        # Single ticker fallback
        df = raw.copy()
        df.index.name = "date"
        df = df.reset_index()
        df.insert(0, "ticker", tickers[0])

    df.columns = [c.lower() for c in df.columns]
    # Keep only the columns we care about
    keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]]
    return df


def _date_slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start).date()) & (
        df["date"] <= pd.Timestamp(end).date()
    )
    return df.loc[mask].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_dir: Path, seed: int) -> None:
    set_seeds(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Fetch constituent list
    print("Fetching S&P 500 constituent list from Wikipedia ...")
    tickers = fetch_sp500_tickers()
    print(f"  Found {len(tickers)} tickers")

    # 2. Download OHLCV in batches
    print(f"\nDownloading OHLCV data in batches of {BATCH_SIZE} ...")
    all_frames: list[pd.DataFrame] = []
    skipped_download: list[str] = []

    batches = [tickers[i : i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for idx, batch in enumerate(batches, 1):
        print(f"  Batch {idx}/{len(batches)}: {batch[0]} ... {batch[-1]}")
        try:
            df = download_batch(batch)
            if df.empty:
                print(
                    f"    WARNING: empty result for batch {idx}, skipping all tickers in batch"
                )
                skipped_download.extend(batch)
            else:
                all_frames.append(df)
        except Exception as exc:
            print(f"    WARNING: batch {idx} failed ({exc}), skipping")
            skipped_download.extend(batch)

    if not all_frames:
        raise RuntimeError(
            "No data downloaded. Check your internet connection or yfinance version."
        )

    ohlcv = pd.concat(all_frames, ignore_index=True)

    # Normalise date column to Python date (not datetime)
    ohlcv["date"] = pd.to_datetime(ohlcv["date"]).dt.date

    # Cast price/volume columns to float
    for col in ["open", "high", "low", "close", "volume"]:
        if col in ohlcv.columns:
            ohlcv[col] = ohlcv[col].astype(float)

    # 3. Fetch sector labels
    print("\nFetching GICS sector labels ...")
    downloaded_tickers = ohlcv["ticker"].unique().tolist()
    sector_map: dict[str, str] = {}
    skipped_sector: list[str] = []

    for tkr in downloaded_tickers:
        sector = fetch_sector(tkr)
        if sector:
            sector_map[tkr] = sector
        else:
            skipped_sector.append(tkr)
            print(f"  WARNING: no sector label for {tkr}, skipping")

    # 4. Filter: must have a sector label
    ohlcv = ohlcv[ohlcv["ticker"].isin(sector_map)].copy()
    ohlcv["sector"] = ohlcv["ticker"].map(sector_map)

    # 5. Filter: must have >= MIN_TRADING_DAYS rows in the full range
    counts = ohlcv.groupby("ticker")["date"].count()
    valid_tickers = counts[counts >= MIN_TRADING_DAYS].index
    skipped_thin = list(
        set(downloaded_tickers) - set(skipped_sector) - set(valid_tickers)
    )
    ohlcv = ohlcv[ohlcv["ticker"].isin(valid_tickers)].copy()

    # 6. Sort deterministically (ticker, then date) so output is byte-stable
    ohlcv = ohlcv.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 7. Chronological split by calendar date
    train_df = _date_slice(ohlcv, TRAIN_START, TRAIN_END).reset_index(drop=True)
    val_df = _date_slice(ohlcv, VAL_START, VAL_END).reset_index(drop=True)
    test_df = _date_slice(ohlcv, TEST_START, TEST_END).reset_index(drop=True)

    # 8. Save
    for split_name, split_df in (
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ):
        out_path = output_dir / f"sp500_{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)
        print(f"\nSaved {split_name}: {out_path}  ({len(split_df):,} rows)")

    # 9. Summary
    final_tickers = sorted(ohlcv["ticker"].unique())
    sector_dist = ohlcv.drop_duplicates("ticker")["sector"].value_counts()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Random seed        : {seed}")
    print(f"  Tickers downloaded : {len(downloaded_tickers)}")
    skipped_all = len(skipped_download) + len(skipped_sector) + len(skipped_thin)
    print(f"  Tickers skipped    : {skipped_all}")
    print(f"    - download error : {len(skipped_download)}")
    print(f"    - missing sector : {len(skipped_sector)}")
    print(f"    - < {MIN_TRADING_DAYS} trading days: {len(skipped_thin)}")
    print(f"  Tickers in dataset : {len(final_tickers)}")
    print(f"  Rows  — train      : {len(train_df):,}  ({TRAIN_START} to {TRAIN_END})")
    print(f"  Rows  — val        : {len(val_df):,}  ({VAL_START} to {VAL_END})")
    print(f"  Rows  — test       : {len(test_df):,}  ({TEST_START} to {TEST_END})")
    print("\n  Sector distribution (unique tickers):")
    for sector, count in sector_dist.items():
        print(f"    {sector:<35s} {count:>4d}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download S&P 500 OHLCV data and save as parquet."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write parquet files (default: scripts/data/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Global random seed (default: 42)",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir, seed=args.seed)
