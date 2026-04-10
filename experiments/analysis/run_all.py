"""Run all analysis: tables and figures, with no duplicated computation.

Execution order:
  1. hyperparameter_selection  → hyperparameter_selection.csv
  2. correlation_analysis      → correlation.csv + correlation_<sweep>.png
  3. early_selection_curve     → early_selection_summary.csv + early_selection_*.png

Usage:
    python experiments/analysis/run_all.py [--log_dir DIR] [--output_dir DIR] [--sweeps SWEEP ...]
"""

import argparse

import hyperparameter_selection
import correlation_analysis
import early_selection_curve
from hyperparameter_selection import SWEEPS, DOWNSTREAM_METRICS, SELECTION_METRICS
from log_reader import get_metric_at_epoch
from correlation_analysis import _pearson


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all analysis scripts once.")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=None,
        help="Sweep names to include (e.g. simclr-tau vicreg-cov). Default: all.",
    )
    args = parser.parse_args()

    kw = dict(log_dir=args.log_dir, output_dir=args.output_dir, sweeps=args.sweeps)

    # ------------------------------------------------------------------
    print("=" * 60)
    print("1/3  Hyperparameter selection …")
    print("=" * 60)
    hyperparameter_selection.main(**kw)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2/3  Correlation analysis + figures …")
    print("=" * 60)
    correlation_analysis.main(**kw)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3/3  Early selection curve figures …")
    print("=" * 60)
    early_selection_curve.main(**kw)

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Summary: which metric won each comparison (epoch=299 Pearson r)")
    print("=" * 60)
    active_sweeps = {
        k: v for k, v in SWEEPS.items() if args.sweeps is None or k in args.sweeps
    }
    for sweep_name, run_names in active_sweeps.items():
        print(f"\n{sweep_name}:")
        for ds_name, ds_metric in DOWNSTREAM_METRICS.items():
            scores = {}
            for sel_name, sel_key in SELECTION_METRICS.items():
                sel_vals, ds_vals = [], []
                for run in run_names:
                    try:
                        sv = get_metric_at_epoch(
                            run, sel_key, 299, log_dir=args.log_dir
                        )
                        dv = get_metric_at_epoch(
                            run, ds_metric, 299, log_dir=args.log_dir
                        )
                        sel_vals.append(sv)
                        ds_vals.append(dv)
                    except (KeyError, FileNotFoundError):
                        pass
                if len(sel_vals) >= 2:
                    scores[sel_name] = _pearson(sel_vals, ds_vals)
            if scores:
                winner = max(scores, key=lambda k: scores[k])
                score_str = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
                print(f"  {ds_name}: {score_str}  → winner: {winner}")

    print("\nDone.")


if __name__ == "__main__":
    main()
