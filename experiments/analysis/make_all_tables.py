"""Entry point: generate all tables (hyperparameter selection + correlation).

Usage:
    python experiments/analysis/make_all_tables.py [--log_dir DIR] [--output_dir DIR]
"""

import argparse

import hyperparameter_selection
import correlation_analysis
from hyperparameter_selection import SWEEPS, DOWNSTREAM_METRICS, SELECTION_METRICS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=None,
        help="Sweep names to include (e.g. simclr-tau vicreg-cov). Default: all.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Running hyperparameter selection analysis …")
    print("=" * 60)
    hyperparameter_selection.main(
        log_dir=args.log_dir, output_dir=args.output_dir, sweeps=args.sweeps
    )

    print("\n" + "=" * 60)
    print("Running correlation analysis …")
    print("=" * 60)
    correlation_analysis.main(
        log_dir=args.log_dir, output_dir=args.output_dir, sweeps=args.sweeps
    )

    # Summary: which selection metric won each comparison at final epoch
    print("\n" + "=" * 60)
    print("Summary: which metric won each comparison (epoch=299 Pearson r)")
    print("=" * 60)
    from log_reader import get_metric_at_epoch
    from correlation_analysis import _pearson

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


if __name__ == "__main__":
    main()
