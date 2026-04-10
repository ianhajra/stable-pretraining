"""Entry point: generate all figures.

Usage:
    python experiments/analysis/make_all_figures.py [--log_dir DIR] [--output_dir DIR]
"""

import argparse

import early_selection_curve
import correlation_analysis


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
    print("Generating early selection curve figures …")
    print("=" * 60)
    early_selection_curve.main(
        log_dir=args.log_dir, output_dir=args.output_dir, sweeps=args.sweeps
    )

    print("=" * 60)
    print("Generating correlation figures …")
    print("=" * 60)
    correlation_analysis.main(
        log_dir=args.log_dir, output_dir=args.output_dir, sweeps=args.sweeps
    )


if __name__ == "__main__":
    main()
