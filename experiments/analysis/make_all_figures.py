"""Entry point: generate all early-selection curve figures.

Usage:
    python experiments/analysis/make_all_figures.py [--log_dir DIR] [--output_dir DIR]
"""

import argparse

import early_selection_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("Generating early selection curve figures …")
    print("=" * 60)
    early_selection_curve.main(log_dir=args.log_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
