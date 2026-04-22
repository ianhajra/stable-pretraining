"""
Ablation script to run SimCLR for a fixed window size, with each run using only a single transform from the set of available transforms (both standard and custom).

This script is designed to be used with sbatch or similar job submission, mirroring the structure and arguments of ablation_window_size.py, but launching one run per transform.
"""

import argparse
import subprocess
from pathlib import Path

# List of transforms to ablate (name, import_path, constructor)
TRANSFORM_SPECS = [
    # ("transform_name", "import_path", "constructor_code")
    ("RandomResizedCrop", "stable_pretraining.data.transforms", "RandomResizedCrop((224, 224))"),
    ("RandomHorizontalFlip", "stable_pretraining.data.transforms", "RandomHorizontalFlip(p=1.0)"),
    ("ColorJitter", "stable_pretraining.data.transforms", "ColorJitter()"),
    ("RandomGrayscale", "stable_pretraining.data.transforms", "RandomGrayscale(p=1.0)"),
    ("GaussianBlur", "stable_pretraining.data.transforms_custom", "GaussianBlur(sigma_range=(0.1, 2.0), p=1.0)"),
    ("MagnitudeScaling", "stable_pretraining.data.transforms_custom", "MagnitudeScaling(scale_range=(0.8, 1.2), p=1.0)"),
    ("GaussianNoiseInjection", "stable_pretraining.data.transforms_custom", "GaussianNoiseInjection(sigma=0.05, p=1.0)"),
    ("RandomTemporalMasking", "stable_pretraining.data.transforms_custom", "RandomTemporalMasking(mask_ratio_range=(0.10, 0.20), p=1.0)"),
]

# Default fixed window size (2nd smallest from ablation_window_size.py)
DEFAULT_WINDOW_SIZE = 63

# Default encoding and dataset
DEFAULT_ENCODING = "gaf_mtf"
DEFAULT_DATASET = "sp500"
DEFAULT_NUM_CLASSES = 11

# Path to ablation_window_size.py
ABLATION_SCRIPT = str(Path(__file__).parent.parent / "train" / "ablation_window_size.py")

# Example: --data_dir, --num_classes, --window_size, --encoding, --dataset, --seed, --batch_size, --num_workers

def main():
    parser = argparse.ArgumentParser(description="SimCLR transform ablation launcher")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to HFDataset root for the encoding and window size")
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE, help="Window size (default: 63)")
    parser.add_argument("--encoding", type=str, default=DEFAULT_ENCODING, help="Encoding type (default: gaf_mtf)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Dataset (default: sp500)")
    parser.add_argument("--num_classes", type=int, default=DEFAULT_NUM_CLASSES, help="Number of classes (default: 11)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--sbatch", action="store_true", help="If set, submit jobs with sbatch; otherwise, run locally.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands but do not execute.")
    args = parser.parse_args()

    for name, import_path, constructor in TRANSFORM_SPECS:
        run_name = f"simclr_ablation_{name.lower()}"
        cmd = [
            "python", ABLATION_SCRIPT,
            f"--data_dir={args.data_dir}",
            f"--num_classes={args.num_classes}",
            f"--window_size={args.window_size}",
            f"--encoding={args.encoding}",
            f"--dataset={args.dataset}",
            f"--seed={args.seed}",
            f"--batch_size={args.batch_size}",
            f"--num_workers={args.num_workers}",
            f"--transform_name={name}",
        ]
        # If using sbatch, wrap in sbatch command
        if args.sbatch:
            sbatch_cmd = [
                "sbatch", "--job-name", run_name,
                "--output", f"logs/{run_name}.out",
                "--wrap", ' '.join(cmd)
            ]
            final_cmd = sbatch_cmd
        else:
            final_cmd = cmd
        print("Launching:", ' '.join(final_cmd))
        if not args.dry_run:
            subprocess.run(final_cmd)

if __name__ == "__main__":
    main()
