"""Hyperparameter selection analysis using Algorithm 1 from the RankMe paper.

For each sweep (SimCLR-tau, VICReg-cov, VICReg-inv), at each checkpoint epoch,
selects the best run using RankMe and ReRankMe scores (Algorithm 1), then records
the final epoch-300 downstream performance of the selected run.

Saves results to experiments/results/hyperparameter_selection.csv and prints a
readable summary.
"""

import csv
from pathlib import Path
from typing import Optional

from log_reader import get_metric_at_epoch

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

SWEEPS = {
    "simclr-tau": [
        "simclr-resnet18-tau0.05",
        "simclr-resnet18-tau0.07",
        "simclr-resnet18-tau0.10",
        "simclr-resnet18-tau0.15",
        "simclr-resnet18-tau0.20",
        "simclr-resnet18-tau0.30",
        "simclr-resnet18-tau0.40",
    ],
    "vicreg-cov": [
        "vicreg-resnet18-cov5",
        "vicreg-resnet18-cov10",
        "vicreg-resnet18-cov15",
        "vicreg-resnet18-cov20",
        "vicreg-resnet18-cov25",
        "vicreg-resnet18-cov30",
        "vicreg-resnet18-cov35",
        "vicreg-resnet18-cov40",
        "vicreg-resnet18-cov50",
    ],
    "vicreg-inv": [
        "vicreg-resnet18-inv5",
        "vicreg-resnet18-inv10",
        "vicreg-resnet18-inv15",
        "vicreg-resnet18-inv20",
        "vicreg-resnet18-inv25",
        "vicreg-resnet18-inv30",
        "vicreg-resnet18-inv35",
        "vicreg-resnet18-inv40",
        "vicreg-resnet18-inv50",
    ],
}

SELECTION_EPOCHS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
FINAL_EPOCH = 300

DOWNSTREAM_METRICS = {
    "linear_probe": "eval/linear_probe_top1",
    "knn_probe": "eval/knn_probe_accuracy",
    "cifar10c": "eval/cifar10c_accuracy",
}

SELECTION_METRICS = {
    "rankme": "rankme",
    "rerankme": "rerankme",
}


# ---------------------------------------------------------------------------
# Algorithm 1 from the RankMe paper
# ---------------------------------------------------------------------------


def algorithm1(run_metric_pairs: list[tuple[str, float]]) -> str:
    """Select the run with the highest metric value (Algorithm 1 from RankMe).

    Algorithm 1 simply picks the hyperparameter configuration that maximises
    the unsupervised selection metric (RankMe or ReRankMe).

    Args:
        run_metric_pairs: List of (run_name, metric_value) pairs.

    Returns:
        The run name with the highest metric value.
    """
    return max(run_metric_pairs, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(log_dir: Optional[str] = None, output_dir: Optional[str] = None) -> None:
    results_dir = (
        Path(output_dir) if output_dir else Path(__file__).parent.parent / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "hyperparameter_selection.csv"

    rows = []
    summary_lines = []

    for sweep_name, run_names in SWEEPS.items():
        summary_lines.append(f"\n=== Sweep: {sweep_name} ===")

        # Compute oracle: best possible final downstream value across the sweep
        oracles = {}
        for ds_name, ds_metric in DOWNSTREAM_METRICS.items():
            best_val = None
            for run in run_names:
                try:
                    val = get_metric_at_epoch(
                        run, ds_metric, FINAL_EPOCH, log_dir=log_dir
                    )
                    if best_val is None or val > best_val:
                        best_val = val
                except (KeyError, FileNotFoundError):
                    pass
            oracles[ds_name] = best_val

        for sel_metric_name, sel_metric_key in SELECTION_METRICS.items():
            for sel_epoch in SELECTION_EPOCHS:
                # Gather selection metric values at sel_epoch for all runs
                pairs = []
                for run in run_names:
                    try:
                        val = get_metric_at_epoch(
                            run, sel_metric_key, sel_epoch, log_dir=log_dir
                        )
                        pairs.append((run, val))
                    except (KeyError, FileNotFoundError):
                        pass

                if not pairs:
                    continue

                selected_run = algorithm1(pairs)

                row = {
                    "sweep": sweep_name,
                    "selection_metric": sel_metric_name,
                    "selection_epoch": sel_epoch,
                    "selected_run": selected_run,
                }

                for ds_name, ds_metric in DOWNSTREAM_METRICS.items():
                    try:
                        val = get_metric_at_epoch(
                            selected_run, ds_metric, FINAL_EPOCH, log_dir=log_dir
                        )
                    except (KeyError, FileNotFoundError):
                        val = None
                    row[f"final_{ds_name}"] = val
                    row[f"oracle_{ds_name}"] = oracles[ds_name]

                rows.append(row)

                if sel_epoch in (100, 200, 300):
                    line = (
                        f"  [{sel_metric_name} @ ep{sel_epoch}] selected={selected_run}"
                    )
                    for ds_name in DOWNSTREAM_METRICS:
                        v = row.get(f"final_{ds_name}")
                        o = oracles.get(ds_name)
                        v_str = f"{v:.4f}" if v is not None else "N/A"
                        o_str = f"{o:.4f}" if o is not None else "N/A"
                        line += f"  {ds_name}={v_str}(oracle={o_str})"
                    summary_lines.append(line)

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    main(log_dir=args.log_dir, output_dir=args.output_dir)
