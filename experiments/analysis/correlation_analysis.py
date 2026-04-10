"""Correlation analysis between unsupervised selection metrics and downstream performance.

At each of {25, 50, 100, 150, 200, 250, 300} epochs, for each sweep, computes
Pearson r and Spearman rho between (RankMe, ReRankMe) and each downstream metric.

Saves to experiments/results/correlation.csv and prints a readable summary.
"""

import csv
import math
from pathlib import Path
from typing import Optional

from log_reader import get_metric_at_epoch
from hyperparameter_selection import SWEEPS, DOWNSTREAM_METRICS, SELECTION_METRICS

CORRELATION_EPOCHS = [25, 50, 100, 150, 200, 250, 300]


# ---------------------------------------------------------------------------
# Statistics helpers (no scipy dependency)
# ---------------------------------------------------------------------------


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = _mean(xs), _mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den > 0 else float("nan")


def _rank(xs: list[float]) -> list[float]:
    sorted_vals = sorted(enumerate(xs), key=lambda t: t[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(sorted_vals):
        j = i
        while j < len(sorted_vals) - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_vals[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_rank(xs), _rank(ys))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    log_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    sweeps: Optional[list[str]] = None,
) -> None:
    results_dir = (
        Path(output_dir) if output_dir else Path(__file__).parent.parent / "results"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "correlation.csv"

    rows = []
    summary_lines = []

    active_sweeps = {k: v for k, v in SWEEPS.items() if sweeps is None or k in sweeps}
    for sweep_name, run_names in active_sweeps.items():
        summary_lines.append(f"\n=== Sweep: {sweep_name} ===")

        for epoch in CORRELATION_EPOCHS:
            for sel_name, sel_key in SELECTION_METRICS.items():
                # Gather (sel_metric, downstream_metric) pairs across runs
                sel_vals: list[float] = []
                ds_vals: dict[str, list[float]] = {k: [] for k in DOWNSTREAM_METRICS}
                valid_runs: list[str] = []

                for run in run_names:
                    try:
                        sv = get_metric_at_epoch(run, sel_key, epoch, log_dir=log_dir)
                    except (KeyError, FileNotFoundError):
                        continue
                    ds_row = {}
                    ok = True
                    for ds_name, ds_metric in DOWNSTREAM_METRICS.items():
                        try:
                            ds_row[ds_name] = get_metric_at_epoch(
                                run, ds_metric, 300, log_dir=log_dir
                            )
                        except (KeyError, FileNotFoundError):
                            ok = False
                            break
                    if not ok:
                        continue
                    sel_vals.append(sv)
                    valid_runs.append(run)
                    for ds_name in DOWNSTREAM_METRICS:
                        ds_vals[ds_name].append(ds_row[ds_name])

                if len(sel_vals) < 2:
                    continue

                row: dict = {
                    "sweep": sweep_name,
                    "epoch": epoch,
                    "selection_metric": sel_name,
                    "n_runs": len(sel_vals),
                }
                for ds_name in DOWNSTREAM_METRICS:
                    r = _pearson(sel_vals, ds_vals[ds_name])
                    rho = _spearman(sel_vals, ds_vals[ds_name])
                    row[f"pearson_{ds_name}"] = round(r, 4)
                    row[f"spearman_{ds_name}"] = round(rho, 4)

                rows.append(row)

                if epoch == 300:
                    line = f"  [{sel_name} @ ep{epoch}]"
                    for ds_name in DOWNSTREAM_METRICS:
                        line += (
                            f"  {ds_name}: r={row[f'pearson_{ds_name}']:.3f}"
                            f" rho={row[f'spearman_{ds_name}']:.3f}"
                        )
                    summary_lines.append(line)

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
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=None,
        help="Sweep names to include (e.g. simclr-tau vicreg-cov). Default: all.",
    )
    args = parser.parse_args()
    main(log_dir=args.log_dir, output_dir=args.output_dir, sweeps=args.sweeps)
