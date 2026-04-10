"""Correlation analysis between unsupervised selection metrics and downstream performance.

At each of {25, 50, 100, 150, 200, 250, 300} epochs, for each sweep, computes
Pearson r and Spearman rho between (RankMe, ReRankMe) and each downstream metric.

Saves to experiments/results/correlation.csv and prints a readable summary.
"""

import csv
import math
from pathlib import Path
from typing import Optional

from log_reader import get_metric_at_epoch, get_logged_epochs
from hyperparameter_selection import SWEEPS, DOWNSTREAM_METRICS, SELECTION_METRICS


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

        for epoch in get_logged_epochs(
            run_names, next(iter(SELECTION_METRICS.values())), log_dir=log_dir
        ):
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
                                run, ds_metric, 299, log_dir=log_dir
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

                if epoch == 299:
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

    # -----------------------------------------------------------------------
    # Figures: one per sweep, subplots = downstream metrics
    # -----------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ds_names = list(DOWNSTREAM_METRICS.keys())
    sel_names = list(SELECTION_METRICS.keys())

    for sweep_name in active_sweeps:
        sweep_rows = [r for r in rows if r["sweep"] == sweep_name]
        if not sweep_rows:
            continue

        fig, axes = plt.subplots(
            1, len(ds_names), figsize=(5 * len(ds_names), 4), sharey=False
        )
        if len(ds_names) == 1:
            axes = [axes]

        for ax, ds_name in zip(axes, ds_names):
            for sel_name in sel_names:
                sel_rows = sorted(
                    [r for r in sweep_rows if r["selection_metric"] == sel_name],
                    key=lambda r: r["epoch"],
                )
                epochs = [r["epoch"] for r in sel_rows]
                pearson = [r[f"pearson_{ds_name}"] for r in sel_rows]
                ax.plot(epochs, pearson, label=sel_name)

            ax.set_title(ds_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Pearson r")
            ax.set_ylim(-1, 1)
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{sweep_name} — correlation with final downstream accuracy")
        fig.tight_layout()
        fig_path = figures_dir / f"correlation_{sweep_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")


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
