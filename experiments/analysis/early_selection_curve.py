"""Early selection curve plots.

For each sweep and downstream measure, produces a plot showing the downstream
accuracy (at final epoch 300) of the run selected by RankMe vs ReRankMe at
each intermediate epoch.  A horizontal dashed oracle line and a ±0.5% band
are included.

Saves plots to experiments/results/figures/ and a summary CSV to
experiments/results/early_selection_summary.csv.
"""

import csv
from pathlib import Path
from typing import Optional

from log_reader import get_metric_at_epoch
from hyperparameter_selection import (
    SWEEPS,
    DOWNSTREAM_METRICS,
    SELECTION_METRICS,
    SELECTION_EPOCHS,
    FINAL_EPOCH,
    algorithm1,
)

ORACLE_BAND = 0.005  # 0.5%


def _compute_selection_curve(
    sweep_name: str,
    run_names: list[str],
    sel_metric_key: str,
    ds_metric_key: str,
    log_dir: Optional[str],
) -> dict[int, Optional[float]]:
    """For each selection epoch, return the final downstream accuracy of the selected run."""
    curve: dict[int, Optional[float]] = {}
    for epoch in SELECTION_EPOCHS:
        pairs = []
        for run in run_names:
            try:
                val = get_metric_at_epoch(run, sel_metric_key, epoch, log_dir=log_dir)
                pairs.append((run, val))
            except (KeyError, FileNotFoundError):
                pass
        if not pairs:
            curve[epoch] = None
            continue
        selected = algorithm1(pairs)
        try:
            final_val = get_metric_at_epoch(
                selected, ds_metric_key, FINAL_EPOCH, log_dir=log_dir
            )
        except (KeyError, FileNotFoundError):
            final_val = None
        curve[epoch] = final_val
    return curve


def main(
    log_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    sweeps: Optional[list[str]] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = (
        Path(output_dir) if output_dir else Path(__file__).parent.parent / "results"
    )
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    active_sweeps = {k: v for k, v in SWEEPS.items() if sweeps is None or k in sweeps}
    for sweep_name, run_names in active_sweeps.items():
        for ds_name, ds_metric in DOWNSTREAM_METRICS.items():
            # Oracle: best possible final downstream value
            oracle = None
            for run in run_names:
                try:
                    val = get_metric_at_epoch(
                        run, ds_metric, FINAL_EPOCH, log_dir=log_dir
                    )
                    if oracle is None or val > oracle:
                        oracle = val
                except (KeyError, FileNotFoundError):
                    pass

            fig, ax = plt.subplots(figsize=(8, 5))

            first_in_band: dict[str, Optional[int]] = {}

            for sel_name, sel_key in SELECTION_METRICS.items():
                curve = _compute_selection_curve(
                    sweep_name, run_names, sel_key, ds_metric, log_dir
                )
                epochs = [e for e, v in curve.items() if v is not None]
                values = [curve[e] for e in epochs]

                ax.plot(epochs, values, marker="o", label=sel_name)

                first_in_band[sel_name] = None
                if oracle is not None:
                    for e in SELECTION_EPOCHS:
                        v = curve.get(e)
                        if v is not None and abs(v - oracle) <= ORACLE_BAND:
                            first_in_band[sel_name] = e
                            break

                summary_rows.append(
                    {
                        "sweep": sweep_name,
                        "downstream": ds_name,
                        "selection_metric": sel_name,
                        "first_epoch_in_band": first_in_band[sel_name],
                    }
                )

            if oracle is not None:
                ax.axhline(oracle, linestyle="--", color="black", label="oracle")
                ax.axhspan(
                    oracle - ORACLE_BAND,
                    oracle + ORACLE_BAND,
                    alpha=0.15,
                    color="gray",
                    label="±0.5% band",
                )

            ax.set_xlabel("Selection epoch")
            ax.set_ylabel(f"Final epoch-{FINAL_EPOCH} {ds_name}")
            ax.set_title(f"{sweep_name} — {ds_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)

            fig_path = figures_dir / f"early_selection_{sweep_name}_{ds_name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fig_path}")

    summary_path = results_dir / "early_selection_summary.csv"
    if summary_rows:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"Saved: {summary_path}")


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
