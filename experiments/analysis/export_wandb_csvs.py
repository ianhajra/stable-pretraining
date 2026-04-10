"""Export WandB run histories to CSV files expected by log_reader.py.

For each run name defined in hyperparameter_selection.SWEEPS, fetches the run
history from WandB and writes ``<log_dir>/<run_name>.csv``.

Usage:
    python experiments/analysis/export_wandb_csvs.py [--log_dir DIR] [--entity ENTITY] [--project PROJECT]

The entity and project default to the values set below (edit them or pass via CLI).
Run ``cat ~/scratch/rerankme/logs/wandb/run-*/files/wandb-metadata.json | grep -E '"entity"|"project"'``
to find your entity and project names.
"""

import argparse
import sys
from pathlib import Path

try:
    import wandb
except ImportError:
    print("wandb is not installed. Run: pip install wandb")
    sys.exit(1)

from hyperparameter_selection import SWEEPS

# ---------------------------------------------------------------------------
# Defaults — edit these or pass via CLI flags
# ---------------------------------------------------------------------------
DEFAULT_ENTITY = ""  # e.g. "ihajra"
DEFAULT_PROJECT = ""  # e.g. "rerankme"
DEFAULT_LOG_DIR = Path.home() / "scratch" / "rerankme" / "logs"


def export_runs(
    entity: str, project: str, log_dir: Path, sweeps: list[str] | None = None
) -> None:
    api = wandb.Api()
    log_dir.mkdir(parents=True, exist_ok=True)

    active_sweeps = {k: v for k, v in SWEEPS.items() if sweeps is None or k in sweeps}
    all_run_names: list[str] = [
        name for runs in active_sweeps.values() for name in runs
    ]

    # Fetch all runs in the project once so we don't make one request per run.
    print(f"Fetching run list from {entity}/{project} …")
    try:
        remote_runs = api.runs(f"{entity}/{project}")
    except Exception as e:
        print(f"Error fetching runs: {e}")
        sys.exit(1)

    name_to_run = {r.name: r for r in remote_runs}

    missing = []
    for run_name in all_run_names:
        csv_path = log_dir / f"{run_name}.csv"
        if csv_path.exists():
            print(f"  [skip]   {run_name}.csv already exists")
            continue

        run = name_to_run.get(run_name)
        if run is None:
            print(f"  [warn]   '{run_name}' not found in {entity}/{project}")
            missing.append(run_name)
            continue

        if run.state != "finished":
            print(f"  [skip]   {run_name} state='{run.state}' (not finished)")
            missing.append(run_name)
            continue

        print(f"  [export] {run_name} …", end=" ", flush=True)
        df = run.history(samples=10_000, pandas=True)
        if df.empty:
            print("empty history, skipping")
            missing.append(run_name)
            continue

        # Normalise epoch column: WandB logs it as "_step" if not explicitly set.
        if "epoch" not in df.columns and "_step" in df.columns:
            df = df.rename(columns={"_step": "epoch"})

        df.to_csv(csv_path, index=False)
        print(f"saved ({len(df)} rows)")

    if missing:
        print(f"\nMissing / empty runs ({len(missing)}): {missing}")
    else:
        print("\nAll runs exported successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WandB runs to CSV")
    parser.add_argument(
        "--entity", default=DEFAULT_ENTITY, help="WandB entity (username or team)"
    )
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="WandB project name")
    parser.add_argument(
        "--log_dir",
        default=None,
        help="Directory to write CSVs (default: ~/scratch/rerankme/logs)",
    )
    parser.add_argument(
        "--sweeps",
        nargs="+",
        default=None,
        help="Sweep names to export (e.g. simclr-tau vicreg-cov). Default: all.",
    )
    args = parser.parse_args()

    entity = args.entity
    project = args.project
    if not entity or not project:
        print(
            "Error: --entity and --project are required (or set DEFAULT_ENTITY/DEFAULT_PROJECT in this script)."
        )
        print(
            'Run: cat ~/scratch/rerankme/logs/wandb/run-*/files/wandb-metadata.json | grep -E \'"entity"|"project"\''
        )
        sys.exit(1)

    log_dir = Path(args.log_dir).expanduser() if args.log_dir else DEFAULT_LOG_DIR
    export_runs(entity, project, log_dir, sweeps=args.sweeps)


if __name__ == "__main__":
    main()
