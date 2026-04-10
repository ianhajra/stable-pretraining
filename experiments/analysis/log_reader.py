"""Log reader utility for rerankme experiments.

All functions accept an optional ``log_dir`` argument defaulting to
``~/scratch/rerankme/logs``.  Logs are expected to be WandB-exported CSV files
named ``<run_name>.csv`` inside ``log_dir``, with columns ``epoch`` and one
column per logged metric.
"""

from pathlib import Path
from typing import Optional
import csv

_DEFAULT_LOG_DIR = Path.home() / "scratch" / "rerankme" / "logs"


def _resolved(log_dir: Optional[Path | str]) -> Path:
    return Path(log_dir).expanduser() if log_dir is not None else _DEFAULT_LOG_DIR


def get_all_run_names(log_dir: Optional[str] = None) -> list[str]:
    """Return all run names found in the log directory."""
    base = _resolved(log_dir)
    return [p.stem for p in sorted(base.glob("*.csv"))]


def get_metric_history(
    run_name: str,
    metric: str,
    log_dir: Optional[str] = None,
) -> dict[int, float]:
    """Return a dict of epoch -> value for a metric across all logged epochs."""
    base = _resolved(log_dir)
    csv_path = base / f"{run_name}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No log file found for run '{run_name}' at {csv_path}")
    history: dict[int, float] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric not in row or row[metric] == "":
                continue
            epoch_str = row.get("epoch", "")
            if epoch_str == "":
                continue
            history[int(float(epoch_str))] = float(row[metric])
    return history


def get_metric_at_epoch(
    run_name: str,
    metric: str,
    epoch: int,
    log_dir: Optional[str] = None,
    tolerance: int = 5,
) -> float:
    """Return the logged value of a metric at or nearest to a specific epoch.

    If the exact epoch is not present, returns the value from the closest
    available epoch within ``tolerance`` epochs.  Raises KeyError if no epoch
    within the tolerance exists.
    """
    history = get_metric_history(run_name, metric, log_dir=log_dir)
    if epoch in history:
        return history[epoch]
    available = sorted(history.keys())
    nearest = min(available, key=lambda e: abs(e - epoch))
    if abs(nearest - epoch) <= tolerance:
        return history[nearest]
    raise KeyError(
        f"Epoch {epoch} not found within tolerance={tolerance} for run '{run_name}', "
        f"metric '{metric}'. Nearest available: {nearest}."
    )


def get_logged_epochs(
    run_names: list[str],
    metric: str,
    log_dir: Optional[str] = None,
) -> list[int]:
    """Return sorted union of all epochs at which metric was logged across runs."""
    epochs: set[int] = set()
    for run in run_names:
        try:
            epochs.update(get_metric_history(run, metric, log_dir=log_dir).keys())
        except FileNotFoundError:
            pass
    return sorted(epochs)


def find_checkpoint(
    run_name: str,
    epoch: int,
    log_dir: Optional[str] = None,
) -> str:
    """Return checkpoint path, checking scratch first then data."""
    base = _resolved(log_dir)
    scratch_root = base.parent  # ~/scratch/rerankme
    candidates = [
        scratch_root / "checkpoints" / run_name / f"epoch={epoch:03d}.ckpt",
        Path.home()
        / "data"
        / "rerankme"
        / "checkpoints"
        / run_name
        / f"epoch={epoch:03d}.ckpt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        f"Checkpoint for run '{run_name}' epoch {epoch} not found. Searched:\n"
        + "\n".join(f"  {p}" for p in candidates)
    )
