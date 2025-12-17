# io_utils.py
from dataclasses import asdict
from pathlib import Path
from datetime import datetime
import json
from typing import Any
from config import Config
import uuid


def make_run_dir(
    cfg: Config,
    base: str = "outputs",
    path_algo_name: str | None = None,
    sharing_algo_name: str | None = None,
    tour_algo_name: str | None = None,
) -> Path:
    """
    Create (if needed) and return a unique directory for this run.

    Parameters
    ----------
    cfg : Config
        The configuration object for this run (grid size, robots, targets, seed, etc.).
    base : str, optional
        Base directory under which the run folder will be created, by default "outputs".
        You can change this (e.g., to "outputs_debug") when calling make_run_dir.
    path_algo_name : str | None, optional
        Name of the pathfinding algorithm. Currently not included in the folder name,
        but kept here so you can easily add it to the name if desired.
    sharing_algo_name : str | None, optional
        Name of the target-sharing (auction) algorithm. Same remark as above.
    tour_algo_name : str | None, optional
        Name of the tour / TSP algorithm. Same remark as above.

    Folder naming
    -------------
    The folder name encodes:
      - grid size (rows x cols)
      - number of robots
      - number of targets
      - random seed
      - a timestamp + short UUID suffix to guarantee uniqueness

    Example:
        outputs/run_R20x20_Rob2_Tgt10_seed1_20251216-213012_ab12cd34/

    Returns
    -------
    Path
        The full path to the newly created run directory.
    """
    # Ensure the base directory (e.g., "outputs") exists.
    base_path = Path(base)
    base_path.mkdir(exist_ok=True)

    # Components that describe the experiment setup.
    parts = [
        f"R{cfg.rows}x{cfg.cols}",
        f"Rob{cfg.n_robots}",
        f"Tgt{cfg.n_targets}",
        f"seed{cfg.seed}",
    ]

    # If you want algorithm names in the folder, uncomment these:
    # if path_algo_name:
    #     parts.append(f"PF-{path_algo_name}")
    # if sharing_algo_name:
    #     parts.append(f"TS-{sharing_algo_name}")
    # if tour_algo_name:
    #     parts.append(f"TR-{tour_algo_name}")

    base_name = "run_" + "_".join(parts)

    # Add a timestamp and a short random suffix so that repeated runs
    # with the same config do not overwrite each other.
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:8]  # short unique ID
    suffix = f"{ts}-{uid}"

    run_dir = base_path / f"{base_name}_{suffix}"

    # exist_ok=False => raise if directory somehow already exists
    run_dir.mkdir(exist_ok=False)
    return run_dir


def save_config(cfg: Config, run_dir: Path, filename: str = "config.json") -> None:
    """
    Serialize the Config object for this run into JSON.

    Parameters
    ----------
    cfg : Config
        Configuration object to save (grid size, robot count, densities, etc.).
    run_dir : Path
        Run directory where the file will be written.
    filename : str, optional
        Name of the JSON file, by default "config.json".

    Notes
    -----
    This provides a full record of how the world was generated and how
    the simulation was parameterized. It is useful for reproducibility
    and for later analysis alongside summary metrics.
    """
    data: dict[str, Any] = asdict(cfg)
    out_path = run_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_summary(summary: dict[str, Any], run_dir: Path, filename: str = "summary.json") -> None:
    """
    Save the summary metrics for a run as a JSON file.

    Parameters
    ----------
    summary : dict[str, Any]
        Nested dictionary of metrics (grid info, robot stats, algorithm runtimes, etc.).
        Typically produced at the end of main.py or inside batch_run.py.
    run_dir : Path
        Run directory where the summary will be written.
    filename : str, optional
        Name of the JSON file, by default "summary.json".

    Notes
    -----
    The JSON structure is intentionally nested (e.g., "grid.rows",
    "pathfinding.total_runtime") so that it is easy to:
      - flatten into CSV columns if needed,
      - or load directly into analysis notebooks for plotting.
    """
    out_path = run_dir / filename
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
