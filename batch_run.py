#!/usr/bin/env python3
"""
batch_run.py

Batch experiment runner.

- Builds a parameter grid (world + algorithm params + meta "purpose").
- For each combination:
    - builds Config + WorldState + Simulator
    - runs one simulation (same logic as in main.py, but without PNG/GIF I/O)
    - builds the same summary dict as main.py
- Uses multiprocessing to parallelize.
- Writes/appends to a single CSV under outputs_batch/, incrementally.

If outputs_batch/batch_results.csv already exists:
    - read its header
    - append new rows with the same columns
"""

from __future__ import annotations

import csv
import itertools
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Optional

from config import Config
from env import WorldState
from sim import Simulator
from batch_config import PARAM_GRID, CPU_COUNT  # user-editable batch settings
import traceback


# ---------------------------------------------------------------------
# 1) Parameter combinations
# ---------------------------------------------------------------------


def iter_param_combinations(grid: Dict[str, List[Any]]):
    """
    Yield dicts for each combination in the parameter grid.

    Example:
        grid = {"rows": [20, 30], "cols": [20, 30]}
        -> yields 4 dicts with all (rows, cols) pairs.
    """
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    for combo in itertools.product(*value_lists):
        yield dict(zip(keys, combo))


# ---------------------------------------------------------------------
# 2) Helper: flatten nested dicts (for CSV columns)
# ---------------------------------------------------------------------


def flatten_dict(
    d: Dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """
    Turn nested dicts into a flat dict with dotted keys:

        {"a": {"b": 1}, "c": 2}
        -> {"a.b": 1, "c": 2}
    """
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# ---------------------------------------------------------------------
# 3) One experiment: like main(), but no images/GIFs, returns metrics
# ---------------------------------------------------------------------


def run_single_experiment(
    purpose: str,                # meta label, not used in sim; only for CSV
    rows: int,
    cols: int,
    n_robots: int,
    n_targets: int,
    known_obstacle_density: float,
    unknown_obstacle_density: float,
    sense_radius: int,
    seed: int,
    max_steps: int,
    path_algo_name: str,
    sharing_algo_name: str,
    tour_algo_name: str,
) -> Dict[str, Any]:
    """
    Run ONE simulation with the given parameters and return a flat dict of metrics.

    This is essentially main(), but:
      - no run_dir, no images/GIFs
      - returns metrics instead of writing files
    """

    # --- build config (mirrors main.py) ---
    cfg = Config(
        rows=rows,
        cols=cols,
        n_robots=n_robots,
        n_targets=n_targets,
        known_obstacle_density=known_obstacle_density,
        unknown_obstacle_density=unknown_obstacle_density,
        sense_radius=sense_radius,
        seed=seed,
        max_steps=max_steps,
    )

    # --- world ---
    world = WorldState.from_config(cfg)

    # record targets for counts
    initial_target_cells = set(world.targets)
    initial_target_count = len(world.targets)

    # --- simulator ---
    # log_events=False in batch to avoid spamming stdout
    sim = Simulator(
        cfg=cfg,
        world=world,
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
        log_events=False,
    )

    # reset timing stats
    sim.path_algo.reset_stats()
    sim.sharing_algo.reset_stats()
    if getattr(sim, "tour_algo", None) is not None:
        sim.tour_algo.reset_stats()

    # run sim
    sim.run()

    # --- build summary (same logic as in main.py, minus images) ---

    pa = sim.path_algo
    sa = sim.sharing_algo
    ta = getattr(sim, "tour_algo", None)

    # per-robot distance stats
    dist_by_robot = sim.robot_distances
    dist_values = list(dist_by_robot.values())
    total_dist_check = sum(dist_values) if dist_values else 0

    max_dist = max(dist_values) if dist_values else 0
    min_dist = min(dist_values) if dist_values else 0
    avg_dist = total_dist_check / len(dist_values) if dist_values else 0
    imbalance = max_dist - min_dist

    summary: Dict[str, Any] = {
        "grid": {"rows": cfg.rows, "cols": cfg.cols},
        "robots": {"count": cfg.n_robots},
        "targets": {
            "initial": initial_target_count,
            "remaining": len(world.targets),
            "collected": initial_target_count - len(world.targets),
        },
        "simulation": {
            "ticks": sim.ticks,
            "max_steps": cfg.max_steps,
            "total_distance": sim.total_distance,
            "total_distance_check_from_robots": total_dist_check,
            "makespan_tick": sim.last_collection_tick,
            "all_targets_collected": (len(world.targets) == 0),
        },
        "per_robot_distance": {
            "by_robot": dist_by_robot,
            "max": max_dist,
            "min": min_dist,
            "avg": avg_dist,
            "imbalance": imbalance,
        },
    }

    # Pathfinding metrics
    summary["pathfinding"] = {
        "algorithm": pa.name,
        "call_count": pa.call_count,
        "total_runtime": pa.total_runtime,
        "avg_runtime": (pa.total_runtime / pa.call_count) if pa.call_count else 0.0,
    }

    # Target sharing metrics (plus hardware if available)
    ts_entry: Dict[str, Any] = {
        "algorithm": sa.name,
        "call_count": sa.call_count,
        "total_runtime": sa.total_runtime,
        "avg_runtime": (sa.total_runtime / sa.call_count) if sa.call_count else 0.0,
    }

    if hasattr(sa, "total_hw_anneal_time"):
        ts_entry["hardware"] = {
            "total_anneal_time_s": sa.total_hw_anneal_time,
            "total_energy_J": sa.total_hw_energy,
            "last_anneal_time_s": sa.last_hw_anneal_time,
            "last_calls": sa.last_hw_calls,
            "spin_limit": sa.HW_SPIN_LIMIT,
            "anneal_time_per_call_s": sa.HW_ANNEAL_TIME_S,
            "power_W": sa.HW_POWER_W,
        }

    summary["target_sharing"] = ts_entry

    # Tours metrics (plus hardware / SNN if available)
    if ta is not None:
        tours_entry: Dict[str, Any] = {
            "algorithm": ta.name,
            "call_count": ta.call_count,
            "total_runtime": ta.total_runtime,
            "avg_runtime": (ta.total_runtime / ta.call_count) if ta.call_count else 0.0,
        }

        if hasattr(ta, "total_hw_anneal_time"):
            tours_entry["hardware"] = {
                "total_anneal_time_s": ta.total_hw_anneal_time,
                "total_energy_J": ta.total_hw_energy,
                "last_anneal_time_s": ta.last_hw_anneal_time,
                "last_calls": ta.last_hw_calls,
                "spin_limit": ta.HW_SPIN_LIMIT,
                "anneal_time_per_call_s": ta.HW_ANNEAL_TIME_S,
                "power_W": ta.HW_POWER_W,
            }

        if hasattr(ta, "total_sim_time"):
            tours_entry["SNN"] = {
                "total_event_count": ta.total_event_count,
                "total_sim_time": ta.total_sim_time,
            }

        summary["tours"] = tours_entry

    # Flatten nested summary for CSV columns
    flat_summary = flatten_dict(summary)

    # IMPORTANT: we do NOT include 'purpose' here;
    # it comes from PARAM_GRID and will be merged in run_one().
    return flat_summary


# ---------------------------------------------------------------------
# 4) Worker wrapper for multiprocessing
# ---------------------------------------------------------------------


def run_one(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Worker function for each process.

    - Calls run_single_experiment(**params).
    - Returns merged {params..., flat_summary...} dict.
    - If the run fails, returns None and prints an error.
    """
    params = dict(params)  # make a shallow copy

    try:
        metrics = run_single_experiment(**params)
    except Exception as e:
        print(f"[ERROR] run_single_experiment failed for params={params}: {e}")
        traceback.print_exc()
        # returning None tells the caller to skip this run
        return None

    if not isinstance(metrics, dict):
        raise TypeError("run_single_experiment must return a dict")

    merged: Dict[str, Any] = {**params, **metrics}
    return merged


# ---------------------------------------------------------------------
# 5) Batch driver: incremental CSV writing in outputs_batch/
# ---------------------------------------------------------------------


def main_batch() -> None:
    combos = list(iter_param_combinations(PARAM_GRID))
    total = len(combos)
    if total == 0:
        print("No parameter combinations to run. Check PARAM_GRID in batch_config.py.")
        return

    print(f"Total experiments to run: {total}")

    out_dir = Path("outputs_batch")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_results.csv"

    file_exists = out_path.exists()

    # If file already exists, read its header to get fieldnames
    if file_exists:
        print(f"Appending to existing CSV: {out_path}")
        with out_path.open("r", newline="") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = []
        fieldnames = existing_header if existing_header else None
    else:
        fieldnames = None

    # If we don't have fieldnames yet (new file or empty existing file),
    # run the first job synchronously to infer them and write header + first row.
    start_index = 0
    if fieldnames is None:
        first_params = combos[0]
        print("Running first job synchronously to infer CSV columns...")
        first_row = run_one(first_params)

        if first_row is None:
            raise RuntimeError("First experiment failed; cannot infer CSV columns.")

        fieldnames = sorted(first_row.keys())
        # Ensure 'purpose' is the first column if present
        if "purpose" in fieldnames:
            fieldnames.remove("purpose")
            fieldnames = ["purpose"] + fieldnames

        # Create file and write header + first row
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(first_row)
            f.flush()
        print(f"Created new CSV and wrote first row to {out_path}")

        start_index = 1  # we already ran combos[0]
    else:
        print(f"Using existing header with {len(fieldnames)} columns.")

    # Now append remaining experiments in parallel
    remaining = combos[start_index:]
    remaining_total = len(remaining)
    if remaining_total == 0:
        print("No remaining experiments to run; done.")
        return

    print(f"Running remaining {remaining_total} experiments in parallel...")

    # Decide how many processes to use:
    # - If CPU_COUNT is None, fall back to mp.cpu_count().
    # - Otherwise, use the user-specified number from batch_config.py.
    n_procs = CPU_COUNT if CPU_COUNT is not None else mp.cpu_count()
    print(f"Using {n_procs} worker processes.")

    done = start_index

    # Open file in append mode and stream rows as they complete
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        with mp.Pool(processes=n_procs) as pool:
            for row in pool.imap_unordered(run_one, remaining):
                if row is None:
                    # this run failed; already logged, so just skip it
                    continue

                writer.writerow(row)
                f.flush()
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"Completed {done}/{total} experiments")

    print(f"All done. Results in {out_path}")


if __name__ == "__main__":
    main_batch()
