#!/usr/bin/env python3
"""
Batch experiment runner.

This script is meant for *offline experiments* where you want to:

- Sweep over many world / algorithm configurations.
- Run one simulation per configuration (no PNG / GIF output).
- Collect all metrics into a single CSV file for analysis.

High-level behavior
-------------------

1. Build a parameter grid (world parameters, algorithms, seeds, and a free-text "purpose").
2. For each combination in the grid:
   - Build Config + WorldState + Simulator.
   - Run one simulation (same core logic as in main.py, but without I/O).
   - Build the same kind of summary dict as main.py.
3. Use multiprocessing to parallelize runs across CPU cores.
4. Flatten the summary dict + parameters into a single row.
5. Append rows to `outputs_batch/batch_results.csv`.

If `outputs_batch/batch_results.csv` already exists:
- The script reads its header.
- Infers the column order from that header.
- Appends new experiment rows using the same schema.

Usage
-----

From the repo root:

    python batch_run.py

Then inspect the resulting CSV:

    outputs_batch/batch_results.csv

You can open it with pandas, Excel, or any plotting tool to compare algorithms.
"""

import csv
import itertools
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Optional
import traceback

from config import Config
from env import WorldState
from sim import Simulator

# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

# Upper bound for how many worker processes we spawn.
# In main_batch() we actually use:
#
#   num_procs = min(CPU_COUNT, mp.cpu_count())
#
# so you can lower this if you're on a shared machine or laptop.
CPU_COUNT = 4

# ---------------------------------------------------------------------------
# Algorithm choices (for documentation / reference)
# ---------------------------------------------------------------------------

# Pathfinding algorithms (see pathfinding/README.md):
#
#   "BFS"      - Breadth-first search on the grid (unweighted shortest paths).
#   "AStar"    - A* search with Manhattan heuristic (optimal).
#   "WAStar"   - Weighted A* (bounded-suboptimal, typically fewer expansions).
#   "GBFS"     - Greedy best-first search (very fast, not optimal).
#   "IDAStar"  - Iterative Deepening A* (low memory, optimal).
#   "SMAStar"  - Simplified memory-bounded A*.
#
# Target-sharing (auction) algorithms (see target_sharing/README.md):
#
#   "RoundRobin"      - Baseline: assigns targets in round-robin order (no distances).
#   "PSA"             - Parallel Single-Item Auctions (one bid per target, per robot).
#   "SSA"             - Sequential Single-Item Auctions (uses tour costs PC(r, S)).
#   "CA_Optimal"      - Single-round Combinatorial Auction, exact (small target sets).
#   "CA_Greedy"       - Greedy Combinatorial Auction (approximate).
#   "CA_IsingFull"    - CA winner determination via full-precision Ising/QUBO (TabuSampler).
#   "CA_IsingLimited" - CA winner determination via integer QUBO [-7,7],
#                       simulating 49-spin, 50µs, 24mW hardware and logging HW stats.
#
# Tour / TSP algorithms (see tours/README.md), used only when sharing algorithm
# needs tour costs:
#
#   Used by:
#     - "SSA"
#     - "CA_Optimal"
#     - "CA_Greedy"
#     - "CA_IsingFull"
#     - "CA_IsingLimited"
#
#   Ignored by:
#     - "RoundRobin"
#     - "PSA"
#
#   Options:
#     "NearestNeighbor"    - Simple nearest-neighbor heuristic (fast, approximate).
#     "ExactBruteForce"    - Exact TSP for small target sets (slow but optimal).
#     "CheapestInsertion"  - Cheapest-insertion heuristic (similar to what many papers use).
#     "IsingFull"          - QUBO Ising tour via TabuSampler (no artificial limits).
#     "IsingLimited"       - Hardware-style Ising: integer coeffs in [-7,7],
#                            simulated 49-spin, 50µs, 24mW hardware.


# ---------------------------------------------------------------------
# 1) Parameter grid
# ---------------------------------------------------------------------

# PARAM_GRID defines the cartesian product of experiments to run.
#
# Each key corresponds to a keyword argument of run_single_experiment().
# Each value is a list; we take the product of all lists.
#
# 'purpose' is a free-text label that will be included in each CSV row.

# PARAM_GRID will run all permutations of given parameters
# add values inside the list, like 
# "rows":[20,30]
# "cols":[20,30]
# this will run 4 different worlds, with rows=20/cols=20, rows=20/cols=30, rows=30/cols=20, rows=30/cols=30
# be careful about adding many trials, experiment count scale exponentially
PARAM_GRID: Dict[str, List[Any]] = {
    # --- meta ---
    "purpose": ["sharing_algorithm_comparison"],  # free-text label to tag this batch of experiments

    # --- world parameters ---
    "rows": [20],                     # number of grid rows in the world
    "cols": [20],                     # number of grid columns in the world
    "n_robots": [1],                  # how many robots are spawned in each world
    "n_targets": [10],                # how many targets must be collected in each run
    "known_obstacle_density": [0.2],  # fraction of cells that start as known (visible) obstacles
    "unknown_obstacle_density": [0.0, 0.1],  # fraction of cells that are hidden obstacles, discovered via sensing
    "sense_radius": [1],              # Manhattan sensing radius for each robot’s local sensor
    "max_steps": [200],               # maximum number of simulation ticks before we force-terminate a run

    # --- algorithms ---
    # "path_algo_name": ["BFS","AStar","WAStar","GBFS","IDAStar","SMAStar"],
    "path_algo_name": ["BFS", "AStar", "WAStar", "GBFS", "IDAStar", "SMAStar"],  # which pathfinding algorithms to compare

    # "sharing_algo_name": ["RoundRobin","PSA","SSA","CA_Optimal","CA_Greedy","CA_IsingFull","CA_IsingLimited"],
    "sharing_algo_name": ["CA_Optimal"],  # target-sharing algorithm (fixed here so we isolate pathfinding effects)

    # "tour_algo_name": ["ExactBruteForce","CheapestInsertion","NearestNeighbor","IsingFull","IsingLimited"],
    "tour_algo_name": ["ExactBruteForce"],  # tour/TSP algorithm used for PC(r, S) by SSA / CA_* (fixed to exact here)

    # --- randomness ---
    "seed": [i for i in range(10)],  # list of RNG seeds to generate different random worlds for each setting
}



def iter_param_combinations(grid: Dict[str, List[Any]]):
    """Yield dicts for each combination in the parameter grid."""
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

        {"a": {"b": 1}, "c": 2}  ->  {"a.b": 1, "c": 2}
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
# 3) One experiment: this is your old main(), minus I/O
# ---------------------------------------------------------------------

def run_single_experiment(
    purpose: str,                # meta label, not used in sim, just for CSV
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

    This is basically your main(), but:
      - no run_dir, no images/GIFs
      - returns metrics instead of writing files
    """

    # --- build config (from your main.py) ---
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

    # record targets
    initial_target_cells = set(world.targets)       # only used for counts
    initial_target_count = len(world.targets)

    # --- simulator ---
    sim = Simulator(
        cfg=cfg,
        world=world,
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
    )

    # reset timing stats
    sim.path_algo.reset_stats()
    sim.sharing_algo.reset_stats()
    
    # run sim
    sim.run()

    # --- build summary (same logic as your main.py, just no saving) ---

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

    # Tours metrics (plus hardware if available)
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

        summary["tours"] = tours_entry

    # Flatten nested summary for CSV
    flat_summary = flatten_dict(summary)

    # We do NOT include 'purpose' here; it comes from params and will be merged outside.
    return flat_summary


# ---------------------------------------------------------------------
# 4) Worker for multiprocessing
# ---------------------------------------------------------------------

def run_one(params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Worker function for each process.

    - Calls run_single_experiment(**params).
    - Returns merged {params..., flat_summary...} dict.
    - If the run fails, returns None and prints an error.
    """
    params = dict(params)

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

def main_batch():
    combos = list(iter_param_combinations(PARAM_GRID))
    total = len(combos)
    if total == 0:
        print("No parameter combinations to run. Check PARAM_GRID.")
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

    print(f"Running remaining {remaining_total} experiments in parallel using {CPU_COUNT} CPUs ...")

    done = start_index
    # Open file in append mode
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        with mp.Pool(processes=mp.cpu_count()) as pool:
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
