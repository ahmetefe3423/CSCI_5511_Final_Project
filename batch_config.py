# batch_config.py
from __future__ import annotations

from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# CPU usage for batch_run.py
# ---------------------------------------------------------------------------
# If CPU_COUNT is None, batch_run.py will use mp.cpu_count().
# Otherwise, it will use exactly this many worker processes.
#
# Example:
#   CPU_COUNT = 32       # use 32 processes
#   CPU_COUNT = None     # auto-detect from the machine
CPU_COUNT: int | None = None

# ---------------------------------------------------------------------------
# Parameter grid for batch_run.py
# ---------------------------------------------------------------------------
# PARAM_GRID will run all permutations (Cartesian product) of the values.
#
# Example:
#   "rows": [20, 30]
#   "cols": [20, 30]
# will generate 4 world shapes:
#   (20x20), (20x30), (30x20), (30x30)
#
# Be careful: experiment count grows exponentially in the number of values
# per key, i.e.  prod(len(v) for v in PARAM_GRID.values()).
PARAM_GRID: Dict[str, List[Any]] = {
    # --- meta ---
    "purpose": ["sharing_algorithm_comparison"],  # free-text label for this batch

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
    # Full options (commented for convenience):
    # "path_algo_name": ["BFS","AStar","WAStar","GBFS","IDAStar","SMAStar"],
    "path_algo_name": ["BFS", "AStar", "WAStar", "GBFS", "IDAStar", "SMAStar"],  # which pathfinding algorithms to compare

    # "sharing_algo_name": ["RoundRobin","PSA","SSA","CA_Optimal","CA_Greedy","CA_IsingFull","CA_IsingLimited"],
    "sharing_algo_name": ["CA_Optimal"],  # target-sharing algorithm (fixed here so we isolate pathfinding effects)

    # "tour_algo_name": ["ExactBruteForce","CheapestInsertion","NearestNeighbor","IsingFull","IsingLimited"],
    "tour_algo_name": ["ExactBruteForce"],  # tour/TSP algorithm used for PC(r, S) by SSA / CA_* (fixed to exact here)

    # --- randomness ---
    "seed": [i for i in range(10)],  # RNG seeds to generate different random worlds for each setting
}


# -----------------------------------------------------------------------
# Algorithm details, no modification is needed below
# -----------------------------------------------------------------------

    # Pathfinding algorithm (see pathfinding/README.md):
    #
    #   "BFS"      - Breadth-first search baseline (unweighted shortest paths).
    #   "AStar"    - A* with Manhattan heuristic (optimal).
    #   "WAStar"   - Weighted A* (bounded-suboptimal, typically fewer expansions than A*).
    #   "GBFS"     - Greedy best-first search (very fast, not optimal).
    #   "IDAStar"  - Iterative Deepening A* (low memory, optimal but may re-expand many nodes).
    #   "SMAStar"  - Simplified memory-bounded A* (limits OPEN size; trades completeness/optimality for memory).

    # Target-sharing (auction) algorithm (see target_sharing/README.md):
    #
    #   "RoundRobin"      - Baseline: assigns targets in round-robin order (ignores distances).
    #
    #   "PSA"             - Parallel Single-Item Auction:
    #                         each target is auctioned independently, in parallel;
    #                         bids are based on single-target path length.
    #
    #   "SSA"             - Sequential Single-Item Auction:
    #                         considers the marginal cost of inserting each target
    #                         into a robot's tour PC(r, S) using a tour algorithm.
    #
    #   "CA_Optimal"      - Single-round Combinatorial Auction:
    #                         enumerates all assignments (small instances only) and
    #                         chooses the assignment that minimizes total tour cost.
    #
    #   "CA_Greedy"       - Greedy Combinatorial Auction:
    #                         approximates CA_Optimal by greedily accepting low-cost
    #                         bundles (size 1 or 2) that don't conflict.
    #
    #   "CA_IsingFull"    - Combinatorial auction where bundle selection is done
    #                         via a full-precision Ising/QUBO model solved by TabuSampler.
    #
    #   "CA_IsingLimited" - Same as above but with integer QUBO coefficients in [-7, 7],
    #                         simulating a 49-spin, 50µs, 24mW hardware constraint and
    #                         tracking approximate anneal time and energy usage.

    # Tour / TSP algorithm for PC(r, S) (see tours/README.md).
    #
    # This is only used when the sharing algorithm needs tour costs:
    #   - "SSA"
    #   - "CA_Optimal"
    #   - "CA_Greedy"
    #   - "CA_IsingFull"
    #   - "CA_IsingLimited"
    #
    # For "RoundRobin" and "PSA", the tour algorithm is ignored.
    #
    # Options:
    #   "NearestNeighbor"    - Greedy nearest-neighbor heuristic (fast, approximate).
    #   "ExactBruteForce"    - Exact TSP solver for small target sets (exponential time).
    #   "CheapestInsertion"  - Insertion heuristic similar to what many papers use.
    #   "IsingFull"          - QUBO / Ising tour via TabuSampler (no artificial limits).
    #   "IsingLimited"       - Hardware-style Ising tour: integer coeffs in [-7, 7],
    #                            simulated 49-spin, 50µs, 24mW hardware with energy stats.