# Target Sharing (Auction) Algorithms

This folder contains **multi-robot target allocation / auction algorithms**.

Given:
- The current world state
- A list of robots (with positions and already-assigned targets)
- A set of remaining targets

a target-sharing algorithm decides **which robot gets which targets**.

The simulator calls the selected target-sharing algorithm:
- At the beginning of the run
- Whenever a replanning event is triggered (for example, robots discover unknown obstacles that change paths or make assignments unreasonable)

The algorithm is selected by the `sharing_algo_name` field of the `Config` dataclass in `config.py` (single-run experiments) and by the `sharing_algo_name` entry in `PARAM_GRID` in `batch_config.py` (batch experiments).

---

## Available algorithms

### `RoundRobin`
- File: `round_robin.py`
- Very simple baseline:
  - Iterates through targets and assigns them in a round-robin fashion over robots.
- Ignores distances; useful as a sanity check and baseline.

---

### `PSA` – Parallel Single-Item Auctions
- File: `psa.py`
- Parallel Single-Item Auction style:
  - Conceptually auctions many single targets at once.
  - Each target is assigned to the robot that can serve it with the lowest marginal cost (based on current belief and pathfinding).
- Faster but more myopic than full combinatorial auctions.

---

### `SSA` – Sequential Single-Item Auctions
- File: `ssa.py`
- Sequential Single-Item Auction:
  - Targets are auctioned **one by one**.
  - For each target, robots bid using a tour cost `PC(r, S)` where `S` is the robot's already-assigned targets plus the new candidate.
  - Requires a **tour algorithm** (from `tours/`) to evaluate bids.
- Captures some combinatorial structure through the tour cost, but remains simpler than full combinatorial auctions.

---

### `CA_Optimal` – Exact Combinatorial Auction
- File: `ca_optimal.py`
- Computes an **optimal allocation** for small instances:
  - Searches over assignments of targets to robots.
  - Uses tour costs to evaluate each robot's bundle.
- Exponential in number of targets/robots – only usable for small problems.
- Serves as a gold standard for comparing heuristics.

---

### `CA_Greedy` – Greedy Combinatorial Auction
- File: `ca_greedy.py`
- Builds small **bundles** (for example 1–2 targets) for each robot.
- Computes the tour cost of each bundle and **greedily** picks non-overlapping bundles with lowest cost.
- Approximates the combinatorial auction with much lower computation than `CA_Optimal`.

---

### `CA_IsingFull` – Combinatorial Auction via Full-Precision Ising/QUBO
- File: `ca_ising_full.py`
- Winner determination is solved as a **QUBO** using `TabuSampler`:
  - One binary variable per bundle `b = (robot, subset_of_targets)`.
  - Cost terms encode bundle costs from the tour solver.
  - Constraint terms penalize assigning the same target in multiple bundles.
- Uses full-precision floating-point coefficients in the QUBO.
- Decodes a sample, repairs conflicts if necessary, and outputs assignments.

Requires the `dwave-tabu` package:
```bash
pip install dwave-tabu
```

---

### `CA_IsingLimited` – Hardware-Limited Ising Combinatorial Auction
- File: `ca_ising_limited.py`
- Same idea as `CA_IsingFull`, but simulates hardware limitations:
  - QUBO coefficients are restricted to **integer** range `[-7, +7]`.
  - **Constraints** (target uniqueness) use ±7 (hard constraints).
  - **Costs** are mapped into `1..6` (never 7) using multiple mapping strategies.
  - For a QUBO with `N` bundle variables, a device is simulated with:
    - spin limit: 49 spins
    - anneal time: 50 µs per hardware call
    - power: 24 mW
  - Estimates number of hardware calls and accumulates:
    - total anneal time
    - total energy consumption
- This information is exported in `summary.json` under `"target_sharing"["hardware"]` and in the batch-results CSV.

Also requires `dwave-tabu`.

---

## Common interface

All target-sharing algorithms conform to `TargetSharingAlgorithm` defined in `base.py`:

```python
class TargetSharingAlgorithm(Protocol):
    name: str
    total_runtime: float
    call_count: int

    def reset_stats(self) -> None: ...

    def assign(
        self,
        world: WorldState,
        robots: list[RobotState],
        targets: set[Pos],
    ) -> dict[int, list[Pos]]:
        ...
```

Behavior:

- `assign(...)` returns a **dictionary**: `robot_id -> list of targets` assigned to that robot.
- Each target should be assigned to **at most one robot** in a single call.
- Some algorithms may leave targets unassigned (depending on design), but the provided implementations aim to cover all reachable targets.
- Each algorithm tracks its own runtime:
  - `total_runtime` – total seconds spent across all calls
  - `call_count` – number of times `assign` was invoked

The simulator logs these stats to `summary.json` (single runs) and to `outputs_batch/batch_results.csv` (batch runs).

Many algorithms (SSA, CA_*) also depend on a `TourAlgorithm` from the `tours/` folder; the simulator injects the chosen tour algorithm via a setter such as `set_tour_algo`.

---

## Target-sharing configuration in single runs (`config.py`)

For single-run simulations driven by `main.py`, the sharing algorithm is selected in `config.py` via `Config.sharing_algo_name`:

```python
# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # ...
    sharing_algo_name: str = "PSA"  # for example: "RoundRobin", "PSA", "SSA", "CA_Optimal", ...
```

The value must match the `.name` field of one of the algorithms in this folder (e.g., `"PSA"`, `"CA_Greedy"`, `"CA_IsingLimited"`).

---

## Target-sharing in batch experiments (`batch_run.py` + `batch_config.py`)

Systematic comparison of target-sharing strategies is carried out by `batch_run.py`, which sweeps over a parameter grid and writes all results to `outputs_batch/batch_results.csv`. The grid is defined in `batch_config.py`.

Example configuration for the sharing dimension:

```python
# batch_config.py
from typing import Dict, List, Any

CPU_COUNT: int | None = None  # None -> use all available cores

PARAM_GRID: Dict[str, List[Any]] = {
    # --- meta ---
    "purpose": ["sharing_algorithm_comparison"],

    # --- world parameters ---
    "rows": [20],
    "cols": [20],
    "n_robots": [3],
    "n_targets": [10],
    "known_obstacle_density": [0.2],
    "unknown_obstacle_density": [0.0, 0.1],
    "sense_radius": [1],
    "max_steps": [200],

    # --- algorithms ---
    "path_algo_name": ["AStar"],  # fixed pathfinding to isolate sharing effects
    "sharing_algo_name": [
        "RoundRobin",
        "PSA",
        "SSA",
        "CA_Optimal",
        "CA_Greedy",
        "CA_IsingFull",
        "CA_IsingLimited",
    ],
    "tour_algo_name": ["CheapestInsertion"],  # or "ExactBruteForce" for small instances

    # --- randomness ---
    "seed": [i for i in range(10)],
}
```

Notes:

- `PARAM_GRID` runs **all combinations** of the lists.  
  Fixing `path_algo_name` and `tour_algo_name` and expanding only `sharing_algo_name` produces a clean target-sharing sweep on matched worlds.
- For each run, the batch driver records:
  - `target_sharing.algorithm` – algorithm name
  - `target_sharing.call_count`, `target_sharing.total_runtime`, `target_sharing.avg_runtime`
  - `targets.collected`, `simulation.total_distance`, `simulation.makespan_tick`
  - For Ising-based methods, `target_sharing.hardware.*` with simulated hardware time and energy.

The script `plot_utils.py` can then be used to compare sharing algorithms, for example with `data_set = "target_sharing"` so that boxplots of `target_sharing.avg_runtime` and `simulation.total_distance` are grouped by `target_sharing.algorithm`.

---

## Selecting a target-sharing algorithm in `main.py`

In `main.py`, the simulator is constructed from the configuration; the sharing algorithm name is taken directly from `cfg.sharing_algo_name`:

```python
cfg = Config()

sharing_algo_name = cfg.sharing_algo_name

sim = Simulator(
    cfg=cfg,
    world=world,
    path_algo_name=cfg.path_algo_name,
    sharing_algo_name=sharing_algo_name,
    tour_algo_name=cfg.tour_algo_name,
    log_events=cfg.log_events,
)
```

`target_sharing/__init__.py` auto-discovers available algorithms at import time and exposes them under their `.name` field.

---

## Adding a new target-sharing algorithm

To add an additional auction / allocation method:

1. Create a new file in this folder, for example `my_auction.py`.
2. Implement a class that follows the interface:

```python
from time import perf_counter
from env import WorldState, RobotState, Pos
from .base import TargetSharingAlgorithm

class MyAuction(TargetSharingAlgorithm):
    name = "MyAuction"

    def __init__(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0

    def assign(
        self,
        world: WorldState,
        robots: list[RobotState],
        targets: set[Pos],
    ) -> dict[int, list[Pos]]:
        t0 = perf_counter()

        assignments: dict[int, list[Pos]] = {r.rid: [] for r in robots}

        # Allocation logic
        # ...

        dt = perf_counter() - t0
        self.total_runtime += dt
        self.call_count += 1
        return assignments
```

3. At the bottom of the file, expose it as:

```python
ALGORITHM = MyAuction()
```

4. `target_sharing/__init__.py` automatically imports this file and registers the algorithm under `MyAuction.name`.
5. The new method then becomes available via `Config.sharing_algo_name = "MyAuction"` in `config.py` and may be added to `PARAM_GRID["sharing_algo_name"]` in `batch_config.py` for batch experiments.

---

## Notes

- All algorithms operate on the **current belief world** (robots only know discovered + known obstacles).
- Replanning events (for example, when an unknown obstacle blocks a path) trigger calls to `assign(...)` again with the remaining targets.
- Heuristic variations (distance thresholds, local vs global auctions, multi-round bidding, etc.) can be introduced as long as the input/output interface is preserved.
- For fair comparison in batch experiments, each implementation should update `total_runtime` and `call_count` consistently, and expose any additional hardware-related statistics (anneal time, energy) under clearly named fields so they appear in `summary.json` and `outputs_batch/batch_results.csv`.
