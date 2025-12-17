# Tour / TSP Algorithms

This folder contains **tour solvers** for a single robot over a small set of targets.

Given:
- The current world state
- A robot start position
- A list of targets assigned to that robot

a `TourAlgorithm` chooses a **visit order** for the targets and returns the total path cost.  
These are essentially small Traveling Salesman Problem (TSP) solvers over the grid.

Tour solvers are used by:
- `SSA` (Sequential Single-Item Auction) in `target_sharing/ssa.py`
- Combinatorial auctions (`CA_Optimal`, `CA_Greedy`, `CA_IsingFull`, `CA_IsingLimited`)

The algorithm is selected by the `tour_algo_name` field of the `Config` dataclass in `config.py` (single-run experiments) and by the `tour_algo_name` entry in `PARAM_GRID` in `batch_config.py` (batch experiments).

---

## Available algorithms

### `NearestNeighbor`
- File: `nearest_neighbor.py`
- Greedy TSP heuristic:
  - From current position, repeatedly visit the closest remaining target
- Fast and simple baseline

---

### `CheapestInsertion`
- File: `cheapest_insertion.py`
- Classic insertion heuristic:
  - Starts from a small tour
  - Repeatedly inserts a new target in the position that causes the smallest increase in tour cost
- Often yields better tour quality than NearestNeighbor, still efficient for small target sets

---

### `ExactBruteForce`
- File: `exact_bruteforce.py`
- Enumerates all permutations of the target set and picks the best tour
- Guarantees the optimal tour cost
- Only usable for **small** sets of targets (factorial complexity)

---

### `IsingFull`
- File: `ising_full.py`
- Encodes the tour problem for one robot as a **QUBO** and solves it with `TabuSampler`:
  - Binary variables x_{i,p} indicate whether target i is at position p in the tour
  - Constraints enforce that each position has exactly one target and each target appears exactly once
  - Cost terms use grid-based shortest path distances
- Uses full-precision floating-point QUBO coefficients
- Decodes the best sample, repairs it to a valid permutation if needed, and computes the **real** tour cost

Requires the `dwave-tabu` package:
```bash
pip install dwave-tabu
```

---

### `IsingLimited`
- File: `ising_limited.py`
- Same modeling idea as `IsingFull`, but simulates hardware limits:
  - QUBO coefficients are restricted to **integer** range `[-7, +7]`
  - **Constraints** (assignment constraints) get ±7 (hard)
  - **Costs** (distances) are mapped into `1..6` (never 7), using multiple mapping strategies
  - Tries several integer QUBO mappings, runs Tabu on each, and keeps the tour with the **lowest real cost**
  - Simulates a device with:
    - spin limit: 49 spins
    - anneal time: 50 µs per call
    - power: 24 mW
  - Estimates number of hardware calls and accumulates:
    - total anneal time
    - total energy

Hardware metrics are stored in `summary.json` under `"tours"["hardware"]` and in the batch-results CSV.

Also requires `dwave-tabu`.

---

## Common interface

All tour algorithms conform to `TourAlgorithm` defined in `base.py`:

```python
class TourAlgorithm(Protocol):
    name: str
    total_runtime: float
    call_count: int

    def reset_stats(self) -> None: ...

    def solve(
        self,
        world: WorldState,
        start: Pos,
        targets: list[Pos],
    ) -> tuple[list[Pos], float]:
        ...
```

Behavior:

- `solve(...)` returns:
  - `order`: list of targets in visit order
  - `cost`: total tour cost (sum of shortest path lengths between legs)
- If no valid tour exists (some targets are unreachable), it returns:
  - `order = []`
  - `cost = float("inf")`
- Runtime measurement:
  - `total_runtime` – total wall-clock seconds spent in `solve`
  - `call_count` – number of times `solve` was called

The simulator records these statistics in `summary.json` (single runs) and in `outputs_batch/batch_results.csv` (batch runs).

---

## Tour configuration in single runs (`config.py`)

For simulations driven by `main.py`, the tour algorithm is selected via `Config.tour_algo_name` in `config.py`:

```python
# config.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # ...
    # Tour (TSP-style) solver, used by SSA / CA_*
    #   None                – no tour model (single-target distances only)
    #   "NearestNeighbor"   – greedy TSP heuristic
    #   "CheapestInsertion" – insertion heuristic TSP
    #   "ExactBruteForce"   – exact tour (small target sets)
    #   "IsingFull"         – Ising-based tour via TabuSampler
    #   "IsingLimited"      – hardware-limited Ising tour
    tour_algo_name: Optional[str] = "CheapestInsertion"
```

If `tour_algo_name` is `None`, sharing algorithms that do not require tours (for example, PSA, RoundRobin) operate without a TSP solver.

---

## Tour algorithms in batch experiments (`batch_run.py` + `batch_config.py`)

Systematic comparison of tour/TSP solvers is carried out by `batch_run.py`, which sweeps over a parameter grid and writes all results to `outputs_batch/batch_results.csv`. The grid is defined in `batch_config.py`.

Example configuration for the tour dimension:

```python
# batch_config.py
from typing import Dict, List, Any

CPU_COUNT: int | None = None  # None -> use all available cores

PARAM_GRID: Dict[str, List[Any]] = {
    # --- meta ---
    "purpose": ["tour_algorithm_comparison"],

    # --- world parameters ---
    "rows": [20],
    "cols": [20],
    "n_robots": [2],
    "n_targets": [8],
    "known_obstacle_density": [0.2],
    "unknown_obstacle_density": [0.0, 0.1],
    "sense_radius": [1],
    "max_steps": [200],

    # --- algorithms ---
    "path_algo_name": ["AStar"],      # fixed pathfinding to isolate tour effects
    "sharing_algo_name": ["SSA"],     # or a CA_* variant that uses tours
    "tour_algo_name": [
        "NearestNeighbor",
        "CheapestInsertion",
        "ExactBruteForce",
        "IsingFull",
        "IsingLimited",
    ],

    # --- randomness ---
    "seed": [i for i in range(10)],
}
```

Notes:

- `PARAM_GRID` runs **all combinations** of the lists.  
  Fixing `path_algo_name` and `sharing_algo_name` and expanding only `tour_algo_name` produces a tour sweep on matched worlds and target allocations.
- For each run, the batch driver records:
  - `tours.algorithm` – algorithm name
  - `tours.call_count`, `tours.total_runtime`, `tours.avg_runtime`
  - `targets.collected`, `simulation.total_distance`, `simulation.makespan_tick`
  - For Ising-based methods, `tours.hardware.*` with simulated hardware time and energy.

The script `plot_utils.py` can then be invoked with `data_set = "tours"` so that boxplots of `tours.avg_runtime` and `simulation.total_distance` are grouped by `tours.algorithm`.

Example programmatic call:

```python
from plot_utils import plot_boxplots_from_csv

plot_boxplots_from_csv(
    csv_path="outputs_batch/batch_results.csv",
    group_by=["tours.algorithm"],
    metrics=["tours.avg_runtime", "simulation.total_distance"],
    output_dir="outputs_batch/plots",
    show=False,
    x_axis_label="Tour algorithm",
)
```

---

## Selecting a tour algorithm in `main.py`

In `main.py`, the simulator is constructed from the configuration; the tour algorithm name is taken from `cfg.tour_algo_name`:

```python
cfg = Config()

sim = Simulator(
    cfg=cfg,
    world=world,
    path_algo_name=cfg.path_algo_name,
    sharing_algo_name=cfg.sharing_algo_name,
    tour_algo_name=cfg.tour_algo_name,
    log_events=cfg.log_events,
)
```

Any `tour_algo_name` value of `None` disables tours (only sharing algorithms that do not require tours should be used in this case). Otherwise, the name must match the `.name` field of one of the algorithms in this folder.

---

## Adding a new tour algorithm

To add an additional TSP / tour solver:

1. Create a new file in this folder, for example `my_tour.py`.
2. Implement a class that follows the interface:

```python
from time import perf_counter
from env import WorldState, Pos
from .base import TourAlgorithm
from pathfinding.base import PathfindingAlgorithm

class MyTour(TourAlgorithm):
    name = "MyTour"

    def __init__(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.path_algo: PathfindingAlgorithm | None = None

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0

    def set_path_algo(self, algo: PathfindingAlgorithm) -> None:
        # optional: Simulator can inject a pathfinding algorithm for distance queries
        self.path_algo = algo

    def solve(
        self,
        world: WorldState,
        start: Pos,
        targets: list[Pos],
    ) -> tuple[list[Pos], float]:
        t0 = perf_counter()

        # Tour construction logic
        order: list[Pos] = list(targets)
        cost: float = 0.0

        dt = perf_counter() - t0
        self.total_runtime += dt
        self.call_count += 1
        return order, cost
```

3. At the bottom of the file, expose the algorithm:

```python
ALGORITHM = MyTour()
```

4. `tours/__init__.py` auto-discovers and registers it under `MyTour.name`.
5. The new solver then becomes available via `Config.tour_algo_name = "MyTour"` in `config.py` and may be added to `PARAM_GRID["tour_algo_name"]` in `batch_config.py` for batch experiments.

---

## Notes

- Tour solvers use **grid distances** computed via the selected pathfinding algorithm (for example, BFS or A*), so they automatically respect discovered obstacles.
- Ising-based algorithms are computationally heavier and are most appropriate for small target sets or for comparison/baseline experiments against classical heuristics.
