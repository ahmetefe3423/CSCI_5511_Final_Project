# Pathfinding Algorithms

This folder contains **single-robot path planners** on the grid world.  
Each algorithm takes a start cell and a goal cell and returns a shortest (or near-shortest) path that avoids known obstacles.

The simulator uses the selected pathfinding algorithm for:
- Robot motion planning on the current **belief** map
- Distance estimates used by tour / auction algorithms (PC(r, S))

The algorithm is selected by the `path_algo_name` field of the `Config` dataclass in `config.py` (single-run experiments) and by the `path_algo_name` entry in `PARAM_GRID` in `batch_config.py` (batch experiments).

---

## Available algorithms

### `BFS` (Breadth-First Search)
- File: `bfs.py`
- Optimal for unweighted 4-connected grids (shortest number of steps)
- Very simple baseline and useful for debugging and sanity checks

### `AStar` (A* Search)
- File: `astar.py`
- Uses Manhattan distance as an admissible heuristic
- Typically expands far fewer nodes than BFS on larger maps
- Still optimal on this grid (no weighted edges)

### `WAStar` (Weighted A*)
- File: `weighted_astar.py`
- Uses modified evaluation function: `f(n) = g(n) + w · h(n)` with `w ≥ 1`
- `w = 1` behaves like standard A*
- `w > 1` trades off optimality for speed (bounded-suboptimal paths, fewer expansions)

### `GBFS` (Greedy Best-First Search)
- File: `greedy_best_first.py`
- Uses only the heuristic value: `f(n) = h(n)` (Manhattan distance)
- Often very fast in practice (explores toward the goal aggressively)
- Not guaranteed to find the shortest path, but complete on finite grids

### `IDAStar` (Iterative Deepening A*)
- File: `ida_star.py`
- Uses iterative deepening on `f(n) = g(n) + h(n)` with Manhattan distance
- Optimal like A*, but with **much lower memory usage** (depth-first with a cost bound)
- May re-expand nodes many times; best for moderate path lengths

### `SMAStar` (Simplified Memory-Bounded A*)
- File: `sma_star.py`
- A memory-bounded best-first search:
  - Uses `f(n) = g(n) + h(n)` with Manhattan distance
  - Keeps the open list under a configurable size (`max_open`)
  - Discards the worst nodes when memory is full
- Not a fully textbook SMA*, but captures the practical behavior of an A*-like planner under tight memory constraints

For single-run simulations, the algorithm name is set in `config.py`, for example:

```python
# config.py
from dataclasses import dataclass

@dataclass
class Config:
    # ...
    path_algo_name: str = "AStar"   # one of: "BFS", "AStar", "WAStar", "GBFS", "IDAStar", "SMAStar"
```

---

## Common interface

All pathfinding algorithms conform to `PathfindingAlgorithm` defined in `base.py`:

```python
class PathfindingAlgorithm(Protocol):
    name: str
    total_runtime: float
    call_count: int

    def reset_stats(self) -> None: ...
    def plan(self, world: WorldState, start: Pos, goal: Pos) -> list[Pos]: ...
```

Behavior:

- `plan(...)` returns a **list of grid cells** from `start` to `goal` (inclusive).
- If `goal` is not reachable, it returns an **empty list**.
- Runtime measurement is handled inside each algorithm:
  - `total_runtime` – accumulated seconds spent in `plan`
  - `call_count` – how many times `plan` was called
  - (Many implementations also track `last_runtime`)
- The simulator uses these fields for performance statistics in `summary.json` and in the batch-results CSV.

---

## Pathfinding in batch experiments (`batch_run.py` + `batch_config.py`)

Systematic comparison of pathfinding algorithms is performed via `batch_run.py`, which sweeps over a parameter grid and writes all results to `outputs_batch/batch_results.csv`.

The relevant configuration lives in `batch_config.py`:

```python
# batch_config.py
from typing import Dict, List, Any

CPU_COUNT: int | None = None  # None -> use all available cores

PARAM_GRID: Dict[str, List[Any]] = {
    # --- meta ---
    "purpose": ["pathfinding_comparison"],

    # --- world parameters ---
    "rows": [20],
    "cols": [20],
    "n_robots": [1],
    "n_targets": [10],
    "known_obstacle_density": [0.2],
    "unknown_obstacle_density": [0.0, 0.1],
    "sense_radius": [1],
    "max_steps": [200],

    # --- algorithms ---
    "path_algo_name": ["BFS", "AStar", "WAStar", "GBFS", "IDAStar", "SMAStar"],
    "sharing_algo_name": ["CA_Optimal"],      # fixed to isolate pathfinding effects
    "tour_algo_name": ["ExactBruteForce"],    # fixed to exact tour costs

    # --- randomness ---
    "seed": [i for i in range(10)],
}
```

Notes:

- `PARAM_GRID` runs **all combinations** of the lists.  
  Keeping `sharing_algo_name` and `tour_algo_name` fixed and expanding only `path_algo_name` produces a clean pathfinding sweep on matched worlds.
- For each run, the batch driver records:
  - `pathfinding.algorithm` – algorithm name
  - `pathfinding.call_count`, `pathfinding.total_runtime`, `pathfinding.avg_runtime`
  - `simulation.total_distance`, `simulation.makespan_tick`, and other metrics

The script `plot_utils.py` can then be used to produce boxplots comparing pathfinding algorithms, by setting `data_set = "pathfinding"`.

Example programmatic call:

```python
from plot_utils import plot_boxplots_from_csv

plot_boxplots_from_csv(
    csv_path="outputs_batch/batch_results.csv",
    group_by=["pathfinding.algorithm"],
    metrics=["pathfinding.avg_runtime", "simulation.total_distance"],
    output_dir="outputs_batch/plots",
    show=False,
    x_axis_label="Pathfinding algorithm",
)
```

---

## Adding a new pathfinding algorithm

To add an additional planner:

1. Create a new file in this folder, for example `my_planner.py`.
2. Implement a class that follows the interface:

```python
from time import perf_counter
from env import WorldState, Pos
from .base import PathfindingAlgorithm

class MyPlanner(PathfindingAlgorithm):
    name = "MyPlanner"

    def __init__(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def plan(self, world: WorldState, start: Pos, goal: Pos) -> list[Pos]:
        t0 = perf_counter()

        # Planning logic
        path: list[Pos] = []

        dt = perf_counter() - t0
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1
        return path
```

3. At the bottom of the file, expose it as `ALGORITHM`:

```python
ALGORITHM = MyPlanner()
```

4. The module `pathfinding/__init__.py` will **auto-discover** this file and register it under `MyPlanner.name`.
5. The new planner can then be selected by setting

   - `Config.path_algo_name = "MyPlanner"` in `config.py` (single runs), and/or
   - adding `"MyPlanner"` to `PARAM_GRID["path_algo_name"]` in `batch_config.py` (batch experiments).

---

## Notes

- All planners operate on the **current belief map**:
  - robots only know discovered obstacles and any globally known static obstacles;
  - hidden obstacles only affect planning once they have been sensed.
- Planners must respect:
  - grid boundaries,
  - obstacles encoded in the world (via `visible_obstacles()` or `neighbors4_visible()`),
  - and the common return convention: a simple list of `(x, y)` positions, `[]` if no path exists.
- More advanced incremental algorithms (e.g., D* Lite, LPA*) can be added as long as they fit the same `plan(world, start, goal) -> list[Pos]` interface and expose timing statistics so they can be compared fairly in `summary.json` and in `outputs_batch/batch_results.csv`.
