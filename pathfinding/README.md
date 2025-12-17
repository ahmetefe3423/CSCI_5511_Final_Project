# Pathfinding Algorithms

This folder contains **single-robot path planners** on the grid world.  
Each algorithm takes a start cell and a goal cell and returns a shortest (or near-shortest) path that avoids known obstacles.

The simulator uses the selected pathfinding algorithm for:
- Robot motion planning on the current **belief** map
- Distance estimates used by tour / auction algorithms (PC(r, S))

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

> To choose the algorithm, set `path_algo_name` in `main.py` to one of:
> `"BFS"`, `"AStar"`, `"WAStar"`, `"GBFS"`, `"IDAStar"`, or `"SMAStar"`.

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
- The simulator uses these fields for performance statistics in `summary.json` and **batch results**.

---

## Using pathfinding algorithms in `batch_run.py`

To compare these algorithms systematically, use `batch_run.py`, which sweeps over a parameter grid and writes all results to `outputs_batch/batch_results.csv`.

In `batch_run.py` the pathfinding dimension is controlled by:

```python
PARAM_GRID: Dict[str, List[Any]] = {
    # ...

    # --- algorithms ---
    # "path_algo_name": ["BFS","AStar","WAStar","GBFS","IDAStar","SMAStar"],
    "path_algo_name": ["BFS", "AStar", "WAStar", "GBFS", "IDAStar", "SMAStar"],  # which pathfinding algorithms to compare

    # Fix sharing/tour algorithms if you want to isolate pathfinding effects:
    "sharing_algo_name": ["CA_Optimal"],
    "tour_algo_name": ["ExactBruteForce"],

    # ...
}
```

Notes:

- `PARAM_GRID` runs **all combinations** of the lists. If you keep `sharing_algo_name` and `tour_algo_name` at length 1 and only expand `path_algo_name`, you get a clean **pathfinding sweep** on the same worlds.
- Each run contributes a row where:
  - `pathfinding.algorithm` is the algorithm’s `name`
  - `pathfinding.total_runtime` / `pathfinding.avg_runtime` capture its planning cost
  - `simulation.total_distance` and `simulation.makespan_tick` capture the effect on overall performance

You can then visualize these comparisons with `plot_utils.py`, for example:

```bash
python plot_utils.py
```

with `data_set = "pathfinding"` to get boxplots of `pathfinding.avg_runtime` and `simulation.total_distance` grouped by `pathfinding.algorithm`.

---

## Adding a new pathfinding algorithm

To add your own planner:

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

        # TODO: your planning logic here
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

4. The `pathfinding/__init__.py` module will **auto-discover** this file and register it by name.
5. You can then select it in `main.py`:

```python
path_algo_name = "MyPlanner"
```

6. To include it in batch experiments, add its name to `PARAM_GRID["path_algo_name"]` in `batch_run.py` and re-run:

```python
"path_algo_name": ["BFS", "AStar", "MyPlanner"]
```

---

## Notes

- All planners operate on the **current belief map**:
  - robots only know discovered obstacles and any globally known static obstacles;
  - hidden obstacles only affect planning once discovered.
- Planners must respect:
  - grid boundaries,
  - obstacles encoded in `world` (via `visible_obstacles()` or `neighbors4_visible()`),
  - and the common return convention: a simple list of `(x, y)` positions, `[]` if no path exists.
- More advanced incremental algorithms (e.g., D* Lite, LPA*) can be added as long as they fit the same `plan(world, start, goal) -> list[Pos]` interface and expose timing stats so they can be compared fairly in `summary.json` and `batch_results.csv`.
