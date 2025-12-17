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
- Better tour quality than NearestNeighbor in many cases, still efficient for small target sets

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

Hardware metrics are stored in `summary.json` under `"tours"["hardware"]`.

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
- If no valid tour exists (some targets unreachable), it should return:
  - `order = []`
  - `cost = float('inf')`
- Runtime measurement:
  - `total_runtime` – total wall-clock seconds spent in `solve`
  - `call_count` – number of times `solve` was called

The simulator uses these statistics in `summary.json`.

---

## Selecting a tour algorithm

In `main.py`, choose the tour algorithm by name, for algorithms that use tours (SSA, CA_*): 

```python
tour_algo_name = "CheapestInsertion"  # or "NearestNeighbor", "ExactBruteForce", "IsingFull", "IsingLimited", ...
```

Set `tour_algo_name = None` to run sharing algorithms that don’t need tours (e.g. PSA, RoundRobin).

---

## Adding a new tour algorithm

To add your own TSP / tour solver:

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

        # TODO: your tour logic here
        order: list[Pos] = list(targets)
        cost: float = 0.0

        dt = perf_counter() - t0
        self.total_runtime += dt
        self.call_count += 1
        return order, cost
```

3. At the bottom of the file, expose your algorithm:

```python
ALGORITHM = MyTour()
```

4. `tours/__init__.py` will auto-discover and register it by `MyTour.name`.
5. You can then select it in `main.py`:

```python
tour_algo_name = "MyTour"
```

---

## Notes

- Tour solvers use **grid distances** computed via the selected pathfinding algorithm (BFS/A*), so they automatically respect discovered obstacles.
- Ising-based algorithms are more computationally heavy and are best used on small target sets, or mainly for comparison/baseline experiments.
