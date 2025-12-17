# Target Sharing (Auction) Algorithms

This folder contains **multi-robot target allocation / auction algorithms**.

Given:
- The current world state
- A list of robots (with positions and already-assigned targets)
- A set of remaining targets

a target-sharing algorithm decides **which robot gets which targets**.

The simulator calls the selected target-sharing algorithm:
- At the beginning of the run
- Whenever a replanning event is triggered (e.g., robots discover unknown obstacles that change paths or make assignments unreasonable)

---

## Available algorithms

### `RoundRobin`
- File: `round_robin.py`
- Very simple baseline:
  - Iterate through targets and assign them in a round-robin fashion over robots.
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
- Useful as a gold standard to compare heuristics.

---

### `CA_Greedy` – Greedy Combinatorial Auction
- File: `ca_greedy.py`
- Builds small **bundles** (e.g. 1–2 targets) for each robot.
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
- Decodes the sample, repairs conflicts, and outputs assignments.

Requires the `dwave-tabu` package:
```bash
pip install dwave-tabu
```

---

### `CA_IsingLimited` – Hardware-Limited Ising Combinatorial Auction
- File: `ca_ising_limited.py`
- Same idea as `CA_IsingFull`, but simulates hardware limitations:
  - QUBO coefficients are restricted to **integer** range `[-7, +7]`.
  - **Constraints** (target uniqueness) get ±7 (hard constraints).
  - **Costs** are mapped into `1..6` (never 7) using multiple mapping strategies.
  - For a QUBO with `N` bundle variables, simulates a device with:
    - spin limit: 49 spins
    - anneal time: 50 µs per hardware call
    - power: 24 mW
  - Estimates number of hardware calls and accumulates:
    - total anneal time
    - total energy consumption
- This information is exported in `summary.json` under `"target_sharing"["hardware"]`.

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

- `assign(...)` must return a **dictionary**: `robot_id -> list of targets` assigned to that robot.
- It should assign **each target to at most one robot** in a single call.
- It may choose to leave some targets unassigned (depending on design), but the current implementations aim to cover all reachable targets.
- Each algorithm tracks its own runtime:
  - `total_runtime` – total seconds spent across all calls
  - `call_count` – number of times `assign` was invoked

The simulator logs these stats to `summary.json`.

Many algorithms (SSA, CA_*) also depend on a `TourAlgorithm` from the `tours/` folder; the simulator injects the chosen tour algorithm via a setter like `set_tour_algo`.

---

## Selecting a target-sharing algorithm

In `main.py`, choose the sharing algorithm by name, e.g.:

```python
sharing_algo_name = "PSA"            # or "SSA", "CA_Greedy", "CA_IsingLimited", ...
```

The `target_sharing/__init__.py` module auto-discovers available algorithms at import time.

---

## Adding a new target-sharing algorithm

To add your own auction / allocation method:

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

        # TODO: your allocation logic here

        dt = perf_counter() - t0
        self.total_runtime += dt
        self.call_count += 1
        return assignments
```

3. At the bottom of the file, expose it as:

```python
ALGORITHM = MyAuction()
```

4. `target_sharing/__init__.py` will **automatically import** this file and register the algorithm under `MyAuction.name`.
5. You can then select it in `main.py`:

```python
sharing_algo_name = "MyAuction"
```

---

## Notes

- All algorithms operate on the **current belief world** (robots only know discovered + known obstacles).
- Replanning events (e.g. when an unknown obstacle blocks a path) trigger calls to `assign(...)` again with the remaining targets.
- If you introduce new heuristics (e.g. distance thresholds, local vs global auctions), keep the input/output interface identical so the simulator can use your algorithm without changes.
