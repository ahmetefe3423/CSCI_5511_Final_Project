# Multi-Robot Gridworld: Auctions & Ising Baselines

This repository is a **course project framework** for experimenting with:

- Multi-robot pathfinding on a grid
- Target allocation via auction algorithms (PSA, SSA, CA, …)
- Tour construction (mini TSPs per robot)
- Optional Ising/QUBO-based solvers using `dwave-tabu`

Robots start near a spawn point in a 2D grid world, must visit targets, and navigate around:

- **Known obstacles** (black): known from the start  
- **Unknown obstacles** (yellow): discovered locally during execution

When robots discover unknown obstacles, the system can **re-plan paths** and **re-run auctions**.

Each run produces a dedicated output folder with:

- World visualizations (PNG)
- A GIF animation of the run
- A JSON summary of metrics and timing
- A JSON dump of the configuration used

---

## Single-run workflow: `main.py` + `config.py`

Single simulations are driven by:

- `config.py` – central place for all world, simulation, and algorithm choices
- `main.py` – entry point that reads `Config`, runs one simulation, and saves outputs

### Configuration (`config.py`)

`config.py` defines a `Config` dataclass that holds:

- Grid size, number of robots, number of targets  
- Known / unknown obstacle densities  
- Sensing radius and random seed  
- Maximum number of simulation steps  
- Algorithm names for:
  - pathfinding (`path_algo_name`)
  - target sharing (`sharing_algo_name`)
  - tour / TSP (`tour_algo_name`)
- Optional toggles such as `log_events`

Example (simplified):

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    rows: int = 20
    cols: int = 20
    n_robots: int = 2
    n_targets: int = 10

    known_obstacle_density: float = 0.25
    unknown_obstacle_density: float = 0.0

    sense_radius: int = 1
    seed: int = 1

    min_connected_ratio: float = 0.5
    max_obstacle_tries: int = 50

    max_steps: int = 200

    # Algorithm names (must match .name fields)
    path_algo_name: str = "AStar"
    sharing_algo_name: str = "PSA"
    tour_algo_name: Optional[str] = "CheapestInsertion"

    # Optional logging flag for main.py runs
    log_events: bool = True
```

### Entry point (`main.py`)

`main.py` obtains a configuration instance and wires up the simulation:

```python
from config import Config
from env import WorldState
from sim import Simulator
from io_utils import make_run_dir, save_config, save_summary
from viz import draw_world
from animate import animate_run

def main() -> None:
    cfg = Config()  # all choices taken from config.py

    path_algo_name = cfg.path_algo_name
    sharing_algo_name = cfg.sharing_algo_name
    tour_algo_name = cfg.tour_algo_name
    log_events = cfg.log_events

    run_dir = make_run_dir(
        cfg,
        base="outputs",
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
    )
    save_config(cfg, run_dir)

    world = WorldState.from_config(cfg)
    initial_target_cells = set(world.targets)

    draw_world(world, cfg, out_path=run_dir / "world_initial.png")

    sim = Simulator(
        cfg=cfg,
        world=world,
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
        log_events=log_events,
    )
    sim.path_algo.reset_stats()
    sim.sharing_algo.reset_stats()
    if getattr(sim, "tour_algo", None) is not None:
        sim.tour_algo.reset_stats()

    sim.run()
    draw_world(world, cfg, out_path=run_dir / "world_final.png")
    # summary.json and animation.gif are generated here
```

All world- and algorithm-level decisions are therefore made via `config.py`; `main.py` only reads them and runs the simulation.

### Quick start

```bash
# (optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# Optional: only needed for Ising-based algorithms
pip install dwave-tabu

python main.py
```

Each run creates a timestamped folder under `outputs/`, for example:

```text
outputs/
└── run_R20x20_Rob2_Tgt10_seed1_20251216-213012_ab12cd34/
    ├── config.json
    ├── world_initial.png
    ├── world_final.png
    ├── summary.json
    └── animation.gif
```

(The base directory can be changed by passing `base=` to `io_utils.make_run_dir`.)

---

## Algorithm choices

Algorithm names in `config.py` must match the `.name` fields exposed by the modules in `pathfinding/`, `target_sharing/`, and `tours/`.

### Pathfinding (`pathfinding/`)

Typical options:

- `"BFS"`      – Breadth-first search baseline (unweighted shortest paths)
- `"AStar"`    – A* with Manhattan heuristic (optimal)
- `"WAStar"`   – Weighted A* (bounded-suboptimal, typically fewer expansions)
- `"GBFS"`     – Greedy best-first search (fast, not optimal)
- `"IDAStar"`  – Iterative Deepening A* (low memory, optimal)
- `"SMAStar"`  – Simplified memory-bounded A*

### Target sharing / auctions (`target_sharing/`)

Typical options:

- `"RoundRobin"`      – simple baseline
- `"PSA"`             – Parallel Single-Item Auction
- `"SSA"`             – Sequential Single-Item Auction (uses tours)
- `"CA_Optimal"`      – Exact combinatorial auction (small instances)
- `"CA_Greedy"`       – Greedy combinatorial auction
- `"CA_IsingFull"`    – CA winner determination via full-precision Ising/QUBO
- `"CA_IsingLimited"` – CA winner determination via integer QUBO [-7, 7]

### Tour / TSP solvers (`tours/`)

Used by SSA and CA-based sharing algorithms:

- `None`                – no tour model (single-target distances only)
- `"NearestNeighbor"`   – greedy TSP heuristic
- `"CheapestInsertion"` – insertion heuristic TSP
- `"ExactBruteForce"`   – exact tour (small target sets)
- `"IsingFull"`         – Ising-based tour via TabuSampler
- `"IsingLimited"`      – hardware-limited Ising tour

Detailed descriptions of individual algorithms are documented in the `README.md` files inside `pathfinding/`, `target_sharing/`, and `tours/`.

---

## Outputs

Each run from `main.py` writes into its run folder:

- **`config.json`** – the full configuration used for that run (written by `io_utils.save_config`)
- **`world_initial.png`** – initial grid with:
  - spawn point  
  - robot starting positions (scattered around spawn)  
  - targets  
  - known obstacles (e.g. black)  
  - unknown obstacles (e.g. yellow)  
- **`world_final.png`** – world snapshot after the simulation has finished
- **`animation.gif`** – step-by-step animation of robots moving, discovering obstacles, and collecting targets  
- **`summary.json`** – metrics such as:
  - total distance traveled  
  - makespan (tick of last target collection)  
  - per-robot distance & imbalance  
  - pathfinding / sharing / tour algorithm runtimes  
  - for Ising-limited variants: simulated hardware anneal time and energy  

---

## Batch experiments (`batch_run.py` + `batch_config.py`)

For running many configurations and collecting results into a single CSV, batch experiments are driven by:

- `batch_config.py` – CPU count and parameter grid (`CPU_COUNT`, `PARAM_GRID`)
- `batch_run.py` – driver script that reads `batch_config.py`, runs simulations, and writes `outputs_batch/batch_results.csv`

### Batch configuration (`batch_config.py`)

`batch_config.py` centralizes all batch-related settings.

```python
from typing import Dict, List, Any

# CPU usage for batch_run.py
# If CPU_COUNT is None, batch_run.py will use mp.cpu_count().
CPU_COUNT: int | None = None

PARAM_GRID: Dict[str, List[Any]] = {
    # --- meta ---
    "purpose": ["sharing_algorithm_comparison"],

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
    "sharing_algo_name": ["CA_Optimal"],
    "tour_algo_name": ["ExactBruteForce"],

    # --- randomness ---
    "seed": [i for i in range(10)],
}
```

Notes:

- `PARAM_GRID` runs the Cartesian product of all lists.  
  For example:
  - `"rows": [20, 30]`  
  - `"cols": [20, 30]`  
  generates 4 world shapes: (20×20), (20×30), (30×20), (30×30).
- The total experiment count is `∏ len(values)` across all keys; small additions can lead to a large number of runs.
- `"purpose"` is a free-text label written into every CSV row for later filtering or grouping.
- `CPU_COUNT` controls the number of worker processes. If set to `None`, `multiprocessing.cpu_count()` is used.

### Batch driver (`batch_run.py`)

`batch_run.py` uses `PARAM_GRID` to generate combinations, builds `Config` objects, and runs simulations without I/O:

```bash
python batch_run.py
```

Behaviour of `batch_run.py`:

- Builds all parameter combinations from `PARAM_GRID`.
- For each combination:
  - creates `Config`, `WorldState`, and `Simulator`
  - runs one simulation with `log_events=False`
  - constructs the same nested `summary` dict as in `main.py`
- Flattens `{params + summary}` into a single row per run.
- Writes/appends rows to a CSV file under `outputs_batch/`:

```text
outputs_batch/
└── batch_results.csv
```

If `outputs_batch/batch_results.csv` already exists, new experiments are appended using the existing header; the first run in a new file is executed synchronously in order to infer the CSV columns.

---

## Plotting batch results (`plot_utils.py`)

Once `outputs_batch/batch_results.csv` exists, algorithm comparison plots can be created using `plot_utils.py`.

### Command-line usage

```bash
python plot_utils.py
```

By default, `plot_utils.py`:

- Uses `data_set = "pathfinding"` (constant at the bottom of the file).
- Reads `outputs_batch/batch_results.csv`.
- Groups runs by `pathfinding.algorithm`.
- Generates boxplots for:
  - `pathfinding.avg_runtime`
  - `simulation.total_distance`
- Saves PDF plots under:

```text
outputs_batch/plots/
```

Changing `data_set` to `"tours"` or `"target_sharing"` switches the focus to tour or target-sharing algorithms; the script then groups by `tours.algorithm` or `target_sharing.algorithm` and adjusts the plotted metrics accordingly.

### Programmatic usage

`plot_utils.py` also exposes a function for use in notebooks or other scripts:

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

This makes it straightforward to script multiple plots, e.g., separate comparisons by `purpose`, grid size, or obstacle density.

---

## High-level structure

```text
main.py         # Entry point: builds Config, runs one simulation, saves outputs
config.py       # Config dataclass (world size, robot count, algorithms, logging)
env.py          # World & robot state, grid generation, obstacles, visibility
sim.py          # Simulator loop, auctions, replanning, metrics, history
viz.py          # Static world visualization (PNGs)
animate.py      # GIF animation from simulation history
io_utils.py     # Run directory helper, config/summary writers

batch_config.py # Batch CPU + PARAM_GRID configuration
batch_run.py    # Batch driver: runs many experiments, writes CSV
plot_utils.py   # Helper for plotting batch_results.csv (boxplots)

pathfinding/    # Path planners (BFS, A*, WAStar, GBFS, IDAStar, SMAStar, ...)
target_sharing/ # Auction / target allocation (RoundRobin, PSA, SSA, CA_*, Ising-based CA_*)
tours/          # Tour/TSP solvers (NearestNeighbor, CheapestInsertion, Ising, ...)

outputs/        # Per-run outputs (config.json, world_initial.png, world_final.png, summary.json, animation.gif)
outputs_batch/  # Batch experiment CSV (batch_results.csv) and plots/
```

Each algorithm folder is auto-discovered: any file that exposes  
`ALGORITHM = YourClass()` with a unique `.name` is picked up and can be selected by that name in `config.py` (`path_algo_name`, `sharing_algo_name`, `tour_algo_name`).

---

## Ising / QUBO-based components

Some algorithms rely on the `dwave-tabu` package:

- `target_sharing/ca_ising_full.py`
- `target_sharing/ca_ising_limited.py`
- `tours/ising_full.py`
- `tours/ising_limited.py`

If `dwave-tabu` (module `tabu`) is not installed:

- The Ising-based algorithms raise a clear `RuntimeError`, and
- Classical baselines such as `"CA_Greedy"` or `"CheapestInsertion"` remain available.

To enable the Ising-based components:

```bash
pip install dwave-tabu
```

The corresponding algorithm names can then be selected in `config.py`.

---

## Extensibility

The framework is designed to be extensible:

- **New pathfinding algorithms**  
  Add a new file in `pathfinding/` that implements the `PathfindingAlgorithm` protocol and exposes `ALGORITHM = Instance`.  
  The `.name` field of the class becomes the value used in `Config.path_algo_name`.

- **New auction / target-sharing strategies**  
  Add a new file in `target_sharing/` implementing `TargetSharingAlgorithm` and exposing `ALGORITHM`.  
  The simulator then treats it as another option for `Config.sharing_algo_name`.

- **New tour solvers**  
  Add a new file in `tours/` implementing `TourAlgorithm` and exposing `ALGORITHM`.  
  The `.name` becomes an option for `Config.tour_algo_name`.

As long as the existing interfaces and naming conventions are respected, new algorithms integrate automatically with `main.py`, `batch_run.py`, and the plotting pipeline.
