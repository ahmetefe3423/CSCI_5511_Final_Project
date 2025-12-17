# Multi-Robot Gridworld: Auctions & Ising Baselines

This repo is a **course project framework** for experimenting with:

- Multi-robot pathfinding on a grid
- Target allocation via auction algorithms (PSA, SSA, CA, …)
- Tour construction (mini TSPs per robot)
- Optional Ising/QUBO-based solvers using `dwave-tabu`

Robots start near a spawn point in a 2D grid world, must visit targets, and navigate around:

- **Known obstacles** (black): known from the start  
- **Unknown obstacles** (yellow): only discovered locally during execution

When robots discover unknown obstacles, the system can **re-plan paths** and **re-run auctions**.

Each run produces:

- A run-specific output folder under `outputs/`
- World visualizations (PNG)
- A GIF animation of the run
- A JSON summary of metrics and timing
- A JSON dump of the config used

---

## Quick start

```bash
# (optional) create a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt

# Optional: only needed if you use Ising-based algorithms
pip install dwave-tabu

python main.py
```

Each run creates a timestamped folder under `outputs/`, for example:

```text
outputs/
└── run_.../
    ├── config.json
    ├── world_initial.png
    ├── world_final.png
    ├── summary.json
    └── animation.gif
```

(You can change the base directory by passing `base=` to `io_utils.make_run_dir`.)

---

## Choosing algorithms

At the top of `main.py` you select algorithms by **name**:

```python
# Pathfinding:
#   "BFS"      - breadth-first search baseline
#   "AStar"    - A* with Manhattan heuristic (optimal)
#   "WAStar"   - Weighted A* (bounded-suboptimal, typically faster)
#   "GBFS"     - Greedy best-first search (fast, not optimal)
#   "IDAStar"  - Iterative Deepening A* (low memory, optimal)
#   "SMAStar"  - Simplified memory-bounded A*
path_algo_name = "AStar"

# Target sharing (auction):
#   "RoundRobin"      - simple baseline
#   "PSA"             - Parallel Single-Item Auction
#   "SSA"             - Sequential Single-Item Auction (uses tours)
#   "CA_Optimal"      - Exact combinatorial auction (small instances)
#   "CA_Greedy"       - Greedy combinatorial auction
#   "CA_IsingFull"    - CA winner determination via full-precision Ising/QUBO
#   "CA_IsingLimited" - CA winner determination via integer QUBO [-7,7]
sharing_algo_name = "PSA"

# Tour (TSP-style) solver, used by SSA / CA_*:
#   None                - no tour model (single-target distances only)
#   "NearestNeighbor"   - greedy TSP heuristic
#   "CheapestInsertion" - insertion heuristic TSP
#   "ExactBruteForce"   - exact tour (small target sets)
#   "IsingFull"         - Ising-based tour via TabuSampler
#   "IsingLimited"      - hardware-limited Ising tour
tour_algo_name = "CheapestInsertion"
```

World / robot / target parameters (grid size, number of robots, number of targets, etc.) live in `config.py` and are read by `main.py`.

For more details on the path planners themselves, see `pathfinding/README.md`.

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

## Batch experiments (`batch_run.py`)

For running many configurations and collecting results into a single CSV, use:

```bash
python batch_run.py
```

`batch_run.py`:

- Builds a parameter grid (world size, number of robots/targets, algorithms, and a free-text `purpose` field).
- For each combination:
  - builds `Config + WorldState + Simulator`
  - runs one simulation (same core logic as in `main.py`, but without per-run PNG/GIF writes)
  - builds the same summary dict as `main.py`
- Uses multiprocessing to parallelize experiments.
- Writes/appends rows to a single CSV file under `outputs_batch/`:

```text
outputs_batch/
└── batch_results.csv
```

If `outputs_batch/batch_results.csv` already exists, new experiments are appended (re-using the existing header).

You can customize the parameter grid by editing the `PARAM_GRID` dictionary near the top of `batch_run.py`.

---

## High-level structure

```text
main.py         # Entry point: builds config, runs one simulation, saves outputs
config.py       # Config dataclass (grid size, robot count, etc.)
env.py          # World & robot state, grid generation, obstacles, visibility
sim.py          # Simulator loop, auctions, replanning, metrics, history
viz.py          # Static world visualization (PNGs)
animate.py      # GIF animation from simulation history
io_utils.py     # Run directory helper, config/summary writers

pathfinding/    # Path planners (BFS, A*, WAStar, GBFS, IDAStar, SMAStar, ...)
target_sharing/ # Auction / target allocation (RoundRobin, PSA, SSA, CA_*, Ising-based CA_*)
tours/          # Tour/TSP solvers (NearestNeighbor, CheapestInsertion, Ising, ...)

outputs/        # Per-run outputs (config.json, world_initial.png, world_final.png, summary.json, animation.gif)
outputs_batch/  # Batch experiment CSV (batch_results.csv)
```

Each of the algorithm folders is auto-discovered: any file that exposes  
`ALGORITHM = YourClass()` with a unique `.name` will be picked up and can be selected by that name in `main.py`.

---

## Ising / QUBO-based components

Some algorithms rely on the `dwave-tabu` package:

- `target_sharing/ca_ising_full.py`
- `target_sharing/ca_ising_limited.py`
- `tours/ising_full.py`
- `tours/ising_limited.py`

If `dwave-tabu` (module `tabu`) is not installed:

- The Ising-based algorithms will raise a clear `RuntimeError`, and
- You can simply switch to classical baselines such as `"CA_Greedy"` or `"CheapestInsertion"`.

To enable them:

```bash
pip install dwave-tabu
```

Then select the corresponding algorithm names in `main.py` as shown above.

---

## Extending the framework

You can extend the framework in several ways:

- **New pathfinding algorithms**: add a new file in `pathfinding/` that implements `PathfindingAlgorithm` and exposes `ALGORITHM`.
- **New auction / sharing strategies**: add a new file in `target_sharing/` that implements `TargetSharingAlgorithm` and exposes `ALGORITHM`.
- **New tour solvers**: add a new file in `tours/` that implements `TourAlgorithm` and exposes `ALGORITHM`.

As long as you follow the existing interfaces and naming conventions, your new algorithms will automatically appear as options via their `.name` in `main.py`.
