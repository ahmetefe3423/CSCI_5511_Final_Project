from pathlib import Path

from config import Config
from env import WorldState
from viz import draw_world
from io_utils import make_run_dir, save_config, save_summary
from sim import Simulator
from animate import animate_run


def main():
    """Run a single simulation and write results to the outputs/ folder.

    High-level steps:
      1. Build a Config object describing the grid, robots, and targets.
      2. Choose algorithms by name (pathfinding, target sharing, tour).
      3. Create a unique run directory under ``outputs/``.
      4. Instantiate the world and simulator.
      5. Run the simulation loop until all targets are collected or max_steps is reached.
      6. Save summary metrics, static PNGs, and an animation GIF into the run directory.
    """

    # -----------------------------------------------------------------------
    # World / experiment configuration
    # -----------------------------------------------------------------------
    cfg = Config(
        rows=20,                 # number of grid rows (y-dimension)
        cols=20,                 # number of grid columns (x-dimension)
        n_robots=3,              # how many robots to spawn near the spawn point
        n_targets=10,             # how many targets must be visited in this run
        known_obstacle_density=0.25,  # fraction of cells that start as known obstacles
        unknown_obstacle_density=0.1, # fraction of cells that are hidden obstacles (discovered when sensed)
        sense_radius=1,          # Manhattan radius of each robot's local sensor
        seed=1,                  # RNG seed for reproducible world generation
        max_steps=200,           # hard cap on simulation ticks; run stops even if targets remain
    )

    # -----------------------------------------------------------------------
    # Algorithm choices
    # -----------------------------------------------------------------------

    # Pathfinding algorithm (see pathfinding/README.md):
    #
    #   "BFS"      - Breadth-first search baseline (unweighted shortest paths).
    #   "AStar"    - A* with Manhattan heuristic (optimal).
    #   "WAStar"   - Weighted A* (bounded-suboptimal, typically fewer expansions than A*).
    #   "GBFS"     - Greedy best-first search (very fast, not optimal).
    #   "IDAStar"  - Iterative Deepening A* (low memory, optimal but may re-expand many nodes).
    #   "SMAStar"  - Simplified memory-bounded A* (limits OPEN size; trades completeness/optimality for memory).
    path_algo_name = "AStar"

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
    sharing_algo_name = "SSA"

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
    tour_algo_name = "CheapestInsertion"

    # From here on:
    #   - create an output directory for this run,
    #   - build the world and simulator,
    #   - execute the simulation loop,
    #   - collect metrics, and
    #   - write JSON + PNG + GIF outputs into the run directory.
    run_dir = make_run_dir(
        cfg,
        base="outputs",
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
    )

    save_config(cfg, run_dir)

    world = WorldState.from_config(cfg)

    # --- record targets both as positions and as count ---
    initial_target_cells = set(world.targets)       # for animation
    initial_target_count = len(world.targets)       # for summary

    # initial snapshot
    draw_world(world, cfg, out_path=run_dir / "world_initial.png")

    sim = Simulator(
        cfg=cfg,
        world=world,
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
    )

    sim.run()

    # --- summarize pathfinding metrics ---
    pa = sim.path_algo
    pf_entry = {
        "algorithm": pa.name,
        "call_count": pa.call_count,
        "total_runtime": pa.total_runtime,
        "avg_runtime": (pa.total_runtime / pa.call_count) if pa.call_count else 0.0,
    }

    # --- summarize target sharing metrics (plus hardware if available) ---
    sa = sim.sharing_algo
    ts_entry = {
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

    summary = {
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
            "terminated_early": sim.ticks < cfg.max_steps,
        },
        "pathfinding": pf_entry,
        "target_sharing": ts_entry,
    }

    # Tours metrics (plus hardware if available, e.g. IsingLimitedTour)
    ta = sim.tour_algo
    if ta is not None:
        tours_entry = {
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

        if hasattr(ta, "total_sim_time"):
            tours_entry["SNN"] = {
                "total_event_count": ta.total_event_count,
                "total_sim_time": ta.total_sim_time,
            }

        summary["tours"] = tours_entry

    save_summary(summary, run_dir)

    # --- build GIF animation ---
    gif_path = run_dir / "animation.gif"
    animate_run(world, cfg, sim.history, initial_target_cells, out_path=gif_path, fps=5)

    print(f"Run directory: {run_dir}")
    print(f"Animation saved to: {gif_path}")
    print(f"Targets collected: {summary['targets']['collected']} / {summary['targets']['initial']}")


if __name__ == "__main__":
    main()
