from pathlib import Path

from config import Config
from env import WorldState
from viz import draw_world
from io_utils import make_run_dir, save_config, save_summary
from sim import Simulator
from animate import animate_run


def main() -> None:
    """
    Single-run entry point for the multi-robot gridworld simulator.

    Typical usage for a user:
      1. Open config.py and edit the Config defaults
         (world size, densities, number of robots/targets, algorithm names, ...).
      2. Run:
             python main.py
      3. Inspect the output folder under outputs/ (PNG, GIF, summary.json).
    """

    # ------------------------------------------------------------------
    # 1) Build configuration from config.py
    # ------------------------------------------------------------------
    # Normal usage: just instantiate Config() and rely on the defaults
    # declared in config.py. Users are expected to edit config.py instead
    # of this file.
    cfg = Config()

    # ------------------------------------------------------------------
    # 2) Algorithm choices (prefer to read from cfg)
    # ------------------------------------------------------------------
    # If config.py has algorithm fields (recommended), use them.
    # For backward compatibility, fall back to sensible defaults if they
    # don't exist yet.
    #
    # Valid names are documented in:
    #   - pathfinding README (BFS, AStar, WAStar, GBFS, IDAStar, SMAStar, ...)
    #   - target_sharing README (RoundRobin, PSA, SSA, CA_*, ...)
    #   - tours README (NearestNeighbor, CheapestInsertion, ExactBruteForce,
    #                  IsingFull, IsingLimited, ...)
    path_algo_name = getattr(cfg, "path_algo_name", "AStar")
    sharing_algo_name = getattr(cfg, "sharing_algo_name", "PSA")
    tour_algo_name = getattr(cfg, "tour_algo_name", "CheapestInsertion")

    # Enable textual logs when running via main.py.
    # If Config has log_events, use that; otherwise default to True here.
    log_events = getattr(cfg, "log_events", True)

    # ------------------------------------------------------------------
    # 3) Create output directory and save config
    # ------------------------------------------------------------------
    # make_run_dir builds something like:
    #   outputs/run_R20x20_Rob2_Tgt10_seed1_YYYYMMDD-HHMMSS-<uid>/
    run_dir = make_run_dir(
        cfg,
        base="outputs",
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
    )

    # Save config.json so every run is reproducible.
    save_config(cfg, run_dir)

    # ------------------------------------------------------------------
    # 4) Build world and simulator
    # ------------------------------------------------------------------
    # WorldState.from_config uses cfg.rows, cfg.cols, obstacle densities,
    # seed, etc. to generate a connected grid with robots, targets, and
    # known/unknown obstacles.
    world = WorldState.from_config(cfg)

    # Keep the initial target set for animation + summary.
    initial_target_cells = set(world.targets)
    initial_target_count = len(world.targets)

    # Draw the initial world BEFORE any robot moves.
    draw_world(world, cfg, out_path=run_dir / "world_initial.png")

    # Create the simulator object, wiring together:
    #   - the world state
    #   - configuration (max_steps, etc.)
    #   - the chosen pathfinding / sharing / tour algorithms by name
    #   - log_events=True so we see tick-by-tick and planning logs
    sim = Simulator(
        cfg=cfg,
        world=world,
        path_algo_name=path_algo_name,
        sharing_algo_name=sharing_algo_name,
        tour_algo_name=tour_algo_name,
        log_events=log_events,
    )

    # Reset timing statistics for this run so total_runtime/call_count
    # reflect only this simulation.
    sim.path_algo.reset_stats()
    sim.sharing_algo.reset_stats()
    if getattr(sim, "tour_algo", None) is not None:
        sim.tour_algo.reset_stats()

    # ------------------------------------------------------------------
    # 5) Run simulation until completion or max_steps
    # ------------------------------------------------------------------
    # The simulator moves robots, runs target-sharing, and replans paths
    # when necessary, until:
    #   - all targets are collected, or
    #   - cfg.max_steps is reached, or
    #   - robots become stuck.
    sim.run()

    # Draw the final world AFTER the simulation completes.
    draw_world(world, cfg, out_path=run_dir / "world_final.png")

    # ------------------------------------------------------------------
    # 6) Build summary dictionary with metrics + algorithm stats
    # ------------------------------------------------------------------
    pa = sim.path_algo
    sa = sim.sharing_algo
    ta = getattr(sim, "tour_algo", None)

    # Per-robot distance stats
    dist_by_robot = sim.robot_distances
    dist_values = list(dist_by_robot.values())
    total_dist_check = sum(dist_values) if dist_values else 0

    max_dist = max(dist_values) if dist_values else 0
    min_dist = min(dist_values) if dist_values else 0
    avg_dist = total_dist_check / len(dist_values) if dist_values else 0
    imbalance = max_dist - min_dist

    summary: dict = {
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
            "total_distance": sim.total_distance,
            # Sanity check: recompute distance from per-robot totals.
            "total_distance_check_from_robots": total_dist_check,
            "makespan_tick": sim.last_collection_tick,
            "all_targets_collected": (len(world.targets) == 0),
        },
        "per_robot_distance": {
            "by_robot": dist_by_robot,
            "max": max_dist,
            "min": min_dist,
            "avg": avg_dist,
            "imbalance": imbalance,
        },
    }

    # ---- Pathfinding metrics ----
    summary["pathfinding"] = {
        "algorithm": pa.name,
        "call_count": pa.call_count,
        "total_runtime": pa.total_runtime,
        "avg_runtime": (pa.total_runtime / pa.call_count) if pa.call_count else 0.0,
    }

    # ---- Target sharing metrics (plus hardware if available) ----
    ts_entry: dict = {
        "algorithm": sa.name,
        "call_count": sa.call_count,
        "total_runtime": sa.total_runtime,
        "avg_runtime": (sa.total_runtime / sa.call_count) if sa.call_count else 0.0,
    }

    # Optional hardware metrics used by Ising-based sharing algorithms.
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

    summary["target_sharing"] = ts_entry

    # ---- Tours metrics (plus hardware / SNN if available) ----
    if ta is not None:
        tours_entry: dict = {
            "algorithm": ta.name,
            "call_count": ta.call_count,
            "total_runtime": ta.total_runtime,
            "avg_runtime": (ta.total_runtime / ta.call_count) if ta.call_count else 0.0,
        }

        # Optional hardware metrics for Ising-based tour solvers.
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

        # Optional SNN-style simulation metrics (if present on tour object).
        if hasattr(ta, "total_sim_time"):
            tours_entry["SNN"] = {
                "total_event_count": ta.total_event_count,
                "total_sim_time": ta.total_sim_time,
            }

        summary["tours"] = tours_entry

    # Save the metrics to summary.json in the run directory.
    save_summary(summary, run_dir)

    # ------------------------------------------------------------------
    # 7) Build GIF animation of the entire run
    # ------------------------------------------------------------------
    gif_path = run_dir / "animation.gif"
    animate_run(world, cfg, sim.history, initial_target_cells, out_path=gif_path, fps=5)

    print(f"Run directory: {run_dir}")
    print(f"Animation saved to: {gif_path}")
    print(
        f"Targets collected: {summary['targets']['collected']} "
        f"/ {summary['targets']['initial']}"
    )


if __name__ == "__main__":
    main()
