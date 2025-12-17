# sim.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional

from config import Config
from env import WorldState, RobotState, Pos
from pathfinding import PATHFINDING_ALGOS
from target_sharing import TARGET_SHARING_ALGOS
from tours import TOUR_ALGOS



@dataclass
class Simulator:
    cfg: Config
    world: WorldState
    path_algo_name: str = "BFS"
    sharing_algo_name: str = "RoundRobin"
    tour_algo_name: str = "NearestNeighbor"

    # control terminal logging
    log_events: bool = False

    # total ticks (time steps) that actually ran
    ticks: int = 0

    # total distance traveled by all robots
    total_distance: int = 0

    # history[step][rid] = position; useful for GIFs
    history: List[List[Pos]] = field(default_factory=list)

    # per-robot metrics
    robot_distances: Dict[int, int] = field(default_factory=dict)
    targets_collected: int = 0
    last_collection_tick: Optional[int] = None  # 1-based tick when last target collected

    def __post_init__(self) -> None:
        # small helper
        self._log(f"[INIT] Simulator with PF={self.path_algo_name}, "
                  f"TS={self.sharing_algo_name}, TR={self.tour_algo_name}")
        
        # pathfinding algorithm
        self.path_algo = PATHFINDING_ALGOS[self.path_algo_name]

        # target-sharing algorithm
        self.sharing_algo = TARGET_SHARING_ALGOS[self.sharing_algo_name]

        # inject path algo into sharing algo if supported (e.g., PSA)
        if hasattr(self.sharing_algo, "set_path_algo"):
            self.sharing_algo.set_path_algo(self.path_algo)

        # tour algorithm (for SSA / CA / etc.), only if requested by sharing_algo
        self.tour_algo = None
        if hasattr(self.sharing_algo, "set_tour_algo"):
            if self.tour_algo_name not in TOUR_ALGOS:
                raise ValueError(f"Unknown tour algorithm: {self.tour_algo_name}")
            self.tour_algo = TOUR_ALGOS[self.tour_algo_name]

            # let tour algorithm use the same pathfinder, if it wants
            if hasattr(self.tour_algo, "set_path_algo"):
                self.tour_algo.set_path_algo(self.path_algo)

            # inject tour algo into sharing algo (SSA)
            self.sharing_algo.set_tour_algo(self.tour_algo)

        # init per-robot distance map and target stats
        self.robot_distances = {r.rid: 0 for r in self.world.robots}
        self.targets_collected = 0
        self.last_collection_tick = None

    # ---------- logging helper ---------- #

    def _log(self, msg: str) -> None:
        if self.log_events:
            print(msg)

    # ---------------- planning ---------------- #

    def _plan_all_paths(self) -> None:
        """
        Use the target-sharing algorithm to assign targets to robots,
        then build a path for each robot visiting its targets in order.
        """
        targets: Set[Pos] = set(self.world.targets)
        robots: List[RobotState] = self.world.robots

        self._log(
            f"[PLAN] Running target sharing ({self.sharing_algo.name}) "
            f"for {len(robots)} robots, {len(targets)} targets"
        )

        assignments: Dict[int, List[Pos]] = self.sharing_algo.assign(
            self.world, robots, targets
        )

        for r in robots:
            tasks = assignments.get(r.rid, [])
            self._log(f"  - Robot {r.rid} assigned {len(tasks)} targets: {tasks}")
            cur = r.pos
            path: List[Pos] = []
            for t in tasks:
                segment = self.path_algo.plan(self.world, cur, t)
                if not segment or len(segment) <= 1:
                    continue
                # drop first cell to avoid duplication
                path.extend(segment[1:])
                cur = t
            r.path = path
            self._log(f"    Path length for robot {r.rid}: {len(r.path)} steps")

    # ---------------- stepping ---------------- #

    def _log_positions(self) -> None:
        self.history.append([r.pos for r in self.world.robots])

    def step_once(self) -> bool:
        """
        One simulation step:
          - sense new obstacles
          - for each robot, try to move one step along its path
          - if the next cell is a true obstacle (known or hidden), do NOT move,
            mark hidden ones as discovered, and replan.
        Returns True if any robot moved or a replan happened, False if everyone is stuck.
        """
        tick_id = self.ticks + 1  # next tick index
        self._log(f"[TICK {tick_id}] remaining targets = {len(self.world.targets)}")
        
        moved = False
        collected_this_step = False

        # robots sense before moving
        self.world.sense_new_obstacles(self.cfg.sense_radius)

        for r in self.world.robots:
            if not r.path:
                continue

            next_pos = r.path[0]

            # If the next cell is a real obstacle in the world, don't step into it
            if self.world.is_blocked_true(next_pos):
                if next_pos in self.world.hidden_obstacles:
                    self.world.discovered_hidden.add(next_pos)
                # Replan for everyone with updated visible obstacles
                self._plan_all_paths()
                # treat as "something happened" so caller doesn't stop the sim
                return True

            # Otherwise, move one step (grid is unit-cost)
            r.pos = r.path.pop(0)
            self.total_distance += 1
            self.robot_distances[r.rid] = self.robot_distances.get(r.rid, 0) + 1
            moved = True

            # Check if we reached a target
            if r.pos in self.world.targets:
                self.world.targets.remove(r.pos)
                self.targets_collected += 1
                collected_this_step = True

        # time advanced by 1 tick
        self.ticks += 1

        # if any target was collected on this tick, record makespan tick
        if collected_this_step:
            self.last_collection_tick = self.ticks  # 1-based tick index

        return moved

    def run(self) -> None:
        """
        Run simulation until all targets are collected or we hit max_steps
        or robots can't move.
        """
        # initial planning
        self._plan_all_paths()
        self._log_positions()

        for _ in range(self.cfg.max_steps):
            if not self.world.targets:
                break

            moved = self.step_once()
            self._log_positions()

            if not moved:
                # all robots stuck (no movement and no replan)
                break
