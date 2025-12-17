# target_sharing/psa.py
from __future__ import annotations

from time import perf_counter
from math import inf
from typing import Dict, List, Set, Tuple, Optional

from env import WorldState, RobotState, Pos
from .base import TargetSharingAlgorithm
from pathfinding.base import PathfindingAlgorithm


class PSAuction(TargetSharingAlgorithm):
    """
    Parallel Single-Item Auctions (Koenig et al., 2006):

    - Every robot r bids on each target t.
    - The bid is the smallest path cost needed to visit t
      from r's *current* location (we approximate this with
      shortest path length on the grid).
    - For each target t, the robot with the smallest bid wins t.

    This implementation:
      * uses whatever pathfinding algorithm is injected via
        set_path_algo (e.g., BFS, A*),
      * ignores synergies between targets (as in PSA),
      * runs in O(|R| * |T| * cost(pathfinding)) time.
    """

    name = "PSA"

    def __init__(self) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # pathfinding algorithm will be injected by the Simulator
        self.path_algo: Optional[PathfindingAlgorithm] = None

    # ---- timing API ----

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def _update_stats(self, dt: float) -> None:
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1

    # ---- dependency injection ----

    def set_path_algo(self, algo: PathfindingAlgorithm) -> None:
        """
        Simulator calls this so PSA uses the same pathfinding algorithm
        (BFS, A*, etc.) that is used for actual execution.
        """
        self.path_algo = algo

    # ---- main auction logic ----

    def assign(
        self,
        world: WorldState,
        robots: List[RobotState],
        targets: Set[Pos],
    ) -> Dict[int, List[Pos]]:
        t0 = perf_counter()

        if self.path_algo is None:
            raise RuntimeError(
                "PSA path_algo is not set. "
                "Simulator must inject it via set_path_algo()."
            )

        # Initialize assignments: robot_id -> list of targets
        assignments: Dict[int, List[Pos]] = {r.rid: [] for r in robots}

        if not robots or not targets:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return assignments

        # For each target, find the robot with the smallest path cost
        for t in targets:
            best_rid: Optional[int] = None
            best_cost: float = inf

            for r in robots:
                path = self.path_algo.plan(world, r.pos, t)
                if not path:
                    # this robot cannot reach t (under current knowledge)
                    continue

                cost = len(path) - 1  # steps as path cost

                if cost < best_cost:
                    best_cost = cost
                    best_rid = r.rid

            if best_rid is not None:
                assignments[best_rid].append(t)
            # if best_rid is None: target remains unassigned (no robot can reach it)

        dt = perf_counter() - t0
        self._update_stats(dt)
        return assignments


ALGORITHM = PSAuction()
