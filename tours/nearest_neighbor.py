# tours/nearest_neighbor.py
from __future__ import annotations

from time import perf_counter
from math import inf
from typing import List, Tuple, Optional

from env import WorldState, Pos
from .base import TourAlgorithm
from pathfinding.base import PathfindingAlgorithm


class NearestNeighborTour(TourAlgorithm):
    """
    Simple TSP heuristic:

      - Start at 'start'.
      - Repeatedly go to the nearest unvisited target (by grid path length),
        using the injected pathfinding algorithm (e.g., BFS, A*).
      - Sum up the step counts.

    Returns:
      (visit_order, cost)

    If any target becomes unreachable under the current visible map,
    the tour cost is set to +inf.
    """

    name = "NearestNeighbor"

    def __init__(self) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # will be injected by the simulator (same path algo it uses)
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
        Optional hook: simulator can inject BFS, A*, etc.
        """
        self.path_algo = algo

    # ---- main solver ----

    def solve(
        self,
        world: WorldState,
        start: Pos,
        targets: List[Pos],
    ) -> Tuple[List[Pos], float]:
        t0 = perf_counter()

        if self.path_algo is None:
            raise RuntimeError(
                "NearestNeighborTour.path_algo is not set. "
                "Simulator must inject a pathfinding algorithm via set_path_algo()."
            )

        targets = list(targets)
        if not targets:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], 0.0

        remaining = set(targets)
        current = start
        order: List[Pos] = []
        total_cost: float = 0.0

        while remaining:
            best_t: Optional[Pos] = None
            best_cost: float = inf
            best_path_len: int = 0

            # find nearest reachable target
            for t in remaining:
                path = self.path_algo.plan(world, current, t)
                if not path:
                    continue
                steps = len(path) - 1
                if steps < best_cost:
                    best_cost = steps
                    best_t = t
                    best_path_len = steps

            if best_t is None:
                # at least one remaining target is unreachable
                total_cost = inf
                break

            order.append(best_t)
            total_cost += best_path_len
            current = best_t
            remaining.remove(best_t)

        dt = perf_counter() - t0
        self._update_stats(dt)
        return order, total_cost


ALGORITHM = NearestNeighborTour()
