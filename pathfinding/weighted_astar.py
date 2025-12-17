# pathfinding/weighted_astar.py
from __future__ import annotations

from time import perf_counter
from heapq import heappush, heappop
from typing import Dict, List, Set, Tuple, Optional

from env import WorldState, Pos
from .base import PathfindingAlgorithm


class WeightedAStarPlanner(PathfindingAlgorithm):
    """
    Weighted A* on a 4-connected grid.

    Uses Manhattan distance as heuristic. The evaluation function is:

        f(n) = g(n) + w * h(n)

    with w >= 1.0.  w = 1.0 reduces to standard A*, while larger w
    typically produce faster but slightly sub-optimal paths.

    This planner searches on the *visible* grid (known + discovered obstacles)
    via world.neighbors4_visible().
    """
    name = "WAStar"

    def __init__(self, weight: float = 1.5) -> None:
        # Algorithm parameter
        self.weight = weight

        # Optional timing statistics
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API required by PathfindingAlgorithm                        #
    # ------------------------------------------------------------------ #
    def plan(self, world: WorldState, start: Pos, goal: Pos) -> Optional[List[Pos]]:
        """
        Plan a path from start to goal on the current belief map.

        Returns:
            - list of positions [start, ..., goal] if a path exists
            - [] if no path exists (never returns None)
        """
        t0 = perf_counter()

        if start == goal:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [start]

        def heuristic(p: Pos) -> int:
            x, y = p
            gx, gy = goal
            return abs(x - gx) + abs(y - gy)

        open_heap: List[Tuple[float, int, Pos]] = []
        g_cost: Dict[Pos, int] = {start: 0}
        parent: Dict[Pos, Pos] = {}
        closed: Set[Pos] = set()

        # Push the start node: (f, g, pos)
        h0 = heuristic(start)
        heappush(open_heap, (self.weight * h0, 0, start))

        neighbors = world.neighbors4_visible

        while open_heap:
            f_cur, g_cur, cur = heappop(open_heap)

            if cur in closed:
                continue
            closed.add(cur)

            if cur == goal:
                # Reconstruct path
                path: List[Pos] = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                dt = perf_counter() - t0
                self._update_stats(dt)
                return path

            for np in neighbors(cur):
                new_g = g_cur + 1  # unit-cost grid

                # Standard graph-search check
                old_g = g_cost.get(np)
                if old_g is not None and new_g >= old_g:
                    continue

                g_cost[np] = new_g
                parent[np] = cur
                f_np = new_g + self.weight * heuristic(np)
                heappush(open_heap, (f_np, new_g, np))

        # No path found
        dt = perf_counter() - t0
        self._update_stats(dt)
        return []

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _update_stats(self, dt: float) -> None:
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1


ALGORITHM = WeightedAStarPlanner()
