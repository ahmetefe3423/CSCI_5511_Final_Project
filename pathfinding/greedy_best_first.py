# pathfinding/greedy_best_first.py
from __future__ import annotations

from heapq import heappush, heappop
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

from env import WorldState, Pos
from .base import PathfindingAlgorithm


class GreedyBestFirstPlanner(PathfindingAlgorithm):
    """
    Greedy Best-First Search (GBFS) on a 4-connected grid.

    Uses only the heuristic value h(n) (Manhattan distance to the goal)
    to guide search:

        f(n) = h(n)

    This often expands far fewer nodes than BFS, but is not guaranteed
    to find an optimal path. It is still complete on finite grids when
    implemented with a closed set, as we do here.

    The planner searches on the *visible* grid only
    (known + discovered obstacles) via world.neighbors4_visible().
    """
    name = "GBFS"

    def __init__(self) -> None:
        # Optional timing statistics
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
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

        open_heap: List[Tuple[int, int, Pos]] = []
        parent: Dict[Pos, Pos] = {}
        visited: Set[Pos] = set()

        h0 = heuristic(start)
        heappush(open_heap, (h0, 0, start))
        visited.add(start)

        neighbors = world.neighbors4_visible

        while open_heap:
            h_cur, depth, cur = heappop(open_heap)

            if cur == goal:
                path: List[Pos] = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                dt = perf_counter() - t0
                self._update_stats(dt)
                return path

            for np in neighbors(cur):
                if np in visited:
                    continue
                visited.add(np)
                parent[np] = cur
                heappush(open_heap, (heuristic(np), depth + 1, np))

        # No path
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


ALGORITHM = GreedyBestFirstPlanner()
