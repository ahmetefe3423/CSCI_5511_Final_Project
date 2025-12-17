# pathfinding/astar.py
from __future__ import annotations

from time import perf_counter
from heapq import heappush, heappop
from typing import Dict, List, Set, Tuple

from env import WorldState, Pos
from .base import PathfindingAlgorithm


class AStarPlanner(PathfindingAlgorithm):
    """
    A* path planner on a 4-connected grid.
    Uses Manhattan distance as heuristic, so paths are still optimal
    (same length as BFS) but usually found with fewer expansions.
    """

    name = "AStar"

    def __init__(self) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

    # ---- stats API ----

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def _update_stats(self, dt: float) -> None:
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1

    # ---- main planning API ----

    def plan(self, world: WorldState, start: Pos, goal: Pos) -> List[Pos]:
        """
        Returns a list of positions from start to goal (inclusive),
        or [] if no path exists under the current visible map.
        """
        t0 = perf_counter()

        if start == goal:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [start]

        rows, cols = world.rows, world.cols
        blocked: Set[Pos] = world.visible_obstacles()

        def in_bounds(p: Pos) -> bool:
            x, y = p
            return 0 <= x < cols and 0 <= y < rows

        def heuristic(p: Pos) -> int:
            x, y = p
            gx, gy = goal
            return abs(x - gx) + abs(y - gy)

        # open set: (f, g, (x, y))
        open_heap: List[Tuple[int, int, Pos]] = []
        heappush(open_heap, (heuristic(start), 0, start))

        g_cost: Dict[Pos, int] = {start: 0}
        parent: Dict[Pos, Pos] = {}
        closed: Set[Pos] = set()

        # 4-connected moves
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while open_heap:
            f_cur, g_cur, cur = heappop(open_heap)

            if cur in closed:
                continue
            if cur == goal:
                # reconstruct path
                path = [cur]
                while cur in parent:
                    cur = parent[cur]
                    path.append(cur)
                path.reverse()
                dt = perf_counter() - t0
                self._update_stats(dt)
                return path

            closed.add(cur)
            cx, cy = cur

            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                np: Pos = (nx, ny)

                if not in_bounds(np):
                    continue
                if np in blocked:
                    continue

                new_g = g_cur + 1  # unit-cost grid

                if np not in g_cost or new_g < g_cost[np]:
                    g_cost[np] = new_g
                    parent[np] = cur
                    heappush(open_heap, (new_g + heuristic(np), new_g, np))

        # no path
        dt = perf_counter() - t0
        self._update_stats(dt)
        return []


ALGORITHM = AStarPlanner()
