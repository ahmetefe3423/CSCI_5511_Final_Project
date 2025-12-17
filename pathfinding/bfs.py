# pathfinding/bfs.py
from collections import deque
from time import perf_counter
from typing import List, Tuple, Optional

from env import WorldState
from .base import PathfindingAlgorithm

Pos = Tuple[int, int]


class BFSPlanner(PathfindingAlgorithm):
    name = "BFS"

    def __init__(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def plan(self, world: WorldState, start: Pos, goal: Pos) -> Optional[List[Pos]]:
        """
        BFS on the VISIBLE world (robots only know visible_obstacles).
        Returns a list of cells from start to goal (inclusive),
        or an empty list [] if unreachable.
        """
        t0 = perf_counter()

        if start == goal:
            path = [start]
            dt = perf_counter() - t0
            self._update_stats(dt)
            return path

        blocked = world.visible_obstacles()
        rows, cols = world.rows, world.cols

        def in_bounds(p: Pos) -> bool:
            x, y = p
            return 0 <= x < cols and 0 <= y < rows

        q = deque([start])
        came_from: dict[Pos, Optional[Pos]] = {start: None}
        path: Optional[List[Pos]] = None

        while q:
            x, y = q.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                np = (nx, ny)
                if not in_bounds(np) or np in came_from:
                    continue
                if np in blocked:
                    continue

                came_from[np] = (x, y)

                if np == goal:
                    # reconstruct
                    path = []
                    cur: Optional[Pos] = np
                    while cur is not None:
                        path.append(cur)
                        cur = came_from[cur]
                    path.reverse()
                    q.clear()
                    break
                q.append(np)

        dt = perf_counter() - t0
        self._update_stats(dt)
        # Normalize: always return a list; [] means "no path found"
        return path or []

    def _update_stats(self, dt: float) -> None:
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1


ALGORITHM = BFSPlanner()
