# pathfinding/ida_star.py
from __future__ import annotations

from math import inf
from time import perf_counter
from typing import Callable, List, Optional

from env import WorldState, Pos
from .base import PathfindingAlgorithm


class IDAStarPlanner(PathfindingAlgorithm):
    """
    IDA* (Iterative Deepening A*) on a 4-connected grid.

    - Memory usage is linear in the maximum depth of the search tree.
    - Like A*, it uses f(n) = g(n) + h(n) with Manhattan distance h.
    - It is optimal on unit-cost grids, but may re-expand nodes many times.
    - To avoid very long runtimes, we cap the total number of node expansions.

    This implementation searches on the *visible* grid only
    (known + discovered obstacles) via world.neighbors4_visible().

    NOTE: This implementation is recursive and therefore best suited to
    moderate grid sizes, where typical optimal path lengths are safely
    below Python's recursion limit.
    """
    name = "IDAStar"

    def __init__(self, max_expansions: int = 50_000) -> None:
        """
        Parameters
        ----------
        max_expansions : int
            Maximum number of node expansions allowed per call to plan().
            If this limit is exceeded, the search aborts and returns [].
        """
        # Optional timing statistics
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # Safety cap on search effort
        self.max_expansions: int = max_expansions
        self._expansions: int = 0

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #
    def plan(self, world: WorldState, start: Pos, goal: Pos) -> Optional[List[Pos]]:
        """
        Plan a path from start to goal on the current belief map.

        Returns:
            - list of positions [start, ..., goal] if a path exists
            - [] if no path exists or the expansion limit is exceeded
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

        neighbors: Callable[[Pos], List[Pos]] = world.neighbors4_visible

        # Initial bound is the heuristic value at the start
        bound: float = heuristic(start)
        path: List[Pos] = [start]

        # Reset expansion counter for this call
        self._expansions = 0

        while True:
            t = self._search(path, 0, bound, goal, heuristic, neighbors)
            if isinstance(t, list):
                # Found a path
                dt = perf_counter() - t0
                self._update_stats(dt)
                return t
            if t == inf:
                # No path exists within the current resource limit
                dt = perf_counter() - t0
                self._update_stats(dt)
                return []
            # Increase cost bound to the minimum f that exceeded it
            bound = t

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    def _search(
        self,
        path: List[Pos],
        g: int,
        bound: float,
        goal: Pos,
        heuristic,
        neighbors,
    ):
        """
        Depth-first search with f = g + h bound.

        Returns either:
          - a List[Pos] representing the path if goal is reached, or
          - the minimum f-cost that exceeded the current bound (float), or
          - inf if the expansion limit has been reached.
        """
        # Expansion cap: if we hit the limit, stop and signal "give up"
        if self._expansions >= self.max_expansions:
            return inf
        self._expansions += 1

        node = path[-1]
        f = g + heuristic(node)
        if f > bound:
            return f
        if node == goal:
            # Return a copy of the current path
            return list(path)

        min_excess = inf
        for succ in neighbors(node):
            if succ in path:
                # Avoid simple cycles; IDA* is tree-search based.
                continue
            path.append(succ)
            t = self._search(path, g + 1, bound, goal, heuristic, neighbors)
            if isinstance(t, list):
                return t
            if t < min_excess:
                min_excess = t
            path.pop()

        return min_excess

    def _update_stats(self, dt: float) -> None:
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1


ALGORITHM = IDAStarPlanner()
