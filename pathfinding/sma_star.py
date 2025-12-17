# pathfinding/sma_star.py
from __future__ import annotations

from heapq import heappush, heappop, heapify
from math import inf
from time import perf_counter
from typing import Dict, List, Optional, Set, Tuple

from env import WorldState, Pos
from .base import PathfindingAlgorithm


class SMAStarPlanner(PathfindingAlgorithm):
    """
    Simplified memory-bounded A* (SMA*) on a 4-connected grid.

    This implementation behaves like A* but keeps the OPEN list
    below a configurable maximum size by discarding the least
    promising frontier nodes when necessary.

    It is *not* a fully textbook-accurate implementation of SMA*
    (which also backs up f-costs into ancestors when pruning
    subtrees), but it captures the practical behavior of a
    memory-bounded best-first search:

      - Uses f(n) = g(n) + h(n) with Manhattan distance h.
      - Expands nodes in increasing f-order.
      - When the OPEN list grows beyond `max_open`, the node
        with the largest f-value is removed, limiting memory use.

    The search is performed on the *visible* grid only
    (known + discovered obstacles) via `world.neighbors4_visible()`.

    Returns an empty list if no path is found.
    """

    name = "SMAStar"

    def __init__(self, max_open: int = 5000) -> None:
        # Maximum number of nodes kept in the OPEN list.
        # Larger values behave more like standard A*.
        self.max_open = max_open

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

        # open_heap stores tuples (f, g, counter, pos)
        open_heap: List[Tuple[int, int, int, Pos]] = []
        g_cost: Dict[Pos, int] = {start: 0}
        parent: Dict[Pos, Pos] = {}
        closed: Set[Pos] = set()

        counter = 0
        f0 = heuristic(start)
        heappush(open_heap, (f0, 0, counter, start))

        neighbors = world.neighbors4_visible

        while open_heap:
            f_cur, g_cur, _, cur = heappop(open_heap)

            if cur in closed:
                continue

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

            closed.add(cur)

            for np in neighbors(cur):
                new_g = g_cur + 1  # unit-cost grid

                # If we've already found a cheaper or equal path to np, skip
                old_g = g_cost.get(np)
                if old_g is not None and new_g >= old_g:
                    continue

                g_cost[np] = new_g
                parent[np] = cur
                counter += 1
                f_np = new_g + heuristic(np)
                heappush(open_heap, (f_np, new_g, counter, np))

                # Enforce memory bound on OPEN list
                if len(open_heap) > self.max_open:
                    self._prune_worst(open_heap, g_cost, parent)

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
    def _prune_worst(
        self,
        open_heap: List[Tuple[int, int, int, Pos]],
        g_cost: Dict[Pos, int],
        parent: Dict[Pos, Pos],
    ) -> None:
        """
        Remove the node with the worst (largest) f-value from the OPEN list.

        This serves as a simple memory-bounding mechanism for the search.
        We also discard its g/parent entries to avoid unbounded growth of
        those dictionaries.

        Note: This is a pragmatic approximation to textbook SMA* that is
        sufficient for many grid-world experiments.
        """
        if not open_heap:
            return

        # Find index of the entry with the largest f
        worst_idx = 0
        worst_f = -inf
        worst_pos: Optional[Pos] = None
        for i, (f, g, c, pos) in enumerate(open_heap):
            if f > worst_f:
                worst_f = f
                worst_idx = i
                worst_pos = pos

        # Remove it from the heap
        last = open_heap.pop()
        if worst_idx < len(open_heap):
            open_heap[worst_idx] = last
            heapify(open_heap)

        # Drop its bookkeeping info (if present)
        if worst_pos is not None:
            if worst_pos in g_cost:
                del g_cost[worst_pos]
            if worst_pos in parent:
                del parent[worst_pos]

    def _update_stats(self, dt: float) -> None:
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1


ALGORITHM = SMAStarPlanner()
