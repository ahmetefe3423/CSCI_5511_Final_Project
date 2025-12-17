# tours/cheapest_insertion.py
from __future__ import annotations

from time import perf_counter
from math import inf
from itertools import permutations  # not strictly needed but fine to keep
from typing import List, Tuple, Optional

from env import WorldState, Pos
from .base import TourAlgorithm
from pathfinding.base import PathfindingAlgorithm


class CheapestInsertionTour(TourAlgorithm):
    """
    Cheapest-insertion TSP heuristic (open tour, no return to start).

    Given:
      - start position
      - targets [t1, ..., tk]

    We:
      - precompute shortest-path distances:
          d_start[i] = dist(start -> targets[i])
          d_tt[i][j] = dist(targets[i] -> targets[j])
      - build an ordered list 'tour' of target indices using cheapest insertion:
          * start from the target closest to 'start'
          * repeatedly insert the remaining target that yields the smallest
            increase in total path cost, considering insertion before,
            between, or after current tour nodes.

    Cost model (open path):
      cost(tour = [i0, i1, ..., ik-1]) =
          d_start[i0] + sum_{m=0}^{k-2} d_tt[im][im+1]

    If at any point a required distance is infinite (no path), the tour
    is considered unreachable and cost = +inf.

    This is a heuristic for PC(r, S); it is not guaranteed optimal,
    but is much cheaper than brute-force for larger |S|.
    """

    name = "CheapestInsertion"

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
        Simulator calls this so the tour heuristic uses the same
        pathfinding algorithm (BFS, A*, etc.).
        """
        self.path_algo = algo

    # ---- helper: compute cost for a given visit order ----

    def _compute_cost(
        self,
        d_start: List[float],
        d_tt: List[List[float]],
        order: List[int],
    ) -> float:
        """Cost of open tour: start -> order[0] -> order[1] -> ..."""
        if not order:
            return 0.0
        i0 = order[0]
        cost = d_start[i0]
        if cost == inf:
            return inf
        for a, b in zip(order, order[1:]):
            step = d_tt[a][b]
            if step == inf:
                return inf
            cost += step
        return cost

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
                "CheapestInsertionTour.path_algo is not set. "
                "Simulator must inject a pathfinding algorithm via set_path_algo()."
            )

        targets = list(targets)
        n = len(targets)

        if n == 0:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], 0.0

        # Precompute distances from start to each target
        d_start: List[float] = [inf] * n
        for i, t in enumerate(targets):
            path = self.path_algo.plan(world, start, t)
            if path:
                d_start[i] = len(path) - 1

        # Precompute distances between targets
        d_tt: List[List[float]] = [[inf] * n for _ in range(n)]
        for i, a in enumerate(targets):
            for j, b in enumerate(targets):
                if i == j:
                    d_tt[i][j] = 0.0
                    continue
                path = self.path_algo.plan(world, a, b)
                if path:
                    d_tt[i][j] = len(path) - 1

        # ----- initialization: pick target closest to start -----
        best_first: Optional[int] = None
        best_first_cost: float = inf
        for i in range(n):
            if d_start[i] < best_first_cost:
                best_first_cost = d_start[i]
                best_first = i

        if best_first is None or best_first_cost == inf:
            # No target reachable from start
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], inf

        tour_idxs: List[int] = [best_first]
        used = {best_first}

        # ----- iterative cheapest insertion -----
        while len(tour_idxs) < n:
            base_cost = self._compute_cost(d_start, d_tt, tour_idxs)
            if base_cost == inf:
                dt = perf_counter() - t0
                self._update_stats(dt)
                return [], inf

            best_u: Optional[int] = None
            best_pos: Optional[int] = None
            best_delta: float = inf

            remaining = [idx for idx in range(n) if idx not in used]

            for u in remaining:
                # try inserting u at every possible position in tour
                k = len(tour_idxs)
                for pos in range(k + 1):
                    # compute incremental Δ cost based on local edges
                    delta = inf

                    if pos == 0:
                        # insert before first node
                        first = tour_idxs[0]
                        old_part = d_start[first]
                        new_part = d_start[u] + d_tt[u][first]
                        if d_start[u] != inf and d_tt[u][first] != inf:
                            delta = new_part - old_part

                    elif pos == k:
                        # insert after last node
                        last = tour_idxs[-1]
                        # old has no outgoing edge from last
                        if d_tt[last][u] != inf:
                            delta = d_tt[last][u]

                    else:
                        # insert between tour_idxs[pos-1] and tour_idxs[pos]
                        a = tour_idxs[pos - 1]
                        b = tour_idxs[pos]
                        old_part = d_tt[a][b]
                        new_part = d_tt[a][u] + d_tt[u][b]
                        if d_tt[a][u] != inf and d_tt[u][b] != inf:
                            delta = new_part - old_part

                    if delta < best_delta:
                        best_delta = delta
                        best_u = u
                        best_pos = pos

            if best_u is None or best_pos is None or best_delta == inf:
                # cannot insert remaining targets feasibly
                dt = perf_counter() - t0
                self._update_stats(dt)
                return [], inf

            # apply best insertion
            tour_idxs.insert(best_pos, best_u)
            used.add(best_u)

        # Final cost
        total_cost = self._compute_cost(d_start, d_tt, tour_idxs)
        if total_cost == inf:
            order: List[Pos] = []
        else:
            order = [targets[i] for i in tour_idxs]

        dt = perf_counter() - t0
        self._update_stats(dt)
        return order, total_cost


ALGORITHM = CheapestInsertionTour()
