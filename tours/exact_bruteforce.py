# tours/exact_bruteforce.py
from __future__ import annotations

from time import perf_counter
from math import inf
from itertools import permutations
from typing import List, Tuple, Optional

from env import WorldState, Pos
from .base import TourAlgorithm
from pathfinding.base import PathfindingAlgorithm


class ExactBruteForceTour(TourAlgorithm):
    """
    Exact TSP-style tour solver for small target sets.

    Given:
      - start position
      - a list of targets [t1, ..., tk]

    It:
      - computes shortest-path distances using the injected pathfinding
        algorithm (e.g. BFS, A*)
      - brute-forces all permutations of the targets
      - returns the *optimal* visiting order and its cost.

    This is exponential in k!, so we restrict to small k.
    """

    name = "ExactBruteForce"

    def __init__(self, max_targets: int = 10) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # pathfinding algorithm (injected)
        self.path_algo: Optional[PathfindingAlgorithm] = None

        # safety limit for brute force
        self.max_targets: int = max_targets

    # ---- timing ----

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
                "ExactBruteForceTour.path_algo is not set. "
                "Simulator must inject a pathfinding algorithm via set_path_algo()."
            )

        targets = list(targets)
        n = len(targets)

        if n == 0:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], 0.0

        if n > self.max_targets:
            raise ValueError(
                f"ExactBruteForceTour supports at most {self.max_targets} "
                f"targets per call, got {n}."
            )

        # Precompute distances from start -> targets
        d_start = [inf] * n
        for i, t in enumerate(targets):
            path = self.path_algo.plan(world, start, t)
            if path:
                d_start[i] = len(path) - 1

        # Precompute distances between targets
        d_tt = [[inf] * n for _ in range(n)]
        for i, a in enumerate(targets):
            for j, b in enumerate(targets):
                if i == j:
                    d_tt[i][j] = 0
                    continue
                path = self.path_algo.plan(world, a, b)
                if path:
                    d_tt[i][j] = len(path) - 1

        best_cost = inf
        best_order: Optional[List[Pos]] = None

        for perm in permutations(range(n)):
            first = perm[0]
            if d_start[first] == inf:
                continue

            cost = d_start[first]
            ok = True

            for i in range(n - 1):
                a = perm[i]
                b = perm[i + 1]
                c = d_tt[a][b]
                if c == inf:
                    ok = False
                    break
                cost += c
                # simple pruning
                if cost >= best_cost:
                    ok = False
                    break

            if not ok:
                continue

            if cost < best_cost:
                best_cost = cost
                best_order = [targets[idx] for idx in perm]

        if best_cost == inf:
            # unreachable set
            order: List[Pos] = []
        else:
            order = best_order if best_order is not None else []

        dt = perf_counter() - t0
        self._update_stats(dt)
        return order, best_cost


ALGORITHM = ExactBruteForceTour()
