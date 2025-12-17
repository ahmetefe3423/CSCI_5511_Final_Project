# tours/ising_full.py
from __future__ import annotations

from time import perf_counter
from math import inf
from typing import Dict, List, Tuple, Optional, Set

from tabu import TabuSampler

from env import WorldState, Pos
from .base import TourAlgorithm
from pathfinding.base import PathfindingAlgorithm


class IsingFullTour(TourAlgorithm):
    """
    Ising / QUBO-based tour solver using D-Wave TabuSampler.

    - Encodes an open-path TSP over the given targets as a QUBO:
        * binary vars x_{i,p} = 1 if target i is at position p
        * constraints:
            - each position p has exactly one target
            - each target i appears at exactly one position
        * objective:
            - minimize d(start, first_target) +
                      sum d(target_k, target_{k+1})

    - Builds the QUBO with floating-point coefficients (no artificial
      quantization / clipping).
    - Calls TabuSampler with default parameters (num_reads configurable).
    - Decodes the best sample into an order of targets and returns
      (order, cost).

    Notes:
      * Uses the injected pathfinding algorithm (BFS, A*, etc.) to
        compute shortest-path distances on the grid.
      * For now, assumes the number of targets is not huge (n^2 vars).
    """

    name = "IsingFull"

    def __init__(self, num_reads: int = 2) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # pathfinding algorithm will be injected by the Simulator
        self.path_algo: Optional[PathfindingAlgorithm] = None

        # Tabu solver config
        self.num_reads: int = num_reads

        # Tabu sampler instance (can be reused across calls)
        self._sampler: Optional[TabuSampler] = None

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
        Simulator calls this so the Ising tour solver uses the same
        pathfinding algorithm (BFS, A*, etc.) as the rest of the system.
        """
        self.path_algo = algo

    # ---- helper: QUBO builder ----

    @staticmethod
    def _var_idx(i: int, p: int, n: int) -> int:
        """
        Map (target_index i, position p) -> single integer variable index.
        Total vars = n * n.
        """
        return i * n + p

    def _add_qubo(self, Q: Dict[Tuple[int, int], float], u: int, v: int, coeff: float) -> None:
        """
        Add 'coeff' to Q[u, v], enforcing u <= v for consistency.
        """
        if coeff == 0.0:
            return
        if u > v:
            u, v = v, u
        Q[(u, v)] = Q.get((u, v), 0.0) + coeff

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
                "IsingFullTour.path_algo is not set. "
                "Simulator must inject a pathfinding algorithm via set_path_algo()."
            )

        targets = list(targets)
        n = len(targets)

        if n == 0:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], 0.0

        # ---- 1) Precompute distances ----

        # d_start[i] = dist(start -> targets[i])
        d_start: List[float] = [inf] * n
        for i, t in enumerate(targets):
            path = self.path_algo.plan(world, start, t)
            if path:
                d_start[i] = len(path) - 1

        # d_tt[i][j] = dist(targets[i] -> targets[j])
        d_tt: List[List[float]] = [[inf] * n for _ in range(n)]
        for i, a in enumerate(targets):
            for j, b in enumerate(targets):
                if i == j:
                    d_tt[i][j] = 0.0
                    continue
                path = self.path_algo.plan(world, a, b)
                if path:
                    d_tt[i][j] = len(path) - 1

        # If all distances from start are inf, no tour is possible
        if all(d == inf for d in d_start):
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], inf

        # maximum finite distance (for penalty scaling)
        finite_vals: List[float] = [
            d for d in d_start if d < inf
        ] + [
            d_tt[i][j]
            for i in range(n)
            for j in range(n)
            if d_tt[i][j] < inf and i != j
        ]
        max_d = max(finite_vals) if finite_vals else 1.0

        # penalty weight (must dominate path costs)
        lambda_pen = 10.0 * max_d

        # ---- 2) Build QUBO: Q[(u,v)] ----
        Q: Dict[Tuple[int, int], float] = {}

        # (a) Assignment constraints:
        #     For each position p: sum_i x_{i,p} = 1
        #     For each target i:  sum_p x_{i,p} = 1
        #   Penalty: lambda * (sum - 1)^2
        #   (sum - 1)^2 = sum x - 2 sum x + 1 + 2 sum_{i<j} x_i x_j
        #               = - sum x + 2 sum_{i<j} x_i x_j + 1
        #   -> linear: -lambda for each var, quadratic: +2 lambda for each pair

        # per-position constraints
        for p in range(n):
            # linear terms: -lambda * x_{i,p}
            for i in range(n):
                k = self._var_idx(i, p, n)
                self._add_qubo(Q, k, k, -lambda_pen)
            # quadratic terms: 2*lambda * x_{i,p} x_{j,p} (i<j)
            for i in range(n):
                for j in range(i + 1, n):
                    k_i = self._var_idx(i, p, n)
                    k_j = self._var_idx(j, p, n)
                    self._add_qubo(Q, k_i, k_j, 2.0 * lambda_pen)

        # per-target constraints
        for i in range(n):
            # linear terms: -lambda * x_{i,p}
            for p in range(n):
                k = self._var_idx(i, p, n)
                self._add_qubo(Q, k, k, -lambda_pen)
            # quadratic terms: 2*lambda * x_{i,p} x_{i,q} (p<q)
            for p in range(n):
                for q in range(p + 1, n):
                    k_p = self._var_idx(i, p, n)
                    k_q = self._var_idx(i, q, n)
                    self._add_qubo(Q, k_p, k_q, 2.0 * lambda_pen)

        # (b) Cost terms:
        #   cost = sum_i d_start[i] * x_{i,0}
        #        + sum_{p=0..n-2} sum_{i,j} d_tt[i][j] * x_{i,p} x_{j,p+1}

        # start -> first target
        for i in range(n):
            if d_start[i] < inf:
                k = self._var_idx(i, 0, n)
                self._add_qubo(Q, k, k, d_start[i])

        # transitions between positions p and p+1
        for p in range(n - 1):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    c = d_tt[i][j]
                    if c == inf:
                        continue
                    k_i = self._var_idx(i, p, n)
                    k_j = self._var_idx(j, p + 1, n)
                    self._add_qubo(Q, k_i, k_j, c)

        # ---- 3) Solve QUBO with TabuSampler ----

        if self._sampler is None:
            self._sampler = TabuSampler()

        sampleset = self._sampler.sample_qubo(Q, num_reads=self.num_reads)
        best = sampleset.first
        sample: Dict[int, int] = best.sample  # var_index -> 0/1

        # ---- 4) Decode sample into an order ----

        # We ideally want:
        #   for each position p, exactly one i with x_{i,p} = 1
        #   each i used exactly once.
        used_targets: Set[int] = set()
        order_idxs: List[int] = []

        for p in range(n):
            # Collect bits for this position
            scores: List[Tuple[int, int]] = []  # (bit, i)
            for i in range(n):
                k = self._var_idx(i, p, n)
                bit = sample.get(k, 0)
                scores.append((bit, i))

            # Sort by bit descending (higher bit first)
            scores.sort(reverse=True)  # (bit, i)

            chosen_i: Optional[int] = None

            # Prefer a var with bit=1 and unused target
            for bit, i in scores:
                if bit == 1 and i not in used_targets:
                    chosen_i = i
                    break

            # If no 1-bit unused, fallback: pick any unused target
            if chosen_i is None:
                unused = [i for i in range(n) if i not in used_targets]
                if not unused:
                    # Can't assign a new target here -> invalid/degenerate
                    dt = perf_counter() - t0
                    self._update_stats(dt)
                    return [], inf
                chosen_i = unused[0]

            used_targets.add(chosen_i)
            order_idxs.append(chosen_i)

        # Sanity: ensure it's a permutation of 0..n-1
        if len(set(order_idxs)) != n:
            # repair best-effort: sort by first occurrence, drop duplicates
            seen: Set[int] = set()
            repaired: List[int] = []
            for i in order_idxs:
                if i not in seen:
                    seen.add(i)
                    repaired.append(i)
            # add any missing targets at the end
            for i in range(n):
                if i not in seen:
                    repaired.append(i)
            order_idxs = repaired[:n]

        # ---- 5) Compute real tour cost using distance matrices ----

        # If any segment is unreachable, treat as failure
        first = order_idxs[0]
        if d_start[first] == inf:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], inf

        total_cost = d_start[first]
        ok = True
        for a, b in zip(order_idxs, order_idxs[1:]):
            step = d_tt[a][b]
            if step == inf:
                ok = False
                break
            total_cost += step

        if not ok:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return [], inf

        order: List[Pos] = [targets[i] for i in order_idxs]

        dt = perf_counter() - t0
        self._update_stats(dt)
        return order, total_cost


ALGORITHM = IsingFullTour()
