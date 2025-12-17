# tours/ising_limited.py
from __future__ import annotations

from time import perf_counter
from math import inf
from typing import Dict, List, Tuple, Optional, Set, Any

# The package is 'dwave-tabu', but the module name is 'tabu'
try:
    from tabu import TabuSampler
except ImportError:
    TabuSampler = None  # type: ignore[assignment]

from env import WorldState, Pos
from .base import TourAlgorithm
from pathfinding.base import PathfindingAlgorithm


class IsingLimitedTour(TourAlgorithm):
    """
    Hardware-limited Ising / QUBO tour solver using TabuSampler.

    Differences from IsingFull:

      - Same TSP-style QUBO structure, but:

          * Constraints (assignment constraints) get integer coefficients ±7.
          * Cost terms (start->first + transitions) get integer coefficients in 1..6.
            We avoid 7 so it's reserved for "hard" constraints.

      - We try multiple cost-mapping strategies (different ways to map real costs
        into 1..6), run Tabu once per mapping, decode each sample to a tour,
        and keep the tour with the lowest REAL grid cost.

      - Hardware model:
            HW_SPIN_LIMIT      = 49 spins
            HW_ANNEAL_TIME_S   = 50e-6 seconds per hardware call
            HW_POWER_W         = 24e-3 W (24 mW)
        For a QUBO with Nvars spins and Nq different mappings tried, we estimate:
            calls_per_qubo = max(1, round(Nvars / HW_SPIN_LIMIT))
            total_hw_calls = Nq * calls_per_qubo

        We track:
            total_hw_anneal_time (seconds)
            total_hw_energy (Joules)
            last_hw_calls
            last_hw_anneal_time
    """

    name = "IsingLimited"

    # hardware parameters (conceptual)
    HW_SPIN_LIMIT: int = 49
    HW_ANNEAL_TIME_S: float = 50e-6  # 50 microseconds per hardware call
    HW_POWER_W: float = 24e-3        # 24 mW

    def __init__(self, num_reads: int = 2) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # pathfinding algorithm will be injected by the Simulator
        self.path_algo: Optional[PathfindingAlgorithm] = None

        # Tabu config
        self.num_reads: int = num_reads
        self._sampler: Optional[Any] = None

        # hardware metrics
        self.total_hw_anneal_time: float = 0.0  # seconds
        self.total_hw_energy: float = 0.0       # Joules
        self.last_hw_anneal_time: float = 0.0   # seconds
        self.last_hw_calls: int = 0

    # ---- stats / hardware API ----

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

        self.total_hw_anneal_time = 0.0
        self.total_hw_energy = 0.0
        self.last_hw_anneal_time = 0.0
        self.last_hw_calls = 0

    def _update_stats(self, dt: float, nvars: int, num_qubos: int) -> None:
        """
        dt: wall-clock time for this solve() call
        nvars: number of spins in the QUBO (here n^2)
        num_qubos: how many different integer QUBO mappings we tried
        """
        # wall-clock runtime
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1

        # simulated hardware usage
        if nvars <= 0 or num_qubos <= 0 or self.HW_SPIN_LIMIT <= 0:
            hw_calls = 0
        else:
            calls_per_qubo = max(1, round(nvars / float(self.HW_SPIN_LIMIT)))
            hw_calls = num_qubos * calls_per_qubo

        hw_time = hw_calls * self.HW_ANNEAL_TIME_S
        hw_energy = hw_time * self.HW_POWER_W

        self.last_hw_calls = hw_calls
        self.last_hw_anneal_time = hw_time

        self.total_hw_anneal_time += hw_time
        self.total_hw_energy += hw_energy

    # ---- dependency injection ----

    def set_path_algo(self, algo: PathfindingAlgorithm) -> None:
        self.path_algo = algo

    # ---- helpers ----

    @staticmethod
    def _var_idx(i: int, p: int, n: int) -> int:
        """Map (target_index i, position p) -> single integer variable index."""
        return i * n + p

    # ---- decoding helper ----

    def _decode_sample(
        self,
        sample: Dict[int, int],
        n: int,
        d_start: List[float],
        d_tt: List[List[float]],
    ) -> Tuple[List[int], float]:
        """
        Decode a Tabu sample into an order of target indices [i0, i1, ..., i_{n-1}]
        and compute the REAL tour cost using d_start and d_tt.

        Returns:
            (order_idxs, cost)
        where cost = inf if the decoded tour is invalid/unreachable.
        """
        used_targets: Set[int] = set()
        order_idxs: List[int] = []

        # 1) Decode one target per position
        for p in range(n):
            scores: List[Tuple[int, int]] = []  # (bit, i)
            for i in range(n):
                k = self._var_idx(i, p, n)
                bit = sample.get(k, 0)
                scores.append((bit, i))

            scores.sort(reverse=True)
            chosen_i: Optional[int] = None

            for bit, i in scores:
                if bit == 1 and i not in used_targets:
                    chosen_i = i
                    break

            if chosen_i is None:
                # fallback: pick any unused target
                unused = [i for i in range(n) if i not in used_targets]
                if not unused:
                    return [], inf
                chosen_i = unused[0]

            used_targets.add(chosen_i)
            order_idxs.append(chosen_i)

        # 2) Repair if not a permutation
        if len(set(order_idxs)) != n:
            seen: Set[int] = set()
            repaired: List[int] = []
            for i in order_idxs:
                if i not in seen:
                    seen.add(i)
                    repaired.append(i)
            for i in range(n):
                if i not in seen:
                    repaired.append(i)
            order_idxs = repaired[:n]

        # 3) Compute real tour cost via distance matrices
        first = order_idxs[0]
        if d_start[first] == inf:
            return [], inf

        total_cost = d_start[first]
        for a, b in zip(order_idxs, order_idxs[1:]):
            step = d_tt[a][b]
            if step == inf:
                return [], inf
            total_cost += step

        return order_idxs, total_cost

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
                "IsingLimitedTour.path_algo is not set. "
                "Simulator must inject a pathfinding algorithm via set_path_algo()."
            )

        if TabuSampler is None:
            raise RuntimeError(
                "IsingLimitedTour requires the 'dwave-tabu' package "
                "(imported as module 'tabu'). "
                "Install it with `pip install dwave-tabu` or choose a "
                "different tour algorithm in main.py."
            )

        targets = list(targets)
        n = len(targets)

        if n == 0:
            dt = perf_counter() - t0
            self._update_stats(dt, nvars=0, num_qubos=0)
            return [], 0.0

        # ---- 1) Precompute distances on the grid ----

        d_start: List[float] = [inf] * n
        for i, t in enumerate(targets):
            path = self.path_algo.plan(world, start, t)
            if path:
                d_start[i] = len(path) - 1

        d_tt: List[List[float]] = [[inf] * n for _ in range(n)]
        for i, a in enumerate(targets):
            for j, b in enumerate(targets):
                if i == j:
                    d_tt[i][j] = 0.0
                    continue
                path = self.path_algo.plan(world, a, b)
                if path:
                    d_tt[i][j] = len(path) - 1

        # If no target is reachable from start, no tour is possible
        if all(d == inf for d in d_start):
            dt = perf_counter() - t0
            self._update_stats(dt, nvars=n * n, num_qubos=0)
            return [], inf

        # ---- 2) Build separate float QUBOs: cost part and constraint part ----

        Q_cost: Dict[Tuple[int, int], float] = {}
        Q_cons: Dict[Tuple[int, int], float] = {}

        def add_cost(u: int, v: int, coeff: float) -> None:
            if coeff == 0.0:
                return
            if u > v:
                u, v = v, u
            Q_cost[(u, v)] = Q_cost.get((u, v), 0.0) + coeff

        def add_cons(u: int, v: int, coeff: float) -> None:
            if coeff == 0.0:
                return
            if u > v:
                u, v = v, u
            Q_cons[(u, v)] = Q_cons.get((u, v), 0.0) + coeff

        # Constraints:
        #   For each position p: sum_i x_{i,p} = 1
        #   For each target i:  sum_p x_{i,p} = 1
        #
        #   Penalty (sum - 1)^2 = -sum x + 2 sum_{i<j} x_i x_j + const

        # per-position constraints
        for p in range(n):
            for i in range(n):
                k = self._var_idx(i, p, n)
                add_cons(k, k, -1.0)
            for i in range(n):
                for j in range(i + 1, n):
                    k_i = self._var_idx(i, p, n)
                    k_j = self._var_idx(j, p, n)
                    add_cons(k_i, k_j, 2.0)

        # per-target constraints
        for i in range(n):
            for p in range(n):
                k = self._var_idx(i, p, n)
                add_cons(k, k, -1.0)
            for p in range(n):
                for q in range(p + 1, n):
                    k_p = self._var_idx(i, p, n)
                    k_q = self._var_idx(i, q, n)
                    add_cons(k_p, k_q, 2.0)

        # Cost terms:
        #   cost = sum_i d_start[i] * x_{i,0}
        #        + sum_{p=0..n-2} sum_{i,j} d_tt[i][j] * x_{i,p} x_{j,p+1}

        # start -> first target
        for i in range(n):
            if d_start[i] < inf:
                k = self._var_idx(i, 0, n)
                add_cost(k, k, d_start[i])

        # transitions
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
                    add_cost(k_i, k_j, c)

        # ---- 3) Build integer QUBOs with limited coefficients [-7, 7] ----
        #
        #    - Constraints all get ±7 (hard).
        #    - Costs get mapped into 1..6 (never 7), several mapping strategies.

        cost_vals = [abs(v) for v in Q_cost.values() if abs(v) > 0.0]
        cost_min = min(cost_vals) if cost_vals else 0.0
        cost_max = max(cost_vals) if cost_vals else 0.0

        def build_qubos_int() -> List[Dict[Tuple[int, int], int]]:
            qubos: List[Dict[Tuple[int, int], int]] = []

            # Helper: add constraint terms with ±7
            def add_cons_int(Q: Dict[Tuple[int, int], int]) -> None:
                for (u, v), coeff in Q_cons.items():
                    if coeff == 0.0:
                        continue
                    sign = 1 if coeff > 0 else -1
                    q_int = 7 * sign
                    if u > v:
                        uu, vv = v, u
                    else:
                        uu, vv = u, v
                    Q[(uu, vv)] = Q.get((uu, vv), 0) + q_int

            # Strategy 1: linear scaling, max cost -> 6
            if cost_vals and cost_max > 0.0:
                Q1: Dict[Tuple[int, int], int] = {}
                add_cons_int(Q1)

                for (u, v), coeff in Q_cost.items():
                    if coeff == 0.0:
                        continue
                    # costs are non-negative here
                    level = int(round(6.0 * coeff / cost_max))  # 0..6
                    if level <= 0:
                        level = 1
                    if level > 6:
                        level = 6
                    if u > v:
                        uu, vv = v, u
                    else:
                        uu, vv = u, v
                    Q1[(uu, vv)] = Q1.get((uu, vv), 0) + level

                qubos.append(Q1)

            # Strategy 2: normalized mapping to 1..6
            if cost_vals and cost_max > 0.0:
                Q2: Dict[Tuple[int, int], int] = {}
                add_cons_int(Q2)

                if cost_max == cost_min:
                    # all costs equal; map them all to 6
                    for (u, v), coeff in Q_cost.items():
                        if coeff == 0.0:
                            continue
                        level = 6
                        if u > v:
                            uu, vv = v, u
                        else:
                            uu, vv = u, v
                        Q2[(uu, vv)] = Q2.get((uu, vv), 0) + level
                else:
                    span = cost_max - cost_min
                    for (u, v), coeff in Q_cost.items():
                        if coeff == 0.0:
                            continue
                        norm = (abs(coeff) - cost_min) / span
                        level = 1 + int(round(5.0 * norm))  # 1..6
                        if level < 1:
                            level = 1
                        if level > 6:
                            level = 6
                        if u > v:
                            uu, vv = v, u
                        else:
                            uu, vv = u, v
                        Q2[(uu, vv)] = Q2.get((uu, vv), 0) + level

                qubos.append(Q2)

            # If there are no cost terms but there are constraints, keep constraints-only QUBO
            if not cost_vals and Q_cons:
                Q_only: Dict[Tuple[int, int], int] = {}
                add_cons_int(Q_only)
                qubos.append(Q_only)

            return qubos

        qubos_int = build_qubos_int()
        nvars = n * n
        num_qubos = len(qubos_int)

        if self._sampler is None:
            self._sampler = TabuSampler()

        # ---- 4) Run Tabu for each mapping, keep best real-cost tour ----

        best_order_idxs: Optional[List[int]] = None
        best_cost: float = inf

        for Q_int in qubos_int:
            if not Q_int:
                continue

            sampleset = self._sampler.sample_qubo(Q_int, num_reads=self.num_reads)
            best_sample = sampleset.first
            sample: Dict[int, int] = best_sample.sample

            order_idxs, cost = self._decode_sample(sample, n, d_start, d_tt)
            if cost < best_cost:
                best_cost = cost
                best_order_idxs = order_idxs

        # If all mappings failed, no feasible tour
        if best_order_idxs is None or best_cost == inf:
            dt = perf_counter() - t0
            self._update_stats(dt, nvars=nvars, num_qubos=num_qubos)
            return [], inf

        # Map indices back to positions
        order: List[Pos] = [targets[i] for i in best_order_idxs]

        dt = perf_counter() - t0
        self._update_stats(dt, nvars=nvars, num_qubos=num_qubos)
        return order, best_cost


ALGORITHM = IsingLimitedTour()
