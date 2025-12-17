# target_sharing/ca_optimal.py
from __future__ import annotations

from time import perf_counter
from math import inf
from typing import Dict, List, Set, Tuple, Optional

from env import WorldState, RobotState, Pos
from .base import TargetSharingAlgorithm
from tours.base import TourAlgorithm


class CAOptimal(TargetSharingAlgorithm):
    """
    Exact Single-Round Combinatorial Auction (small instances only).

    Conceptually equivalent to Koenig et al.'s single-round CA:

      - Each target must be visited by exactly one robot.
      - Each robot gets at most one bundle T(r) of targets
        (possibly empty).
      - Cost for a robot r is PC(r, T(r)) = shortest tour
        covering all targets in T(r) from r.pos.
      - We choose the assignment that minimizes the sum of
        all PC(r, T(r)).

    Implementation:

      - Brute-force over all assignments of targets to robots:
            f: targets -> robots
      - For each robot r, we use an injected TourAlgorithm
        (e.g., ExactBruteForce) to compute PC(r, T(r)).
      - Keep the best assignment.

    This is exponential in |targets|, so we restrict it to
    small |targets| via max_targets_exact.
    """

    name = "CA_Optimal"

    def __init__(self, max_targets_exact: int = 10) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # injected tour algorithm, set by the simulator
        self.tour_algo: Optional[TourAlgorithm] = None

        # limit for exact search
        self.max_targets_exact: int = max_targets_exact

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

    def set_tour_algo(self, algo: TourAlgorithm) -> None:
        """
        Simulator calls this so CA_Optimal can use the chosen tour
        solver (ExactBruteForce, Ising, SNN, etc.).
        """
        self.tour_algo = algo

    # ---- main logic ----

    def assign(
        self,
        world: WorldState,
        robots: List[RobotState],
        targets: Set[Pos],
    ) -> Dict[int, List[Pos]]:
        t0 = perf_counter()

        if self.tour_algo is None:
            raise RuntimeError(
                "CA_Optimal requires a TourAlgorithm. "
                "Call set_tour_algo() before assign()."
            )

        assignments: Dict[int, List[Pos]] = {r.rid: [] for r in robots}

        # quick exits
        if not robots or not targets:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return assignments

        target_list = list(targets)
        num_targets = len(target_list)

        if num_targets > self.max_targets_exact:
            raise ValueError(
                f"CA_Optimal supports at most {self.max_targets_exact} targets "
                f"for exact search, got {num_targets}."
            )

        rid_list = [r.rid for r in robots]
        start_pos: Dict[int, Pos] = {r.rid: r.pos for r in robots}

        # cache for PC(r, S) keyed by (robot id, frozenset of targets).
        # We store BOTH the optimal visiting order and its cost so that
        # we can later return the same order to the simulator.
        pc_cache: Dict[Tuple[int, frozenset[Pos]], Tuple[List[Pos], float]] = {}

        def get_pc(rid: int, S: List[Pos]) -> float:
            """Return tour cost for robot rid visiting bundle S.

            Uses the injected tour algorithm and caches results. The
            cache is keyed by the *set* of targets, so the order of S
            is irrelevant here.
            """
            if not S:
                return 0.0
            key = (rid, frozenset(S))
            if key in pc_cache:
                return pc_cache[key][1]
            order, cost = self.tour_algo.solve(world, start_pos[rid], S)
            pc_cache[key] = (order, cost)
            return cost

        # backtracking over assignments
        best_total: float = inf
        best_owned: Optional[Dict[int, List[Pos]]] = None

        owned: Dict[int, List[Pos]] = {rid: [] for rid in rid_list}

        def backtrack(i: int) -> None:
            nonlocal best_total, best_owned

            # all targets assigned
            if i == num_targets:
                total = 0.0
                for rid in rid_list:
                    bundle = owned[rid]
                    if not bundle:
                        continue
                    cost = get_pc(rid, bundle)
                    if cost == inf:
                        # this assignment is infeasible for this robot
                        return
                    total += cost
                    if total >= best_total:
                        # prune
                        return
                if total < best_total:
                    best_total = total
                    # store a deep copy of current bundles
                    best_owned = {rid: list(owned[rid]) for rid in rid_list}
                return

            t = target_list[i]

            # try assigning t to each robot
            for rid in rid_list:
                owned[rid].append(t)
                backtrack(i + 1)
                owned[rid].pop()

        backtrack(0)

        if best_owned is not None:
            for rid, S in best_owned.items():
                if not S:
                    continue
                key = (rid, frozenset(S))
                # ensure tour order is available in cache
                if key not in pc_cache:
                    order, cost = self.tour_algo.solve(world, start_pos[rid], S)
                    pc_cache[key] = (order, cost)
                order, _ = pc_cache[key]
                # return the optimal visiting order for this bundle
                assignments[rid] = list(order)
        # else: everything stays empty (no feasible assignment)

        dt = perf_counter() - t0
        self._update_stats(dt)
        return assignments


ALGORITHM = CAOptimal()
