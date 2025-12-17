# target_sharing/ssa.py
from __future__ import annotations

from time import perf_counter
from math import inf
from typing import Dict, List, Set, Tuple, Optional

from env import WorldState, RobotState, Pos
from .base import TargetSharingAlgorithm
from tours.base import TourAlgorithm


class SSAuction(TargetSharingAlgorithm):
    """
    Sequential Single-Item Auctions (Koenig et al., 2006):

    - Maintain for each robot r a set T(r) of owned targets (initially empty).
    - While there are unallocated targets:
        * For each robot r and each unallocated target t:
              bid(r, t) = PC(r, T(r) ∪ {t}) - PC(r, T(r))
          where PC(...) is approximated by the injected TourAlgorithm.
        * Choose (r*, t*) with smallest bid(r, t).
        * Assign t* to r*, update T(r*), remove t* from unallocated.

    Here:
      - The tour cost PC(r, S) is computed starting from r.pos.
      - TourAlgorithm (e.g. NearestNeighbor) measures its own runtime.

    This implementation:
      - Caches (tour order, cost) for each (robot, set of targets) pair
        so we don't recompute the same tour multiple times.
      - Returns assignments in the *tour order* for each robot, not
        just in the order targets were added.
    """

    name = "SSA"

    def __init__(self) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # Tour algorithm will be injected by the Simulator
        self.tour_algo: Optional[TourAlgorithm] = None

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

    def set_tour_algo(self, algo: TourAlgorithm) -> None:
        """
        Simulator calls this so SSA can use the chosen tour heuristic
        (NearestNeighbor, Ising, SNN, etc.).
        """
        self.tour_algo = algo

    # ---- main auction logic ----

    def assign(
        self,
        world: WorldState,
        robots: List[RobotState],
        targets: Set[Pos],
    ) -> Dict[int, List[Pos]]:
        t0 = perf_counter()

        if self.tour_algo is None:
            raise RuntimeError(
                "SSA.tour_algo is not set. Simulator must inject a TourAlgorithm via set_tour_algo()."
            )

        # Initialize assignments: robot_id -> (final ordered list of targets)
        assignments: Dict[int, List[Pos]] = {r.rid: [] for r in robots}
        if not robots or not targets:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return assignments

        # For convenience: robot start position for PC(r, S)
        start_pos: Dict[int, Pos] = {r.rid: r.pos for r in robots}

        # For each robot r:
        #   current_sets[r]  = frozenset of targets T(r)
        #   current_costs[r] = PC(r, T(r))
        #   current_orders[r] = visiting order for T(r) (tour)
        current_sets: Dict[int, frozenset[Pos]] = {r.rid: frozenset() for r in robots}
        current_costs: Dict[int, float] = {r.rid: 0.0 for r in robots}
        current_orders: Dict[int, List[Pos]] = {r.rid: [] for r in robots}

        # Cache tours so we don't recompute the same (robot, set-of-targets) problem
        # multiple times in one assign() call.
        # Key: (robot id, frozenset of targets) -> (order, cost)
        tour_cache: Dict[Tuple[int, frozenset[Pos]], Tuple[List[Pos], float]] = {}

        def get_tour(
            rid: int,
            S_set: frozenset[Pos],
        ) -> Tuple[List[Pos], float]:
            """Return (order, cost) for robot rid visiting S_set from its start."""
            if not S_set:
                return [], 0.0
            key = (rid, S_set)
            if key in tour_cache:
                return tour_cache[key]
            # We can give targets to the tour algo in any order; it will compute its own tour.
            S_list = list(S_set)
            order, cost = self.tour_algo.solve(world, start=start_pos[rid], targets=S_list)
            tour_cache[key] = (order, cost)
            return order, cost

        unallocated: Set[Pos] = set(targets)

        while unallocated:
            best_rid: Optional[int] = None
            best_t: Optional[Pos] = None
            best_delta: float = inf

            # For each robot and each unallocated target, compute marginal cost
            for r in robots:
                rid = r.rid
                base_cost = current_costs[rid]  # PC(r, T(r))
                if base_cost == inf:
                    # Robot already has an infeasible tour; skip
                    continue

                current_set = current_sets[rid]

                for t in unallocated:
                    new_set = current_set | {t}
                    _, new_cost = get_tour(rid, new_set)

                    if new_cost == inf:
                        # Can't build a tour including this t for this robot
                        continue

                    delta = new_cost - base_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_rid = rid
                        best_t = t

            if best_rid is None or best_t is None or best_delta == inf:
                # No further useful assignments possible (unreachable targets remain)
                break

            # Commit: assign best_t to best_rid
            unallocated.remove(best_t)

            # Update that robot's set and tour using the cached tour
            old_set = current_sets[best_rid]
            new_set = old_set | {best_t}
            current_sets[best_rid] = new_set

            order, new_cost = get_tour(best_rid, new_set)
            current_orders[best_rid] = order
            current_costs[best_rid] = new_cost

        # Build final assignments using the *tour order* for each robot
        for rid in assignments.keys():
            if current_sets[rid]:
                assignments[rid] = list(current_orders[rid])
            # if current_sets[rid] is empty, leave assignments[rid] as []

        dt = perf_counter() - t0
        self._update_stats(dt)
        return assignments


ALGORITHM = SSAuction()
