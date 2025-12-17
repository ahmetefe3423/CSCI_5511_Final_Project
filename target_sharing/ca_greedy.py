# target_sharing/ca_greedy.py
from __future__ import annotations

from time import perf_counter
from math import inf
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

from env import WorldState, RobotState, Pos
from .base import TargetSharingAlgorithm
from tours.base import TourAlgorithm


@dataclass
class BundleBid:
    """A bid for a bundle of targets by a single robot."""
    rid: int                  # robot id
    targets: Tuple[Pos, ...]  # bundle of targets, IN TOUR ORDER for this robot
    cost: float               # tour cost to cover this bundle from r.pos


class GreedyCombinatorialAuction(TargetSharingAlgorithm):
    """
    Greedy Combinatorial Auction (CA-inspired):

    - For each robot r and each bundle S of targets (size 1 or 2 here),
      compute PC(r, S) using the injected TourAlgorithm.
    - Treat each (r, S) as a "bid" with cost = PC(r, S).
    - Sort bids by increasing cost, and greedily accept bids whose targets
      are not already assigned to someone else.
    - This approximates the combinatorial auction idea without solving
      the full NP-hard winner determination problem.

    Notes:
      * Uses the robot's current position r.pos as the start for PC(r, S).
      * Any leftover targets that never appear in winning bundles
        remain unassigned (you could add a fallback if desired).
    """

    name = "CA_Greedy"

    def __init__(self) -> None:
        # timing stats
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # Tour algorithm will be injected by the Simulator
        self.tour_algo: Optional[TourAlgorithm] = None

        # You can tweak this if you want larger bundles
        self.max_bundle_size: int = 2

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
        Simulator calls this so CA_Greedy can use the chosen tour heuristic
        (NearestNeighbor, ExactBruteForce, CheapestInsertion, etc.).
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
                "CA_Greedy.tour_algo is not set. "
                "Simulator must inject a TourAlgorithm via set_tour_algo()."
            )

        assignments: Dict[int, List[Pos]] = {r.rid: [] for r in robots}
        if not robots or not targets:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return assignments

        target_list = list(targets)

        # Cache tours so we don't solve the same (robot, bundle) problem twice.
        # Keyed by (robot id, frozenset(targets)) -> (order, cost)
        tour_cache: Dict[Tuple[int, frozenset[Pos]], Tuple[List[Pos], float]] = {}

        def get_tour(
            rid: int,
            start: Pos,
            bundle: Tuple[Pos, ...],
        ) -> Tuple[List[Pos], float]:
            """Return (order, cost) for robot rid visiting 'bundle' from 'start'."""
            if not bundle:
                return [], 0.0
            key = (rid, frozenset(bundle))
            if key in tour_cache:
                return tour_cache[key]
            order, cost = self.tour_algo.solve(world, start, list(bundle))
            tour_cache[key] = (order, cost)
            return order, cost

        bids: List[BundleBid] = []

        # Generate bundle bids (size 1 and 2)
        for r in robots:
            rid = r.rid
            start = r.pos

            # Size-1 bundles
            for i in range(len(target_list)):
                t = target_list[i]
                order, cost = get_tour(rid, start, (t,))
                if cost == inf:
                    continue
                # store the order given by the tour algorithm
                bids.append(BundleBid(rid=rid, targets=tuple(order), cost=cost))

            # Size-2 bundles (if enabled)
            if self.max_bundle_size >= 2 and len(target_list) >= 2:
                for i in range(len(target_list)):
                    for j in range(i + 1, len(target_list)):
                        S = (target_list[i], target_list[j])
                        order, cost = get_tour(rid, start, S)
                        if cost == inf:
                            continue
                        # targets in this bid are in tour order for this robot
                        bids.append(BundleBid(rid=rid, targets=tuple(order), cost=cost))

        # Sort bids by increasing cost (cheapest bundles first)
        bids.sort(key=lambda b: b.cost)

        assigned_targets: Set[Pos] = set()

        # Greedy winner determination:
        # pick non-overlapping bundles in order of increasing cost
        for bid in bids:
            # skip if any target already assigned
            if any(t in assigned_targets for t in bid.targets):
                continue

            # accept this bid; append targets in the tour order
            for t in bid.targets:
                assignments[bid.rid].append(t)
                assigned_targets.add(t)

        # At this point, some targets may remain unassigned.
        # They will stay in world.targets and might be handled on replans
        # or simply remain uncollected.

        dt = perf_counter() - t0
        self._update_stats(dt)
        return assignments


ALGORITHM = GreedyCombinatorialAuction()
