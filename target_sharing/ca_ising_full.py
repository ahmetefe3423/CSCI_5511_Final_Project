# target_sharing/ca_ising_full.py
from __future__ import annotations

from dataclasses import dataclass
from math import inf
from time import perf_counter
from typing import Dict, List, Set, Tuple, Optional, Any

# D-Wave Tabu: package is 'dwave-tabu', module name is 'tabu'
try:
    from tabu import TabuSampler
except ImportError:
    TabuSampler = None  # type: ignore[assignment]

from env import WorldState, RobotState, Pos
from .base import TargetSharingAlgorithm
from tours.base import TourAlgorithm


@dataclass
class BundleBid:
    """A bundle of targets assigned to a robot, with a tour cost.

    IMPORTANT: 'targets' is stored in some visiting order produced by
    the TourAlgorithm for this robot (for that bundle).
    """
    rid: int                  # robot id
    targets: Tuple[Pos, ...]  # bundle of targets (tour order for this bundle)
    cost: float               # tour cost to cover this bundle from r.pos


class CAIsingFull(TargetSharingAlgorithm):
    """
    Combinatorial Auction with Ising-based winner determination.

    Objective (conceptual):

        Minimize   sum_r PC(r, T(r))
        subject to each target t is assigned to at most (ideally exactly) one robot,

    where PC(r, S) is the tour cost given by an injected TourAlgorithm.

    Implementation:

      1) For each robot r and each small bundle S of targets (size 1 or 2),
         compute a tour PC(r, S) and cache (order, cost).

      2) For each bundle b = (r, S), create a binary variable z_b:
           z_b = 1  -> we select bundle b.

      3) Build a QUBO for each penalty scaling λ in a small set Λ:
             E_λ(z) = sum_b cost(b) * z_b
                      + λ * sum_t (sum_{b: t in b.targets} z_b - 1)^2
         and solve each with TabuSampler.

      4) For each λ, decode the best sample into an assignment, repair conflicts
         greedily, and get per-robot target sets T_λ(r).

      5) Evaluate each assignment with a "true" cost:
             F(assignment) = sum_r PC(r, T_λ(r)) + μ * (#unassigned targets)
         using the same TourAlgorithm and cached tours.

      6) Pick the λ whose assignment has the smallest F, and return for each robot
         the tour order PC(r, T(r)) gives as assignments[rid].
    """

    name = "CA_IsingFull"

    def __init__(self, max_bundle_size: int = 2, num_reads: int = 2) -> None:
        # statistics for the auction itself
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # Tour algorithm injected by Simulator
        self.tour_algo: Optional[TourAlgorithm] = None

        # bundle / solver parameters
        self.max_bundle_size: int = max_bundle_size  # 1 or 2 typically
        self.num_reads: int = num_reads

        # Tabu sampler instance (reused across calls)
        self._sampler: Optional[Any] = None

        # Lambda multipliers for penalty search (relative to max bundle cost)
        self.lambda_factors: List[float] = [2.0, 5.0, 10.0, 20.0]

        # Penalty multiplier for unassigned targets in final scoring
        self.unassigned_penalty_factor: float = 10.0

    # ---- stats API ----

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
        Simulator calls this so CA_IsingFull can use the chosen tour solver
        (NearestNeighbor, CheapestInsertion, ExactBruteForce, IsingFull, etc.)
        to evaluate bundle costs PC(r, S).
        """
        self.tour_algo = algo

    # ---- QUBO helper ----

    @staticmethod
    def _add_qubo(Q: Dict[Tuple[int, int], float], u: int, v: int, coeff: float) -> None:
        """Add 'coeff' to Q[u, v], enforcing u <= v for consistency."""
        if coeff == 0.0:
            return
        if u > v:
            u, v = v, u
        Q[(u, v)] = Q.get((u, v), 0.0) + coeff

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
                "CA_IsingFull.tour_algo is not set. "
                "Simulator must inject a TourAlgorithm via set_tour_algo()."
            )

        if TabuSampler is None:
            raise RuntimeError(
                "CA_IsingFull requires the 'dwave-tabu' package "
                "(imported as module 'tabu'). "
                "Install it with `pip install dwave-tabu` or choose a "
                "different target-sharing algorithm in main.py."
            )

        assignments: Dict[int, List[Pos]] = {r.rid: [] for r in robots}

        if not robots or not targets:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return assignments

        target_list = list(targets)
        start_pos: Dict[int, Pos] = {r.rid: r.pos for r in robots}

        # ---- Tour cache: (rid, frozenset(targets)) -> (order, cost) ----

        tour_cache: Dict[Tuple[int, frozenset[Pos]], Tuple[List[Pos], float]] = {}

        def get_tour_for_set(
            rid: int,
            S_set: frozenset[Pos],
        ) -> Tuple[List[Pos], float]:
            """Return (order, cost) for robot rid visiting S_set from its start."""
            if not S_set:
                return [], 0.0
            key = (rid, S_set)
            if key in tour_cache:
                return tour_cache[key]
            S_list = list(S_set)
            order, cost = self.tour_algo.solve(
                world,
                start=start_pos[rid],
                targets=S_list,
            )
            tour_cache[key] = (order, cost)
            return order, cost

        def get_tour_for_bundle(
            rid: int,
            bundle: Tuple[Pos, ...],
        ) -> Tuple[List[Pos], float]:
            """Convenience wrapper for small bundles."""
            S_set = frozenset(bundle)
            return get_tour_for_set(rid, S_set)

        # ---- 1) Generate bundle bids (size 1 and 2) ----

        bids: List[BundleBid] = []
        for r in robots:
            rid = r.rid

            # size-1 bundles
            for i in range(len(target_list)):
                t = target_list[i]
                order, cost = get_tour_for_bundle(rid, (t,))
                if cost == inf:
                    continue
                # store the bundle in tour order (usually just (t,))
                bids.append(BundleBid(rid=rid, targets=tuple(order), cost=cost))

            # size-2 bundles
            if self.max_bundle_size >= 2 and len(target_list) >= 2:
                for i in range(len(target_list)):
                    for j in range(i + 1, len(target_list)):
                        S = (target_list[i], target_list[j])
                        order, cost = get_tour_for_bundle(rid, S)
                        if cost == inf:
                            continue
                        # targets in this bid are in tour order for this robot
                        bids.append(
                            BundleBid(
                                rid=rid,
                                targets=tuple(order),
                                cost=cost,
                            )
                        )

        if not bids:
            # No feasible bundles; nothing gets assigned.
            dt = perf_counter() - t0
            self._update_stats(dt)
            return assignments

        num_bids = len(bids)
        max_cost = max(b.cost for b in bids)
        base_cost_scale = max_cost if max_cost > 0.0 else 1.0

        # Precompute which bundles contain which targets (independent of λ)
        target_to_bids: Dict[Pos, List[int]] = {t: [] for t in targets}
        for k, b in enumerate(bids):
            for t in b.targets:
                if t in target_to_bids:
                    target_to_bids[t].append(k)

        # Ensure we have a sampler
        if self._sampler is None:
            self._sampler = TabuSampler()

        # ---- 2) Try multiple penalty scalings λ and pick best assignment ----

        best_score: float = inf
        best_robot_targets: Optional[Dict[int, Set[Pos]]] = None

        # Helper to build robot_targets from selected bundles
        def build_robot_targets(
            selected_idxs: List[int],
        ) -> Dict[int, Set[Pos]]:
            final_assigned: Set[Pos] = set()
            robot_targets: Dict[int, Set[Pos]] = {r.rid: set() for r in robots}

            for k in sorted(selected_idxs, key=lambda idx: bids[idx].cost):
                b = bids[k]
                for t in b.targets:
                    if t in final_assigned:
                        continue
                    final_assigned.add(t)
                    robot_targets[b.rid].add(t)
            return robot_targets

        # Helper to evaluate an assignment using true tour costs + unassigned penalty
        def evaluate_assignment(
            robot_targets: Dict[int, Set[Pos]],
        ) -> float:
            # union of all assigned targets
            assigned_union: Set[Pos] = set()
            total_cost: float = 0.0

            for rid, tset in robot_targets.items():
                if not tset:
                    continue
                assigned_union.update(tset)
                S_set = frozenset(tset)
                order, cost = get_tour_for_set(rid, S_set)
                if cost == inf:
                    return inf  # unreachable assignment
                total_cost += cost

            num_unassigned = len(targets - assigned_union)
            unassigned_penalty = self.unassigned_penalty_factor * base_cost_scale
            total_cost += unassigned_penalty * num_unassigned
            return total_cost

        for factor in self.lambda_factors:
            lambda_pen = factor * base_cost_scale

            # ---- 2a) Build QUBO for this λ ----
            Q: Dict[Tuple[int, int], float] = {}

            # (a) cost terms: linear
            for k, b in enumerate(bids):
                self._add_qubo(Q, k, k, b.cost)

            # (b) constraint terms: for each target
            for t, bid_idxs in target_to_bids.items():
                if not bid_idxs:
                    continue
                # linear: -lambda * z_k for each k in B_t
                for k in bid_idxs:
                    self._add_qubo(Q, k, k, -lambda_pen)
                # quadratic: 2 * lambda * z_k z_l for k<l in B_t
                for i in range(len(bid_idxs)):
                    for j in range(i + 1, len(bid_idxs)):
                        k_i = bid_idxs[i]
                        k_j = bid_idxs[j]
                        self._add_qubo(Q, k_i, k_j, 2.0 * lambda_pen)

            # ---- 2b) Solve QUBO with TabuSampler ----
            sampleset = self._sampler.sample_qubo(Q, num_reads=self.num_reads)
            best = sampleset.first
            sample: Dict[int, int] = best.sample  # k -> 0/1

            # ---- 2c) Decode sample: pick bundles, repair conflicts ----

            # Initially, keep all bundles with z_k == 1
            chosen_idxs = [k for k in range(num_bids) if sample.get(k, 0) == 1]

            # Greedy conflict resolution:
            #  - process chosen bundles in order of increasing cost
            #  - accept a bundle only if none of its targets are already assigned
            assigned_targets: Set[Pos] = set()
            selected_idxs: List[int] = []

            for k in sorted(chosen_idxs, key=lambda idx: bids[idx].cost):
                b = bids[k]
                if any(t in assigned_targets for t in b.targets):
                    continue
                selected_idxs.append(k)
                for t in b.targets:
                    assigned_targets.add(t)

            # Some targets may remain unassigned; try to assign them greedily
            unassigned = set(targets) - assigned_targets

            for t in unassigned:
                # candidate bundles that contain t and don't conflict
                cand_idxs: List[int] = []
                for k, b in enumerate(bids):
                    if t not in b.targets:
                        continue
                    if any(tt in assigned_targets for tt in b.targets):
                        continue
                    cand_idxs.append(k)

                if not cand_idxs:
                    # No feasible bundle for this target
                    continue

                # choose cheapest candidate
                best_k = min(cand_idxs, key=lambda idx: bids[idx].cost)
                selected_idxs.append(best_k)
                for tt in bids[best_k].targets:
                    assigned_targets.add(tt)

            # ---- 2d) Build per-robot target sets and evaluate ----

            robot_targets = build_robot_targets(selected_idxs)
            score = evaluate_assignment(robot_targets)

            if score < best_score:
                best_score = score
                # deep copy sets so they aren't mutated in later iterations
                best_robot_targets = {rid: set(s) for rid, s in robot_targets.items()}

        # ---- 3) Convert best_robot_targets into final assignments ----

        if best_robot_targets is not None:
            for rid, tset in best_robot_targets.items():
                if not tset:
                    continue
                S_set = frozenset(tset)
                order, _ = get_tour_for_set(rid, S_set)
                assignments[rid] = list(order)
        # else: if somehow everything failed, assignments stay empty

        dt = perf_counter() - t0
        self._update_stats(dt)
        return assignments


ALGORITHM = CAIsingFull()
