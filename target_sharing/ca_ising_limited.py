# target_sharing/ca_ising_limited.py
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

    IMPORTANT: 'targets' is stored in the visiting order of the tour
    computed by the injected TourAlgorithm for this robot and bundle.
    """
    rid: int                  # robot id
    targets: Tuple[Pos, ...]  # bundle of targets (tour order for this bundle)
    cost: float               # tour cost to cover this bundle from r.pos


class CAIsingLimited(TargetSharingAlgorithm):
    """
    Combinatorial Auction with Ising-based winner determination
    (hardware-limited / integer-QUBO version).

    Differences from CA_IsingFull:

      - QUBO coefficients are limited to INTEGER range [-7, +7].
      - We simulate a hardware device with:
            * spin_limit      = 49 spins
            * anneal_time_us  = 50 microseconds per hardware call
            * power_mW        = 24 mW
        For a QUBO with Nvars variables (bundles), we estimate:
            hardware_calls = max(1, round(Nvars / spin_limit))

        and accumulate:
            total_hw_anneal_time  (seconds)
            total_hw_energy       (Joules).

      - We construct multiple integer QUBO mappings (cost encodings into 1..6)
        and pick the assignment whose *true* objective

            F = sum_r PC(r, T(r)) + μ * (#unassigned targets)

        is minimal, reusing the same TourAlgorithm and tour cache.
    """

    name = "CA_IsingLimited"

    # hardware parameters (conceptual)
    HW_SPIN_LIMIT: int = 49
    HW_ANNEAL_TIME_S: float = 50e-6   # 50 microseconds per hardware call
    HW_POWER_W: float = 24e-3         # 24 mW

    def __init__(self, max_bundle_size: int = 2, num_reads: int = 2) -> None:
        # timing stats for the auction itself
        self.total_runtime: float = 0.0
        self.call_count: int = 0
        self.last_runtime: float = 0.0

        # Tour algorithm injected by Simulator
        self.tour_algo: Optional[TourAlgorithm] = None

        # bundle parameters
        self.max_bundle_size: int = max_bundle_size
        self.num_reads: int = num_reads

        # Tabu sampler instance (reused across calls)
        self._sampler: Optional[Any] = None

        # hardware metrics (aggregated over calls)
        self.total_hw_anneal_time: float = 0.0   # seconds
        self.total_hw_energy: float = 0.0        # Joules
        self.last_hw_anneal_time: float = 0.0    # seconds
        self.last_hw_calls: int = 0

        # penalty factor for unassigned targets in final scoring
        self.unassigned_penalty_factor: float = 10.0

    # ---- stats API ----

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

        self.total_hw_anneal_time = 0.0
        self.total_hw_energy = 0.0
        self.last_hw_anneal_time = 0.0
        self.last_hw_calls = 0

    def _update_stats(self, dt: float, nvars: int) -> None:
        # wall-clock runtime
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1

        # simulated hardware usage
        ratio = nvars / float(self.HW_SPIN_LIMIT) if self.HW_SPIN_LIMIT > 0 else 0.0
        hw_calls = max(1, round(ratio)) if nvars > 0 else 0

        hw_time = hw_calls * self.HW_ANNEAL_TIME_S  # seconds
        hw_energy = hw_time * self.HW_POWER_W       # Joules

        self.last_hw_calls = hw_calls
        self.last_hw_anneal_time = hw_time

        self.total_hw_anneal_time += hw_time
        self.total_hw_energy += hw_energy

    # ---- dependency injection ----

    def set_tour_algo(self, algo: TourAlgorithm) -> None:
        """
        Simulator calls this so CA_IsingLimited can use the chosen tour solver
        (NearestNeighbor, CheapestInsertion, ExactBruteForce, IsingFull, etc.)
        to evaluate bundle costs PC(r, S).
        """
        self.tour_algo = algo

    # ---- QUBO helpers ----

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
                "CA_IsingLimited.tour_algo is not set. "
                "Simulator must inject a TourAlgorithm via set_tour_algo()."
            )

        if TabuSampler is None:
            raise RuntimeError(
                "CA_IsingLimited requires the 'dwave-tabu' package "
                "(imported as module 'tabu'). "
                "Install it with `pip install dwave-tabu` or choose a "
                "different target-sharing algorithm in main.py."
            )

        assignments: Dict[int, List[Pos]] = {r.rid: [] for r in robots}

        if not robots or not targets:
            dt = perf_counter() - t0
            self._update_stats(dt, nvars=0)
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
                bids.append(BundleBid(rid=rid, targets=tuple(order), cost=cost))

            # size-2 bundles
            if self.max_bundle_size >= 2 and len(target_list) >= 2:
                for i in range(len(target_list)):
                    for j in range(i + 1, len(target_list)):
                        S = (target_list[i], target_list[j])
                        order, cost = get_tour_for_bundle(rid, S)
                        if cost == inf:
                            continue
                        bids.append(
                            BundleBid(
                                rid=rid,
                                targets=tuple(order),
                                cost=cost,
                            )
                        )

        if not bids:
            dt = perf_counter() - t0
            self._update_stats(dt, nvars=0)
            return assignments

        num_bids = len(bids)
        nvars = num_bids  # one spin per bundle

        # ---- 2) Build separate float QUBOs: cost part and constraint part ----

        # cost part: linear terms cost_k * z_k
        Q_cost: Dict[Tuple[int, int], float] = {}

        def add_cost(u: int, v: int, coeff: float) -> None:
            if coeff == 0.0:
                return
            if u > v:
                u, v = v, u
            Q_cost[(u, v)] = Q_cost.get((u, v), 0.0) + coeff

        # constraint part: from (sum z_k - 1)^2 expansion
        Q_cons: Dict[Tuple[int, int], float] = {}

        def add_cons(u: int, v: int, coeff: float) -> None:
            if coeff == 0.0:
                return
            if u > v:
                u, v = v, u
            Q_cons[(u, v)] = Q_cons.get((u, v), 0.0) + coeff

        # (a) cost terms
        for k, b in enumerate(bids):
            add_cost(k, k, b.cost)

        # (b) constraint terms: for each target, penalize (sum_{b: t in b} z_b - 1)^2
        #    (sum - 1)^2 = -sum z_k + 2 sum_{k<l} z_k z_l + const
        lambda_pen = 1.0  # base magnitude for constraints (we'll saturate to ±7 later)

        target_to_bids: Dict[Pos, List[int]] = {t: [] for t in targets}
        for k, b in enumerate(bids):
            for t in b.targets:
                if t in target_to_bids:
                    target_to_bids[t].append(k)

        for t, bid_idxs in target_to_bids.items():
            if not bid_idxs:
                continue
            # linear: -lambda * z_k
            for k in bid_idxs:
                add_cons(k, k, -lambda_pen)
            # quadratic: 2 * lambda * z_k z_l
            for i in range(len(bid_idxs)):
                for j in range(i + 1, len(bid_idxs)):
                    k_i = bid_idxs[i]
                    k_j = bid_idxs[j]
                    add_cons(k_i, k_j, 2.0 * lambda_pen)

        # ---- 3) Build integer QUBOs with limited coefficients [-7, 7] ----
        #      Strategy: constraints get ±7 (hard), costs get 1..6 (soft).
        #      Try multiple cost mappings and later pick the assignment
        #      with best true objective.

        cost_vals = [abs(v) for v in Q_cost.values() if abs(v) > 0.0]
        cost_min = min(cost_vals) if cost_vals else 0.0
        cost_max = max(cost_vals) if cost_vals else 0.0
        base_cost_scale = cost_max if cost_max > 0.0 else 1.0

        def build_qubos_int() -> List[Dict[Tuple[int, int], int]]:
            """
            Return a list of integer QUBOs, each implementing a different
            mapping strategy for cost terms. Constraints always use ±7.
            """
            qubos: List[Dict[Tuple[int, int], int]] = []

            # Helper to add constraint terms (always ±7)
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

            # Strategy 1: linear scaling so max cost -> 6
            if cost_vals and cost_max > 0.0:
                Q1: Dict[Tuple[int, int], int] = {}
                add_cons_int(Q1)

                for (u, v), coeff in Q_cost.items():
                    if coeff == 0.0:
                        continue
                    level = int(round(6.0 * coeff / cost_max))  # 0..6
                    if level <= 0:
                        level = 1  # ensure non-zero costs stay visible
                    if level > 6:
                        level = 6
                    if u > v:
                        uu, vv = v, u
                    else:
                        uu, vv = u, v
                    Q1[(uu, vv)] = Q1.get((uu, vv), 0) + level

                qubos.append(Q1)

            # Strategy 2: rank-based / normalized mapping to 1..6
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

            # If there are no cost terms, we still might have constraints.
            if not cost_vals and Q_cons:
                Q_only: Dict[Tuple[int, int], int] = {}
                add_cons_int(Q_only)
                qubos.append(Q_only)

            return qubos

        qubos_int = build_qubos_int()

        if self._sampler is None:
            self._sampler = TabuSampler()

        # ---- Helpers for decoding and evaluating assignments ----

        def build_robot_targets(selected_idxs: List[int]) -> Dict[int, Set[Pos]]:
            """Turn selected bundles into per-robot sets of targets."""
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

        def evaluate_assignment(robot_targets: Dict[int, Set[Pos]]) -> float:
            """Evaluate an assignment using true tour costs + unassigned penalty."""
            assigned_union: Set[Pos] = set()
            total_cost: float = 0.0

            for rid, tset in robot_targets.items():
                if not tset:
                    continue
                assigned_union.update(tset)
                S_set = frozenset(tset)
                order, cost = get_tour_for_set(rid, S_set)
                if cost == inf:
                    return inf
                total_cost += cost

            num_unassigned = len(targets - assigned_union)
            unassigned_penalty = self.unassigned_penalty_factor * base_cost_scale
            total_cost += unassigned_penalty * num_unassigned
            return total_cost

        # ---- 4) Run Tabu for each integer QUBO and pick the best assignment ----

        best_score: float = inf
        best_robot_targets: Optional[Dict[int, Set[Pos]]] = None

        for Q_int in qubos_int:
            if not Q_int:
                continue

            sampleset = self._sampler.sample_qubo(Q_int, num_reads=self.num_reads)
            best_sample = sampleset.first
            sample: Dict[int, int] = best_sample.sample  # k -> 0/1

            # Decode this sample to selected bundles
            chosen_idxs = [k for k in range(num_bids) if sample.get(k, 0) == 1]

            assigned_targets: Set[Pos] = set()
            selected_idxs: List[int] = []

            # First pass: keep non-conflicting bundles, cheapest first
            for k in sorted(chosen_idxs, key=lambda idx: bids[idx].cost):
                b = bids[k]
                if any(t in assigned_targets for t in b.targets):
                    continue
                selected_idxs.append(k)
                for t in b.targets:
                    assigned_targets.add(t)

            # Second pass: greedily assign any remaining targets
            unassigned = set(targets) - assigned_targets
            for t in unassigned:
                cand_idxs: List[int] = []
                for k, b in enumerate(bids):
                    if t not in b.targets:
                        continue
                    if any(tt in assigned_targets for tt in b.targets):
                        continue
                    cand_idxs.append(k)

                if not cand_idxs:
                    continue

                best_k = min(cand_idxs, key=lambda idx: bids[idx].cost)
                selected_idxs.append(best_k)
                for tt in bids[best_k].targets:
                    assigned_targets.add(tt)

            # Build robot_targets and evaluate with true objective
            robot_targets = build_robot_targets(selected_idxs)
            score = evaluate_assignment(robot_targets)

            if score < best_score:
                best_score = score
                best_robot_targets = {rid: set(s) for rid, s in robot_targets.items()}

        # Fallback: if all QUBO runs failed somehow, use simple greedy
        if best_robot_targets is None:
            assigned_targets: Set[Pos] = set()
            greedy_idxs: List[int] = []
            for k in sorted(range(num_bids), key=lambda idx: bids[idx].cost):
                b = bids[k]
                if any(t in assigned_targets for t in b.targets):
                    continue
                greedy_idxs.append(k)
                for t in b.targets:
                    assigned_targets.add(t)
            best_robot_targets = build_robot_targets(greedy_idxs)

        # ---- 5) Convert best_robot_targets into final assignments in tour order ----

        for rid, tset in best_robot_targets.items():
            if not tset:
                continue
            S_set = frozenset(tset)
            order, _ = get_tour_for_set(rid, S_set)
            assignments[rid] = list(order)

        dt = perf_counter() - t0
        self._update_stats(dt, nvars=nvars)
        return assignments


ALGORITHM = CAIsingLimited()
