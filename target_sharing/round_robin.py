# target_sharing/round_robin.py
from time import perf_counter
from typing import Dict, List, Set, Tuple

from env import WorldState, RobotState
from .base import TargetSharingAlgorithm

Pos = Tuple[int, int]


class RoundRobinSharing(TargetSharingAlgorithm):
    name = "RoundRobin"

    def __init__(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def reset_stats(self) -> None:
        self.total_runtime = 0.0
        self.call_count = 0
        self.last_runtime = 0.0

    def assign(
        self,
        world: WorldState,
        robots: List[RobotState],
        targets: Set[Pos],
    ) -> Dict[int, List[Pos]]:
        t0 = perf_counter()

        assignments: Dict[int, List[Pos]] = {r.rid: [] for r in robots}
        if not robots:
            dt = perf_counter() - t0
            self._update_stats(dt)
            return assignments

        t_list = list(targets)
        # (optionally: rng.shuffle(t_list) if you want randomness here)
        for i, t in enumerate(t_list):
            rid = robots[i % len(robots)].rid
            assignments[rid].append(t)

        dt = perf_counter() - t0
        self._update_stats(dt)
        return assignments

    def _update_stats(self, dt: float) -> None:
        self.last_runtime = dt
        self.total_runtime += dt
        self.call_count += 1


ALGORITHM = RoundRobinSharing()
