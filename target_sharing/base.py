# target_sharing/base.py
from typing import Protocol, Dict, List, Set, Tuple
from env import WorldState, RobotState

Pos = Tuple[int, int]


class TargetSharingAlgorithm(Protocol):
    name: str
    total_runtime: float
    call_count: int
    last_runtime: float

    def assign(
        self,
        world: WorldState,
        robots: List[RobotState],
        targets: Set[Pos],
    ) -> Dict[int, List[Pos]]:
        ...

    def reset_stats(self) -> None:
        ...
