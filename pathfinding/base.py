# pathfinding/base.py
from typing import Protocol, List, Tuple, Optional
from env import WorldState

Pos = Tuple[int, int]


class PathfindingAlgorithm(Protocol):
    name: str
    # Optional timing stats (per algorithm implementation)
    total_runtime: float
    call_count: int
    last_runtime: float

    def plan(self, world: WorldState, start: Pos, goal: Pos) -> Optional[List[Pos]]:
        ...

    def reset_stats(self) -> None:
        ...
