# tours/base.py
from __future__ import annotations

from typing import Protocol, List, Tuple
from env import WorldState, Pos


class TourAlgorithm(Protocol):
    """
    Interface for tour / TSP solvers.

    Given:
      - world
      - start position
      - list of target positions

    Return:
      - a visit order (permutation of targets)
      - the cost of the tour in steps (grid moves)
    """

    name: str

    # timing stats (seconds)
    total_runtime: float
    call_count: int
    last_runtime: float

    def solve(
        self,
        world: WorldState,
        start: Pos,
        targets: List[Pos],
    ) -> Tuple[List[Pos], float]:
        ...

    def reset_stats(self) -> None:
        ...
