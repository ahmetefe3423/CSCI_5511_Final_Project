# pathfinding/__init__.py
import importlib
import pkgutil
from typing import Dict
from .base import PathfindingAlgorithm

PATHFINDING_ALGOS: Dict[str, PathfindingAlgorithm] = {}


def load_algorithms() -> None:
    global PATHFINDING_ALGOS
    PATHFINDING_ALGOS = {}
    package = __name__
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        if name in {"base", "__init__"}:
            continue
        module = importlib.import_module(f"{package}.{name}")
        algo = getattr(module, "ALGORITHM", None)
        if algo is None:
            continue
        if algo.name in PATHFINDING_ALGOS:
            raise ValueError(f"Duplicate pathfinding name: {algo.name}")
        PATHFINDING_ALGOS[algo.name] = algo


load_algorithms()
