# tours/__init__.py
import importlib
import pkgutil
from typing import Dict
from .base import TourAlgorithm

TOUR_ALGOS: Dict[str, TourAlgorithm] = {}


def load_algorithms() -> None:
    global TOUR_ALGOS
    TOUR_ALGOS = {}
    package = __name__
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        if name in {"base", "__init__"}:
            continue
        module = importlib.import_module(f"{package}.{name}")
        algo = getattr(module, "ALGORITHM", None)
        if algo is None:
            continue
        if algo.name in TOUR_ALGOS:
            raise ValueError(f"Duplicate tour algorithm name: {algo.name}")
        TOUR_ALGOS[algo.name] = algo


load_algorithms()
