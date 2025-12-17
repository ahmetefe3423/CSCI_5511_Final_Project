# target_sharing/__init__.py
import importlib
import pkgutil
from typing import Dict
from .base import TargetSharingAlgorithm

TARGET_SHARING_ALGOS: Dict[str, TargetSharingAlgorithm] = {}


def load_algorithms() -> None:
    global TARGET_SHARING_ALGOS
    TARGET_SHARING_ALGOS = {}
    package = __name__
    for info in pkgutil.iter_modules(__path__):
        name = info.name
        if name in {"base", "__init__"}:
            continue
        module = importlib.import_module(f"{package}.{name}")
        algo = getattr(module, "ALGORITHM", None)
        if algo is None:
            continue
        if algo.name in TARGET_SHARING_ALGOS:
            raise ValueError(f"Duplicate target-sharing name: {algo.name}")
        TARGET_SHARING_ALGOS[algo.name] = algo


load_algorithms()
