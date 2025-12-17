# config.py
from dataclasses import dataclass

@dataclass
class Config:
    rows: int = 20
    cols: int = 20

    n_robots: int = 4
    n_targets: int = 10

    # Densities are fractions of non-spawn cells
    known_obstacle_density: float = 0.10
    unknown_obstacle_density: float = 0.05

    sense_radius: int = 1  # robots can see 1 tile around them
    seed: int = 0

    min_connected_ratio: float = 0.5      # like old World.regen_connected
    max_obstacle_tries: int = 50          # avoid infinite loop

    # simulation horizon
    max_steps: int = 200
