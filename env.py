# env.py
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import List, Set, Tuple
import random

from config import Config

Pos = Tuple[int, int]  # (x, y) with x = col, y = row


@dataclass
class RobotState:
    """State of a single robot."""
    rid: int
    pos: Pos
    path: List[Pos] = field(default_factory=list)  # planned path (sequence of cells)


@dataclass
class WorldState:
    """
    Grid world with:
      - spawn: central point
      - robots: placed near spawn but not on the same cell
      - targets: placed on reachable free cells
      - known_obstacles: robots know these from t=0
      - hidden_obstacles: exist in the true world but initially unknown
      - discovered_hidden: subset of hidden_obstacles discovered via sensing

    True obstacles  = known_obstacles ∪ hidden_obstacles
    Visible obstacles (for planning) = known_obstacles ∪ discovered_hidden
    """
    rows: int
    cols: int
    spawn: Pos

    robots: List[RobotState]
    targets: Set[Pos]

    known_obstacles: Set[Pos]
    hidden_obstacles: Set[Pos]

    # subset of hidden_obstacles that robots have discovered
    discovered_hidden: Set[Pos] = field(default_factory=set)

    # ------------------------------------------------------------------ #
    # World generation                                                   #
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Config) -> "WorldState":
        rng = random.Random(cfg.seed)
        rows, cols = cfg.rows, cfg.cols

        # Spawn at the center
        spawn: Pos = (cols // 2, rows // 2)

        def in_bounds(p: Pos) -> bool:
            x, y = p
            return 0 <= x < cols and 0 <= y < rows

        def manhattan(a: Pos, b: Pos) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # BFS over TRUE world (known + hidden obstacles) to get reachable cells
        def bfs_reach_mask(known: Set[Pos], hidden: Set[Pos]) -> Set[Pos]:
            blocked = known | hidden
            reachable: Set[Pos] = set()
            visited: Set[Pos] = set()
            q = deque()

            if spawn in blocked:
                return reachable

            visited.add(spawn)
            q.append(spawn)

            while q:
                x, y = q.popleft()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    np = (nx, ny)
                    if not in_bounds(np) or np in visited:
                        continue
                    visited.add(np)
                    if np in blocked:
                        continue
                    reachable.add(np)
                    q.append(np)

            return reachable

        # 1) Sample obstacles until connectivity conditions are met
        known_obstacles: Set[Pos] = set()
        hidden_obstacles: Set[Pos] = set()
        reachable: Set[Pos] = set()

        for attempt in range(cfg.max_obstacle_tries):
            # all cells except spawn
            all_cells = [
                (x, y)
                for y in range(rows)
                for x in range(cols)
                if (x, y) != spawn
            ]
            rng.shuffle(all_cells)

            # maximum obstacles so we still have room for robots+targets
            max_obstacles = len(all_cells) - (cfg.n_robots + cfg.n_targets)
            max_obstacles = max(max_obstacles, 0)

            # sample known & hidden obstacles by density, but clamp to max_obstacles
            n_known = min(int(cfg.known_obstacle_density * len(all_cells)), max_obstacles)
            n_hidden = min(
                int(cfg.unknown_obstacle_density * len(all_cells)),
                max_obstacles - n_known,
            )

            known_obstacles = set(all_cells[:n_known])
            hidden_obstacles = set(all_cells[n_known : n_known + n_hidden])

            # BFS reachability in TRUE world
            reachable = bfs_reach_mask(known_obstacles, hidden_obstacles)

            # free cells in TRUE world (excluding spawn)
            total_free = sum(
                1
                for (x, y) in all_cells
                if (x, y) not in known_obstacles and (x, y) not in hidden_obstacles
            )
            reachable_free = len(reachable)
            ratio = reachable_free / total_free if total_free > 0 else 0.0

            if (
                total_free >= cfg.n_robots + cfg.n_targets
                and reachable_free >= cfg.n_robots + cfg.n_targets
                and ratio >= cfg.min_connected_ratio
            ):
                # good enough
                break
        else:
            raise RuntimeError(
                f"Failed to generate a connected world after {cfg.max_obstacle_tries} attempts."
            )

        # 2) Place robots and targets on reachable free cells

        reachable_free_cells = list(reachable)  # these are all free & reachable
        if len(reachable_free_cells) < cfg.n_robots + cfg.n_targets:
            raise RuntimeError(
                f"Not enough reachable cells ({len(reachable_free_cells)}) "
                f"for {cfg.n_robots} robots and {cfg.n_targets} targets."
            )

        # --- Robots: near spawn but not stacked ---
        # Find reachable cells sorted by distance to spawn
        reachable_sorted = sorted(reachable_free_cells, key=lambda p: manhattan(p, spawn))

        # Candidate pool: first up to 5 * n_robots cells nearest to spawn
        k = min(len(reachable_sorted), 5 * cfg.n_robots)
        candidate_robot_cells = reachable_sorted[:k]

        if len(candidate_robot_cells) < cfg.n_robots:
            # fallback: use all reachable cells
            candidate_robot_cells = reachable_free_cells

        # Randomly pick n_robots distinct positions from the candidate pool
        robot_cells = rng.sample(candidate_robot_cells, cfg.n_robots)

        robots = [RobotState(rid=i, pos=robot_cells[i]) for i in range(cfg.n_robots)]

        # --- Targets: random from remaining reachable cells ---
        remaining_cells = list(
            set(reachable_free_cells) - set(robot_cells)
        )

        if len(remaining_cells) < cfg.n_targets:
            raise RuntimeError("Not enough remaining reachable cells for targets.")

        target_cells = rng.sample(remaining_cells, cfg.n_targets)
        targets = set(target_cells)

        return cls(
            rows=rows,
            cols=cols,
            spawn=spawn,
            robots=robots,
            targets=targets,
            known_obstacles=known_obstacles,
            hidden_obstacles=hidden_obstacles,
        )

    # ------------------------------------------------------------------ #
    # Basic queries & helpers                                            #
    # ------------------------------------------------------------------ #
    def in_bounds(self, p: Pos) -> bool:
        x, y = p
        return 0 <= x < self.cols and 0 <= y < self.rows

    def true_obstacles(self) -> Set[Pos]:
        """All obstacles in the ground-truth world (known + hidden)."""
        return self.known_obstacles | self.hidden_obstacles

    def visible_obstacles(self) -> Set[Pos]:
        """
        Obstacles robots are aware of:
          - all known_obstacles
          - hidden_obstacles that have been discovered via sensing
        This set should be used by planning algorithms.
        """
        return self.known_obstacles | self.discovered_hidden

    def is_blocked_true(self, p: Pos) -> bool:
        return p in self.true_obstacles()

    def is_blocked_visible(self, p: Pos) -> bool:
        return p in self.visible_obstacles()

    def neighbors4_true(self, p: Pos) -> List[Pos]:
        """4-connected neighbors, blocked by TRUE obstacles."""
        x, y = p
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [q for q in candidates if self.in_bounds(q) and not self.is_blocked_true(q)]

    def neighbors4_visible(self, p: Pos) -> List[Pos]:
        """4-connected neighbors, blocked only by VISIBLE (known) obstacles."""
        x, y = p
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [q for q in candidates if self.in_bounds(q) and not self.is_blocked_visible(q)]

    # ------------------------------------------------------------------ #
    # Sensing / discovery of hidden obstacles                            #
    # ------------------------------------------------------------------ #
    def sense_new_obstacles(self, sense_radius: int) -> None:
        """
        Robots scan within Manhattan distance 'sense_radius'.
        Any hidden obstacle in that region becomes 'discovered' and thus
        added to the visible obstacle map (while still being part of the true world).
        """
        newly_discovered: Set[Pos] = set()

        for r in self.robots:
            rx, ry = r.pos
            for dy in range(-sense_radius, sense_radius + 1):
                for dx in range(-sense_radius, sense_radius + 1):
                    if abs(dx) + abs(dy) > sense_radius:
                        continue
                    p = (rx + dx, ry + dy)
                    if not self.in_bounds(p):
                        continue
                    if p in self.hidden_obstacles:
                        newly_discovered.add(p)

        if newly_discovered:
            self.discovered_hidden |= newly_discovered
