# viz.py
from __future__ import annotations
from typing import Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from env import WorldState
from config import Config

Pos = Tuple[int, int]


def draw_world(world: WorldState, cfg: Config, out_path: str | Path) -> None:
    """
    Draw a snapshot of the world:
      - free cells: light background
      - known obstacles: dark gray
      - hidden (unknown) obstacles: soft amber
      - discovered hidden obstacles: muted red
      - spawn: purple star
      - robots: blue circles
      - targets: green crosses
    """
    rows, cols = world.rows, world.cols
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Color palette (RGB in 0–1) ---
    bgcolor          = np.array([0.96, 0.96, 0.96])  # light gray background
    known_color      = np.array([0.30, 0.30, 0.30])  # dark gray
    hidden_color     = np.array([1.00, 0.78, 0.37])  # soft amber
    discovered_color = np.array([0.84, 0.37, 0.30])  # muted red

    # base: all light background
    img = np.zeros((rows, cols, 3), dtype=float)
    img[:, :, :] = bgcolor

    # known obstacles
    for (x, y) in world.known_obstacles:
        img[y, x] = known_color

    # hidden obstacles (unknown to robots)
    for (x, y) in world.hidden_obstacles:
        img[y, x] = hidden_color

    # discovered hidden obstacles (now known to robots)
    for (x, y) in world.discovered_hidden:
        img[y, x] = discovered_color

    fig, ax = plt.subplots(figsize=(cols / 2.0, rows / 2.0))
    ax.imshow(img, origin="lower")

    # Grid lines (subtle)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="0.85", linestyle="-", linewidth=0.4)

    # Spawn
    sx, sy = world.spawn
    h_spawn = ax.scatter(
        [sx],
        [sy],
        marker="*",
        s=150,
        c="#9467bd",          # purple
        edgecolors="white",
        linewidths=1.0,
        label="spawn",
    )

    # Robots
    h_robots = None
    if world.robots:
        rx = [r.pos[0] for r in world.robots]
        ry = [r.pos[1] for r in world.robots]
        h_robots = ax.scatter(
            rx,
            ry,
            marker="o",
            s=80,
            c="#1f77b4",       # blue
            edgecolors="white",
            linewidths=0.7,
            label="robots",
        )

    # Targets
    h_targets = None
    if world.targets:
        tx = [p[0] for p in world.targets]
        ty = [p[1] for p in world.targets]
        h_targets = ax.scatter(
            tx,
            ty,
            marker="x",
            s=80,
            c="#2ca02c",       # green
            linewidths=1.5,
            label="targets",
        )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Title (larger, figure-level) ---
    fig.suptitle("Multi-robot gridworld", fontsize=18, y=0.98)

    # --- Legend: single line, below the title ---
    handles = []

    handles.append(h_spawn)
    if h_robots is not None:
        handles.append(h_robots)
    if h_targets is not None:
        handles.append(h_targets)

    # obstacle legend entries (color patches)
    handles.extend(
        [
            Patch(facecolor=known_color, edgecolor="black", label="known obstacle"),
            Patch(facecolor=hidden_color, edgecolor="black", label="hidden obstacle"),
            Patch(facecolor=discovered_color, edgecolor="black", label="discovered obstacle"),
        ]
    )

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),  # just below the title
        ncol=len(handles),            # single line
        fontsize=9,
        frameon=False,
    )

    # Leave space at top for title + legend
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
