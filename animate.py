# animate.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Set

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

from env import WorldState, Pos
from config import Config


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _compute_target_collection_steps(
    initial_targets: Set[Pos],
    history: List[List[Pos]],
) -> Dict[Pos, int | None]:
    """
    For each target cell, find the earliest step index where any robot stands on it.
    If never visited, value is None.
    """
    coll_step: Dict[Pos, int | None] = {t: None for t in initial_targets}
    if not history:
        return coll_step

    for step_idx, positions in enumerate(history):
        pos_set = set(positions)
        for t in initial_targets:
            if coll_step[t] is None and t in pos_set:
                coll_step[t] = step_idx
    return coll_step


def _compute_obstacle_discovery_steps(
    world: WorldState,
    history: List[List[Pos]],
    cfg: Config,
) -> Dict[Pos, int | None]:
    """
    For each obstacle that ended up in discovered_hidden, estimate
    the earliest step when a robot was within sensing range.
    If never "close enough" in history, value is None (shouldn't
    happen for actually discovered ones, but safe).
    """
    sense_radius = cfg.sense_radius
    discovered = set(world.discovered_hidden)
    disc_step: Dict[Pos, int | None] = {p: None for p in discovered}

    if not history or not discovered:
        return disc_step

    for step_idx, positions in enumerate(history):
        for rpos in positions:
            for obs in discovered:
                if disc_step[obs] is not None:
                    continue
                if _manhattan(rpos, obs) <= sense_radius:
                    disc_step[obs] = step_idx

    return disc_step


def animate_run(
    world: WorldState,
    cfg: Config,
    history: List[List[Pos]],
    initial_targets: Set[Pos],
    out_path: str | Path,
    fps: int = 10,
) -> None:
    """
    Build a GIF showing the robots moving, targets disappearing when
    collected, and hidden obstacles turning 'discovered' when robots
    get close.

    - world: final WorldState (after sim.run())
    - cfg: Config
    - history: sim.history  (list of [robot positions] per step)
    - initial_targets: positions of all targets at the start
    - out_path: path to save the GIF
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        print("No history to animate; skipping GIF.")
        return

    rows, cols = world.rows, world.cols

    # ----- Color palette (same family as draw_world) -----
    bgcolor          = np.array([0.96, 0.96, 0.96])  # light gray background
    known_color      = np.array([0.30, 0.30, 0.30])  # dark gray
    hidden_color     = np.array([1.00, 0.78, 0.37])  # soft amber
    discovered_color = np.array([0.84, 0.37, 0.30])  # muted red

    # ----- Precompute "when things happen" -----
    # Targets: when each target gets collected (first time any robot stands on it)
    coll_step = _compute_target_collection_steps(initial_targets, history)

    # Obstacles: when each discovered obstacle is first "seen" (robot within sense radius)
    disc_step = _compute_obstacle_discovery_steps(world, history, cfg)

    # ----- Base image: background + known + hidden (all yellow initially) -----
    base_img = np.zeros((rows, cols, 3), dtype=float)
    base_img[:, :, :] = bgcolor

    for (x, y) in world.known_obstacles:
        base_img[y, x] = known_color

    for (x, y) in world.hidden_obstacles:
        base_img[y, x] = hidden_color

    # We'll overlay discovered_color frame-by-frame using disc_step.

    # ----- Matplotlib setup -----
    fig, ax = plt.subplots(figsize=(cols / 2.0, rows / 2.0))
    im = ax.imshow(base_img, origin="lower", animated=True)

    # initial robot positions
    xs0 = [p[0] for p in history[0]]
    ys0 = [p[1] for p in history[0]]
    robots_scatter = ax.scatter(
        xs0,
        ys0,
        marker="o",
        s=80,
        c="#1f77b4",   # blue
        edgecolors="white",
        linewidths=0.7,
        label="robots",
        animated=True,
    )

    # initial targets (all still present at step 0)
    tx0 = [t[0] for t in initial_targets]
    ty0 = [t[1] for t in initial_targets]
    targets_scatter = ax.scatter(
        tx0,
        ty0,
        marker="x",
        s=80,
        c="#2ca02c",   # green
        linewidths=1.3,
        label="targets",
        animated=True,
    )

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # title + legend (similar to draw_world)
    fig.suptitle("Multi-robot gridworld (animation)", fontsize=16, y=0.98)

    from matplotlib.lines import Line2D
    spawn_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="#9467bd",
        markersize=10,
        label="spawn",
    )
    robot_handle = Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="#1f77b4",
        markeredgecolor="white",
        markersize=7,
        label="robots",
    )
    target_handle = Line2D(
        [0],
        [0],
        marker="x",
        color="#2ca02c",
        markersize=7,
        label="targets",
    )

    handles = [
        spawn_handle,
        robot_handle,
        target_handle,
        Patch(facecolor=known_color, edgecolor="black", label="known obstacle"),
        Patch(facecolor=hidden_color, edgecolor="black", label="hidden obstacle"),
        Patch(facecolor=discovered_color, edgecolor="black", label="discovered obstacle"),
    ]

    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(handles),
        fontsize=9,
        frameon=False,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    n_frames = len(history)

    # ----- init + update functions for FuncAnimation -----
    def init():
        im.set_array(base_img)
        robots_scatter.set_offsets(np.column_stack([xs0, ys0]))
        targets_scatter.set_offsets(np.column_stack([tx0, ty0]) if initial_targets else np.empty((0, 2)))
        return (im, robots_scatter, targets_scatter)

    def update(frame: int):
        # 1) update background for discovered obstacles
        img = base_img.copy()
        for p, s in disc_step.items():
            if s is not None and frame >= s:
                x, y = p
                img[y, x] = discovered_color
        im.set_array(img)

        # 2) update robot positions
        positions = history[frame]
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        robots_scatter.set_offsets(np.column_stack([xs, ys]))

        # 3) update targets (hide those collected before this frame)
        active_targets = [
            t for t, s in coll_step.items()
            if s is None or frame <= s
        ]
        if active_targets:
            tx = [t[0] for t in active_targets]
            ty = [t[1] for t in active_targets]
            targets_scatter.set_offsets(np.column_stack([tx, ty]))
        else:
            targets_scatter.set_offsets(np.empty((0, 2)))

        return (im, robots_scatter, targets_scatter)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        init_func=init,
        interval=1000 / fps,
        blit=True,
    )

    writer = animation.PillowWriter(fps=fps)
    ani.save(out_path, writer=writer)
    plt.close(fig)
    print(f"Saved animation GIF to {out_path}")
