import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Left room bounds (room_length=15, wall_thickness=0.2, origin at left wall)
_ROOM_X = (-15.1, -0.1)
_ROOM_Z = (-7.5, 7.5)
_MIRROR_Z = (1.0, 3.0)

_CUBE_COLORS = {"red": "red", "green": "limegreen", "blue": "royalblue"}


def save_trajectory_heatmap(
    run_id: int,
    trajectory: list,
    agent_color: str,
    object_positions_2d: dict,
    output_dir: str,
) -> str:
    """Save a 2D trajectory plot of a single run as a PNG file.

    Args:
        run_id: Run identifier used in the filename.
        trajectory: List of (x, z) tuples, one per step.
        agent_color: The robot's color label (e.g. "red").
        object_positions_2d: Dict mapping color name to (x, z) position.
        output_dir: Directory to write the PNG into.

    Returns:
        Absolute path to the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Room rectangle
    w = _ROOM_X[1] - _ROOM_X[0]
    h = _ROOM_Z[1] - _ROOM_Z[0]
    ax.add_patch(patches.Rectangle(
        (_ROOM_X[0], _ROOM_Z[0]), w, h,
        lw=2, edgecolor="black", facecolor="#f8f8f8",
    ))

    # Mirror gap on the right wall (x ≈ -0.1)
    ax.add_patch(patches.Rectangle(
        (-0.35, _MIRROR_Z[0]), 0.25, _MIRROR_Z[1] - _MIRROR_Z[0],
        lw=1, edgecolor="deepskyblue", facecolor="deepskyblue", alpha=0.5,
        label="mirror",
    ))

    # Cube positions
    for cname, (cx, cz) in object_positions_2d.items():
        ax.scatter(cx, cz, c=_CUBE_COLORS.get(cname, "gray"), s=220, marker="s",
                   zorder=5, edgecolors="black", linewidths=0.8, label=f"{cname} cube")

    # Trajectory
    if trajectory:
        xs, zs = zip(*trajectory)
        n = len(xs)
        sc = ax.scatter(xs, zs, c=range(n), cmap="plasma", s=40, zorder=4, label="steps")
        ax.plot(xs, zs, color="gray", lw=0.8, alpha=0.4, zorder=3)
        ax.scatter(xs[0], zs[0], c="green", s=120, marker="^", zorder=6, label="start")
        ax.scatter(xs[-1], zs[-1], c="red", s=120, marker="v", zorder=6, label="end")
        plt.colorbar(sc, ax=ax, label="step index")

    ax.set_xlim(_ROOM_X[0] - 0.5, _ROOM_X[1] + 0.5)
    ax.set_ylim(_ROOM_Z[0] - 0.5, _ROOM_Z[1] + 0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal")
    ax.set_title(f"Run {run_id} trajectory  (agent={agent_color})")
    ax.legend(loc="upper left", fontsize=8)

    out = Path(output_dir) / f"run{run_id:02d}_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(out)
