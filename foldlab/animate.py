"""Draw an HP fold or an annealing trajectory and save a GIF.

Hydrophobic beads are rust, polar beads are blue, and non-sequential H–H
contacts are dashed. If Pillow cannot encode a GIF, the same clip is written
as an MP4 with the stem kept and the suffix changed.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from foldlab.model import contacts, energy

H_COLOR = "#c2410c"
P_COLOR = "#0369a1"
BACKBONE = "#292524"
CONTACT = "#9a3412"
PAPER = "#fffbf5"
INK = "#1c1917"
MUTED = "#57534e"


def _gif_supported() -> bool:
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def _limits(coords):
    xs = coords[:, 0]
    ys = coords[:, 1]
    xmin, xmax = int(xs.min()) - 1, int(xs.max()) + 1
    ymin, ymax = int(ys.min()) - 1, int(ys.max()) + 1
    if xmax - xmin < 6:
        mid = (xmin + xmax) / 2
        xmin, xmax = int(mid - 3), int(mid + 3)
    if ymax - ymin < 6:
        mid = (ymin + ymax) / 2
        ymin, ymax = int(mid - 3), int(mid + 3)
    return xmin, xmax, ymin, ymax


def _draw(ax, sequence: str, coords, title: str, show_contacts: bool) -> None:
    coords = np.asarray(coords, dtype=int)
    ax.clear()
    ax.set_facecolor(PAPER)
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=BACKBONE,
        linewidth=2.4,
        zorder=1,
        solid_capstyle="round",
    )
    if show_contacts:
        for i, j in contacts(sequence, coords):
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=CONTACT,
                linewidth=1.6,
                linestyle=(0, (2.2, 2.2)),
                zorder=2,
            )
    for i, char in enumerate(sequence[: len(coords)]):
        color = H_COLOR if char == "H" else P_COLOR
        ax.scatter(
            [coords[i, 0]],
            [coords[i, 1]],
            s=320,
            c=color,
            zorder=3,
            edgecolors="white",
            linewidths=1.0,
        )
        ax.text(
            coords[i, 0],
            coords[i, 1],
            str(i),
            color="white",
            ha="center",
            va="center",
            fontsize=7.5,
            zorder=4,
        )
    xmin, xmax, ymin, ymax = _limits(coords)
    pad = 0.4
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xticks(range(xmin, xmax + 1))
    ax.set_yticks(range(ymin, ymax + 1))
    ax.grid(True, color="#e7e5e4", linewidth=0.7, zorder=0)
    ax.tick_params(labelbottom=False, labelleft=False, length=0)
    ax.set_aspect("equal")
    ax.set_title(title, color=INK, fontsize=12, pad=10)
    ax.set_xlabel(
        "rust  H     blue  P     dashed  H–H contact",
        color=MUTED,
        fontsize=9,
    )
    for spine in ax.spines.values():
        spine.set_color("#e7e5e4")


def _write(fig, path: Path, draw_frames, fps: float) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    from matplotlib.animation import FFMpegWriter, PillowWriter

    def run(writer, dest: Path) -> None:
        with writer.saving(fig, str(dest), dpi=fig.dpi):
            for draw in draw_frames:
                draw()
                writer.grab_frame()

    if path.suffix.lower() == ".gif" and _gif_supported():
        run(PillowWriter(fps=fps), path)
        return path
    dest = path.with_suffix(".mp4") if path.suffix.lower() == ".gif" else path
    run(FFMpegWriter(fps=fps), dest)
    return dest


def _figure():
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    fig.patch.set_facecolor(PAPER)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.1)
    return fig, ax


def save_fold_gif(path, sequence: str, coords, title: str = "HP fold", fps: int = 8) -> Path:
    """Draw the final fold one bead at a time, then hold with H–H contacts."""
    coords = np.asarray(coords, dtype=int)
    fig, ax = _figure()
    n = len(sequence)
    draws = []
    for k in range(1, n + 1):

        def draw(k=k):
            _draw(ax, sequence, coords[:k], f"{title}    beads {k}/{n}", False)

        draws.append(draw)
    score = energy(sequence, coords)
    hold = f"{title}    best fold    E={score}"
    for _ in range(8):

        def draw(hold=hold):
            _draw(ax, sequence, coords, hold, True)

        draws.append(draw)
    try:
        return _write(fig, Path(path), draws, fps)
    finally:
        plt.close(fig)


def save_trajectory_gif(path, result, title: str = "HP fold", fps: float = 8) -> Path:
    """Animate the annealing trajectory, then hold on the best fold."""
    if not result.trajectory:
        raise ValueError("trajectory is empty")
    fig, ax = _figure()
    sequence = result.sequence
    draws = []
    for frame in result.trajectory:

        def draw(frame=frame):
            _draw(
                ax,
                sequence,
                frame.coords,
                (
                    f"{title}    step {frame.step}    T={frame.temperature:.2f}"
                    f"    E={frame.energy}    best={frame.best_energy}"
                ),
                True,
            )

        draws.append(draw)
    hold = f"{title}    best fold    E={result.energy}"
    for _ in range(10):

        def draw(hold=hold):
            _draw(ax, sequence, result.coords, hold, True)

        draws.append(draw)
    try:
        return _write(fig, Path(path), draws, fps)
    finally:
        plt.close(fig)
