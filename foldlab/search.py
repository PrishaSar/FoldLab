"""Exhaustive enumeration and simulated annealing on the 2D HP lattice.

Enumeration fixes the first bead at the origin and the first step along +x.
Translation and rotation do not change the energy, and every self-avoiding walk
has one rotation with that first step, so the search still finds a lowest-energy
fold. The number of conformations is c_{N-1}/4.

Lengths above EXHAUSTIVE_MAX_LENGTH are annealed. Annealing is deterministic
given the sequence, the seed, and the step count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np

from foldlab.model import DIRS, energy, parse_sequence

# Past this length the self-avoiding-walk count leaves interactive exhaustive search.
EXHAUSTIVE_MAX_LENGTH = 14

DEFAULT_ANNEAL_STEPS = 30_000
T_START = 2.0
T_END = 0.05
_MAX_TRIES = 20

# Non-identity symmetries of the square. Identity would leave the chain unmoved.
_PIVOTS = (
    lambda x, y: (-y, x),
    lambda x, y: (-x, -y),
    lambda x, y: (y, -x),
    lambda x, y: (-x, y),
    lambda x, y: (x, -y),
    lambda x, y: (y, x),
    lambda x, y: (-y, -x),
)


@dataclass
class Frame:
    step: int
    temperature: float
    energy: int
    best_energy: int
    coords: np.ndarray


@dataclass
class SearchResult:
    sequence: str
    coords: np.ndarray
    energy: int
    method: str
    seed: int | None = None
    steps: int | None = None
    n_conformations: int | None = None
    elapsed_s: float | None = None
    trajectory: list[Frame] = field(default_factory=list)


def _coords_array(xs, ys) -> np.ndarray:
    return np.column_stack((np.array(xs, dtype=int), np.array(ys, dtype=int)))


def _enumerate(sequence: str, visit) -> int:
    """Visit each symmetry-reduced conformation.

    ``visit(xs, ys, energy)`` sees the working coordinate lists. Copy them
    before returning if the values are needed later. Returns the number of
    conformations.
    """
    sequence = parse_sequence(sequence)
    n = len(sequence)
    if n >= 255:
        raise ValueError("exhaustive search supports chains shorter than 255 beads")
    is_h = [char == "H" for char in sequence]
    count = 0

    if n == 1:
        visit([0], [0], 0)
        return 1

    xs = [0] * n
    ys = [0] * n
    xs[1] = 1
    offset = n
    side = 2 * n + 1
    occ = bytearray(side * side)
    occ[offset * side + offset] = 1
    occ[(1 + offset) * side + offset] = 2

    def rec(i: int, current: int) -> None:
        nonlocal count
        if i == n:
            count += 1
            visit(xs, ys, current)
            return
        x = xs[i - 1]
        y = ys[i - 1]
        h_here = is_h[i]
        for dx, dy in DIRS:
            nx = x + dx
            ny = y + dy
            slot = (nx + offset) * side + (ny + offset)
            if occ[slot]:
                continue
            delta = 0
            if h_here:
                for ex, ey in DIRS:
                    if ex == -dx and ey == -dy:
                        continue
                    marked = occ[(nx + ex + offset) * side + (ny + ey + offset)]
                    if marked and is_h[marked - 1] and i - (marked - 1) > 1:
                        delta -= 1
            xs[i] = nx
            ys[i] = ny
            occ[slot] = i + 1
            rec(i + 1, current + delta)
            occ[slot] = 0

    rec(2, 0)
    return count


def exhaustive_search(sequence: str) -> SearchResult:
    """Lowest-energy fold by full enumeration. The chain may be any length."""
    sequence = parse_sequence(sequence)
    started = time.perf_counter()
    best_energy = 0
    best = np.column_stack(
        (np.arange(len(sequence), dtype=int), np.zeros(len(sequence), dtype=int))
    )

    def visit(xs, ys, current: int) -> None:
        nonlocal best_energy, best
        if current < best_energy:
            best_energy = current
            best = _coords_array(xs, ys)

    n_conformations = _enumerate(sequence, visit)
    return SearchResult(
        sequence=sequence,
        coords=best,
        energy=best_energy,
        method="exhaustive",
        n_conformations=n_conformations,
        elapsed_s=time.perf_counter() - started,
    )


def iter_scored_conformations(sequence: str):
    """Yield ``(coords, incremental_energy)`` for every symmetry-reduced fold."""
    found = []

    def visit(xs, ys, current: int) -> None:
        found.append((_coords_array(xs, ys), current))

    _enumerate(sequence, visit)
    return found


def _temperature(step_index: int, steps: int) -> float:
    if steps <= 1:
        return T_END
    frac = step_index / (steps - 1)
    return T_START * (T_END / T_START) ** frac


def _random_saw(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    for _ in range(10_000):
        coords = [(0, 0)]
        occupied = {(0, 0)}
        stuck = False
        for _bead in range(1, n):
            x, y = coords[-1]
            options = [
                (x + dx, y + dy) for dx, dy in DIRS if (x + dx, y + dy) not in occupied
            ]
            if not options:
                stuck = True
                break
            coords.append(options[int(rng.integers(0, len(options)))])
            occupied.add(coords[-1])
        if not stuck:
            return coords
    return [(i, 0) for i in range(n)]


def _end_flip(coords, rng: np.random.Generator):
    n = len(coords)
    if n < 2:
        return None
    end = 0 if int(rng.integers(0, 2)) == 0 else n - 1
    anchor = coords[1] if end == 0 else coords[n - 2]
    occupied = set(coords)
    options = [
        (anchor[0] + dx, anchor[1] + dy)
        for dx, dy in DIRS
        if (anchor[0] + dx, anchor[1] + dy) not in occupied
    ]
    if not options:
        return None
    moved = list(coords)
    moved[end] = options[int(rng.integers(0, len(options)))]
    return moved


def _corner_flip(coords, rng: np.random.Generator):
    n = len(coords)
    occupied = set(coords)
    options = []
    for i in range(1, n - 1):
        px, py = coords[i - 1]
        nx, ny = coords[i + 1]
        if abs(px - nx) == 1 and abs(py - ny) == 1:
            alt = (px + nx - coords[i][0], py + ny - coords[i][1])
            if alt not in occupied:
                options.append((i, alt))
    if not options:
        return None
    i, alt = options[int(rng.integers(0, len(options)))]
    moved = list(coords)
    moved[i] = alt
    return moved


def _pivot(coords, rng: np.random.Generator):
    n = len(coords)
    if n < 3:
        return None
    pivot = int(rng.integers(0, n - 1))
    transform = _PIVOTS[int(rng.integers(0, len(_PIVOTS)))]
    ox, oy = coords[pivot]
    moved = list(coords[: pivot + 1])
    seen = set(moved)
    for x, y in coords[pivot + 1 :]:
        nx, ny = transform(x - ox, y - oy)
        point = (ox + nx, oy + ny)
        if point in seen:
            return None
        seen.add(point)
        moved.append(point)
    return moved


def _propose(coords, rng: np.random.Generator):
    for _ in range(_MAX_TRIES):
        kind = int(rng.integers(0, 3))
        if kind == 0:
            proposal = _end_flip(coords, rng)
        elif kind == 1:
            proposal = _corner_flip(coords, rng)
        else:
            proposal = _pivot(coords, rng)
        if proposal is not None:
            return proposal
    return None


def simulated_annealing(
    sequence: str,
    seed: int,
    steps: int = DEFAULT_ANNEAL_STEPS,
) -> SearchResult:
    """Metropolis search. The returned fold is the best energy seen."""
    sequence = parse_sequence(sequence)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    current = _random_saw(len(sequence), rng)
    current_energy = energy(sequence, current)
    best = list(current)
    best_energy = current_energy
    record_every = max(1, steps // 80)
    trajectory = [
        Frame(0, T_START, current_energy, best_energy, np.array(current, dtype=int))
    ]

    for step_index in range(steps):
        temp = _temperature(step_index, steps)
        proposal = _propose(current, rng)
        if proposal is not None:
            proposed_energy = energy(sequence, proposal)
            delta = proposed_energy - current_energy
            if delta <= 0 or float(rng.random()) < math.exp(-delta / temp):
                current = proposal
                current_energy = proposed_energy
                if current_energy < best_energy:
                    best = list(current)
                    best_energy = current_energy
        done = step_index + 1
        if done % record_every == 0 or done == steps:
            trajectory.append(
                Frame(
                    done,
                    temp,
                    current_energy,
                    best_energy,
                    np.array(current, dtype=int),
                )
            )

    return SearchResult(
        sequence=sequence,
        coords=np.array(best, dtype=int),
        energy=best_energy,
        method="annealing",
        seed=int(seed),
        steps=int(steps),
        elapsed_s=time.perf_counter() - started,
        trajectory=trajectory,
    )
