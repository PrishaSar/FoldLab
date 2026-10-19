"""HP chain on the 2D square lattice.

Energy is −1 per non-sequential H–H contact: orthogonal neighbors on the grid
whose sequence indices differ by more than 1. Chain bonds do not count, and a
pair is counted once.
"""

from __future__ import annotations

import numpy as np

DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def parse_sequence(sequence: str) -> str:
    text = "".join(str(sequence).split()).upper()
    if not text or any(char not in "HP" for char in text):
        raise ValueError("sequence must be a non-empty string of H and P")
    return text


def as_coords(coords) -> np.ndarray:
    arr = np.asarray(coords)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")
    if arr.size and not np.all(np.mod(arr, 1) == 0):
        raise ValueError("coords must be integer lattice sites")
    return arr.astype(int, copy=False)


def is_self_avoiding(coords) -> bool:
    """True when beads occupy distinct sites and successive beads are one step apart."""
    arr = as_coords(coords)
    if len(arr) == 0:
        return False
    if len({(int(x), int(y)) for x, y in arr}) != len(arr):
        return False
    if len(arr) == 1:
        return True
    steps = np.abs(np.diff(arr, axis=0)).sum(axis=1)
    return bool(np.all(steps == 1))


def contacts(sequence: str, coords) -> list[tuple[int, int]]:
    """Non-sequential H–H contacts as index pairs (i, j) with i < j."""
    sequence = parse_sequence(sequence)
    arr = as_coords(coords)
    if len(arr) != len(sequence):
        raise ValueError("coords and sequence must have the same length")
    if not is_self_avoiding(arr):
        raise ValueError("coords are not a self-avoiding lattice chain")
    index_at = {(int(arr[i, 0]), int(arr[i, 1])): i for i in range(len(arr))}
    found: list[tuple[int, int]] = []
    for i, char in enumerate(sequence):
        if char != "H":
            continue
        x, y = int(arr[i, 0]), int(arr[i, 1])
        for dx, dy in DIRS:
            j = index_at.get((x + dx, y + dy))
            if j is not None and j > i + 1 and sequence[j] == "H":
                found.append((i, j))
    return found


def energy(sequence: str, coords) -> int:
    """Lattice energy. Lower means more H–H contacts. A straight chain scores 0."""
    return -len(contacts(sequence, coords))
