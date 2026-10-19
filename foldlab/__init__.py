"""FoldLab Phase 1: 2D hydrophobic–polar lattice folds."""

from foldlab.model import contacts, energy, is_self_avoiding, parse_sequence
from foldlab.search import (
    EXHAUSTIVE_MAX_LENGTH,
    SearchResult,
    exhaustive_search,
    simulated_annealing,
)

__all__ = [
    "EXHAUSTIVE_MAX_LENGTH",
    "SearchResult",
    "contacts",
    "energy",
    "exhaustive_search",
    "is_self_avoiding",
    "parse_sequence",
    "simulated_annealing",
]
