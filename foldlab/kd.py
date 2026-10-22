"""Kyte–Doolittle hydropathy and the H/P class used by the lattice model.

Scores are the 1982 scale. A residue is H when its score is greater than 0
and P otherwise. No standard amino acid scores exactly 0.
"""

from __future__ import annotations

import numpy as np

# J. Kyte and R. F. Doolittle, J. Mol. Biol. 157:105 (1982).
KYTE_DOOLITTLE = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# Greater than this score is hydrophobic (H). Otherwise the residue is P.
HP_THRESHOLD = 0.0
PROFILE_WINDOW = 9


def kd_score(residue: str) -> float:
    """Score a one-letter code or a standard three-letter name."""
    text = residue.strip().upper()
    if len(text) == 3:
        if text not in AA3_TO_1:
            raise ValueError(f"unknown residue {residue!r}")
        text = AA3_TO_1[text]
    if text not in KYTE_DOOLITTLE:
        raise ValueError(f"unknown residue {residue!r}")
    return KYTE_DOOLITTLE[text]


def hp_class(residue: str) -> str:
    """H when the Kyte–Doolittle score is above the threshold, else P."""
    return "H" if kd_score(residue) > HP_THRESHOLD else "P"


def window_profile(scores, window: int = PROFILE_WINDOW) -> np.ndarray:
    """Centered sliding mean. Ends use the residues that exist."""
    values = np.asarray(scores, dtype=float)
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    half = window // 2
    out = np.empty(len(values), dtype=float)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out[i] = values[lo:hi].mean()
    return out
