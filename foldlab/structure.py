"""Published PDB structures: Cα contacts and HELIX/SHEET labels.

Contacts are Cα pairs in one chain at most 8 Å apart, with sequence numbers
at least 4 apart. Secondary structure is read from the file's own HELIX and
SHEET records. A residue listed as both is recorded as helix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np
from Bio.PDB import PDBParser, is_aa

from foldlab.kd import AA3_TO_1

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RCSB = "https://files.rcsb.org/download/{pdb_id}.pdb"

CONTACT_CUTOFF = 8.0
MIN_SEPARATION = 4


@dataclass(frozen=True)
class Residue:
    chain: str
    resseq: int
    name3: str
    name1: str
    ca: np.ndarray
    ss: str


def cached_pdb(pdb_id: str) -> Path:
    """Return the cached PDB file, downloading it once if the cache is empty."""
    path = DATA_DIR / f"{pdb_id.lower()}.pdb"
    if path.is_file() and path.stat().st_size > 0:
        return path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(RCSB.format(pdb_id=pdb_id.upper()), path)
    return path


def _span(start: int, end: int) -> range:
    lo, hi = (start, end) if start <= end else (end, start)
    return range(lo, hi + 1)


def secondary_structure(pdb_text: str) -> dict[tuple[str, int], str]:
    """Map (chain, residue number) to 'H' or 'E' from HELIX and SHEET records."""
    assigned: dict[tuple[str, int], str] = {}
    for raw in pdb_text.splitlines():
        line = raw.ljust(80)
        if line.startswith("SHEET"):
            chain = line[21]
            start = int(line[22:26])
            end_chain = line[32]
            end = int(line[33:37])
            if chain != end_chain:
                continue
            for resseq in _span(start, end):
                assigned.setdefault((chain, resseq), "E")
        elif line.startswith("HELIX"):
            chain = line[19]
            start = int(line[21:25])
            end_chain = line[31]
            end = int(line[33:37])
            if chain != end_chain:
                continue
            for resseq in _span(start, end):
                assigned[(chain, resseq)] = "H"
    return assigned


def load_chains(path: Path) -> dict[str, list[Residue]]:
    """Standard amino acids with a Cα, grouped by chain id, first model only."""
    text = Path(path).read_text()
    ss = secondary_structure(text)
    parser = PDBParser(QUIET=True)
    model = next(parser.get_structure(Path(path).stem, str(path)).get_models())
    chains: dict[str, list[Residue]] = {}
    for chain in model:
        residues: list[Residue] = []
        for res in chain:
            if res.id[0] != " " or not is_aa(res, standard=True):
                continue
            name3 = res.get_resname()
            if name3 not in AA3_TO_1 or "CA" not in res:
                continue
            resseq = int(res.id[1])
            residues.append(
                Residue(
                    chain=chain.id,
                    resseq=resseq,
                    name3=name3,
                    name1=AA3_TO_1[name3],
                    ca=np.array(res["CA"].get_coord(), dtype=float),
                    ss=ss.get((chain.id, resseq), "C"),
                )
            )
        if residues:
            chains[chain.id] = residues
    return chains


def ca_contacts(
    residues: list[Residue],
    cutoff: float = CONTACT_CUTOFF,
    min_separation: int = MIN_SEPARATION,
) -> list[tuple[int, int]]:
    """Pairs (i, j) with i < j, Cα distance ≤ cutoff, and |seq_i − seq_j| ≥ min_separation."""
    coords = np.vstack([res.ca for res in residues])
    numbers = np.array([res.resseq for res in residues], dtype=int)
    delta = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(delta * delta, axis=-1))
    sep = np.abs(numbers[:, None] - numbers[None, :])
    hits = np.argwhere((dist <= cutoff) & (sep >= min_separation))
    pairs = [(int(i), int(j)) for i, j in hits if i < j]
    return pairs


def contact_matrix(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=float)
    for i, j in pairs:
        matrix[i, j] = 1.0
        matrix[j, i] = 1.0
    return matrix


def mean_centroid_distance(residues: list[Residue], hp: list[str]) -> tuple[float, float]:
    """Mean Cα distance to the chain centroid, for H residues and for P residues."""
    coords = np.vstack([res.ca for res in residues])
    center = coords.mean(axis=0)
    dist = np.sqrt(np.sum((coords - center) ** 2, axis=1))
    labels = np.array(hp)
    h = dist[labels == "H"]
    p = dist[labels == "P"]
    h_mean = float(h.mean()) if h.size else float("nan")
    p_mean = float(p.mean()) if p.size else float("nan")
    return h_mean, p_mean
