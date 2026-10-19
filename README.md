# FoldLab

FoldLab folds a hydrophobic–polar (HP) chain on the 2D square lattice. Phase 1 searches that toy model and writes an animation of the fold. It does not read PDB files or score real proteins.

A chain is a string of `H` and `P`. Beads occupy distinct lattice sites, and successive beads are orthogonal neighbors. The energy is −1 for every pair of H beads that sit on orthogonal neighboring sites and are not consecutive in the sequence. A straight chain scores 0. Lower is better.

Chains of length **14 or shorter** are solved by exhaustive enumeration. Longer chains are folded by simulated annealing. A chain of \(N\) beads is an \((N-1)\)-step self-avoiding walk, and that count grows exponentially (Levinthal). Enumeration fixes the first bead at the origin and the first step along \(+x\), which does not change the lowest energy. Length 16 is already past a short exact search, so it is timed by the bench and is not the default.

Annealing is deterministic for a given sequence, seed, and step count. The reported fold is the best energy seen.

## Sequences

| Name | Sequence | How Phase 1 folds it |
| --- | --- | --- |
| 14.1 | `HPHPHHPHPHHPPH` | Exhaustive. The designing 14-mer from Schiemann, Bachmann, and Janke, arXiv:0710.4074. |
| HP-20 | `HPHPPHHPHPPHPHHPPHPH` | Annealing. Unger and Moult, J. Mol. Biol. 235:75 (1993). |
| HP-24 | `HHPPHPPHPPHPPHPPHPPHPPHH` | Annealing, same source. Not the default clip. |

Energies in the measured section are the ones this package computed. They are not copied from papers.

## Run Phase 1

Python 3.12 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m foldlab demo-short
python -m foldlab demo-long
python -m foldlab bench
```

`demo-short` enumerates the 14-mer and writes `artifacts/demo-short.gif`: the optimal chain drawn one bead at a time, then its H–H contacts.

`demo-long` anneals HP-20 for 30,000 steps at seed 0 and writes `artifacts/demo-long.gif`: the search trajectory (chain, contacts, step, temperature, current energy, best energy), then a hold on the best fold.

```bash
python -m foldlab exhaustive HPHPHHPHPHHPPH --gif artifacts/hp14.gif
python -m foldlab anneal HPHPPHHPHPPHPHHPPHPH --seed 0 --steps 30000 --gif artifacts/hp20.gif
python -m foldlab anneal HHPPHPPHPPHPPHPPHPPHPPHH --seed 0
```

`exhaustive` refuses a chain longer than 14. Dependencies are NumPy and matplotlib only. Pillow comes in with matplotlib and encodes the GIF; if it is missing, the writer saves an `.mp4` with the same stem.

## Measured on this machine

The enumeration table below is filled from `python -m foldlab bench` (all-H chains, so every bead can score). Counts are symmetry-reduced conformations. Seconds include scoring every conformation. Demo energies are filled from `demo-short` and `demo-long` on the same machine.

_Not yet recorded._

## Out of scope

Phase 2 is not in this tree: no PDB, BioPython, py3Dmol, lysozyme, sickle-cell hemoglobin, contact maps, or Kyte–Doolittle hydropathy. No database, auth, or web service.
