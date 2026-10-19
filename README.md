# FoldLab

FoldLab folds a hydrophobic–polar (HP) chain on the 2D square lattice. Phase 1 searches that toy model and writes an animation of the fold. It does not read PDB files or score real proteins.

A chain is a string of `H` and `P`. Beads occupy distinct lattice sites, and successive beads are orthogonal neighbors. The energy is −1 for every pair of H beads that sit on orthogonal neighboring sites and are not consecutive in the sequence. A straight chain scores 0. Lower is better.

Chains of length **14 or shorter** are solved by exhaustive enumeration. Longer chains are folded by simulated annealing. A chain of \(N\) beads is an \((N-1)\)-step self-avoiding walk, and that count grows exponentially (Levinthal). Enumeration fixes the first bead at the origin and the first step along \(+x\), which does not change the lowest energy. Length 16 is already past a short exact search, so it is timed by the bench and is not the default.

Annealing is deterministic for a given sequence, seed, and step count. The reported fold is the best energy seen.

## Sequences

| Name | Sequence | How Phase 1 folds it |
| --- | --- | --- |
| 14.1 | `HPHPHHPHPHHPPH` | Exhaustive. Sequence 14.1 from Bachmann and Janke, arXiv:0710.4074, folded here on the square lattice. |
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

`python -m foldlab bench` enumerates all-H chains, so every bead can score. Counts are symmetry-reduced conformations (first bead at the origin, first step along +x), which is \(c_{N-1}/4\). Seconds are wall-clock time for that count and the scoring, from one run, rounded to milliseconds. Length 8 finished in under 1 ms.

| Length | Conformations | All-H energy | Seconds |
| --- | ---: | ---: | ---: |
| 8 | 543 | −3 | < 0.001 |
| 10 | 4,067 | −4 | 0.003 |
| 12 | 30,073 | −6 | 0.022 |
| 14 | 220,375 | −7 | 0.165 |
| 16 | 1,604,149 | −9 | 1.222 |

Two extra beads take the count from 220,375 at length 14 to 1,604,149 at length 16, about 7.3×, which is why exhaustive search stops at 14. Length 16 all-H scored −9. A filled 4×4 has 24 lattice edges and the chain uses 15 of them as bonds, so nine is the most contacts that square can hold.

Demos from the same machine:

- `demo-short` on 14.1: exhaustive, energy **−6**, 220,375 conformations, 0.138 s. Six H–H contacts.
- `demo-long` on HP-20: annealing, seed 0, 30,000 steps, energy **−8**, 1.554 s. Eight H–H contacts. The same schedule at seeds 1 and 4 scored −9.

## Out of scope

Phase 2 is not in this tree: no PDB, BioPython, py3Dmol, lysozyme, sickle-cell hemoglobin, contact maps, or Kyte–Doolittle hydropathy. No database, auth, or web service.
