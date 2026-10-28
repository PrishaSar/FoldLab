# FoldLab

FoldLab reads published crystal structures, scores them on the Kyte–Doolittle scale, and draws the same H/P classes on the real fold.

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

`demo-long` anneals HP-20 for 30,000 steps at seed 0 and writes `artifacts/demo-long.gif`: the search trajectory (chain, contacts, step, temperature, current energy, best energy), then a hold on the best fold. Each conformation stays on screen for 800 ms. The final hold stays up for 6 s.

```bash
python -m foldlab exhaustive HPHPHHPHPHHPPH --gif artifacts/hp14.gif
python -m foldlab anneal HPHPPHHPHPPHPHHPPHPH --seed 0 --steps 30000 --gif artifacts/hp20.gif
python -m foldlab anneal HHPPHPPHPPHPPHPPHPPHPPHH --seed 0
```

`exhaustive` refuses a chain longer than 14. Phase 1 uses NumPy and matplotlib. Pillow comes in with matplotlib and encodes the GIF; if it is missing, the writer saves an `.mp4` with the same stem. Phase 2 also uses BioPython and py3Dmol.

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

## Phase 2

`python -m foldlab phase2` reads the cached files `data/1lyz.pdb` and `data/2hhb.pdb` and writes four figures under `artifacts/`. If a cache file is missing it is downloaded from RCSB once. The 3Dmol script is cached at `data/3dmol-min.js` and inlined into the HTML, so the page opens on its own.

Hen egg-white lysozyme is 1LYZ. A contact is a Cα pair at most 8 Å apart with sequence separation at least 4. On this file that is **311** contacts. A residue is H when its Kyte–Doolittle score is greater than 0, and P otherwise. 1LYZ has 45 H and 84 P. Mean Cα distance to the centroid is 11.28 Å for H and 14.08 Å for P, so the hydrophobic set sits closer to the core. HELIX and SHEET records label 35 helix residues, 20 sheet residues, and 74 coil.

Sperm whale myoglobin is 1MBN, drawn with that same contact rule. Chain A has 153 residues and **262** contacts. HELIX and SHEET records label 118 helix residues and no sheet. Of the 262 contacts, 147 join residues only 4 to 6 apart. The side-by-side figure is `artifacts/contact-compare.png`.

The sickle-cell figure uses 2HHB, deoxy hemoglobin A, where both β chains still have glutamate at residue 6. Valine scores 4.2 on the same scale, so the class of that site would change from P to H. The drawing relabels the published site. It does not rebuild a side chain, and it is not a recipe for making the variant.

```bash
python -m foldlab phase2
```

## Energy model

`python -m foldlab.learn` builds `data/hp_labels.csv` and trains a small transformer that predicts the lattice energy. Lengths 8, 10, 12, and 14 are labeled by exhaustive search. Lengths 16 and 20 are labeled by annealing for 4,000 steps at seed 0, which is the best energy seen, not a proven optimum. The hold-out uses split seed 0. The binary view calls a chain low-energy when its label is at or below the training-set median. PyTorch is the extra dependency. The unit test loads the saved csv and does not retrain.

```bash
python -m foldlab.learn
```

## Local screen

```bash
python -m foldlab.screen
```

Opens [http://127.0.0.1:8765](http://127.0.0.1:8765). The protein box takes a 4-character code, such as `1LYZ` or `1MBN`, downloads that structure once, and draws the contact map, the oiliness chart, and a 3D view. Add a chain when the file has several, for example `2HHB B`. The other buttons run the same short fold, the same long fold, and the same Phase 2 pictures. Transformer score shows the saved test numbers. Train again is a separate button and is the only one that trains.

## Out of scope

No expression, purification, synthesis, or wet-lab mutagenesis. No database or auth. The screen is only this computer.

