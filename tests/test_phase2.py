"""Kyte–Doolittle scores, the H/P threshold, and the cached 1LYZ contact count."""

import unittest

from foldlab.kd import HP_THRESHOLD, hp_class, kd_score
from foldlab.structure import (
    CONTACT_CUTOFF,
    DATA_DIR,
    MIN_SEPARATION,
    ca_contacts,
    load_chains,
)


class KyteDoolittleTests(unittest.TestCase):
    def test_published_scores(self):
        self.assertEqual(kd_score("I"), 4.5)
        self.assertEqual(kd_score("V"), 4.2)
        self.assertEqual(kd_score("A"), 1.8)
        self.assertEqual(kd_score("G"), -0.4)
        self.assertEqual(kd_score("E"), -3.5)
        self.assertEqual(kd_score("R"), -4.5)
        self.assertEqual(kd_score("GLU"), -3.5)
        self.assertEqual(kd_score("VAL"), 4.2)

    def test_hp_threshold(self):
        self.assertEqual(HP_THRESHOLD, 0.0)
        self.assertEqual(hp_class("A"), "H")
        self.assertEqual(hp_class("V"), "H")
        self.assertEqual(hp_class("I"), "H")
        self.assertEqual(hp_class("G"), "P")
        self.assertEqual(hp_class("E"), "P")
        self.assertEqual(hp_class("K"), "P")


class ContactMapTests(unittest.TestCase):
    def test_cached_lysozyme_contacts(self):
        chains = load_chains(DATA_DIR / "1lyz.pdb")
        residues = chains["A"]
        pairs = ca_contacts(residues)
        self.assertEqual(CONTACT_CUTOFF, 8.0)
        self.assertEqual(MIN_SEPARATION, 4)
        self.assertEqual(len(residues), 129)
        self.assertEqual(len(pairs), 311)
        by_number = {res.resseq: res.ss for res in residues}
        self.assertEqual(by_number[10], "H")
        self.assertEqual(by_number[1], "E")
        self.assertEqual(by_number[20], "C")

    def test_cached_myoglobin_contacts(self):
        chains = load_chains(DATA_DIR / "1mbn.pdb")
        residues = chains["A"]
        pairs = ca_contacts(residues)
        seps = [abs(residues[i].resseq - residues[j].resseq) for i, j in pairs]
        self.assertEqual(len(residues), 153)
        self.assertEqual(len(pairs), 262)
        self.assertEqual(sum(sep <= 6 for sep in seps), 147)
        self.assertEqual(sum(res.ss == "H" for res in residues), 118)
        self.assertEqual(sum(res.ss == "E" for res in residues), 0)


if __name__ == "__main__":
    unittest.main()
