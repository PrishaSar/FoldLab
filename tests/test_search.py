"""Enumeration counts, optima of the tiny chains, and annealing determinism."""

import unittest

import numpy as np

from foldlab.cli import main
from foldlab.model import energy, is_self_avoiding
from foldlab.search import (
    EXHAUSTIVE_MAX_LENGTH,
    exhaustive_search,
    iter_scored_conformations,
    simulated_annealing,
)


class ExhaustiveTests(unittest.TestCase):
    def test_cutoff_is_14(self):
        self.assertEqual(EXHAUSTIVE_MAX_LENGTH, 14)

    def test_symmetry_reduced_counts(self):
        # c_{N-1}/4 for the square lattice, first step fixed along +x.
        expected = {4: 9, 6: 71, 8: 543, 10: 4067}
        for length, count in expected.items():
            result = exhaustive_search("H" * length)
            self.assertEqual(result.n_conformations, count, msg=f"N={length}")

    def test_hhhh_optimum(self):
        result = exhaustive_search("HHHH")
        self.assertEqual(result.energy, -1)
        self.assertEqual(energy(result.sequence, result.coords), -1)
        self.assertEqual(tuple(result.coords[0]), (0, 0))
        self.assertEqual(tuple(result.coords[1]), (1, 0))

    def test_hairpin_optimum(self):
        result = exhaustive_search("HHPPHH")
        self.assertEqual(result.energy, -2)
        self.assertEqual(energy(result.sequence, result.coords), -2)
        self.assertTrue(is_self_avoiding(result.coords))

    def test_incremental_energy_matches_spec(self):
        sequence = "HHPPHH"
        scored = iter_scored_conformations(sequence)
        self.assertEqual(len(scored), 71)
        for coords, incremental in scored:
            self.assertEqual(incremental, energy(sequence, coords))


class AnnealTests(unittest.TestCase):
    def test_same_seed_reproduces_the_fold(self):
        sequence = "HPHPPHHPHP"
        first = simulated_annealing(sequence, seed=1, steps=200)
        second = simulated_annealing(sequence, seed=1, steps=200)
        self.assertEqual(first.energy, second.energy)
        self.assertTrue(np.array_equal(first.coords, second.coords))
        self.assertEqual(len(first.trajectory), len(second.trajectory))
        for left, right in zip(first.trajectory, second.trajectory):
            self.assertEqual(left.step, right.step)
            self.assertEqual(left.energy, right.energy)
            self.assertTrue(np.array_equal(left.coords, right.coords))
        self.assertTrue(is_self_avoiding(first.coords))
        self.assertLessEqual(first.energy, 0)
        self.assertEqual(first.energy, energy(sequence, first.coords))

    def test_cli_rejects_exhaustive_past_the_cutoff(self):
        with self.assertRaises(SystemExit) as caught:
            main(["exhaustive", "H" * (EXHAUSTIVE_MAX_LENGTH + 1)])
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
