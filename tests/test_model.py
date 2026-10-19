"""Hand-checked HP contacts and the public energy function."""

import unittest

from foldlab.model import contacts, energy, is_self_avoiding, parse_sequence


class EnergyTests(unittest.TestCase):
    def test_square_hhhh_has_one_contact(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertEqual(contacts("HHHH", coords), [(0, 3)])
        self.assertEqual(energy("HHHH", coords), -1)

    def test_straight_hhhh_has_no_contact(self):
        coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
        self.assertEqual(contacts("HHHH", coords), [])
        self.assertEqual(energy("HHHH", coords), 0)

    def test_hhppph_hairpin_contacts(self):
        coords = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]
        self.assertEqual(contacts("HHPPHH", coords), [(0, 5), (1, 4)])
        self.assertEqual(energy("HHPPHH", coords), -2)

    def test_phph_square_is_not_an_hh_contact(self):
        coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
        self.assertEqual(contacts("PHPH", coords), [])
        self.assertEqual(energy("PHPH", coords), 0)

    def test_diagonal_step_is_rejected(self):
        with self.assertRaises(ValueError):
            energy("HH", [(0, 0), (1, 1)])

    def test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            energy("HHH", [(0, 0), (1, 0), (0, 0)])

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            energy("HH", [(0, 0), (1, 0), (2, 0)])

    def test_parse_sequence(self):
        self.assertEqual(parse_sequence(" h pH "), "HPH")
        with self.assertRaises(ValueError):
            parse_sequence("HPX")
        with self.assertRaises(ValueError):
            parse_sequence("   ")

    def test_self_avoiding_flag(self):
        self.assertTrue(is_self_avoiding([(0, 0), (1, 0), (1, 1)]))
        self.assertFalse(is_self_avoiding([(0, 0), (1, 0), (1, 0)]))


if __name__ == "__main__":
    unittest.main()
