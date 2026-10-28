"""A protein code loads one chain with the same contact rule as lysozyme."""

import tempfile
import unittest
from pathlib import Path

from foldlab.intake import ProteinError, _molecule_name, build_protein, parse_code
from foldlab.structure import DATA_DIR


class ParseTests(unittest.TestCase):
    def test_code_and_chain(self):
        self.assertEqual(parse_code("1lyz"), ("1LYZ", None))
        self.assertEqual(parse_code("2HHB B"), ("2HHB", "B"))
        self.assertEqual(parse_code("2hhb:d"), ("2HHB", "D"))

    def test_chain_name_follows_the_file(self):
        from foldlab.intake import _molecule_name

        text = (DATA_DIR / "2hhb.pdb").read_text()
        self.assertEqual(_molecule_name(text, "B"), "Hemoglobin (Deoxy) (Beta Chain)")
        self.assertEqual(_molecule_name(text, "A"), "Hemoglobin (Deoxy) (Alpha Chain)")
        lysozyme = (DATA_DIR / "1lyz.pdb").read_text()
        self.assertEqual(_molecule_name(lysozyme, "A"), "Hen Egg White Lysozyme")
        with self.assertRaises(ProteinError):
            parse_code("lysozyme")


class BuildTests(unittest.TestCase):
    def test_lysozyme_code_matches_the_cached_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = build_protein("1LYZ", Path(tmp))
            self.assertEqual(info["code"], "1LYZ")
            self.assertEqual(info["chain"], "A")
            self.assertEqual(info["residues"], 129)
            self.assertEqual(info["contacts"], 311)
            self.assertEqual(info["n_h"], 45)
            self.assertEqual(info["n_p"], 84)
            folder = Path(tmp) / "1lyz-a"
            view = (folder / "view.html").read_text()
            self.assertIn("3Dmol", view)
            self.assertGreater((folder / "contact.png").stat().st_size, 0)
            self.assertGreater((folder / "oiliness.png").stat().st_size, 0)

    def test_unknown_chain_names_the_real_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProteinError) as caught:
                build_protein("1LYZ Z", Path(tmp))
        self.assertIn("Chains in this file: A", str(caught.exception))
