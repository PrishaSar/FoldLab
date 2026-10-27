"""The energy transformer runs, and the training-set baseline is well defined."""

import unittest

import torch

from foldlab.learn import (
    FEATURE_NAMES,
    EnergyTransformer,
    baselines,
    load_dataset,
    split_indices,
    tokenize,
)


class LearnTests(unittest.TestCase):
    def test_forward_pass_and_baseline(self):
        rows = load_dataset()
        train_idx, _test_idx = split_indices(len(rows))
        train_energy = [rows[i]["energy"] for i in train_idx]
        reference = baselines(torch.tensor(train_energy).numpy())
        self.assertIn("mean_energy", reference)
        self.assertIn("majority_class", reference)
        self.assertLessEqual(reference["threshold"], max(train_energy))

        tokens, mask = tokenize([row["sequence"] for row in rows[:4]])
        features = torch.zeros(len(tokens), len(FEATURE_NAMES))
        model = EnergyTransformer()
        model.eval()
        with torch.no_grad():
            predicted = model(torch.as_tensor(tokens), features, torch.as_tensor(mask))
        self.assertEqual(tuple(predicted.shape), (len(tokens),))
        self.assertTrue(torch.isfinite(predicted).all())


if __name__ == "__main__":
    unittest.main()
