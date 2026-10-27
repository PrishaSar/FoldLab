"""Supervised energy prediction on HP sequences.

Short chains are labeled by exhaustive search. Longer chains are labeled by
simulated annealing with one fixed seed and step count. The model is a small
transformer over H/P tokens plus a few sequence counts.

Run from the repository root:

    python -m foldlab.learn
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from foldlab.search import EXHAUSTIVE_MAX_LENGTH, exhaustive_search, simulated_annealing

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "hp_labels.csv"

DATA_SEED = 0
SPLIT_SEED = 0
TRAIN_SEED = 0
TEST_FRACTION = 0.2

SHORT_LENGTHS = (8, 10, 12, 14)
SHORT_PER_LENGTH = 48
LONG_LENGTHS = (16, 20)
LONG_PER_LENGTH = 36
ANNEAL_SEED = 0
ANNEAL_STEPS = 4000

D_MODEL = 32
NHEAD = 4
N_LAYERS = 2
FEEDFORWARD = 64
EPOCHS = 80
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_LENGTH = 20

FEATURE_NAMES = (
    "h_fraction",
    "h_count",
    "p_count",
    "longest_h_run",
    "n_h_runs",
)


def sequence_features(sequence: str) -> dict[str, float]:
    """Counts that summarize an H/P string. The string itself is the token input."""
    h_count = sequence.count("H")
    p_count = sequence.count("P")
    longest = 0
    runs = 0
    current = 0
    for char in sequence:
        if char == "H":
            current += 1
            longest = max(longest, current)
        else:
            if current:
                runs += 1
            current = 0
    if current:
        runs += 1
    return {
        "h_fraction": h_count / len(sequence),
        "h_count": float(h_count),
        "p_count": float(p_count),
        "longest_h_run": float(longest),
        "n_h_runs": float(runs),
    }


def _random_sequences(rng: np.random.Generator, length: int, count: int, seen: set[str]) -> list[str]:
    sequences = []
    while len(sequences) < count:
        bits = rng.integers(0, 2, size=length)
        sequence = "".join("H" if bit else "P" for bit in bits)
        if sequence in seen:
            continue
        seen.add(sequence)
        sequences.append(sequence)
    return sequences


def generate_dataset(path: Path = DATASET_PATH) -> Path:
    """Label random chains and write the csv. Exhaustive labels are optima."""
    rng = np.random.default_rng(DATA_SEED)
    seen: set[str] = set()
    rows = []
    for length in SHORT_LENGTHS:
        for sequence in _random_sequences(rng, length, SHORT_PER_LENGTH, seen):
            result = exhaustive_search(sequence)
            rows.append(_row(sequence, result.energy, "exhaustive", "", ""))
    for length in LONG_LENGTHS:
        for sequence in _random_sequences(rng, length, LONG_PER_LENGTH, seen):
            result = simulated_annealing(sequence, seed=ANNEAL_SEED, steps=ANNEAL_STEPS)
            rows.append(
                _row(sequence, result.energy, "annealing", str(ANNEAL_SEED), str(ANNEAL_STEPS))
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "length",
        *FEATURE_NAMES,
        "energy",
        "method",
        "anneal_seed",
        "anneal_steps",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(sequence: str, energy: int, method: str, anneal_seed: str, anneal_steps: str) -> dict:
    features = sequence_features(sequence)
    return {
        "sequence": sequence,
        "length": len(sequence),
        **features,
        "energy": energy,
        "method": method,
        "anneal_seed": anneal_seed,
        "anneal_steps": anneal_steps,
    }


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["length"] = int(row["length"])
        row["energy"] = float(row["energy"])
        for name in FEATURE_NAMES:
            row[name] = float(row[name])
    return rows


def split_indices(n: int, test_fraction: float = TEST_FRACTION, seed: int = SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(n)
    rng.shuffle(order)
    n_test = max(1, int(round(n * test_fraction)))
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def binary_labels(energies: np.ndarray, threshold: float) -> np.ndarray:
    """1 when the energy is at or below the threshold (more contacts)."""
    return (energies <= threshold).astype(np.int64)


def baselines(train_energy: np.ndarray) -> dict[str, float]:
    """Training-mean energy, training-median threshold, and majority class."""
    threshold = float(np.median(train_energy))
    labels = binary_labels(train_energy, threshold)
    n_low = int(labels.sum())
    n_high = int(len(labels) - n_low)
    majority = 1 if n_low >= n_high else 0
    return {
        "mean_energy": float(train_energy.mean()),
        "threshold": threshold,
        "majority_class": majority,
        "n_low": n_low,
        "n_high": n_high,
    }


def tokenize(sequences: list[str], max_length: int = MAX_LENGTH) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((len(sequences), max_length), dtype=np.int64)
    mask = np.ones((len(sequences), max_length), dtype=bool)
    for i, sequence in enumerate(sequences):
        for j, char in enumerate(sequence):
            tokens[i, j] = 1 if char == "H" else 2
            mask[i, j] = False
    return tokens, mask


def numeric_matrix(rows: list[dict]) -> np.ndarray:
    return np.array([[row[name] for name in FEATURE_NAMES] for row in rows], dtype=np.float32)


class EnergyTransformer(nn.Module):
    """Encoder over H/P tokens. Numeric counts are concatenated after pooling."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        layers: int = N_LAYERS,
        feedforward: int = FEEDFORWARD,
        n_features: int = len(FEATURE_NAMES),
        max_length: int = MAX_LENGTH,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(3, d_model)
        self.position = nn.Embedding(max_length, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=feedforward,
            dropout=0.1,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Sequential(
            nn.Linear(d_model + n_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens: torch.Tensor, features: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)[None, :, :]
        hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        keep = (~pad_mask).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.head(torch.cat([pooled, features], dim=1)).squeeze(-1)


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    return (train - mean) / std, (test - mean) / std


def train_and_evaluate(rows: list[dict] | None = None) -> dict[str, float]:
    rows = rows if rows is not None else load_dataset()
    train_idx, test_idx = split_indices(len(rows))
    train_rows = [rows[i] for i in train_idx]
    test_rows = [rows[i] for i in test_idx]
    train_energy = np.array([row["energy"] for row in train_rows], dtype=np.float32)
    test_energy = np.array([row["energy"] for row in test_rows], dtype=np.float32)
    reference = baselines(train_energy)
    threshold = reference["threshold"]
    train_binary = binary_labels(train_energy, threshold)
    test_binary = binary_labels(test_energy, threshold)

    train_num, test_num = _standardize(numeric_matrix(train_rows), numeric_matrix(test_rows))
    train_tokens, train_mask = tokenize([row["sequence"] for row in train_rows])
    test_tokens, test_mask = tokenize([row["sequence"] for row in test_rows])

    torch.manual_seed(TRAIN_SEED)
    model = EnergyTransformer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    tokens_t = torch.as_tensor(train_tokens)
    mask_t = torch.as_tensor(train_mask)
    features_t = torch.as_tensor(train_num, dtype=torch.float32)
    target_t = torch.as_tensor(train_energy)
    model.train()
    for _epoch in range(EPOCHS):
        order = torch.randperm(len(train_rows))
        for start in range(0, len(train_rows), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            pred = model(tokens_t[batch], features_t[batch], mask_t[batch])
            loss = loss_fn(pred, target_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predicted = model(
            torch.as_tensor(test_tokens),
            torch.as_tensor(test_num, dtype=torch.float32),
            torch.as_tensor(test_mask),
        ).numpy()
    mae = float(np.mean(np.abs(predicted - test_energy)))
    baseline_pred = np.full_like(test_energy, reference["mean_energy"])
    baseline_mae = float(np.mean(np.abs(baseline_pred - test_energy)))
    pred_binary = binary_labels(predicted, threshold)
    accuracy = float(np.mean(pred_binary == test_binary))
    majority = np.full_like(test_binary, reference["majority_class"])
    baseline_accuracy = float(np.mean(majority == test_binary))
    return {
        "n": float(len(rows)),
        "n_train": float(len(train_rows)),
        "n_test": float(len(test_rows)),
        "threshold": threshold,
        "mean_energy": reference["mean_energy"],
        "majority_class": float(reference["majority_class"]),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
    }


def main() -> None:
    if any(length > EXHAUSTIVE_MAX_LENGTH for length in SHORT_LENGTHS):
        raise SystemExit("short chains must stay within exhaustive search")
    path = generate_dataset()
    metrics = train_and_evaluate(load_dataset(path))
    print(f"dataset:            {path}")
    print(f"n:                  {int(metrics['n'])}")
    print(f"train / test:       {int(metrics['n_train'])} / {int(metrics['n_test'])}")
    print(f"threshold:          {metrics['threshold']:.4f}")
    print(f"mae:                {metrics['mae']:.4f}")
    print(f"baseline_mae:       {metrics['baseline_mae']:.4f}")
    print(f"accuracy:           {metrics['accuracy']:.4f}")
    print(f"baseline_accuracy:  {metrics['baseline_accuracy']:.4f}")
    print(f"majority_class:     {int(metrics['majority_class'])}")
    print(f"train_mean_energy:  {metrics['mean_energy']:.4f}")


if __name__ == "__main__":
    main()
