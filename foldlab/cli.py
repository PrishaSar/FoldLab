"""Command line for the Phase 1 demos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NoReturn

from foldlab.animate import save_fold_gif, save_trajectory_gif
from foldlab.model import contacts, parse_sequence
from foldlab.search import (
    DEFAULT_ANNEAL_STEPS,
    EXHAUSTIVE_MAX_LENGTH,
    exhaustive_search,
    simulated_annealing,
)
from foldlab.sequences import DESIGNING_14, HP20, HP24

BENCH_LENGTHS = (8, 10, 12, 14, 16)
# GIF timing for demo-long. Matplotlib stores int(1000 / fps) milliseconds,
# then the merged final hold is set on its own.
DEMO_LONG_FRAME_MS = 800
DEMO_LONG_HOLD_MS = 6000


def _print_result(result) -> None:
    print(f"sequence:       {result.sequence}")
    print(f"length:         {len(result.sequence)}")
    print(f"method:         {result.method}")
    if result.seed is not None:
        print(f"seed:           {result.seed}")
    if result.steps is not None:
        print(f"steps:          {result.steps}")
    if result.n_conformations is not None:
        print(f"conformations:  {result.n_conformations}")
    print(f"energy:         {result.energy}")
    pairs = contacts(result.sequence, result.coords)
    print(f"contacts:       {len(pairs)}  {pairs}")
    folded = " ".join(f"({int(x)},{int(y)})" for x, y in result.coords)
    print(f"coords:         {folded}")
    if result.elapsed_s is not None:
        print(f"seconds:        {result.elapsed_s:.3f}")


def _die(message: str) -> NoReturn:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def _exhaustive(sequence: str, gif: Path | None, title: str | None = None):
    try:
        parsed = parse_sequence(sequence)
    except ValueError as exc:
        _die(str(exc))
    if len(parsed) > EXHAUSTIVE_MAX_LENGTH:
        _die(
            f"length {len(parsed)} is above the exhaustive cutoff "
            f"({EXHAUSTIVE_MAX_LENGTH}); use anneal"
        )
    result = exhaustive_search(parsed)
    _print_result(result)
    if gif is not None:
        written = save_fold_gif(gif, result.sequence, result.coords, title=title or parsed)
        print(f"wrote:          {written}")
    return result


def _anneal(
    sequence: str,
    seed: int,
    steps: int,
    gif: Path | None,
    title: str | None = None,
    frame_ms: int | None = None,
    hold_ms: int | None = None,
):
    try:
        result = simulated_annealing(sequence, seed=seed, steps=steps)
    except ValueError as exc:
        _die(str(exc))
    _print_result(result)
    if gif is not None:
        fps = 8 if frame_ms is None else 1000 / frame_ms
        written = save_trajectory_gif(
            gif,
            result,
            title=title or result.sequence,
            fps=fps,
            hold_ms=hold_ms,
        )
        print(f"wrote:          {written}")
    return result


def _bench() -> list[dict]:
    rows = []
    print(f"{'length':>6}  {'conformations':>14}  {'energy':>6}  {'seconds':>8}")
    for length in BENCH_LENGTHS:
        result = exhaustive_search("H" * length)
        row = {
            "length": length,
            "sequence": result.sequence,
            "conformations": result.n_conformations,
            "energy": result.energy,
            "seconds": round(result.elapsed_s, 3),
        }
        rows.append(row)
        print(
            f"{length:6d}  {result.n_conformations:14d}  {result.energy:6d}  "
            f"{result.elapsed_s:8.3f}"
        )
    out = Path("artifacts") / "bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote: {out}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foldlab",
        description="FoldLab Phase 1: 2D HP lattice folds.",
        epilog=(
            f"Named sequences: 14.1 {DESIGNING_14}; HP-20 {HP20}; HP-24 {HP24}."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    exhaustive = sub.add_parser("exhaustive", help="exact search for a chain of length <= 14")
    exhaustive.add_argument("sequence")
    exhaustive.add_argument("--gif", type=Path)

    anneal = sub.add_parser("anneal", help="simulated annealing, deterministic given --seed")
    anneal.add_argument("sequence")
    anneal.add_argument("--seed", type=int, default=0)
    anneal.add_argument("--steps", type=int, default=DEFAULT_ANNEAL_STEPS)
    anneal.add_argument("--gif", type=Path)

    sub.add_parser("demo-short", help="exhaustive fold of sequence 14.1")
    sub.add_parser("demo-long", help="anneal HP-20 at seed 0 for 30000 steps")
    sub.add_parser("bench", help="time exhaustive search at lengths 8, 10, 12, 14, and 16")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "exhaustive":
        _exhaustive(args.sequence, args.gif)
    elif args.cmd == "anneal":
        _anneal(args.sequence, args.seed, args.steps, args.gif)
    elif args.cmd == "demo-short":
        _exhaustive(DESIGNING_14, Path("artifacts/demo-short.gif"), title="14.1")
    elif args.cmd == "demo-long":
        _anneal(
            HP20,
            seed=0,
            steps=DEFAULT_ANNEAL_STEPS,
            gif=Path("artifacts/demo-long.gif"),
            title="HP-20",
            frame_ms=DEMO_LONG_FRAME_MS,
            hold_ms=DEMO_LONG_HOLD_MS,
        )
    elif args.cmd == "bench":
        _bench()
    else:
        parser.error(f"unknown command {args.cmd}")
    return 0

